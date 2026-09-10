#!/usr/bin/env python3
"""make_capture.py — frames + masks -> a capture/v1 dir, with Pi3X cameras.

The ONE converter for hand-made inputs. Give it a folder of video frames and
binary masks; it picks keyframes, runs Pi3X jointly over them for per-frame
camera poses + intrinsics, and writes the capture layout every harness tool
reads (datasets/common/capture.py). Runs in the `pi3x` micromamba env
(tools/install_pi3x.sh), from the repo root:

    micromamba run -n pi3x python tools/make_capture.py \\
        --src examples/garden_shears/input --out captures/garden_shears \\
        [--every N | --frames a.jpg,b.jpg] [--depth] [--fps 30] [--name NAME]

Input contract (--src):

    <src>/frames/<NNNNNN>.jpg|png      numeric stems = temporal order (required)
    <src>/mask_object/<NNNNNN>.png     binary object masks, stem-joined (required)
    <src>/mask_hand/<NNNNNN>.png       binary hand/occluder masks (optional)

The alternative names mask_object_1/ (object) and mask_object_2/ (hand) are
accepted too. A frame without an object mask is dropped with a warning.

Output (--out):

    capture.json                       manifest, depth = "none" unless --depth
    frames/ mask_object/ [mask_hand/]  the selected keyframes, copied
    tracking/cameras.npz               c2w (K,4,4) f32 OpenCV cam-to-world,
                                       intrinsics (K,3,3) f32 in Pi3X's pixel
                                       grid, size [Wp,Hp] i32 (the harness
                                       rescales K from `size` to the frame size)
    tracking/keyframes.json            [{view, frame_index, frame_name, time_s}]
    depth/<stem>.npz                   only with --depth: local (H,W,3), conf, keep

By default the capture carries CAMERAS ONLY: the agent poses the object against
the known camera motion and the masks (masks-only mode; run.sh turns depth
cost/report off automatically). --depth additionally keeps Pi3X's per-frame
pointmaps so the depth scorer and `analysis.measure_depth` seeding have
observed geometry to compare against.

Pi3X recipe (the one used to build the bundled example capture):
Pi3X.from_pretrained("yyfz233/Pi3X") with the multimodal head off, images
resized to a ~255k-pixel budget on a multiple-of-14 grid, bf16 autocast,
conf = sigmoid(conf), keep = conf > 0.1 minus depth/normal edges,
intrinsics recovered from the local rays with a centred principal point.
"""

import argparse
import json
import math
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from datasets.common import capture as cap  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
PIXEL_BUDGET = 255_000
CONF_KEEP = 0.1
EDGE_RTOL = 0.03
HF_MODEL = "yyfz233/Pi3X"


# --------------------------------------------------------------------------- #
# input layout
# --------------------------------------------------------------------------- #
def resolve_src(src):
    """(frames_dir, object_masks_dir, hand_masks_dir|None) of an input dir."""
    src = os.path.abspath(src)
    frames = os.path.join(src, "frames")
    if not os.path.isdir(frames):
        raise SystemExit(f"make_capture: no frames/ under {src}")
    obj = None
    for name in (cap.MASK_OBJECT_DIR, "mask_object_1", "masks"):
        if os.path.isdir(os.path.join(src, name)):
            obj = os.path.join(src, name)
            break
    if obj is None:
        raise SystemExit(f"make_capture: no mask_object/ under {src}")
    hand = None
    for name in (cap.MASK_HAND_DIR, "mask_object_2"):
        if os.path.isdir(os.path.join(src, name)):
            hand = os.path.join(src, name)
            break
    return frames, obj, hand


def list_frames(frames_dir):
    """Frame filenames in temporal order (integer stem order)."""
    names = [n for n in os.listdir(frames_dir)
             if n.lower().endswith(IMAGE_EXTS)]
    if not names:
        raise SystemExit(f"make_capture: no images in {frames_dir}")

    def key(n):
        stem = cap.stem_of(n)
        return (0, int(stem)) if stem.isdigit() else (1, stem)
    return sorted(names, key=key)


def frame_index_of(name):
    """The integer a numeric stem encodes, else None."""
    stem = cap.stem_of(name)
    return int(stem) if stem.isdigit() else None


def select_frames(all_frames, masks_dir, every=None, frames=None):
    """The keyframes to reconstruct: --frames, else every Nth by frame NUMBER
    (stride on the stem, like run.sh --every), else all. Frames without an
    object mask are dropped with a warning."""
    if frames:
        wanted = [f.strip() for f in frames.split(",") if f.strip()]
        known = set(all_frames)
        missing = [f for f in wanted if f not in known]
        if missing:
            raise SystemExit(f"make_capture: --frames not in frames/: {missing}")
        picked = wanted
    elif every and every > 1:
        picked = []
        for f in all_frames:
            idx = frame_index_of(f)
            if idx is None or idx % every == 0:
                picked.append(f)
    else:
        picked = list(all_frames)

    kept = []
    for f in picked:
        if os.path.isfile(cap.mask_for(masks_dir, f)):
            kept.append(f)
        else:
            print(f"warning: no object mask for {f}; dropped", file=sys.stderr)
    if not kept:
        raise SystemExit("make_capture: no keyframe has an object mask")
    return kept


# --------------------------------------------------------------------------- #
# Pi3X
# --------------------------------------------------------------------------- #
def _pi3_size(width, height, pixel_limit=PIXEL_BUDGET):
    """The (W, H) Pi3X sees: ~pixel_limit pixels on a multiple-of-14 grid."""
    scale = math.sqrt(pixel_limit / (width * height))
    target_w = max(1, round(width * scale / 14))
    target_h = max(1, round(height * scale / 14))
    while (target_w * 14) * (target_h * 14) > pixel_limit:
        if target_w / target_h > width / height:
            target_w -= 1
        else:
            target_h -= 1
    return max(1, target_w) * 14, max(1, target_h) * 14


def _load_images(paths):
    """(images (N,3,Hp,Wp) float tensor in [0,1], (Wp, Hp), (W, H))."""
    import torch
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    images = [Image.open(p).convert("RGB") for p in paths]
    width, height = images[0].size
    for p, im in zip(paths, images):
        if im.size != (width, height):
            raise SystemExit(f"make_capture: {p} is {im.size}, expected "
                             f"{(width, height)} (all frames must match)")
    size = _pi3_size(width, height)
    tensors = [pil_to_tensor(im.resize(size, Image.Resampling.LANCZOS))
               .float().div_(255.0) for im in images]
    return torch.stack(tensors), size, (width, height)


def run_pi3x(paths, device=None, checkpoint=None):
    """Pi3X over `paths` jointly -> dict(c2w, intrinsics, local, conf, keep,
    size). Arrays are numpy; c2w (K,4,4), intrinsics (K,3,3) in the (Wp,Hp)
    grid, local (K,Hp,Wp,3), conf/keep (K,Hp,Wp)."""
    import torch
    import torch.nn.functional as F

    pi3_repo = os.environ.get("PI3_REPO_DIR",
                              os.path.join(REPO, "third_party", "Pi3"))
    if os.path.isdir(os.path.join(pi3_repo, "pi3")) and pi3_repo not in sys.path:
        sys.path.insert(0, pi3_repo)
    try:
        from pi3.models.pi3x import Pi3X
        from pi3.utils.geometry import (depth_normal_edge,
                                        recover_intrinsic_from_rays_d)
    except ImportError as e:
        raise SystemExit(
            f"make_capture: Pi3 is not importable ({e}). Run "
            "tools/install_pi3x.sh, or point PI3_REPO_DIR at a yyfz/Pi3 checkout.")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    images, size, _ = _load_images(paths)
    checkpoint = checkpoint or os.environ.get("PI3X_CHECKPOINT")
    if checkpoint:
        from safetensors.torch import load_file
        model = Pi3X(use_multimodal=False)
        model.load_state_dict(load_file(checkpoint), strict=False)
    else:
        model = Pi3X.from_pretrained(HF_MODEL)
        model.disable_multimodal()
    model = model.eval().to(device)

    amp = device.type == "cuda"
    dtype = (torch.bfloat16 if amp and torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)
    with torch.inference_mode(), torch.amp.autocast(device.type, dtype=dtype,
                                                    enabled=amp):
        result = model(imgs=images.unsqueeze(0).to(device))

    local = result["local_points"][0].float()
    conf = torch.sigmoid(result["conf"][0, ..., 0].float())
    keep = conf > CONF_KEEP
    keep &= ~depth_normal_edge(local, rtol=EDGE_RTOL, mask=keep)
    rays = F.normalize(local, dim=-1)
    K = recover_intrinsic_from_rays_d(rays, force_center_principal_point=True)
    return {
        "c2w": result["camera_poses"][0].float().cpu().numpy(),
        "intrinsics": K.float().cpu().numpy(),
        "local": local.cpu().numpy(),
        "conf": conf.cpu().numpy(),
        "keep": keep.cpu().numpy(),
        "size": size,
    }


# --------------------------------------------------------------------------- #
# writing the capture
# --------------------------------------------------------------------------- #
def write_tracking(out, pi3, keyframes):
    tracking = os.path.join(out, cap.TRACKING_DIR)
    os.makedirs(tracking, exist_ok=True)
    np.savez(os.path.join(tracking, cap.CAMERAS_NPZ),
             c2w=np.asarray(pi3["c2w"], dtype=np.float32),
             intrinsics=np.asarray(pi3["intrinsics"], dtype=np.float32),
             size=np.asarray(pi3["size"], dtype=np.int32))
    with open(os.path.join(tracking, cap.KEYFRAMES_JSON), "w") as f:
        json.dump(keyframes, f, indent=2)
        f.write("\n")


def keyframes_for(frames, fps):
    """The keyframes.json entries: view = Pi3X view index, frame_index = the
    numeric stem (position when stems are not numeric), time_s = index / fps."""
    entries = []
    for view, name in enumerate(frames):
        idx = frame_index_of(name)
        if idx is None:
            idx = view
        entries.append({"view": view, "frame_index": idx, "frame_name": name,
                        "time_s": round(idx / fps, 4)})
    return entries


def copy_inputs(out, frames, frames_dir, masks_dir, hand_dir):
    os.makedirs(os.path.join(out, cap.FRAMES_DIR), exist_ok=True)
    os.makedirs(os.path.join(out, cap.MASK_OBJECT_DIR), exist_ok=True)
    have_hand = False
    for f in frames:
        stem = cap.stem_of(f)
        shutil.copy2(os.path.join(frames_dir, f), os.path.join(out, cap.FRAMES_DIR, f))
        shutil.copy2(cap.mask_for(masks_dir, f),
                     os.path.join(out, cap.MASK_OBJECT_DIR, stem + ".png"))
        if hand_dir:
            src = cap.mask_for(hand_dir, f)
            if os.path.isfile(src):
                os.makedirs(os.path.join(out, cap.MASK_HAND_DIR), exist_ok=True)
                shutil.copy2(src, os.path.join(out, cap.MASK_HAND_DIR, stem + ".png"))
                have_hand = True
    return have_hand


def write_depth(out, pi3, frames):
    for view, f in enumerate(frames):
        cap.save_depth(os.path.join(out, cap.DEPTH_DIR, cap.stem_of(f) + ".npz"),
                       pi3["local"][view], conf=pi3["conf"][view],
                       keep=pi3["keep"][view])


def image_size_of(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def make_capture(src, out, every=None, frames=None, depth=False, fps=30.0,
                 name=None, device=None, checkpoint=None, pi3_runner=run_pi3x):
    """Build the capture at `out`. Returns the manifest dict."""
    frames_dir, masks_dir, hand_dir = resolve_src(src)
    selected = select_frames(list_frames(frames_dir), masks_dir, every, frames)
    paths = [os.path.join(frames_dir, f) for f in selected]

    out = os.path.abspath(out)
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"make_capture: {out} exists and is not empty")
    os.makedirs(out, exist_ok=True)

    print(f"[make_capture] {len(selected)} keyframes: {selected[0]} .. {selected[-1]}")
    pi3 = pi3_runner(paths, device=device, checkpoint=checkpoint)
    if not bool(np.asarray(pi3["keep"]).any()):
        raise SystemExit("make_capture: Pi3X produced no confident geometry "
                         "(textureless / flat input?)")

    have_hand = copy_inputs(out, selected, frames_dir, masks_dir, hand_dir)
    write_tracking(out, pi3, keyframes_for(selected, fps))
    if depth:
        write_depth(out, pi3, selected)

    source = {"producer": "tools/make_capture.py", "src": src,
              "pi3x_model": HF_MODEL, "pi3x_size": list(pi3["size"]),
              "hand_masks": have_hand}
    if name:
        source["name"] = name
    if every:
        source["every"] = int(every)
    manifest = cap.write_manifest(
        out, dataset="pi3x", frames=selected,
        image_size=image_size_of(paths[0]), source=source,
        depth="pi3x" if depth else "none")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n", 1)[1])
    p.add_argument("--src", required=True, help="input dir: frames/ + mask_object/ [+ mask_hand/]")
    p.add_argument("--out", required=True, help="capture dir to create (must not exist / be empty)")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--every", type=int, help="keep every Nth frame by frame number")
    sel.add_argument("--frames", help="comma-separated frame filenames to keep")
    p.add_argument("--depth", action="store_true",
                   help="also write per-frame Pi3X pointmaps to depth/ (default: cameras only)")
    p.add_argument("--fps", type=float, default=30.0, help="source video fps for time_s (default 30)")
    p.add_argument("--name", help="capture name recorded in capture.json source")
    p.add_argument("--device", help="torch device (default cuda if available)")
    p.add_argument("--checkpoint", help="local Pi3X model.safetensors (default: HF hub yyfz233/Pi3X)")
    args = p.parse_args(argv)

    manifest = make_capture(args.src, args.out, every=args.every, frames=args.frames,
                            depth=args.depth, fps=args.fps, name=args.name,
                            device=args.device, checkpoint=args.checkpoint)
    print(f"[make_capture] wrote {len(manifest['frames'])} frames -> {args.out} "
          f"(depth: {manifest['depth']})")

    # validate with the harness's own checker (needs cv2 — present in the pi3x env)
    from datasets.common import validate
    problems = validate.validate(args.out, deep=True)
    if problems:
        raise SystemExit(f"make_capture: produced an INVALID capture "
                         f"({len(problems)} problems — see above)")
    print("[make_capture] capture validates OK. Next:")
    print(f"  ./run.sh --capture {args.out} --agent claude")


if __name__ == "__main__":
    main()
