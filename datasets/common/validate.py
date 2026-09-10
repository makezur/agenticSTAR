"""validate.py — check a capture dir against the capture/v1 contract.

    micromamba run -n artscript python -m datasets.common.validate <capture> \
        [--deep]

Fast checks (always): manifest schema, per-frame file existence, cameras.npz
shapes/finiteness, keyframes <-> frames agreement, depth stems ⊆ frame stems.
--deep additionally opens every image/mask/depth file and checks decoded sizes
against the manifest's image_size (slow on big captures).

Exit code 0 = valid; 1 = problems (each printed as 'ERROR: ...').
"""

import argparse
import os
import sys

import cv2
import numpy as np

from . import capture as cap
from . import capture_io


def _err(problems, msg):
    problems.append(msg)
    print(f"ERROR: {msg}")


def validate(root, deep=False):
    """Validate one capture dir. Returns a list of problem strings (empty = ok)."""
    problems = []
    try:
        c = cap.Capture(root)
    except cap.CaptureError as e:
        return [str(e)]

    m = c.manifest
    # -- manifest schema -------------------------------------------------------
    if m.get("dataset", "") == "":
        _err(problems, "manifest: missing/empty 'dataset'")
    if m.get("depth") not in cap.DEPTH_KINDS:
        _err(problems, f"manifest: depth {m.get('depth')!r} not in {cap.DEPTH_KINDS}")
    frames = c.frames
    if not frames:
        _err(problems, "manifest: empty 'frames'")
    if c.ref_frame not in frames:
        _err(problems, f"manifest: ref_frame {c.ref_frame!r} not in frames")
    if c.image_size is None:
        _err(problems, "manifest: missing 'image_size'")
    if len(set(cap.stem_of(f) for f in frames)) != len(frames):
        _err(problems, "manifest: duplicate frame stems")

    # -- per-frame files --------------------------------------------------------
    for fr in frames:
        if not os.path.isfile(c.frame_path(fr)):
            _err(problems, f"missing frame: {c.frame_path(fr)}")
        if not os.path.isfile(c.mask_path(fr)):
            _err(problems, f"missing object mask: {c.mask_path(fr)}")
    if c.hand_masks_dir:
        n_hand = sum(1 for fr in frames if c.hand_mask_path(fr))
        if n_hand == 0:
            _err(problems, f"{cap.MASK_HAND_DIR}/ exists but matches no frame")

    # -- tracking ----------------------------------------------------------------
    cams_path = os.path.join(c.tracking_dir, cap.CAMERAS_NPZ)
    if not os.path.isfile(cams_path):
        _err(problems, f"missing {cams_path}")
    else:
        cams = c.cameras()
        c2w, K = cams["c2w"], cams["intrinsics"]
        if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
            _err(problems, f"cameras.npz c2w shape {c2w.shape} != (K,4,4)")
        if K.ndim != 3 or K.shape[1:] != (3, 3):
            _err(problems, f"cameras.npz intrinsics shape {K.shape} != (K,3,3)")
        if c2w.shape[0] != K.shape[0]:
            _err(problems, f"cameras.npz c2w/intrinsics count mismatch "
                           f"({c2w.shape[0]} vs {K.shape[0]})")
        if not (np.isfinite(c2w).all() and np.isfinite(K).all()):
            _err(problems, "cameras.npz contains non-finite values")
        keyframes = cap.load_keyframes(c.tracking_dir)
        if keyframes is None:
            _err(problems, f"missing {os.path.join(c.tracking_dir, cap.KEYFRAMES_JSON)}")
        else:
            if len(keyframes) != c2w.shape[0]:
                _err(problems, f"keyframes ({len(keyframes)}) != cameras "
                               f"({c2w.shape[0]})")
            kf_stems = {cap.stem_of(k.get("frame_name", "")) for k in keyframes}
            for fr in frames:
                if cap.stem_of(fr) not in kf_stems:
                    _err(problems, f"frame {fr} has no keyframes.json entry")

    # -- depth --------------------------------------------------------------------
    depth_dir = os.path.join(c.root, cap.DEPTH_DIR)
    if c.depth_kind == "none":
        if os.path.isdir(depth_dir) and os.listdir(depth_dir):
            _err(problems, "manifest says depth='none' but depth/ is non-empty")
    else:
        if not c.depth_dir:
            _err(problems, f"manifest says depth={c.depth_kind!r} but no depth/ dir")
        else:
            frame_stems = {cap.stem_of(f) for f in frames}
            entries = [e for e in os.listdir(c.depth_dir) if e.endswith(".npz")]
            if not entries:
                _err(problems, "depth/ contains no npz files")
            for e in entries:
                if cap.stem_of(e) not in frame_stems:
                    _err(problems, f"depth/{e} has no matching frame")

    # -- deep: decode every raster -------------------------------------------------
    if deep and c.image_size is not None and not problems:
        W, H = c.image_size
        for fr in frames:
            img = cv2.imread(c.frame_path(fr), cv2.IMREAD_UNCHANGED)
            if img is None:
                _err(problems, f"unreadable frame: {c.frame_path(fr)}")
            elif img.shape[1] != W or img.shape[0] != H:
                _err(problems, f"{fr}: image {img.shape[1]}x{img.shape[0]} != "
                               f"manifest {W}x{H}")
            msk = cv2.imread(c.mask_path(fr), cv2.IMREAD_GRAYSCALE)
            if msk is None:
                _err(problems, f"unreadable mask: {c.mask_path(fr)}")
            elif msk.shape[1] != W or msk.shape[0] != H:
                _err(problems, f"{fr}: mask {msk.shape[1]}x{msk.shape[0]} != "
                               f"manifest {W}x{H}")
            dp = c.depth_path(fr)
            if dp:
                local, depth, conf, keep, _ = capture_io.load_depth(dp)
                if conf.shape != depth.shape or keep.shape != depth.shape:
                    _err(problems, f"{fr}: depth field shapes disagree")

    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", help="capture dir (contains capture.json)")
    p.add_argument("--deep", action="store_true",
                   help="also decode every image/mask/depth file")
    args = p.parse_args()

    problems = validate(args.capture, deep=args.deep)
    if problems:
        print(f"INVALID: {args.capture} ({len(problems)} problem(s))")
        sys.exit(1)
    c = cap.Capture(args.capture)
    print(f"OK: {args.capture} — dataset={c.dataset} frames={len(c.frames)} "
          f"depth={c.depth_kind} size={c.image_size} downscale={c.downscale}")


if __name__ == "__main__":
    main()
