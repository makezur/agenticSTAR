"""Shared render-pass primitives for shape and composite workflows.

This module owns only the neutral mechanics: resolve a multi-frame layout,
invoke ``render.sh`` into an iteration-owned pass directory, and construct the
standard per-frame composite command. Shape-only policy such as turntables,
depth scoring, aggregation, and finalization stays in ``shape_pass``.
"""

from collections import deque
import os
import subprocess
import sys
import time

from core import captures, run_layout


HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(HARNESS)
RENDER_SH = os.path.join(HARNESS, "render.sh")
COMPOSITE_MODULE = "analysis.viz.composite"


def resolve_layout(run_dir, args, tool_name="pass"):
    """Resolve layout.json plus the standard explicit CLI overrides."""
    manifest = run_layout.load_run_layout(run_dir)
    if manifest is not None:
        if manifest.get("kind") == "single":
            raise SystemExit(
                f"[{tool_name}] {os.path.join(run_dir, 'layout.json')} is a "
                "single-image run; this command drives a multi-frame pass.")
    elif not (args.frames_dir and args.masks_dir and args.frames):
        raise SystemExit(
            f"[{tool_name}] no {os.path.join(run_dir, 'layout.json')} and the "
            "layout was not fully given on the CLI. Pass --tracking, "
            "--frames-dir, --masks-dir, --frames, and --ref-frame as needed.")
    manifest = manifest or {}

    def pick(cli, key):
        return cli if cli else manifest.get(key, "")

    layout = {
        "capture": manifest.get("capture", ""),
        "frames_dir": pick(args.frames_dir, "frames_dir"),
        "masks_dir": pick(args.masks_dir, "masks_dir"),
        "hand_masks_dir": pick(args.hand_masks_dir, "hand_masks_dir"),
        "ref_frame": pick(args.ref_frame, "ref_frame"),
    }
    tracking = args.tracking or manifest.get("tracking", "")
    layout["tracking"] = tracking if tracking and os.path.isdir(tracking) else ""
    if args.frames:
        layout["frames"] = [
            frame.strip() for frame in args.frames.split(",") if frame.strip()]
    else:
        layout["frames"] = list(manifest.get("frames", []))
    if not layout["ref_frame"] and layout["frames"]:
        layout["ref_frame"] = layout["frames"][0]

    missing = [
        key for key in ("frames_dir", "masks_dir", "ref_frame")
        if not layout[key]]
    if not layout["frames"]:
        missing.append("frames")
    if missing:
        raise SystemExit(
            f"[{tool_name}] layout incomplete (missing: {', '.join(missing)}).")

    for key in ("capture", "tracking", "frames_dir", "masks_dir",
                "hand_masks_dir"):
        if layout[key]:
            layout[key] = os.path.abspath(layout[key])
    return layout


def check_inputs(layout, tool_name="pass"):
    """Fail before rendering when a requested source image or mask is absent."""
    problems = []
    for frame in layout["frames"]:
        image = os.path.join(layout["frames_dir"], frame)
        mask = captures.mask_for(layout["masks_dir"], frame)
        if not os.path.isfile(image):
            problems.append(f"  image missing: {image}")
        if not os.path.isfile(mask):
            problems.append(f"  mask missing:  {mask}")
    if problems:
        raise SystemExit(
            f"[{tool_name}] input files not found:\n" + "\n".join(problems))


def _scan_render_log(path, tail_lines=50):
    pass_dir = None
    tail = deque(maxlen=tail_lines)
    with open(path, errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            tail.append(line)
            index = line.find("OUTPUT DIR:")
            if index != -1:
                pass_dir = line[index + len("OUTPUT DIR:"):].strip()
    return pass_dir, list(tail)


def render_pass(run_dir, layout, iteration, views, frames, engine, samples,
                pass_label="", intent="", rollback="", extra_args=(),
                log=print, run=subprocess.run, tool_name="pass"):
    """Run render.sh and return the pass directory announced in its log."""
    scene = os.path.join(run_dir, "scene.py")
    if not os.path.isfile(scene):
        raise SystemExit(f"[{tool_name}] no scene.py at {scene}")
    views_root = (os.path.join(iteration["path"], "renders")
                  if iteration else os.path.join(run_dir, "views"))
    reference = os.path.join(layout["frames_dir"], layout["ref_frame"])
    command = [
        "bash", RENDER_SH, scene, views_root,
        "--views", views,
        "--frames", frames,
        "--ref-frame", layout["ref_frame"],
        "--match-res", reference,
        "--engine", engine,
        "--samples", str(samples),
        "--export", os.path.join(run_dir, "mesh", "object.glb"),
    ]
    if layout["tracking"]:
        command += ["--tracking", layout["tracking"]]
    if pass_label:
        command += ["--pass-label", pass_label]
    if intent.strip():
        command += ["--intent", intent.strip()]
    if rollback.strip():
        command += ["--rollback", rollback.strip()]
    command += [str(value) for value in extra_args]
    log("render: " + " ".join(command))

    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = time.time_ns() % 1_000_000_000
    log_path = os.path.join(
        logs_dir, f"render-{stamp}-{os.getpid()}-{suffix:09d}.log")
    log(f"Blender output: {log_path}")
    with open(log_path, "w") as render_log:
        result = run(
            command, cwd=REPO, stdout=render_log, stderr=subprocess.STDOUT,
            text=True)

    pass_dir, tail = _scan_render_log(log_path)
    if result.returncode != 0:
        if tail:
            sys.stderr.write(f"[{tool_name}] Blender log tail:\n")
            sys.stderr.write("\n".join(tail) + "\n")
        raise SystemExit(
            f"[{tool_name}] render.sh FAILED. Full Blender log: {log_path}")
    if not pass_dir or not os.path.isdir(pass_dir):
        raise SystemExit(
            f"[{tool_name}] render.sh did not announce a valid pass directory. "
            f"Full log: {log_path}")
    log(f"PASS_DIR = {pass_dir}")
    return pass_dir


def composite_paths(pass_dir, layout, frame):
    stem = os.path.splitext(frame)[0]
    hand = ""
    if layout["hand_masks_dir"]:
        candidate = captures.mask_for(layout["hand_masks_dir"], frame)
        if os.path.isfile(candidate):
            hand = candidate

    def inside(name):
        return os.path.join(pass_dir, name)

    return {
        "stem": stem,
        "img": os.path.join(layout["frames_dir"], frame),
        "mask": captures.mask_for(layout["masks_dir"], frame),
        "hand": hand,
        "match": inside(f"match_{stem}.png"),
        "metrics_out": inside(f"metrics_{stem}.json"),
        "overlap_out": inside(f"overlap_{stem}.png"),
        "composite_out": inside(f"composite_{stem}.png"),
        "side_by_side_out": inside(f"side_by_side_{stem}.png"),
    }


def composite_command(paths, bg_mode="black", tracking="", render_depth="",
                      depth_residual_out="", side_by_side_max_dimension=0):
    """Build the canonical analysis.viz.composite invocation."""
    command = [
        "python", "-m", COMPOSITE_MODULE,
        "--source", paths["img"],
        "--render", paths["match"],
        "--mask", paths["mask"],
        "--metrics-out", paths["metrics_out"],
        "--overlap-out", paths["overlap_out"],
        "--out", paths["composite_out"],
        "--side-by-side-out", paths["side_by_side_out"],
        "--side-by-side-max-dimension", str(side_by_side_max_dimension),
        "--bg-mode", bg_mode,
    ]
    if paths.get("hand"):
        command += ["--hand-mask", paths["hand"]]
    if tracking:
        command += ["--tracking", tracking]
    if render_depth:
        command += ["--render-depth", render_depth]
    if depth_residual_out:
        command += ["--depth-residual-out", depth_residual_out]
    return command
