"""run_layout.py — the ONE reader for a run's layout.json.

A run's RUN_DIR/layout.json (written by run.sh) records the run's capture dir,
frame dir, masks, and reference frame. Every consumer — shape_pass, measure_depth, frames,
the render pool, multiagent.windows/pool_session — reads it through
`load_run_layout` so the schema, path absolutization, and tracking derivation
live in exactly one place (bpy-free core/, importable from Blender's python
and the analysis env alike).

`find_layout`/`resolve_tracking` walk UP from a render dir instead: the render
views write into RUN_DIR/iterations/NNNNNN/renders/NNNN, so from any output path the layout is a
few levels up. That pair is a SAFETY NET, not a convention: a manual
`render.sh --views sweep` that forgets `--tracking` would otherwise silently score
against the iPhone-13 placeholder K. An explicit `--tracking` / `--intrinsics`
always wins — it only fills the gap.
"""

import json
import os

# dir-valued layout keys, absolutized (relative to RUN_DIR) on load
_DIR_KEYS = ("capture", "tracking", "frames_dir", "masks_dir", "hand_masks_dir")


def load_run_layout(run_dir):
    """RUN_DIR/layout.json as a normalized dict, or None when absent/invalid.

    Normalization (shared by every consumer):
    - dir paths absolutized against run_dir (run.sh writes absolute paths;
      hand-edited relative ones resolve against the run, not the caller's cwd);
    - "tracking" = the explicit manifest value, else <capture>/tracking —
      kept only when the directory actually exists;
    - "ref_frame" defaults to frames[0];
    - "frames" is always a list.

    Validation policy stays with the caller: this returns whatever the manifest
    declares (including kind=single) and never raises on missing fields."""
    run_dir = os.path.abspath(run_dir)
    try:
        with open(os.path.join(run_dir, "layout.json")) as f:
            layout = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(layout, dict):
        return None
    for k in _DIR_KEYS:
        if layout.get(k) and not os.path.isabs(layout[k]):
            layout[k] = os.path.join(run_dir, layout[k])
    tracking = layout.get("tracking") or (
        os.path.join(layout["capture"], "tracking") if layout.get("capture") else "")
    layout["tracking"] = tracking if tracking and os.path.isdir(tracking) else ""
    layout["frames"] = list(layout.get("frames") or [])
    if not layout.get("ref_frame") and layout["frames"]:
        layout["ref_frame"] = layout["frames"][0]
    return layout


def find_layout(start_path):
    """The nearest layout.json dict walking UP from start_path, or None.

    start_path may be a file or a directory; the search begins at its directory
    (or itself, if it's already a dir) and climbs to the filesystem root. Same
    nearest-ancestor resolution as core.depth_config's walk-up."""
    d = os.path.dirname(os.path.abspath(start_path))
    if os.path.isdir(os.path.abspath(start_path)):
        d = os.path.abspath(start_path)
    prev = None
    while d and d != prev:
        cand = os.path.join(d, "layout.json")
        if os.path.isfile(cand):
            try:
                with open(cand) as f:
                    cfg = json.load(f)
                if isinstance(cfg, dict):
                    return cfg
            except (ValueError, OSError):
                pass
        prev, d = d, os.path.dirname(d)
    return None


def resolve_tracking(start_path):
    """The run's tracking dir (absolute — <capture>/tracking), or None.

    Resolved from the layout's "capture" entry. Only an existing directory is
    returned — so the caller safely falls back to the placeholder rather than
    pointing at a dir with no cameras.npz."""
    layout = find_layout(start_path)
    if not isinstance(layout, dict):
        return None
    capture = layout.get("capture")
    if not capture:
        return None
    tracking = os.path.join(os.path.abspath(capture), "tracking")
    return tracking if os.path.isdir(tracking) else None
