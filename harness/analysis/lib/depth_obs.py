"""depth_obs — load & resample a capture's per-frame OBSERVED-depth pointmap.

The tracking/depth I/O the depth scorer and the pose-seeder (`measure_depth`)
both need: pick the tracking view for a frame, load its camera-frame pointmap
/ confidence / intrinsics / extrinsics, and resample the observed grid onto the
render grid (rescaling K to match). No scoring — that stays in `scorers/depth`.
Strictly the OBSERVED side: our own render's depth is `rasters.load_render_depth`,
so a monocular tool needs nothing from this module.

Data comes from the canonical CAPTURE layout (datasets/common/capture.py):
tracking/cameras.npz + tracking/keyframes.json for cameras, depth/<stem>.npz
for per-frame pointmaps. Every function here accepts EITHER the capture root
or its tracking/ subdir; both resolve to the same capture.

The pointmap convention is Pi3X's (OpenCV camera-frame pointmaps, planar-Z
depth, [0,1] confidence); the producing backend is recorded in the `backend`
field of depth_config.json.

Imports the shared pure-numpy camera core (`core.cam_math`) for the K rescale —
the SAME conventions the Blender render side uses. Runs with `cwd=harness`, so
`core` is importable without any sys.path hack.
"""

import os

import numpy as np

from core import cam_math, captures
from datasets.common import capture_io


def _tracking_dir(path):
    """Normalize a capture root OR tracking dir to the tracking dir."""
    if os.path.isfile(os.path.join(path, captures.MANIFEST)):
        return os.path.join(path, captures.TRACKING_DIR)
    return path


def _depth_dir(path):
    """The capture's per-frame depth dir for a capture root or tracking dir."""
    return os.path.join(os.path.dirname(_tracking_dir(path).rstrip(os.sep)),
                        captures.DEPTH_DIR)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def resolve_view(tracking_dir, view, frame_name, source):
    """Pick the tracking view index from --view, --frame-name, or the source
    name.

    Returns (view_index, frame_name_str). Matches by the source frame filename
    via keyframes.json (view <-> frame), tolerant of the .jpg/.png extension
    difference between the capture frames and the mask/render.
    """
    tracking = _tracking_dir(tracking_dir)
    target = frame_name or (os.path.basename(source) if source else "")
    if view is None and not target:
        raise SystemExit("depth_obs: give --view, --frame-name, or --source to "
                         "select the tracking view")
    try:
        return captures.resolve_view(tracking, view=view, frame_name=target)
    except captures.CaptureError as e:
        raise SystemExit(f"depth_obs: {e}")


def _stem_for_view(tracking_dir, v):
    """The frame stem tracking view `v` maps to (via keyframes.json)."""
    keyframes = captures.load_keyframes(_tracking_dir(tracking_dir))
    if keyframes:
        for k in keyframes:
            if int(k["view"]) == int(v):
                return captures.stem_of(k.get("frame_name", ""))
    return None


def load_view(tracking_dir, v):
    """Load the per-view observed-depth arrays for tracking view `v`.

    Returns (xyz (Hp,Wp,3), depth (Hp,Wp), conf (Hp,Wp), keep bool (Hp,Wp),
    K (3,3)). Reads the capture's per-frame depth/<stem>.npz; a capture with
    no depth for this frame fails loudly (callers gate on the run's
    depth_config backend before asking).
    """
    stem = _stem_for_view(tracking_dir, v)
    if stem is None:
        raise SystemExit(f"depth_obs: no keyframes.json entry for view {v} "
                         f"in {tracking_dir}")
    depth_path = os.path.join(_depth_dir(tracking_dir), stem + ".npz")
    if not os.path.isfile(depth_path):
        raise SystemExit(
            f"depth_obs: no observed depth for frame {stem} ({depth_path}); "
            "this capture ships no depth (manifest depth: none?) — depth "
            "scoring/measuring is unavailable for it")
    local, depth, conf, keep, K = captures.load_depth(depth_path)
    if K is None:
        cams = captures.load_cameras(_tracking_dir(tracking_dir))
        K = cams["intrinsics"][int(v)]
    return local, depth, conf, keep, np.asarray(K, dtype=np.float64)


def load_c2w(tracking_dir):
    """Per-view camera-to-world extrinsics (K,4,4) from cameras.npz (OpenCV).

    Consumed by the pose seeding in measure_depth — the relative pose between
    two views is inv(c2w[k]) @ c2w[ref]."""
    return np.asarray(
        captures.load_cameras(_tracking_dir(tracking_dir))["c2w"],
        dtype=np.float64)


def relative_pose(c2w, ref, k):
    """Rigid transform carrying a point from view `ref`'s camera frame into view
    `k`'s: inv(c2w[k]) @ c2w[ref]. ref==k -> identity."""
    return np.linalg.inv(c2w[k]) @ c2w[ref]


# --------------------------------------------------------------------------- #
# resampling the observed grid -> render grid
# --------------------------------------------------------------------------- #
def resample_to(shape_hw, obs_xyz, obs_depth, conf, keep, K_grid):
    """Resample every observed-depth field onto the render grid and rescale K.

    depth/conf/xyz -> INTER_LINEAR (smooth fields); keep -> INTER_NEAREST (bool).
    K is scaled by (W/Wp, H/Hp) so unprojection uses pixel coordinates on the
    render grid. Returns (obs_xyz, obs_depth, conf, keep, K) at the render res.

    Delegates to the single owner of the resample convention
    (datasets.common.capture_io.resample_depth_to), passing in the harness's
    cam_math.rescale_K — the SAME plain-linear op the Blender render side uses.
    """
    return capture_io.resample_depth_to(
        shape_hw, obs_xyz, obs_depth, conf, keep, K_grid, cam_math.rescale_K)
