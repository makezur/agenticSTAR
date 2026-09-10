"""capture_io.py — cv2-dependent capture readers (analysis side).

The array-loading half of the capture API: per-frame depth npz, images/masks,
and the depth-grid -> render-grid resample. Kept out of capture.py so the
bpy-safe core stays numpy+stdlib-only (Blender's python has no cv2).

Depth npz contract (one file per frame, <capture>/depth/<stem>.npz):
    local (H,W,3) f32   camera-frame pointmap, [...,2] = planar Z
    conf  (H,W)   f32   per-pixel confidence in [0,1]
    keep  (H,W)   bool  gate (False = never score this pixel)
    K     (3,3)   f32   OPTIONAL intrinsics of the depth grid; when absent the
                        tracking intrinsics for the frame's view are used
"""

import cv2
import numpy as np

from . import capture as cap


# the npz load/save pair is numpy-only and lives in the bpy-safe core (the
# in-Blender sweep engine reads depth supervision through it); re-exported
# here so analysis-side callers keep one import home.
load_depth = cap.load_depth
save_depth = cap.save_depth


def load_frame_depth(capture, frame_name):
    """Depth arrays for one frame of a Capture, K falling back to the frame's
    tracking intrinsics. Returns (local, depth, conf, keep, K) or None when the
    capture ships no depth for this frame."""
    path = capture.depth_path(frame_name)
    if not path:
        return None
    local, depth, conf, keep, K = load_depth(path)
    if K is None:
        cams = capture.cameras()
        K = cams["intrinsics"][capture.view_of(frame_name)]
    return local, depth, conf, keep, np.asarray(K, dtype=np.float64)


def resample_depth_to(shape_hw, local, depth, conf, keep, K_grid, rescale_K):
    """Resample every depth field onto the render grid, rescaling K to match.

    depth/conf/local -> INTER_LINEAR (smooth fields); keep -> INTER_NEAREST
    (bool). `rescale_K` is core.cam_math.rescale_K passed in by the caller —
    this module deliberately doesn't import harness code (datasets/ stands
    alone; the harness depends on it, not the reverse)."""
    H, W = shape_hw
    Hp, Wp = depth.shape
    wh = (W, H)
    local_r = cv2.resize(local, wh, interpolation=cv2.INTER_LINEAR)
    depth_r = cv2.resize(depth, wh, interpolation=cv2.INTER_LINEAR)
    conf_r = cv2.resize(conf, wh, interpolation=cv2.INTER_LINEAR)
    keep_r = cv2.resize(keep.astype(np.uint8), wh,
                        interpolation=cv2.INTER_NEAREST) > 0
    K = rescale_K(K_grid, (Wp, Hp), (W, H))
    return local_r, depth_r, conf_r, keep_r, K
