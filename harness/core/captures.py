"""captures.py — the harness's bridge to datasets.common.capture (bpy-safe).

The capture format (datasets/common/capture.py) is the ONE input layout every
tool reads; this shim makes it importable from BOTH pythons that run harness
code — Blender's bundled interpreter (render_wrapper --python, cwd=harness)
and the artscript env (analysis tools, cwd=harness) — by putting the repo
root on sys.path exactly once, here.

    from core import captures
    cap = captures.Capture(capture_root)
    captures.mask_for(masks_dir, frame_name)

cv2-dependent readers (per-frame depth resample etc.) live in
datasets.common.capture_io; analysis-side code imports that DIRECTLY —
`from core import captures` first guarantees the repo root is importable:

    from core import captures            # noqa: F401  (path side-effect)
    from datasets.common import capture_io
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.common.capture import (  # noqa: E402,F401
    CAMERAS_NPZ,
    DEPTH_DIR,
    DEPTH_KINDS,
    FORMAT,
    FRAMES_DIR,
    KEYFRAMES_JSON,
    MANIFEST,
    MASK_HAND_DIR,
    MASK_OBJECT_DIR,
    TRACKING_DIR,
    Capture,
    CaptureError,
    find_capture,
    load_cameras,
    load_depth,
    load_keyframes,
    mask_for,
    resolve_view,
    save_depth,
    stem_of,
    write_manifest,
)
