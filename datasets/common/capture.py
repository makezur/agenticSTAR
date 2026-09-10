"""capture.py — the canonical capture format: schema + bpy-safe reader.

A *capture* is one video sequence written by tools/make_capture.py into the
ONE on-disk layout every harness tool reads.

    <capture>/
      capture.json               # versioned manifest (FORMAT, below)
      frames/<NNNNNN>.jpg        # numeric stems; temporal order = the integer
      mask_object/<NNNNNN>.png   # binary object mask (visible region)
      mask_hand/<NNNNNN>.png     # binary hand/occluder mask (optional dir)
      tracking/
        cameras.npz              # c2w (K,4,4) f32 OpenCV cam-to-world,
                                 # intrinsics (K,3,3) f32, size [W,H] i64
        keyframes.json           # [{view, frame_index, frame_name,
                                 #   time_s}, ...] (time_s optional; readers
                                 #   fall back to frame_index / 30)
      depth/<NNNNNN>.npz         # OPTIONAL per-frame observed geometry:
                                 # local (H,W,3) f32 camera-frame pointmap
                                 # ([...,2] = planar Z), conf (H,W) f32 [0,1],
                                 # keep (H,W) bool

capture.json:
    {
      "format": "capture/v1",
      "dataset": "pi3x",
      "source": {...},                   # backend provenance (sub/seq/view/...)
      "downscale": 1.0,
      "undistorted": false,              # true when frames were undistorted
      "depth": "pi3x" | "none",
      "image_size": [W, H],
      "frames": ["000000.jpg", ...],     # full ordered list
      "ref_frame": "000000.jpg"
    }

IMPORTANT: this module is imported from Blender's bundled python (the rig) —
numpy + stdlib ONLY. cv2-dependent readers live in capture_io.py.
"""

import json
import os

import numpy as np

FORMAT = "capture/v1"
DEPTH_KINDS = ("pi3x", "none")

MANIFEST = "capture.json"
FRAMES_DIR = "frames"
MASK_OBJECT_DIR = "mask_object"
MASK_HAND_DIR = "mask_hand"
TRACKING_DIR = "tracking"
DEPTH_DIR = "depth"
CAMERAS_NPZ = "cameras.npz"
KEYFRAMES_JSON = "keyframes.json"


class CaptureError(ValueError):
    """A capture dir that doesn't satisfy the format contract."""


def stem_of(frame_name):
    """'000123.jpg' -> '000123' (the per-frame join key everywhere)."""
    return os.path.splitext(os.path.basename(frame_name))[0]


def mask_for(masks_dir, frame_name):
    """The mask path for a frame: <masks_dir>/<stem>.png.

    THE owner of the stem-join convention — harness tools that hold a bare
    masks/hand-masks dir (layout overrides, sweep args) join through this
    instead of re-deriving `stem + '.png'` locally."""
    return os.path.join(masks_dir, stem_of(frame_name) + ".png")


def load_depth(path):
    """Load a per-frame depth npz -> (local (H,W,3) f32, depth (H,W) f32,
    conf (H,W) f32, keep (H,W) bool, K (3,3) f64 | None).

    Depth npz contract (<capture>/depth/<stem>.npz):
        local (H,W,3) f32   camera-frame pointmap, [...,2] = planar Z
        conf  (H,W)   f32   per-pixel confidence in [0,1] (absent -> 1)
        keep  (H,W)   bool  gate; False = never score (absent -> all True)
        K     (3,3)   f32   OPTIONAL depth-grid intrinsics; when absent the
                            tracking intrinsics for the frame's view apply
    numpy-only (the bpy-side sweep loads depth supervision through this)."""
    z = np.load(path)
    local = np.asarray(z["local"], dtype=np.float32)
    depth = local[..., 2]
    conf = (np.asarray(z["conf"], dtype=np.float32) if "conf" in z
            else np.ones(depth.shape, dtype=np.float32))
    keep = (np.asarray(z["keep"], dtype=bool) if "keep" in z
            else np.ones(depth.shape, dtype=bool))
    K = np.asarray(z["K"], dtype=np.float64) if "K" in z else None
    return local, depth, conf, keep, K


def save_depth(path, local, conf=None, keep=None, K=None):
    """Write one per-frame depth npz (the single owner of the write format)."""
    local = np.asarray(local, dtype=np.float32)
    if local.ndim != 3 or local.shape[2] != 3:
        raise CaptureError(f"depth local must be (H,W,3), got {local.shape}")
    arrays = {"local": local}
    hw = local.shape[:2]
    arrays["conf"] = (np.ones(hw, np.float32) if conf is None
                      else np.asarray(conf, dtype=np.float32))
    arrays["keep"] = (np.ones(hw, bool) if keep is None
                      else np.asarray(keep, dtype=bool))
    if K is not None:
        arrays["K"] = np.asarray(K, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_cameras(tracking_dir):
    """cameras.npz -> {'c2w': (K,4,4) f64, 'intrinsics': (K,3,3) f64,
    'size': (W, H) | None}. numpy-only (works inside Blender)."""
    cams = np.load(os.path.join(tracking_dir, CAMERAS_NPZ))
    size = cams["size"] if "size" in cams else None
    return {
        "c2w": np.asarray(cams["c2w"], dtype=float),
        "intrinsics": np.asarray(cams["intrinsics"], dtype=float),
        "size": (int(size[0]), int(size[1])) if size is not None else None,
    }


def load_keyframes(tracking_dir):
    """keyframes.json list, or None when absent (single-view legacy tracking)."""
    path = os.path.join(tracking_dir, KEYFRAMES_JSON)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def resolve_view(tracking_dir, view=None, frame_name=""):
    """Map a frame filename (or explicit index) to a tracking view index.

    Returns (view_index, frame_name|None). Matching is by stem, tolerant of the
    .jpg/.png extension difference between frames and masks/renders."""
    keyframes = load_keyframes(tracking_dir)
    if view is not None:
        name = None
        if keyframes:
            for k in keyframes:
                if k["view"] == view:
                    name = k.get("frame_name")
        return int(view), name
    if not frame_name:
        raise CaptureError("resolve_view: give view or frame_name")
    if not keyframes:
        raise CaptureError(f"no {KEYFRAMES_JSON} in {tracking_dir}; pass view")
    stem = stem_of(frame_name)
    for k in keyframes:
        if stem_of(k.get("frame_name", "")) == stem:
            return int(k["view"]), k.get("frame_name")
    raise CaptureError(
        f"no tracking view matches frame '{stem}' "
        f"(views: {[k.get('frame_name') for k in keyframes]})")


class Capture:
    """Read-only handle on a capture dir: manifest + per-frame path resolution.

    Cheap to construct (reads capture.json only); array loading lives in
    capture_io (cv2 side) and load_cameras (numpy side)."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        path = os.path.join(self.root, MANIFEST)
        if not os.path.isfile(path):
            raise CaptureError(f"not a capture (no {MANIFEST}): {self.root}")
        with open(path) as f:
            self.manifest = json.load(f)
        fmt = self.manifest.get("format")
        if fmt != FORMAT:
            raise CaptureError(f"{path}: format {fmt!r} != {FORMAT!r}")

    # -- manifest fields ------------------------------------------------------
    @property
    def dataset(self):
        return self.manifest.get("dataset", "")

    @property
    def frames(self):
        return list(self.manifest.get("frames", []))

    @property
    def ref_frame(self):
        return self.manifest.get("ref_frame", "")

    @property
    def depth_kind(self):
        return self.manifest.get("depth", "none")

    @property
    def image_size(self):
        wh = self.manifest.get("image_size")
        return (int(wh[0]), int(wh[1])) if wh else None

    @property
    def downscale(self):
        return float(self.manifest.get("downscale", 1.0))

    # -- directories ----------------------------------------------------------
    @property
    def frames_dir(self):
        return os.path.join(self.root, FRAMES_DIR)

    @property
    def masks_dir(self):
        return os.path.join(self.root, MASK_OBJECT_DIR)

    @property
    def hand_masks_dir(self):
        """The hand-mask dir, or '' when the capture ships no hand masks."""
        d = os.path.join(self.root, MASK_HAND_DIR)
        return d if os.path.isdir(d) else ""

    @property
    def tracking_dir(self):
        return os.path.join(self.root, TRACKING_DIR)

    @property
    def depth_dir(self):
        """The per-frame depth dir, or '' when the capture ships no depth."""
        d = os.path.join(self.root, DEPTH_DIR)
        return d if self.depth_kind != "none" and os.path.isdir(d) else ""

    # -- per-frame paths (the ONE owner of the stem-join convention) -----------
    def frame_path(self, frame_name):
        return os.path.join(self.frames_dir, frame_name)

    def mask_path(self, frame_name):
        return os.path.join(self.masks_dir, stem_of(frame_name) + ".png")

    def hand_mask_path(self, frame_name):
        """Path of the frame's hand mask, or '' (no dir / no file)."""
        d = self.hand_masks_dir
        if not d:
            return ""
        p = os.path.join(d, stem_of(frame_name) + ".png")
        return p if os.path.isfile(p) else ""

    def depth_path(self, frame_name):
        """Path of the frame's depth npz, or '' (no depth / no file)."""
        d = self.depth_dir
        if not d:
            return ""
        p = os.path.join(d, stem_of(frame_name) + ".npz")
        return p if os.path.isfile(p) else ""

    # -- tracking -------------------------------------------------------------
    def cameras(self):
        return load_cameras(self.tracking_dir)

    def view_of(self, frame_name):
        """The tracking view index for a frame (via keyframes.json)."""
        v, _ = resolve_view(self.tracking_dir, frame_name=frame_name)
        return v


def find_capture(start_path):
    """The nearest capture root walking UP from start_path, or None.

    Mirrors core/run_layout.py:find_layout — 'nearest ancestor holding the
    file wins'."""
    d = os.path.abspath(start_path)
    if not os.path.isdir(d):
        d = os.path.dirname(d)
    prev = None
    while d and d != prev:
        if os.path.isfile(os.path.join(d, MANIFEST)):
            return d
        prev, d = d, os.path.dirname(d)
    return None


def write_manifest(root, dataset, frames, image_size, source=None,
                   downscale=1.0, undistorted=False, depth="none",
                   ref_frame=None):
    """Write <root>/capture.json. `frames` must already be in temporal order."""
    if depth not in DEPTH_KINDS:
        raise CaptureError(f"depth {depth!r} not in {DEPTH_KINDS}")
    if not frames:
        raise CaptureError("write_manifest: empty frame list")
    manifest = {
        "format": FORMAT,
        "dataset": dataset,
        "source": source or {},
        "downscale": float(downscale),
        "undistorted": bool(undistorted),
        "depth": depth,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "frames": list(frames),
        "ref_frame": ref_frame or frames[0],
    }
    path = os.path.join(root, MANIFEST)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest
