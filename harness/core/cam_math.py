"""cam_math.py — pure-numpy camera intrinsics: the ONE home for K conventions.

No bpy, no mathutils — importable from Blender and the artscript env alike
(see core/__init__.py).

The recurring source of bugs this module retires: a pinhole K is measured on ONE
pixel grid (a Pi3X-resized image, a source photo, a sweep's low-res scoring
raster) but consumed on ANOTHER. Carrying K as a RESOLUTION-AGNOSTIC quantity —
`NormIntrinsics`, the K normalized by its own (W, H) — makes "the K at any
resolution" a single exact operation (`.at(W, H)`), so the same K drives the
full-res match render, the low-res sweep, and the depth unprojection without
ever silently pinning the render resolution to the grid it happened to be
measured on.

Convention notes (canonical for the whole repo):
  * Rescale is PLAIN-LINEAR: fx' = fx * W'/W, cx' = cx * W'/W (y analogous).
    This matches analysis.lib.depth_obs.resample_to, so normalized
    round-tripping is EXACT.
  * Principal point uses the OpenCV pixel-INDEX convention: pixel centers at
    integer coords u=0..W-1, image center at (W-1)/2. unproject() and the
    Blender lens-shift math (rig.camera.apply_K) both honor this; the (W-1)/2
    term is a shift concern applied at the target resolution, NOT part of the
    linear rescale above.
"""

import math
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# iPhone 13 (main/wide lens) intrinsics — PLACEHOLDER
# --------------------------------------------------------------------------- #
# Models the NORMAL rear lens (26 mm-equivalent main/"Wide", ~69° across the long
# side) — NOT the 13 mm ultra-wide fisheye (~120°). A plain pinhole: square
# pixels, centered principal point, ZERO lens distortion. Synthesized from the
# render resolution (so it stays correct at any resolution — this is exactly why
# the placeholder path never suffered the resolution-clobber bug). SWAP for a
# real K via --intrinsics / --camera-json / --tracking.
IPHONE13_LONGSIDE_FOV_DEG = 69.0  # main/wide lens (26mm-eq), not the ultra-wide


# --------------------------------------------------------------------------- #
# FOV <-> focal (de-duplicates the formula that lived in 3 places)
# --------------------------------------------------------------------------- #
def focal_from_fov(fov_deg, long_side):
    """Pinhole focal length (px) for a field of view measured across `long_side`
    pixels. Orientation-independent when `long_side = max(W, H)`."""
    return (long_side / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def fov_from_focal(focal, long_side):
    """Inverse of focal_from_fov: the FOV (deg) a focal length subtends across
    `long_side` pixels."""
    return math.degrees(2.0 * math.atan((long_side / 2.0) / focal))


# --------------------------------------------------------------------------- #
# framing: the distance that FITS a bounding sphere
# --------------------------------------------------------------------------- #
# `--fov` states ONE angle, and Blender's sensor_fit=AUTO applies it across the
# LONGER render dimension. On a landscape frame the other axis is therefore
# NARROWER (840x600 at fov 45 gives a 45 deg width but only 33 deg height), so a
# distance that fits the sphere across the width overflows the height by W/H.
# Framing off the stated FOV alone is what cropped every mechanism tile: the
# object ran off the top and bottom of all 18 while the width looked fine.
#
# So fit BOTH axes: the binding constraint is the SMALLER half-angle, and the
# exact distance for a sphere of radius r is r/sin(half_angle) — sin, not tan,
# because the limiting ray is TANGENT to the sphere (tan fits a box of half-width
# r and still clips the bulge, ~4% at these angles).
FIT_MARGIN = 1.12       # ~12% breathing room so a silhouette never grazes the edge


def fit_distance(radius, fov_deg, width, height, margin=FIT_MARGIN):
    """Camera distance that fits a sphere of `radius` in a (width, height) frame
    whose `fov_deg` is measured across the longer dimension (Blender AUTO).

    Both axes, with the tighter one binding — see the note above. `margin`
    multiplies the exact tangent distance."""
    radius = max(float(radius), 1e-9)
    w, h = float(width), float(height)
    half = math.radians(float(fov_deg)) / 2.0
    long_side = max(w, h)
    # the stated FOV lives on the long side; the short side's half-angle follows
    # from the shared focal length (same pinhole, fewer pixels).
    short_half = math.atan(math.tan(half) * (min(w, h) / long_side))
    binding = min(half, short_half)
    return radius / math.sin(binding) * float(margin)


def fit_ortho_scale(radius, width, height, margin=FIT_MARGIN):
    """Blender `ortho_scale` that fits a sphere of `radius` in a (width, height)
    frame. Same short-axis trap as fit_distance: ortho_scale is measured across
    the LONGER dimension under sensor_fit=AUTO, so fitting the diameter into the
    SHORTER one means scaling the diameter up by the aspect ratio."""
    radius = max(float(radius), 1e-9)
    w, h = float(width), float(height)
    aspect = max(w, h) / max(min(w, h), 1e-9)
    return 2.0 * radius * aspect * float(margin)


# --------------------------------------------------------------------------- #
# resolution-agnostic intrinsics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NormIntrinsics:
    """A pinhole K normalized by the grid it was measured on: fx_n = fx/W etc.

    Resolution-agnostic by construction — `.at(W, H)` re-absolutizes to ANY grid
    with a single exact plain-linear scaling. `src_wh` is retained only for
    provenance/logging (the math never uses it). No skew (Blender can't encode
    it, and none of our K sources produce it)."""
    fx_n: float
    fy_n: float
    cx_n: float
    cy_n: float
    src_wh: tuple = None      # (W, H) the K was measured on — record-keeping only
    provenance: str = None    # "explicit" | "tracking" (None -> unspecified)

    @classmethod
    def from_pixels(cls, fx, fy, cx, cy, width, height, provenance=None):
        """Normalize a pixel-space K stated on a (width, height) grid."""
        w, h = float(width), float(height)
        return cls(fx / w, fy / h, cx / w, cy / h,
                   (int(width), int(height)), provenance)

    @classmethod
    def from_K(cls, K, width, height, provenance=None):
        """Normalize a 3x3 pinhole matrix K stated on a (width, height) grid.

        Accepts anything indexable as K[i][j] (numpy array, nested list)."""
        return cls.from_pixels(float(K[0][0]), float(K[1][1]),
                               float(K[0][2]), float(K[1][2]), width, height,
                               provenance)

    def at(self, width, height):
        """Re-absolutize to a (width, height) grid: the pixel-space K dict
        {fx,fy,cx,cy,width,height} that rig.camera.apply_K consumes."""
        w, h = float(width), float(height)
        return {"fx": self.fx_n * w, "fy": self.fy_n * h,
                "cx": self.cx_n * w, "cy": self.cy_n * h,
                "width": int(width), "height": int(height)}


def iphone13_intrinsics(width, height):
    """iPhone-13-wide PLACEHOLDER pixel K for a (width, height) render.

    fx == fy (square pixels), principal point at the image center, focal from
    IPHONE13_LONGSIDE_FOV_DEG across the longer dimension (orientation-
    independent). Returned as a pixel dict at (width, height) — NOT NormIntrinsics
    — because it is defined by a long-side FOV and is meant to be recomputed at
    whatever resolution is being rendered."""
    width, height = int(width), int(height)
    f = focal_from_fov(IPHONE13_LONGSIDE_FOV_DEG, max(width, height))
    return {"fx": f, "fy": f,
            "cx": (width - 1) / 2.0, "cy": (height - 1) / 2.0,
            "width": width, "height": height}


def rescale_K(K, src_wh, dst_wh):
    """Rescale a 3x3 pinhole K from its (Ws, Hs) grid to a (Wd, Hd) grid.

    The canonical PLAIN-LINEAR rescale (fx*sx, cx*sx, fy*sy, cy*sy) shared by the
    render side (rig.camera) and the depth side (analysis.depth), so both agree.
    Returns a new 3x3 (same container semantics as numpy .copy() when K is an
    ndarray; otherwise a nested list)."""
    Ws, Hs = src_wh
    Wd, Hd = dst_wh
    sx, sy = Wd / float(Ws), Hd / float(Hs)
    try:
        out = K.copy()  # numpy ndarray
    except AttributeError:
        out = [list(row) for row in K]
    out[0][0] = K[0][0] * sx
    out[0][2] = K[0][2] * sx
    out[1][1] = K[1][1] * sy
    out[1][2] = K[1][2] * sy
    return out


def parse_intrinsics_str(s):
    """'fx,fy,cx,cy,W,H' -> {fx,fy,cx,cy,width,height} dict, or None if empty.

    The exact format analysis.measure_depth.py emits as `intrinsics_for_render`."""
    if not s:
        return None
    v = [float(x) for x in s.split(",")]
    if len(v) != 6:
        raise ValueError("--intrinsics needs 'fx,fy,cx,cy,W,H'")
    return {"fx": v[0], "fy": v[1], "cx": v[2], "cy": v[3],
            "width": v[4], "height": v[5]}


def overscan_intrinsics(intr, margin):
    """Widen a pixel-space intrinsics dict by `margin` (fraction of W/H) per side.

    Keeps fx/fy and shifts the principal point so the ORIGINAL frame is the
    centered inner rectangle of a wider film — same camera rays, bigger canvas.
    Returns (over_intr, (padx, pady)). margin<=0 -> unchanged (padx=pady=0).
    Operates purely in absolute pixels on whatever grid `intr` is stated on."""
    W, H = int(intr["width"]), int(intr["height"])
    m = max(0.0, float(margin))
    padx, pady = int(round(m * W)), int(round(m * H))
    over = {"fx": intr["fx"], "fy": intr["fy"],
            "cx": intr["cx"] + padx, "cy": intr["cy"] + pady,
            "width": W + 2 * padx, "height": H + 2 * pady}
    return over, (padx, pady)


def unproject(depth, K):
    """Back-project a planar-depth map to camera-frame XYZ (OpenCV: +Z forward).

    X=(u-cx)/fx*Z, Y=(v-cy)/fy*Z, Z=depth, with u=0..W-1 pixel indices. Returns
    (H,W,3) float32 (NaN where depth is non-finite). numpy required (present in
    every env that calls this)."""
    import numpy as np
    H, W = depth.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u = np.arange(W, dtype=np.float32)[None, :]
    v = np.arange(H, dtype=np.float32)[:, None]
    Z = depth
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)
