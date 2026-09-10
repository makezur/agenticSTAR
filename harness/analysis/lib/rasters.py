"""rasters — image/mask loading, silhouette extraction, keep-region.

The shared raster I/O every analysis tool builds on. Kept free of any scoring or
panel logic so `scorers/` and `viz/` can both depend on it without a cycle.

Renders go through `load_render` (load + present on the shared backdrop), photos
through `load_bgr`. Keeping those distinct is the point: a render carries alpha with
colour underneath it, and reading one as a plain image is how the same render came to
look slightly different depending on which tool drew it.

`load_render_depth` is here for the same reason: it reads OUR OWN render (the
`depth` view's .npy), so it belongs with the other render loaders rather than in
the observed-depth module — a monocular tool that needs our depth should not have
to import the pointmap loader to get at it.
"""

import cv2
import numpy as np

from core import renderer_settings


# --------------------------------------------------------------------------- #
# silhouette extraction
# --------------------------------------------------------------------------- #
def render_silhouette(render_path):
    """Binary silhouette (uint8 0/255) from a render.

    Prefer the alpha channel (exact, color-independent) since the harness
    renders on a transparent film; fall back to Otsu on luminance.
    """
    rgba = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
    if rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4:
        alpha = rgba[:, :, 3]
        if alpha.min() < 250:  # a genuinely transparent background exists
            _, m = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
            return m
    bgr = cv2.imread(render_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read render: {render_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


def resize_mask_to(mask, target_shape):
    """`mask` nearest-resized onto `target_shape`'s (h, w) grid; a no-op if it
    already matches. NEAREST because a mask is labels, not intensities —
    interpolating one invents boundary values that were never labeled."""
    h, w = target_shape[:2]
    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def build_keep(hand_mask, target_shape, dilate=0.01):
    """Keep-region K = ¬hand as a 0/1 uint8 array at `target_shape` (h, w).

    The hand is the OCCLUDER: the object continues behind it, so hand pixels are
    scored as "don't care" (excluded from every metric). We nearest-resize the
    hand mask onto the scoring grid, optionally DILATE it by `dilate` (a fraction
    of the longer side; elliptical kernel) to absorb the thin rim where the SAM
    hand boundary doesn't line up with the true occlusion edge, then invert.
    `dilate <= 0` uses the hand mask as given.
    """
    h, w = target_shape[:2]
    hm = cv2.resize((hand_mask > 0).astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST)
    if dilate > 0:
        r = int(round(dilate * max(h, w)))
        if r >= 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            hm = cv2.dilate(hm, k)
    return (hm == 0).astype(np.uint8)


def load_bgr(path):
    """A plain BGR image, alpha DISCARDED. For photos (which have none).

    Not for renders: see `load_render`, because dropping a render's alpha keeps the
    colour Blender left underneath it."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img  # BGR


def load_render(path, bg_mode="black"):
    """A render loaded AND presented on `bg_mode` — the ONE way to read a render.

    The harness renders on a transparent film (`FILM_TRANSPARENT`), and Blender
    leaves colour in the pixels under `alpha == 0` — an anti-aliased rim about a
    pixel wide around the silhouette. `load_bgr` throws the alpha away and keeps
    that colour, so a tool that reads a render as a plain image shows a faint fringe
    the same render presented properly does not. That is why this exists: every
    panel and every score reads a render through here, so they cannot disagree about
    what the render looks like.

    'black' (the default) composites onto the single opaque `BACKDROP_BGR`, so
    every viewer sees the same background and no hidden RGB fringe leaks through.
    """
    bgr, alpha = load_rgba(path)
    return renderer_settings.present(bgr, alpha, bg_mode)


def load_render_depth(path):
    """Load the harness depth render (float32 (H,W), +inf where no geometry).

    The `depth` view's .npy — planar camera-space Z of our own render. `+inf`
    (not a sentinel) marks no-geometry pixels, so every consumer selects the
    object with a plain `np.isfinite`."""
    d = np.load(path)
    return np.asarray(d, dtype=np.float32)


def load_rgba(path):
    """(rgb_bgr, alpha) from a PNG. Alpha is the object silhouette (transparent
    film), as in render_silhouette."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not read ID render: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3], img[:, :, 3]
    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return bgr, np.full(bgr.shape[:2], 255, dtype=np.uint8)
