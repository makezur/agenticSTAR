"""imaging.py — pixel helpers shared by the ids / crop / visibility views.

Pure numpy on Blender-loaded pixels (no PIL/cv2 inside Blender): PNG load, box
drawing, the ID-pass palette + emission materials + color-management state, and
nearest-palette part assignment.
"""

import colorsys
from contextlib import contextmanager

import bpy
import numpy as np


# --------------------------------------------------------------------------- #
# PNG load + box drawing
# --------------------------------------------------------------------------- #
def image_pixels(img):
    """C-level readback of a Blender image's flat pixel buffer as float32.

    `foreach_get` into a preallocated array is ~20x faster than the obvious
    `np.array(img.pixels[:])`, which materializes every float through a Python
    list (measured 49ms -> 2.7ms on a 288x512 RGBA read). That per-image list
    conversion — NOT the PNG disk round trip — is what dominates the sweep views'
    per-candidate cost, so this is the hot path they all go through.
    """
    buf = np.empty(len(img.pixels), dtype=np.float32)
    img.pixels.foreach_get(buf)
    return buf


def load_png_rgba(path):
    """Load a PNG as a numpy (H, W, 4) float array in 0..1, top row first."""
    img = bpy.data.images.load(path, check_existing=False)
    w, h = int(img.size[0]), int(img.size[1])
    px = image_pixels(img).reshape(h, w, 4)
    bpy.data.images.remove(img)
    return px[::-1]  # Blender stores bottom-up; flip to top-down (image y-down)


@contextmanager
def edit_png_overlay(path):
    """Load `path` as a TOP-DOWN (H, W, 4) editable float array, yield it for
    in-place drawing, then write it back to the same PNG (color management
    round-trips through the image datablock). Shared by the crop + visibility
    overlays so the load/reshape/flip/save dance lives in one place."""
    img = bpy.data.images.load(path, check_existing=False)
    ow, oh = int(img.size[0]), int(img.size[1])
    bottom_up = image_pixels(img).reshape(oh, ow, 4)
    top = bottom_up[::-1].copy()  # top-down, image y-down (matches bbox coords)
    try:
        yield top
    finally:
        img.pixels[:] = top[::-1].reshape(-1)
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        bpy.data.images.remove(img)


def frame_clip_stats(mask_bool, padx, pady, W, H):
    """Framing stats for a silhouette in the OVERSCAN frame — the primitive the
    crop (whole-object) and visibility (per-part) views share.

    Measures how much of `mask_bool` falls inside the true match frame (the inner
    [pady:pady+H, padx:padx+W] rectangle) vs. spills past each edge; all fractions
    are of the silhouette's total (overscan) pixels. Returns total_px, inside_px,
    per-edge overflow fractions, and whether the silhouette is fully caught by the
    overscan (else clipping is a lower bound). Callers layer their own extra
    signals (crop's suggested_zoom_out, visibility's occlusion) on top."""
    total = int(mask_bool.sum())
    if total == 0:
        return {"total_px": 0, "inside_px": 0,
                "edges": {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0},
                "fully_captured_in_overscan": False}
    inside = int(mask_bool[pady:pady + H, padx:padx + W].sum())

    def frac(region):
        return round(int(region.sum()) / total, 4)

    edges = {
        "left": frac(mask_bool[:, :padx]),
        "right": frac(mask_bool[:, padx + W:]),
        "top": frac(mask_bool[:pady, :]),
        "bottom": frac(mask_bool[pady + H:, :]),
    }
    touches = bool(mask_bool[0, :].any() or mask_bool[-1, :].any()
                   or mask_bool[:, 0].any() or mask_bool[:, -1].any())
    return {"total_px": total, "inside_px": inside, "edges": edges,
            "fully_captured_in_overscan": not touches}


def draw_box(px, x0, y0, x1, y1, rgba, thickness=2, dashed=False, dash=12):
    """Draw a rectangle outline onto a TOP-DOWN (H, W, 4) float array in place.

    Coordinates are image pixels (x right, y down), inclusive corners; clamped to
    the array. `rgba` is a 0..1 tuple. `dashed` draws a dash/gap pattern (used for
    the full/predicted extent of occluded parts vs. the solid visible box). Pure
    numpy so it works inside Blender (no PIL/cv2). Used by the visibility overlay.
    """
    h, w = px.shape[:2]
    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))
    x0c, x1c = max(0, x0), min(w - 1, x1)
    y0c, y1c = max(0, y0), min(h - 1, y1)
    if x0c > x1c or y0c > y1c:
        return
    col = np.array(rgba, dtype=px.dtype)
    t = max(1, int(thickness))

    def _on(coord, anchor):
        # dashed: draw only where the distance along the edge is in the "on" half
        return (not dashed) or (((coord - anchor) // dash) % 2 == 0)

    for yy in range(y0c, y1c + 1):
        for edge_x in (x0, x1):
            for k in range(t):
                xx = edge_x + (k if edge_x == x0 else -k)
                if 0 <= xx < w and _on(yy, y0c):
                    px[yy, xx] = col
    for xx in range(x0c, x1c + 1):
        for edge_y in (y0, y1):
            for k in range(t):
                yy = edge_y + (k if edge_y == y0 else -k)
                if 0 <= yy < h and _on(xx, x0c):
                    px[yy, xx] = col


# --------------------------------------------------------------------------- #
# ID pass: flat-color each part + a name/color legend ("which bit is which")
# --------------------------------------------------------------------------- #
GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def id_palette(n):
    """n visually distinct sRGB-8bit (r, g, b) tuples.

    Golden-ratio hue spacing gives well-separated hues for any n; parity-
    alternated saturation/value pushes neighboring indices further apart so they
    survive 8-bit rounding and read as different colors. Deterministic in n, so
    keyed by index into the SORTED part-name list the mapping is stable across
    iterations (a given name keeps its color as long as the name set is stable).
    """
    out = []
    for i in range(max(n, 1)):
        h = (i * GOLDEN_RATIO_CONJUGATE) % 1.0
        s = 0.90 if i % 2 == 0 else 0.62
        v = 0.98 if i % 3 else 0.74
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((round(r * 255), round(g * 255), round(b * 255)))
    return out[:n]


def _srgb_to_linear(c):
    """Scalar sRGB (0..1) -> scene-linear (Blender color sockets are linear)."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def make_id_material(part_name, rgb255):
    """Unlit Emission material whose 8-bit sRGB pixel equals rgb255.

    Emission ignores scene lights and the gray world, so the color is flat and
    view-independent. We linearize because the socket is scene-linear and, under
    the 'Standard' view transform (set by the ID pass), the 8-bit pixel is the
    sRGB-encoding of that linear value.
    """
    mat = bpy.data.materials.new(f"id_pass_{part_name}")
    mat.use_nodes = True
    nt = mat.node_tree
    for node in list(nt.nodes):  # strip the default Principled + Output
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emis = nt.nodes.new("ShaderNodeEmission")
    lin = tuple(_srgb_to_linear(c / 255.0) for c in rgb255)
    emis.inputs["Color"].default_value = (*lin, 1.0)
    emis.inputs["Strength"].default_value = 1.0
    nt.links.new(emis.outputs[0], out.inputs["Surface"])
    return mat


def snapshot_id_state(scene, args):
    """Save the color-management / render state the ID pass overwrites."""
    vs, ds = scene.view_settings, scene.display_settings
    saved = {
        "view_transform": vs.view_transform,
        "look": vs.look,
        "exposure": vs.exposure,
        "gamma": vs.gamma,
        "display_device": ds.display_device,
        "filter_size": scene.render.filter_size,
    }
    if args.engine == "CYCLES":
        saved["samples"] = scene.cycles.samples
        saved["use_denoising"] = scene.cycles.use_denoising
    return saved


def apply_id_state(scene, args):
    """Force a linear, unfiltered, low-sample render so flat colors come out
    exact and edges stay crisp (nearest-color assignment needs no bleed)."""
    vs, ds = scene.view_settings, scene.display_settings
    vs.view_transform = "Standard"   # default AgX would remap our chosen colors
    vs.look = "None"
    vs.exposure = 0.0
    vs.gamma = 1.0
    ds.display_device = "sRGB"
    scene.render.filter_size = 0.01  # avoid AA color-bleed across boundaries
    if args.engine == "CYCLES":
        scene.cycles.samples = 4      # emission is noiseless; samples only = AA
        scene.cycles.use_denoising = False  # a denoiser would smear edge colors


def restore_id_state(scene, saved, args):
    vs, ds = scene.view_settings, scene.display_settings
    vs.view_transform = saved["view_transform"]
    vs.look = saved["look"]
    vs.exposure = saved["exposure"]
    vs.gamma = saved["gamma"]
    ds.display_device = saved["display_device"]
    scene.render.filter_size = saved["filter_size"]
    if args.engine == "CYCLES":
        scene.cycles.samples = saved["samples"]
        scene.cycles.use_denoising = saved["use_denoising"]


def assign_parts(rgba, palette, return_label_map=False):
    """Nearest-palette-color assignment of near-opaque pixels.

    Returns per palette index: (pixel_count, centroid[x, y] or None). Uses a
    tolerance of half the minimum pairwise palette distance so anti-aliased
    edge pixels that blend two parts are dropped rather than miscounted.

    If `return_label_map` is True, also returns an int32 (H, W) array giving the
    assigned palette index per pixel (-1 for background / dropped edge pixels) —
    the front-most part label at each pixel, used by the visibility view to
    attribute occlusion. Backward-compatible: the label map is only appended to
    the tuple when requested.
    """
    h, w, _ = rgba.shape
    n = len(palette)
    counts = [0] * n
    centroids = [None] * n
    label_map = np.full((h, w), -1, dtype=np.int32) if return_label_map else None
    opaque = rgba[:, :, 3] >= 0.5
    if not opaque.any():
        return (counts, centroids, label_map) if return_label_map \
            else (counts, centroids)

    pal = np.array(palette, dtype=np.float32)              # (n, 3) 0..255
    ys, xs = np.nonzero(opaque)
    cols = rgba[ys, xs, :3] * 255.0                        # (k, 3)
    d = np.linalg.norm(cols[:, None, :] - pal[None, :, :], axis=2)  # (k, n)
    nearest = np.argmin(d, axis=1)
    dmin = d[np.arange(len(nearest)), nearest]

    if n > 1:
        pd = np.linalg.norm(pal[:, None, :] - pal[None, :, :], axis=2)
        pd[pd == 0] = np.inf
        tol = max(0.5 * float(pd.min()), 24.0)
    else:
        tol = np.inf
    keep = dmin <= tol

    if label_map is not None:
        label_map[ys[keep], xs[keep]] = nearest[keep]

    for idx in range(n):
        sel = keep & (nearest == idx)
        c = int(sel.sum())
        counts[idx] = c
        if c:
            centroids[idx] = [float(xs[sel].mean()), float(ys[sel].mean())]
    return (counts, centroids, label_map) if return_label_map \
        else (counts, centroids)


def cleanup_id_materials():
    """Remove orphaned id_pass_* materials left after an ID/visibility pass."""
    for m in list(bpy.data.materials):
        if m.name.startswith("id_pass_") and m.users == 0:
            bpy.data.materials.remove(m)


# --------------------------------------------------------------------------- #
# silhouette IoU on a mask grid — the numpy proxy shared by the sweep views
# --------------------------------------------------------------------------- #
# These mirror analysis.scorers.silhouette's iou_raw / iou_visible definitions in pure
# numpy (no cv2 inside Blender), so the unified sweep (views/sweeps/) can score a
# rendered silhouette against an object mask in-process. It is a FAST PROXY at low
# resolution: re-verify a winner with the authoritative cv2 silhouette.py on the
# full-res match render.
def resize_nearest(arr, target_shape):
    """Nearest-neighbour resample a 2-D array to (H, W) via integer index maps
    (reproduces cv2.INTER_NEAREST). dtype-agnostic: bool masks and the float
    Pi3X depth/conf fields go through the same index math."""
    th, tw = target_shape[:2]
    sh, sw = arr.shape[:2]
    if (sh, sw) == (th, tw):
        return arr
    yi = (np.arange(th) * sh // th).clip(0, sh - 1)
    xi = (np.arange(tw) * sw // tw).clip(0, sw - 1)
    return arr[yi[:, None], xi[None, :]]


# legacy name from when this was bool-only — same function.
resize_nearest_bool = resize_nearest


def dilate_bool(mask, r):
    """Square-window binary dilation by radius r (separable numpy max-filter)."""
    m = mask
    for axis in (0, 1):
        acc = m.copy()
        for k in range(1, r + 1):
            acc |= np.roll(m, k, axis=axis)
            acc |= np.roll(m, -k, axis=axis)
        m = acc
    return m


def iou_on_grid(mask_bool, rmask_bool, keep=None):
    """Raw + visible IoU on the MASK grid, mirroring metrics.silhouette_iou.

    The render mask is nearest-resampled onto the mask grid. Empty union -> 0.0
    (NOT metrics.iou()'s 1.0: a blank render must never win a ranking). Visible
    IoU is over K=not-hand; None when that region is empty (mirrors silhouette.py's
    visible_region_empty). Returns (iou_raw, iou_visible)."""
    rm = resize_nearest_bool(rmask_bool, mask_bool.shape)
    inter = int(np.logical_and(mask_bool, rm).sum())
    union = int(np.logical_or(mask_bool, rm).sum())
    iou_raw = (inter / union) if union else 0.0
    iou_vis = None
    if keep is not None:
        mk = mask_bool & keep
        rk = rm & keep
        union_k = int(np.logical_or(mk, rk).sum())
        iou_vis = (int(np.logical_and(mk, rk).sum()) / union_k) if union_k else None
    return iou_raw, iou_vis


class GridIoU:
    """Precomputed EXACT iou_on_grid for a fixed (render grid, mask, keep) trio.

    iou_on_grid nearest-upsamples every candidate's render mask onto the mask
    grid and runs bool ops there — ~2.3ms/candidate on an 840x600 mask, which
    dominates the loop once the render itself is a ~0.4ms GPU raster. But the
    upsample REPLICATES each render pixel (y, x) into a fixed block of mask-grid
    pixels, so all the counts collapse onto the render grid: precompute, per
    render pixel, how many mask-grid pixels it covers (w) and how many of those
    are mask-true (a); then intersection = a·r and upsampled-render-area = w·r
    for a candidate's flat mask r — two small dot products (float64, integer-
    exact), byte-identical to iou_on_grid by construction.

    score(rmask) == iou_on_grid(mask_bool, rmask, keep) for rmask of the fixed
    `render_hw` shape; any other shape falls back to iou_on_grid verbatim.

    The dot products run in float32 when every count fits: all values are
    small non-negative integers whose partial sums are bounded by the mask-grid
    pixel count, so as long as that stays under 2^24 every intermediate is an
    exactly-representable integer in float32 and the result is exact under ANY
    summation order (BLAS blocking/FMA included) — same numbers, less memory
    traffic. A mask grid of 16.7M+ pixels keeps float64. The dots go through
    np.einsum, not `@`: Blender bundles a BLAS-less numpy where dot() is a
    slow reference loop while einsum is SIMD-vectorized (measured 1.8ms ->
    0.5ms/candidate for the 4 dots on an 840x600 grid, identical results)."""

    def __init__(self, mask_bool, keep, render_hw):
        self.mask_bool = mask_bool     # kept only for the odd-shape fallback
        self.keep = keep
        self.shape = (int(render_hw[0]), int(render_hw[1]))
        sh, sw = self.shape
        th, tw = mask_bool.shape
        dt = np.float32 if th * tw < 2 ** 24 else np.float64
        self._dt = dt
        yi = (np.arange(th) * sh // th).clip(0, sh - 1)
        xi = (np.arange(tw) * sw // tw).clip(0, sw - 1)
        flat = (yi[:, None] * sw + xi[None, :]).ravel()
        n = sh * sw
        self.w = np.bincount(flat, minlength=n).astype(dt)
        self.a = np.bincount(flat, weights=mask_bool.ravel().astype(np.float64),
                             minlength=n).astype(dt)
        self.m_count = float(mask_bool.sum())
        if keep is not None:
            mk = mask_bool & keep
            self.wk = np.bincount(flat, weights=keep.ravel().astype(np.float64),
                                  minlength=n).astype(dt)
            self.ak = np.bincount(flat, weights=mk.ravel().astype(np.float64),
                                  minlength=n).astype(dt)
            self.mk_count = float(mk.sum())

    def score(self, rmask_bool):
        """(iou_raw, iou_visible), exactly iou_on_grid's values."""
        if rmask_bool.shape != self.shape:
            return iou_on_grid(self.mask_bool, rmask_bool, self.keep)
        r = rmask_bool.ravel().astype(self._dt)
        dot = np.einsum
        inter = float(dot("i,i->", self.a, r))
        union = self.m_count + float(dot("i,i->", self.w, r)) - inter
        iou_raw = (inter / union) if union else 0.0
        iou_vis = None
        if self.keep is not None:
            inter_k = float(dot("i,i->", self.ak, r))
            union_k = self.mk_count + float(dot("i,i->", self.wk, r)) - inter_k
            iou_vis = (inter_k / union_k) if union_k else None
        return iou_raw, iou_vis


def load_mask_grid(mask_path, hand_path="", hand_dilate=0.0):
    """(object mask bool (H,W), keep=not-hand bool | None) at the mask grid.

    The object mask's native resolution is the common scoring grid. An optional
    hand/occluder mask is nearest-resampled onto it, dilated by `hand_dilate`
    (fraction of the longer side), and inverted to keep=not-hand — so its region
    scores as don't-care and iou_visible ranks over K=not-hand (mirrors silhouette.py).
    """
    mask_bool = load_png_rgba(mask_path)[:, :, 0] > 0.5
    keep = None
    if hand_path:
        hand = load_png_rgba(hand_path)[:, :, 0] > 0.5
        hand = resize_nearest_bool(hand, mask_bool.shape)
        r = int(round(max(0.0, hand_dilate) * max(mask_bool.shape)))
        if r >= 1:
            hand = dilate_bool(hand, r)
        keep = ~hand
    return mask_bool, keep
