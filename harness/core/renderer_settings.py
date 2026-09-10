"""renderer_settings.py — the ONE home for how renders treat their background.

The harness renders on a TRANSPARENT film (`FILM_TRANSPARENT`) so the alpha
channel is an exact, colour-independent object silhouette (dark parts still
segment cleanly — see rig/render.py `configure_render`). That single choice then
ripples into every tool that later has to *present* a transparent render on some
visible background:

  * critic.py  — composites the match/turntable renders (and the masked source)
                 onto ONE backdrop before sending them to the VLM, so a dark
                 object never washes out on an uncontrolled background.
  * composite.py — paints its silhouette-overlap panel on that backdrop.

Before this module each of those re-declared its own background colour and
re-implemented the same alpha-over-solid blend. They now share the constants +
the blend here, so the render flag and the backdrop are defined ONCE.

PURE NUMPY — no bpy, no cv2 — so it is importable from Blender's bundled python
(rig/render.py) AND the artscript env (analysis/*) alike, per core/__init__.py.
Colours are **BGR** (the cv2/OpenCV order every analysis tool already uses).
"""

import numpy as np


# --------------------------------------------------------------------------- #
# render-time film flag (consumed by rig/render.py configure_render)
# --------------------------------------------------------------------------- #
# Render on a transparent film so the alpha channel is an exact object
# silhouette, independent of the object's colour. The silhouette/depth scorers
# rely on this alpha; do not flip it off lightly.
FILM_TRANSPARENT = True


# --------------------------------------------------------------------------- #
# present-time backdrop (consumed by critic.py / composite.py)
# --------------------------------------------------------------------------- #
# The uniform solid backdrop a transparent render is composited onto when a tool
# wants a controlled, opaque background (BGR). Swap here to change it everywhere.
BACKDROP_BGR = (0, 0, 0)

# The two ways a tool may present a transparent render:
#   "black" — composite onto BACKDROP_BGR (a uniform, opaque, controlled canvas)
#   "alpha" — keep the transparent film as-is (RGBA out)
BG_MODES = ("alpha", "black")


def backdrop_phrase(bg_mode):
    """Human wording for a bg_mode, for a prompt/label describing the backdrop."""
    return ("a uniform black studio backdrop" if bg_mode == "black"
            else "a transparent backdrop")


def composite_over(bgr, alpha, bg=BACKDROP_BGR):
    """Alpha-composite a BGR image over a solid BGR colour. Pure numpy.

    bgr   — (H, W, 3) uint8 colour.
    alpha — (H, W) uint8 [0..255] object coverage (the render's transparent film).
    bg    — BGR tuple for the solid backdrop (defaults to BACKDROP_BGR).
    Returns (H, W, 3) uint8. An already-opaque image (alpha all 255) is a no-op.
    """
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    back = np.array(bg, dtype=np.float32)
    out = bgr.astype(np.float32) * a + back * (1.0 - a)
    return out.astype(np.uint8)


def present(bgr, alpha, bg_mode):
    """Present a transparent render in the chosen bg_mode (see BG_MODES).

    "black" -> a (H, W, 3) BGR image composited onto BACKDROP_BGR.
    "alpha" -> a (H, W, 4) BGRA image. Fully-masked-out pixels (alpha == 0) also
               get their BGR zeroed, so the region is truly transparent — no
               original photo content bleeds through if a viewer flattens the
               alpha channel. Partial-alpha edges keep their colour.
    """
    if bg_mode == "black":
        return composite_over(bgr, alpha)
    a = alpha.astype(np.uint8)
    bgr = np.where((a == 0)[:, :, None], 0, bgr).astype(np.uint8)
    return np.dstack([bgr, a])
