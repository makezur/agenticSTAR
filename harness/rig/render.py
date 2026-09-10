"""render.py — resolution, engine config, the render call + match camera.

Also defines the small `Context` bundle every view.render(ctx) receives, and
`requested_views()` which parses/validates the --views list in order.
"""

import math
import os
from dataclasses import dataclass
from typing import Any

import bpy
from mathutils import Matrix

from core import renderer_settings
from rig import camera, imaging


# --------------------------------------------------------------------------- #
# per-view call context
# --------------------------------------------------------------------------- #
@dataclass
class Context:
    """Everything a view.render(ctx) needs, in one uniform bundle.

    args         — the parsed argparse namespace (also carries .base_w/.base_h
                   set during setup).
    cam_obj      — the rig camera object.
    center,radius— world-space bbox of the posed object (this frame).
    frame        — the FrameSpec for the frame currently posed (name, view_index,
                   intrinsics, pose, joint_states); drives per-frame output names
                   and the camera intrinsics. See render_wrapper.FrameSpec.
    """
    args: Any
    cam_obj: Any
    center: Any
    radius: float
    frame: Any = None
    spec: Any = None            # the SceneSpec (shared scale / joints / frames)
    canonical: Any = None       # {part_obj: canonical world matrix} (rest pose)
    frames: Any = None          # the FULL resolved FrameSpec list (whole-run views
                                # like osweep/oapply that score every frame; per-frame
                                # views leave this None and use `frame`)

    @property
    def suffix(self):
        """Per-frame output suffix, e.g. '_000000' -> match_000000.png."""
        if self.frame is None or getattr(self.frame, "name", None) is None:
            return ""
        return "_" + os.path.splitext(str(self.frame.name))[0]

    @property
    def placement(self):
        """The full placement (with shared scale) of the frame currently posed —
        for the pose diagnostics (sweep) that think in whole placements."""
        if self.frame is None:
            return None
        return self.frame.placement

    @property
    def frame_intr(self):
        return None if self.frame is None else getattr(self.frame, "intrinsics", None)

    @property
    def frame_intr_provenance(self):
        """Where this frame's K came from: "explicit" | "tracking" |
        "iphone-placeholder" (the None fallback place_match_camera uses)."""
        if self.frame is None:
            return None
        return getattr(self.frame, "intr_provenance", None) or "iphone-placeholder"


def requested_views(args):
    """Ordered, de-duplicated list of requested view names from --views."""
    seen = []
    for v in args.views.split(","):
        v = v.strip()
        if v and v not in seen:
            seen.append(v)
    return seen


# --------------------------------------------------------------------------- #
# resolution + engine config
# --------------------------------------------------------------------------- #
def image_resolution(path):
    """(width, height) of an image, read via Blender's own loader (no deps)."""
    img = bpy.data.images.load(path, check_existing=True)
    w, h = int(img.size[0]), int(img.size[1])
    bpy.data.images.remove(img)
    if w <= 0 or h <= 0:
        raise ValueError(f"could not read image size from {path}")
    return w, h


def resolve_base_resolution(args):
    """Base (width, height) for rendering, in precedence order:

      1. --match-res <image> -> that image's exact pixel size (use the source
         image so match.png lines up 1:1 with it).
      2. an explicit --intrinsics/--camera-json K's stated W,H (a.explicit_intr_wh)
         -> render at the grid the K was measured on (the K is still rescaled to
         whatever resolution is active, so this is only a sensible default).
      3. --res -> 'N' (square) or 'WxH'.

    The K is always re-absolutized to the resolution actually rendered, so this
    only picks the DEFAULT canvas; --match-res at a different size still works.
    """
    if args.match_res:
        w, h = image_resolution(args.match_res)
        print(f"[render_wrapper] matching source resolution {w}x{h} "
              f"(from {args.match_res})")
        return w, h
    wh = getattr(args, "explicit_intr_wh", None)
    if wh is not None:
        w, h = int(wh[0]), int(wh[1])
        print(f"[render_wrapper] rendering at explicit intrinsics resolution "
              f"{w}x{h}")
        return w, h
    s = str(args.res).lower().replace("×", "x")
    if "x" in s:
        ws, hs = s.split("x", 1)
        return int(ws), int(hs)
    n = int(s)
    return n, n


def configure_render(args):
    scene = bpy.context.scene
    scene.render.engine = args.engine
    base_w, base_h = resolve_base_resolution(args)
    args.base_w, args.base_h = base_w, base_h  # reused by the turntable reset
    scene.render.resolution_x = base_w
    scene.render.resolution_y = base_h
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    # Transparent film -> the alpha channel is an exact object silhouette,
    # independent of the object's color (so dark parts still segment cleanly).
    # The flag lives in core.renderer_settings, shared with the tools that later
    # present these transparent renders on a backdrop (critic/composite).
    scene.render.film_transparent = renderer_settings.FILM_TRANSPARENT
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    # Reuse mesh sync / BVH across frames in one process (free win for the
    # multi-frame turntable; no effect on image quality).
    scene.render.use_persistent_data = True

    if args.engine == "BLENDER_EEVEE_NEXT":
        # EEVEE is a rasterizer: TAA samples only affect anti-aliasing, not
        # noise/GI, so it stays fast. Cap them so we don't over-sample AA.
        scene.eevee.taa_render_samples = min(args.samples, 16)
    elif args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        if args.device == "CPU":
            scene.cycles.device = "CPU"
        else:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            wanted = args.device  # OPTIX or CUDA
            prefs.compute_device_type = wanted
            try:
                prefs.refresh_devices()
            except Exception:
                pass
            enabled = 0
            for d in prefs.devices:
                d.use = d.type in {"OPTIX", "CUDA"}
                enabled += int(d.use)
            if enabled == 0:  # no GPU visible -> fall back so we still render
                print("[render_wrapper] WARNING: no GPU device; using CPU")
                scene.cycles.device = "CPU"
            else:
                scene.cycles.device = "GPU"
                print(f"[render_wrapper] Cycles GPU via {wanted}: "
                      f"{enabled} device(s) enabled")


def render_to(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print(f"[render_wrapper] wrote {path}")


def render_silhouette(path):
    """Render the current scene to `path` and return the alpha>=0.5 bool mask.

    The scoring-loop counterpart to render_to(): folds the PNG decode + alpha
    threshold both sweep views repeat, and drops render_to()'s per-call
    os.makedirs + print (this runs once per candidate — the scratch dir is the
    already-created output dir). Still writes to disk (background-mode in-memory
    pixel reads are unreliable), but the caller sets image compression to 0 for a
    fast encode of the throwaway scratch PNG (see apply_scoring_state).
    """
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return imaging.load_png_rgba(path)[:, :, 3] >= 0.5


# --------------------------------------------------------------------------- #
# sweep scoring helpers — low-res render + non-destructive camera/render state
# --------------------------------------------------------------------------- #
def snapshot_scoring_state():
    """Save the render samples + PNG compression the sweep scoring loop lowers.

    Engine-aware and independent of snapshot_camera_render (which owns resolution
    + intrinsics): samples/compression are a separate concern, kept apart so the
    full-res winner re-render restores them cleanly on their own.
    """
    scn = bpy.context.scene
    s = {"engine": scn.render.engine,
         "compression": scn.render.image_settings.compression}
    if scn.render.engine == "BLENDER_EEVEE_NEXT":
        s["taa_render_samples"] = scn.eevee.taa_render_samples
    elif scn.render.engine == "CYCLES":
        s["cycles_samples"] = scn.cycles.samples
    return s


def apply_scoring_state(samples=1):
    """Drop AA/path samples to `samples` and PNG compression to 0 for scoring.

    Silhouette IoU is a binary alpha>=0.5 coverage test, so AA beyond 1 sample
    only slows the raster without changing the mask; compression=0 trades file
    size (a reused throwaway scratch PNG) for the fastest possible encode.
    """
    scn = bpy.context.scene
    scn.render.image_settings.compression = 0
    if scn.render.engine == "BLENDER_EEVEE_NEXT":
        scn.eevee.taa_render_samples = max(1, int(samples))
    elif scn.render.engine == "CYCLES":
        scn.cycles.samples = max(1, int(samples))


def restore_scoring_state(s):
    """Restore a snapshot_scoring_state() (call BEFORE the full-res winner
    re-render so the winner isn't a 1-sample aliased raster). Idempotent."""
    scn = bpy.context.scene
    scn.render.image_settings.compression = s["compression"]
    if s["engine"] == "BLENDER_EEVEE_NEXT":
        scn.eevee.taa_render_samples = s["taa_render_samples"]
    elif s["engine"] == "CYCLES":
        scn.cycles.samples = s["cycles_samples"]



# floor for a scoring render's long side (px) so a low quality ratio can't
# degrade the silhouette into noise.
_MIN_SWEEP_LONGSIDE = 96
# ceiling for a scoring render's long side (px). The dominant per-candidate cost
# is the PNG encode/decode round trip, which scales with pixel count (measured:
# ~40ms/candidate at 410px long side, ~250ms at 1024px). The silhouette-IoU proxy
# does not benefit from more than a few hundred px, and the winner is always
# re-rendered + re-verified at full match-res — so cap the scoring side here to
# keep a hi-res frame from making every candidate a slow full-res raster.
_MAX_SWEEP_LONGSIDE = 512


def clamp_quality(q, default=1.0):
    """Clamp a --*-quality ratio into (0.05, 1.0]; fall back to `default`."""
    try:
        q = float(q)
    except (TypeError, ValueError):
        q = default
    return min(1.0, max(0.05, q))


def sweep_resolution(base_w, base_h, quality):
    """Scoring render resolution = quality x match-res, clamped on the long side.

    Shared by the pose sweep and joint sweep: they render candidates at a low
    resolution for speed, then re-render only the winner at full match-res. The
    long side is floored (a low quality can't degrade the silhouette into noise)
    and CAPPED (a hi-res frame can't make every candidate a slow full-res raster —
    the PNG round trip per candidate scales with pixel count).
    """
    q = clamp_quality(quality)
    w = max(1, int(round(base_w * q)))
    h = max(1, int(round(base_h * q)))
    long_side = max(w, h)
    if long_side < _MIN_SWEEP_LONGSIDE:
        s = _MIN_SWEEP_LONGSIDE / long_side
        w, h = max(1, int(round(w * s))), max(1, int(round(h * s)))
    elif long_side > _MAX_SWEEP_LONGSIDE:
        s = _MAX_SWEEP_LONGSIDE / long_side
        w, h = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return w, h


def snapshot_camera_render(cam_obj):
    """Save the render resolution + camera intrinsics a sweep view overwrites."""
    scn = bpy.context.scene
    cd = cam_obj.data
    return {
        "res_x": scn.render.resolution_x, "res_y": scn.render.resolution_y,
        "pax": scn.render.pixel_aspect_x, "pay": scn.render.pixel_aspect_y,
        "type": cd.type, "lens_unit": cd.lens_unit, "lens": cd.lens,
        "angle": cd.angle, "shift_x": cd.shift_x, "shift_y": cd.shift_y,
    }


def restore_camera_render(cam_obj, s):
    """Restore a snapshot_camera_render() state (non-destructive sweep)."""
    scn = bpy.context.scene
    cd = cam_obj.data
    scn.render.resolution_x = s["res_x"]
    scn.render.resolution_y = s["res_y"]
    scn.render.pixel_aspect_x = s["pax"]
    scn.render.pixel_aspect_y = s["pay"]
    cd.type = s["type"]
    cd.lens_unit = s["lens_unit"]
    cd.lens = s["lens"]
    cd.angle = s["angle"]
    cd.shift_x = s["shift_x"]
    cd.shift_y = s["shift_y"]


def place_match_camera(args, cam_obj, center, radius, frame_intr=None):
    """Set up the fixed camera-0 (identity extrinsic) with a frame's intrinsics.

    frame_intr — a cam_math.NormIntrinsics (explicit --intrinsics/--camera-json
    K, or a Pi3X K[k]) for this frame, or None. None -> synthesize the iPhone-13
    PLACEHOLDER K from the CURRENT render resolution (and warn loudly).

    RESOLUTION-AGNOSTIC: the K is re-absolutized to whatever
    scene.render.resolution is set to RIGHT NOW (the caller owns resolution — the
    match view uses the base res; the sweeps set a low scoring res; crop/vis grow
    to an overscan).

    Returns the spec with 'intr_effective' = the pixel-K actually applied (at the
    current resolution) and 'intr_provenance'."""
    scn = bpy.context.scene
    W, H = scn.render.resolution_x, scn.render.resolution_y
    cam = camera.resolve_camera(args, frame_intr)
    cam_obj.matrix_world = cam["c2w"]           # camera-0 identity extrinsic
    cam_obj.data.type = "PERSP"
    norm = cam["intr"]
    if norm is None and cam.get("intr_default"):
        intr = camera.iphone13_intrinsics(W, H)  # synthesized at current res
        _warn_placeholder_intrinsics(intr)
        prov = "iphone-placeholder"
    else:
        intr = norm.at(W, H)                      # re-absolutize to current res
        prov = getattr(norm, "provenance", None) or "tracking"
    camera.apply_intrinsics(cam_obj.data, intr)
    cam["intr_effective"] = intr          # the pixel K applied at (W, H)
    cam["intr_provenance"] = prov
    return cam


_WARNED_PLACEHOLDER = False


def _warn_placeholder_intrinsics(intr):
    """Prominent one-time stderr banner when NO real intrinsics are available and
    the iPhone-13 placeholder is used (so a guessed K is never silent)."""
    global _WARNED_PLACEHOLDER
    if _WARNED_PLACEHOLDER:
        return
    _WARNED_PLACEHOLDER = True
    import sys
    bar = "[render_wrapper] " + "!" * 58
    print(bar, file=sys.stderr)
    print("[render_wrapper] WARNING: NO real intrinsics — using iPhone-13 "
          "PLACEHOLDER K", file=sys.stderr)
    print(f"[render_wrapper]   fx=fy={intr['fx']:.1f}px @ "
          f"{intr['width']}x{intr['height']}; pass --intrinsics / --camera-json "
          "/ --tracking for a real K", file=sys.stderr)
    print(bar, file=sys.stderr)
