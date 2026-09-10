"""raster.py — direct GPU silhouette rasterizer for the sweep scoring loops.

The sweeps score CANDIDATE POSES by silhouette IoU (+ optional depth error): a
binary coverage test that needs no shading, AA, or PNG. Rendering each candidate
through `bpy.ops.render.render` pays EEVEE's full per-frame pipeline (measured
33ms/candidate at 294x210, 83% of the loop); drawing the same evaluated meshes
into a `gpu.types.GPUOffScreen` with a flat depth-tested shader produces a
PIXEL-IDENTICAL alpha>=0.5 mask in ~0.4ms (measured 50x), and the framebuffer's
depth attachment linearizes to the same camera-Z the EXR Z-pass rider carried —
so DepthSupervision needs no compositor File Output node on this path either.

Contract with the engines:
  * `maybe_create(cam_obj, objs, w, h)` -> a SilhouetteRaster or None. None means
    "fall back to render.render_silhouette" — creation self-probes (the gpu
    module needs a GL context, which exists in --background only after one real
    render; maybe_create pays that warmup render itself) and NEVER raises.
    SWEEP_GPU_RASTER=0 disables the path (byte-level A/B against EEVEE).
  * Meshes are captured ONCE at creation (evaluated, modifiers applied, in
    OBJECT-LOCAL coords): the scoring loop only moves `matrix_world`
    (scene.pose_frame), never edits geometry, so batches stay valid across
    candidates, refine passes, and frames.
  * `update_camera()` re-reads the view/projection off the scene camera — call it
    after each render.place_match_camera (per-frame intrinsics; honors the
    anamorphic pixel_aspect apply_K encodes).
  * `silhouette()` -> bool (H, W) top-down mask, exactly render_silhouette's
    alpha>=0.5 semantics for opaque parts. `silhouette_and_depth()` also returns
    the float32 camera-Z grid (background = clip_end, like the Z pass).

Limitation: coverage of the drawn triangles IS the silhouette — a part whose
EEVEE material is alpha-blended below 0.5 would differ. Harness scenes build
opaque solids, and the full-res winner re-render still goes through EEVEE.
"""

import os

import bpy
import numpy as np


def _linearize_depth(dbuf, proj):
    """Framebuffer depth in [0,1] -> camera-space Z distance (positive).

    For a GL perspective projection P (row-major mathutils), clip.z/clip.w =
    (P22*z_e + P23) / (-z_e) with z_e the (negative) eye-space z, so
    z_e = -P23 / (P22 + ndc) and the camera-plane distance is -z_e. Background
    (depth=1.0) comes out exactly at the far clip, matching the Z pass's
    clip_end fill. Pure numpy (unit-testable without a GPU)."""
    ndc = 2.0 * dbuf - 1.0
    c, d = float(proj[2][2]), float(proj[2][3])
    return d / (c + ndc)


class SilhouetteRaster:
    """One offscreen + prebuilt mesh batches; draw at the current matrix_world."""

    def __init__(self, cam_obj, objs, width, height):
        import gpu
        from gpu_extras.batch import batch_for_shader

        self._gpu = gpu
        self.cam_obj = cam_obj
        self.w, self.h = int(width), int(height)
        self.offs = gpu.types.GPUOffScreen(self.w, self.h)
        self.shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        self._ubuf = np.empty(self.w * self.h, dtype=np.uint8)
        self._dbuf = np.empty(self.w * self.h, dtype=np.float32)
        self.view_m = None
        self.proj_m = None

        # capture the evaluated meshes ONCE (object-local coords; the loop only
        # moves matrix_world). Empty parts (no triangles) simply don't draw.
        deps = bpy.context.evaluated_depsgraph_get()
        self.batches = []
        for o in objs:
            ev = o.evaluated_get(deps)
            me = ev.to_mesh()
            me.calc_loop_triangles()
            n_v, n_t = len(me.vertices), len(me.loop_triangles)
            if n_v and n_t:
                co = np.empty(n_v * 3, dtype=np.float32)
                me.vertices.foreach_get("co", co)
                idx = np.empty(n_t * 3, dtype=np.int32)
                me.loop_triangles.foreach_get("vertices", idx)
                self.batches.append((batch_for_shader(
                    self.shader, "TRIS", {"pos": co.reshape(-1, 3)},
                    indices=idx.reshape(-1, 3)), o))
            ev.to_mesh_clear()
        self.update_camera()

    def update_camera(self):
        """Re-read view/projection off the scene camera (call after each
        place_match_camera: per-frame K -> a new projection; pixel_aspect is the
        anamorphic-K encoding apply_K uses, folded in via scale_x/scale_y)."""
        scn = bpy.context.scene
        deps = bpy.context.evaluated_depsgraph_get()
        self.view_m = self.cam_obj.matrix_world.inverted()
        self.proj_m = self.cam_obj.calc_matrix_camera(
            deps, x=self.w, y=self.h,
            scale_x=scn.render.pixel_aspect_x,
            scale_y=scn.render.pixel_aspect_y)

    def _draw(self):
        gpu = self._gpu
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(True)
        gpu.state.face_culling_set("NONE")   # mirrored parts (negative scale)
        with gpu.matrix.push_pop():
            gpu.matrix.load_matrix(self.view_m)
            gpu.matrix.load_projection_matrix(self.proj_m)
            self.shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
            for batch, o in self.batches:
                with gpu.matrix.push_pop():
                    gpu.matrix.multiply_matrix(o.matrix_world)
                    batch.draw(self.shader)
        gpu.state.depth_test_set("NONE")
        return fb

    def _read_mask(self, fb):
        """Coverage mask off the bound framebuffer via a 1-channel UBYTE read.

        The color buffer only ever holds 0.0 or 1.0 (one flat color, no
        blending/AA), so red as a byte (0|255, thresholded at 128) equals the
        float alpha>=0.5 test — verified byte-identical — while the readback
        drops 16x in size (measured 5.3ms -> 0.4ms/candidate at 840x600; the
        4-channel FLOAT read was ~90% of silhouette())."""
        pix = fb.read_color(0, 0, self.w, self.h, 1, 0, "UBYTE")
        pix.dimensions = self.w * self.h
        self._ubuf[:] = pix
        return self._ubuf.reshape(self.h, self.w)[::-1] >= 128

    def silhouette(self):
        """bool (H, W) top-down coverage mask of the parts as currently posed."""
        with self.offs.bind():
            fb = self._draw()
            return self._read_mask(fb)

    def silhouette_and_depth(self):
        """(mask, camera-Z float32 (H, W) top-down; background = clip_end)."""
        with self.offs.bind():
            fb = self._draw()
            mask = self._read_mask(fb)
            dep = fb.read_depth(0, 0, self.w, self.h)
        dep.dimensions = self.w * self.h
        self._dbuf[:] = dep
        z = _linearize_depth(self._dbuf.reshape(self.h, self.w)[::-1],
                             self.proj_m)
        return mask, z.astype(np.float32, copy=False)

    def free(self):
        self.offs.free()
        self.batches = []


def maybe_create(cam_obj, objs, width, height, label="sweep"):
    """A SilhouetteRaster, or None -> caller uses the per-candidate EEVEE path.

    Never raises: the gpu module has no GL context in --background until the
    first real render, so on a cold process this pays ONE warmup render (at the
    caller's already-applied scoring res/samples), then retries. Any other
    failure logs and falls back. SWEEP_GPU_RASTER=0 forces the fallback."""
    if os.environ.get("SWEEP_GPU_RASTER", "1") == "0":
        print(f"[render_wrapper] {label}: SWEEP_GPU_RASTER=0 -> per-candidate "
              "EEVEE renders")
        return None
    if not objs:
        return None
    try:
        return SilhouetteRaster(cam_obj, objs, width, height)
    except Exception:
        try:
            bpy.ops.render.render(write_still=False)   # create the GL context
            return SilhouetteRaster(cam_obj, objs, width, height)
        except Exception as e:
            print(f"[render_wrapper] {label}: GPU raster unavailable ({e}); "
                  "falling back to per-candidate EEVEE renders")
            return None
