"""depth view — render a per-pixel DEPTH map from the fixed camera 0 -> depth.npy.

The geometric counterpart of `match`: same posed object, same camera 0 (identity
extrinsic + the resolved intrinsics), but instead of the RGBA silhouette this
writes the camera-space Z of every pixel as a float32 numpy array. That depth is
what `analysis.scorers.depth` compares against the capture's observed pointmap (whose
channel-2 is the same planar camera-frame depth), so a part floating at the wrong
DEPTH — which silhouette IoU is blind to — finally shows up as a number.

Blender's Z pass is PLANAR camera-space depth (distance along the view axis, not
euclidean ray length), which matches the observed `local[...,2]`. Background (no geometry)
pixels are set to +inf using the render's alpha as the foreground mask, so the
analysis side selects the object with a plain `np.isfinite`, independent of the
camera clip planes (EEVEE fills empty pixels with clip_end, not a huge sentinel).

Output is .npy rather than .exr because the analysis env's OpenCV cannot decode
OpenEXR; Blender has numpy, so we read the Z pass back and save a float32 array
that `np.load` opens with no codec dependency.

Multi-frame: every frame renders from the SAME fixed camera 0 (identity
extrinsic) — we MOVE THE OBJECT into each frame's camera (M_k = T_{k<-ref} @ M_ref,
plus that frame's joint states) and only swap in the frame's intrinsics K[k]. So
the rendered depth is already in frame-k's camera frame and compares RAW against
the observed local[k] — no per-view pose baked into the camera here.

Non-destructive: the compositor tree, view-layer pass flags, and camera/render
resolution+intrinsics are all snapshotted and restored, so running `depth`
alongside `match`/`turntable` in one Blender process leaves those outputs, the
pose.json, and the exported GLB untouched (mirrors views/sweep.py).
"""

import glob
import os

import bpy
import numpy as np

from rig import camera, imaging, render


def render_view(ctx):
    a = ctx.args
    saved = _snapshot(ctx.cam_obj)
    try:
        render.place_match_camera(a, ctx.cam_obj, ctx.center, ctx.radius,
                                  ctx.frame_intr)
        if a.debug_project:
            camera.debug_project(ctx.cam_obj, a.debug_project)
        out_path = os.path.join(a.out, f"depth{ctx.suffix}.npy")
        depth = _render_depth(a.out)
        np.save(out_path, depth)
        n_fg = int(np.isfinite(depth).sum())
        print(f"[render_wrapper] wrote {out_path} "
              f"({n_fg}/{depth.size} foreground px)")
    finally:
        _restore(ctx.cam_obj, saved)


# --------------------------------------------------------------------------- #
# depth pass via the compositor File Output node (robust headless in 4.2)
# --------------------------------------------------------------------------- #
def _render_depth(out_dir):
    """Render the Z + Alpha passes to EXR, read them back, return a float32
    (H, W) planar-depth array with background (alpha < 0.5) set to +inf.

    The File Output node appends the frame number to each slot filename (e.g.
    'depth0000.exr'); we render frame 0 into a temp subdir, read both channels
    back through Blender's own image loader (numpy, no cv2), then clean up.
    """
    scene = bpy.context.scene
    vl = bpy.context.view_layer
    vl.use_pass_z = True

    scene.use_nodes = True
    tree = scene.node_tree
    rlayers = tree.nodes.new("CompositorNodeRLayers")
    fout = tree.nodes.new("CompositorNodeOutputFile")
    fout.format.file_format = "OPEN_EXR"
    fout.format.color_mode = "RGB"
    fout.format.color_depth = "32"
    tmp_dir = os.path.join(out_dir, "_depth_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    fout.base_path = tmp_dir
    # slot 0 = camera-space Z (the 'Depth' pass); slot 1 = coverage alpha. The
    # input sockets keep their default names ("Image", "alpha"); the slot .path
    # sets only the output filename, so link by index/socket, not by path.
    fout.file_slots[0].path = "depth"
    fout.file_slots.new("alpha")
    tree.links.new(rlayers.outputs["Depth"], fout.inputs[0])
    tree.links.new(rlayers.outputs["Alpha"], fout.inputs[1])

    scene.frame_set(0)
    bpy.ops.render.render(write_still=False)  # File Output writes on render

    depth = _read_exr_channel(tmp_dir, "depth")
    alpha = _read_exr_channel(tmp_dir, "alpha")
    for leftover in glob.glob(os.path.join(tmp_dir, "*")):
        os.remove(leftover)
    os.rmdir(tmp_dir)

    depth = depth.astype(np.float32)
    # Background pixels carry the camera far-plane (clip_end) as their Z, and
    # anti-aliased silhouette edges — kept by alpha >= 0.5 — get a TAA-averaged
    # depth pulled toward clip_end (>> the object's real few-unit depth). Reject
    # both: uncovered (alpha < 0.5) OR any depth in the far half of the frustum,
    # which real geometry filling the frame never reaches.
    clip_end = float(bpy.context.scene.camera.data.clip_end)
    invalid = (alpha < 0.5) | (depth >= 0.5 * clip_end)
    depth[invalid] = np.inf  # no reliable geometry here
    return depth


def _read_exr_channel(tmp_dir, slot):
    """Load '<slot>####.exr' from tmp_dir into a top-down (H, W) float32 array."""
    hits = sorted(glob.glob(os.path.join(tmp_dir, f"{slot}*.exr")))
    if not hits:
        raise RuntimeError(f"depth pass produced no '{slot}' EXR in {tmp_dir}")
    img = bpy.data.images.load(hits[0], check_existing=False)
    w, h = int(img.size[0]), int(img.size[1])
    ch = img.channels
    px = imaging.image_pixels(img).reshape(h, w, ch)
    bpy.data.images.remove(img)
    return px[::-1, :, 0]  # Blender stores bottom-up; flip to top-down, take R


class RenderedDepthRider:
    """Capture camera-Z beside an existing RGB render, without rendering twice.

    ``setup`` adds a compositor File Output node. After the caller's normal
    ``render.render_to(...)`` call, ``save`` reads that render's Z pass, masks it
    with the PNG alpha, and writes a float32 ``.npy``. ``teardown`` removes only
    the nodes this rider added.

    This is presentation-only machinery for sweep candidate panels. It does not
    load observed depth and cannot affect candidate scoring or ranking.
    """

    def __init__(self, out_dir, stem="_panel_depth"):
        self.tmp_dir = os.path.join(out_dir, stem)
        self._saved = None
        self._nodes = ()
        self._far = None

    def setup(self):
        scn = bpy.context.scene
        vl = bpy.context.view_layer
        self._saved = {"use_nodes": scn.use_nodes, "use_pass_z": vl.use_pass_z}
        vl.use_pass_z = True
        scn.use_nodes = True
        tree = scn.node_tree
        rl = tree.nodes.new("CompositorNodeRLayers")
        self._nodes += (rl,)
        comp = tree.nodes.new("CompositorNodeComposite")
        self._nodes += (comp,)
        comp.use_alpha = True
        tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
        tree.links.new(rl.outputs["Alpha"], comp.inputs["Alpha"])
        fout = tree.nodes.new("CompositorNodeOutputFile")
        self._nodes += (fout,)
        fout.format.file_format = "OPEN_EXR"
        fout.format.color_mode = "RGB"
        fout.format.color_depth = "32"
        os.makedirs(self.tmp_dir, exist_ok=True)
        fout.base_path = self.tmp_dir
        fout.file_slots[0].path = "depth"
        tree.links.new(rl.outputs["Depth"], fout.inputs[0])
        scn.frame_set(0)
        self._far = 0.5 * float(scn.camera.data.clip_end)

    def save(self, render_path, out_path):
        depth = _read_exr_channel(self.tmp_dir, "depth").astype(np.float32)
        alpha = imaging.load_png_rgba(render_path)[:, :, 3]
        if alpha.shape != depth.shape:
            raise RuntimeError(
                f"panel depth shape {depth.shape} != render alpha {alpha.shape}")
        depth[(alpha < 0.5) | (depth >= self._far)] = np.inf
        np.save(out_path, depth)
        for leftover in glob.glob(os.path.join(self.tmp_dir, "*")):
            os.remove(leftover)
        print(f"[render_wrapper] wrote {out_path}")

    def teardown(self):
        if self._saved is None:
            return
        scn = bpy.context.scene
        tree = scn.node_tree
        if tree is not None:
            for node in self._nodes:
                try:
                    tree.nodes.remove(node)
                except (ReferenceError, RuntimeError):
                    pass
        scn.use_nodes = self._saved["use_nodes"]
        bpy.context.view_layer.use_pass_z = self._saved["use_pass_z"]
        for leftover in glob.glob(os.path.join(self.tmp_dir, "*")):
            os.remove(leftover)
        if os.path.isdir(self.tmp_dir):
            os.rmdir(self.tmp_dir)
        self._saved = None
        self._nodes = ()


# --------------------------------------------------------------------------- #
# snapshot / restore  (keep match/turntable/pose.json/GLB unaffected)
# --------------------------------------------------------------------------- #
def _snapshot(cam_obj):
    # the camera/render state apply_intrinsics touches is the shared sweep set;
    # the depth view additionally toggles the compositor + Z pass, saved here.
    scn = bpy.context.scene
    saved = render.snapshot_camera_render(cam_obj)
    saved["use_nodes"] = scn.use_nodes
    saved["use_pass_z"] = bpy.context.view_layer.use_pass_z
    return saved


def _restore(cam_obj, s):
    scn = bpy.context.scene
    # Drop any compositor nodes we added, then restore the prior use_nodes flag +
    # Z pass, and finally the shared camera/render state.
    if scn.node_tree is not None:
        for node in list(scn.node_tree.nodes):
            scn.node_tree.nodes.remove(node)
    scn.use_nodes = s["use_nodes"]
    bpy.context.view_layer.use_pass_z = s["use_pass_z"]
    render.restore_camera_render(cam_obj, s)
