"""ids view — per-part flat-color ID map (match view) + a name/color legend.

Each part in the 'parts' collection is flat-colored with a distinct unlit
emission color and rendered from the match camera, so the agent can see WHICH
primitive produced which pixels. Materials, color management, and Cycles
sampling are all snapshotted and restored, so this leaves the scene
byte-for-byte ready for GLB export.
"""

import json
import os

import bpy

from rig import imaging, render, scene


def _write_legend(out_dir, manifest, resolution):
    """Write ids.json (machine) + ids.txt (human) legend files."""
    report = {
        "view": "match",
        "image": "ids.png",
        "resolution": list(resolution),
        "parts": manifest,
    }
    with open(os.path.join(out_dir, "ids.json"), "w") as f:
        json.dump(report, f, indent=2)

    name_w = max((len(p["name"]) for p in manifest), default=4)
    lines = [
        f"ID pass legend ({len(manifest)} parts) — match view, "
        f"{resolution[0]}x{resolution[1]}",
        f"{'idx':>3}  {'name':<{name_w}}  {'RGB':<15}  {'pixels':>8}  status",
    ]
    for p in manifest:
        r, g, b = p["color"]
        status = "OCCLUDED" if p["occluded"] else ""
        lines.append(
            f"{p['index']:>3}  {p['name']:<{name_w}}  "
            f"({r:>3},{g:>3},{b:>3})  {p['pixel_count']:>8}  {status}"
        )
    with open(os.path.join(out_dir, "ids.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def render_view(ctx):
    a = ctx.args
    scene_data = bpy.context.scene
    parts = scene.ensure_parts_collection()
    objs = sorted((o for o in parts.objects if o.type == "MESH"),
                  key=lambda o: o.name)
    if not objs:
        print("[render_wrapper] ids: no parts to render")
        return
    palette = imaging.id_palette(len(objs))

    # NOTE: swap materials at the mesh-data level (mirrors set_color); parts have
    # distinct meshes, so this does not cross-contaminate. Per-face material_index
    # is preserved on the polygons, so multi-slot parts restore cleanly.
    saved_mats = {o: list(o.data.materials) for o in objs}
    saved_state = imaging.snapshot_id_state(scene_data, a)
    try:
        for i, o in enumerate(objs):
            mat = imaging.make_id_material(o.name, palette[i])
            o.data.materials.clear()
            o.data.materials.append(mat)
        imaging.apply_id_state(scene_data, a)
        render.place_match_camera(a, ctx.cam_obj, ctx.center, ctx.radius,
                                  ctx.frame_intr)
        ids_path = os.path.join(a.out, "ids.png")
        render.render_to(ids_path)

        # Recover per-part pixel counts + centroids straight from the render.
        rgba = imaging.load_png_rgba(ids_path)
        counts, centroids = imaging.assign_parts(rgba, palette)
        res = (scene_data.render.resolution_x, scene_data.render.resolution_y)
        manifest = [
            {
                "index": i,
                "name": o.name,
                "color": list(palette[i]),
                "pixel_count": counts[i],
                "centroid": centroids[i],
                "occluded": counts[i] == 0,
            }
            for i, o in enumerate(objs)
        ]
        _write_legend(a.out, manifest, res)
        n_occ = sum(1 for p in manifest if p["occluded"])
        print(f"[render_wrapper] ids: {len(objs)} parts "
              f"({n_occ} occluded) -> ids.png, ids.json, ids.txt")
    finally:
        for o in objs:
            o.data.materials.clear()
            for m in saved_mats[o]:
                o.data.materials.append(m)
        imaging.restore_id_state(scene_data, saved_state, a)
        imaging.cleanup_id_materials()
