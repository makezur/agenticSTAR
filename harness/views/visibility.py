"""visibility view — per-part where-it-lands + occlusion + framing.

Extends the ID pass from a binary occluded flag to a full per-part report:
visible AND full (unoccluded) bbox, visible_fraction, which parts occlude it,
and per-part frame clipping.

Method (N+1 fast EEVEE renders, all restored afterwards):
  * one COMPOSITE ID render (all parts, distinct emission colors) -> the
    front-most part label at every pixel (reuses the ID palette + label map).
  * one SOLO render per part (others hidden) -> the transparent-film alpha is
    that part's FULL, unoccluded silhouette, so we recover where it WOULD land
    if nothing were in front, even for a fully hidden part.
Comparing the two gives visible_fraction and occluder attribution; a modest
overscan frame (like the crop view) also measures per-part frame clipping.
"""

import json
import os

import bpy
import numpy as np

from rig import camera, imaging, render, scene


def _clip_stats(mask_bool, padx, pady, W, H):
    """Per-part framing stats from a solo silhouette in the OVERSCAN frame.

    Thin wrapper over imaging.frame_clip_stats (shared with the crop view) that
    adds the per-part in_frame_fraction / clipped fields the visibility report
    uses. All fractions are of the part's total (overscan) silhouette."""
    s = imaging.frame_clip_stats(mask_bool, padx, pady, W, H)
    total, inside = s["total_px"], s["inside_px"]
    return {"in_frame_fraction": round(inside / total, 4) if total else 0.0,
            "clip": s["edges"],
            "clipped": total > 0 and inside < total,
            "fully_captured_in_overscan": total > 0 and s["fully_captured_in_overscan"],
            "in_frame_px": inside, "total_px": total}


def _occlusion_status(visible_px, full_px):
    """Occlusion enum from in-frame visible vs. full footprint pixel counts.
    Framing is reported separately (clip/in_frame_fraction), so this is purely
    about what's IN FRONT of the part."""
    vf = (visible_px / full_px) if full_px > 0 else None
    if visible_px == 0:
        return "OCCLUDED", (0.0 if full_px > 0 else None)
    if vf is not None and vf >= 0.99:
        return "VISIBLE", round(vf, 4)
    return "PARTIAL", (round(vf, 4) if vf is not None else None)


def render_view(ctx):
    a = ctx.args
    scene_data = bpy.context.scene
    parts = scene.ensure_parts_collection()
    objs = sorted((o for o in parts.objects if o.type == "MESH"),
                  key=lambda o: o.name)
    if not objs:
        print("[render_wrapper] visibility: no parts to render")
        return
    n = len(objs)
    palette = imaging.id_palette(n)

    saved_mats = {o: list(o.data.materials) for o in objs}
    saved_hide = {o: o.hide_render for o in objs}
    saved_state = imaging.snapshot_id_state(scene_data, a)
    saved_pax = scene_data.render.pixel_aspect_x
    saved_pay = scene_data.render.pixel_aspect_y
    saved_rx = scene_data.render.resolution_x
    saved_ry = scene_data.render.resolution_y

    records = None
    try:
        # 1) flat-color every part with a distinct emission ID color.
        for i, o in enumerate(objs):
            mat = imaging.make_id_material(o.name, palette[i])
            o.data.materials.clear()
            o.data.materials.append(mat)
        imaging.apply_id_state(scene_data, a)

        # 2) resolve the match camera + its intrinsics, then widen to an overscan
        #    frame so parts spilling off-frame are still captured.
        cam = render.place_match_camera(a, ctx.cam_obj, ctx.center, ctx.radius,
                                        ctx.frame_intr)
        intr = camera.effective_intrinsics(ctx.cam_obj, cam, scene_data)
        W, H = int(intr["width"]), int(intr["height"])
        over, (padx, pady) = camera.overscan_intrinsics(intr, a.visibility_margin)
        OW, OH = int(over["width"]), int(over["height"])
        # grow the film to the overscan.
        scene_data.render.resolution_x, scene_data.render.resolution_y = OW, OH
        camera.apply_intrinsics(ctx.cam_obj.data, over)

        # 3) composite render (all parts) -> front-most part label per pixel.
        comp_path = os.path.join(a.out, "_vis_composite.png")
        for o in objs:
            o.hide_render = False
        render.render_to(comp_path)
        comp_rgba = imaging.load_png_rgba(comp_path)  # (OH, OW, 4) top-down
        _, _, label_map = imaging.assign_parts(comp_rgba, palette, return_label_map=True)
        label_inner = label_map[pady:pady + H, padx:padx + W]

        # 4) one solo render per part -> its full (unoccluded) silhouette.
        records = []
        for i, o in enumerate(objs):
            for other in objs:
                other.hide_render = (other is not o)
            solo_path = os.path.join(a.out, "_vis_solo.png")
            render.render_to(solo_path)
            solo = imaging.load_png_rgba(solo_path)[:, :, 3] >= 0.5  # alpha silhouette
            foot_inner = solo[pady:pady + H, padx:padx + W]   # in-frame footprint
            full_px = int(foot_inner.sum())

            # full extent (true-frame coords; may be <0 or >W/H if off-frame).
            ys, xs = np.where(solo)
            if len(xs):
                bbox_full = [int(xs.min()) - padx, int(ys.min()) - pady,
                             int(xs.max()) + 1 - padx, int(ys.max()) + 1 - pady]
                centroid_full = [round(float(xs.mean()) - padx, 2),
                                 round(float(ys.mean()) - pady, 2)]
            else:
                bbox_full, centroid_full = None, None

            # visible (front-most) extent from the composite label map.
            vis_mask = label_inner == i
            visible_px = int(vis_mask.sum())
            if visible_px:
                vy, vx = np.where(vis_mask)
                bbox_visible = [int(vx.min()), int(vy.min()),
                                int(vx.max()) + 1, int(vy.max()) + 1]
                centroid_visible = [round(float(vx.mean()), 2),
                                    round(float(vy.mean()), 2)]
            else:
                bbox_visible, centroid_visible = None, None

            # occluder attribution: over the in-frame footprint, which OTHER part
            # is front-most (nearest occluder per pixel; see _notes).
            occluded_by = []
            if full_px:
                labs = label_inner[foot_inner]
                labs = labs[(labs >= 0) & (labs != i)]
                if labs.size:
                    idxs, cnts = np.unique(labs, return_counts=True)
                    order = np.argsort(cnts)[::-1]
                    for j, c in zip(idxs[order], cnts[order]):
                        occluded_by.append({
                            "name": objs[int(j)].name, "px": int(c),
                            "frac": round(int(c) / full_px, 4)})
                    occluded_by = [e for e in occluded_by if e["frac"] >= 0.005][:6]

            status, visible_fraction = _occlusion_status(visible_px, full_px)
            clip = _clip_stats(solo, padx, pady, W, H)
            records.append({
                "index": i, "name": o.name, "color": list(palette[i]),
                "status": status,
                "visible_px": visible_px, "full_px": full_px,
                "visible_fraction": visible_fraction,
                "bbox_visible": bbox_visible, "bbox_full": bbox_full,
                "centroid_visible": centroid_visible, "centroid_full": centroid_full,
                "occluded_by": occluded_by,
                "clip": clip["clip"], "in_frame_fraction": clip["in_frame_fraction"],
                "clipped": clip["clipped"],
                "fully_captured_in_overscan": clip["fully_captured_in_overscan"],
            })

        # restore visibility + the true match frame before the overlay render.
        for o in objs:
            o.hide_render = saved_hide[o]
        scene_data.render.resolution_x = saved_rx
        scene_data.render.resolution_y = saved_ry
        camera.apply_intrinsics(ctx.cam_obj.data, intr)
        scene_data.render.pixel_aspect_x = saved_pax
        scene_data.render.pixel_aspect_y = saved_pay
    finally:
        for o in objs:
            o.data.materials.clear()
            for m in saved_mats[o]:
                o.data.materials.append(m)
            o.hide_render = saved_hide[o]
        imaging.restore_id_state(scene_data, saved_state, a)
        scene_data.render.resolution_x = saved_rx
        scene_data.render.resolution_y = saved_ry
        scene_data.render.pixel_aspect_x = saved_pax
        scene_data.render.pixel_aspect_y = saved_pay
        imaging.cleanup_id_materials()
        for tmp in ("_vis_composite.png", "_vis_solo.png"):
            p = os.path.join(a.out, tmp)
            if os.path.exists(p):
                os.remove(p)

    if records is None:
        return

    # 5) overlay: a real match render (restored materials) + per-part boxes.
    vis_png = os.path.join(a.out, "visibility.png")
    render.place_match_camera(a, ctx.cam_obj, ctx.center, ctx.radius,
                              ctx.frame_intr)
    render.render_to(vis_png)
    _draw_visibility_overlay(vis_png, records, palette)

    res = (W, H)
    summary = {
        "n_parts": n,
        "n_fully_visible": sum(1 for r in records if r["status"] == "VISIBLE"),
        "n_partial": sum(1 for r in records if r["status"] == "PARTIAL"),
        "n_occluded": sum(1 for r in records if r["status"] == "OCCLUDED"),
        "n_clipped": sum(1 for r in records if r["clipped"]),
        "most_occluded": [
            r["name"] for r in sorted(
                records,
                key=lambda r: (1.0 - (r["visible_fraction"]
                                      if r["visible_fraction"] is not None else 0.0)),
                reverse=True)
            if r["status"] in ("PARTIAL", "OCCLUDED")][:3],
    }
    _write_visibility_legend(a.out, records, summary, res, a.visibility_margin)
    print(f"[render_wrapper] visibility: {n} parts "
          f"({summary['n_fully_visible']} full, {summary['n_partial']} partial, "
          f"{summary['n_occluded']} occluded, {summary['n_clipped']} clipped) "
          f"-> visibility.png, visibility.json, visibility.txt")


def _draw_visibility_overlay(png_path, records, palette):
    """Draw per-part boxes on the match render: SOLID at the visible extent,
    DASHED at the full/unoccluded extent (dashed-only = fully occluded, shown at
    its predicted location). Colored by the ID palette so box <-> region <->
    legend row line up. The load/save round-trip (color management included) lives
    in imaging.edit_png_overlay."""
    with imaging.edit_png_overlay(png_path) as top:  # top-down (y-down) array
        for r in records:
            col = tuple(c / 255.0 for c in r["color"]) + (1.0,)
            if r["bbox_full"] and (r["bbox_visible"] is None
                                   or r["bbox_full"] != r["bbox_visible"]):
                x0, y0, x1, y1 = r["bbox_full"]
                imaging.draw_box(top, x0, y0, x1 - 1, y1 - 1, col, thickness=2,
                                 dashed=True)
            if r["bbox_visible"]:
                x0, y0, x1, y1 = r["bbox_visible"]
                imaging.draw_box(top, x0, y0, x1 - 1, y1 - 1, col, thickness=2,
                                 dashed=False)


def _write_visibility_legend(out_dir, records, summary, resolution, margin):
    """visibility.json (machine) + visibility.txt (human) legend."""
    report = {
        "view": "visibility",
        "image": "visibility.png",
        "resolution": list(resolution),
        "visibility_margin": margin,
        "summary": summary,
        "parts": records,
        "_notes": "Per-part visibility in the match view. visible_* = the "
                  "front-most (actually visible) extent; bbox_full/centroid_full = "
                  "the part's UNOCCLUDED extent (from a solo render), in true-frame "
                  "pixels (may be <0 or >W/H if the part spills off-frame). full_px "
                  "= the part's IN-FRAME unoccluded footprint (the visible_fraction "
                  "denominator), so clipping does not count as occlusion. "
                  "visible_fraction = visible_px/full_px (in-frame occlusion; 1.0 "
                  "= nothing in front). status is occlusion-only "
                  "(VISIBLE/PARTIAL/OCCLUDED); framing is the separate clipped / "
                  "in_frame_fraction fields. occluded_by lists the parts FRONT-MOST "
                  "over this one (nearest occluder per pixel — a part hidden behind "
                  "two layers is attributed to the front one). A part with "
                  "fully_captured_in_overscan=false exceeds even the overscan, so "
                  "its bbox_full/clip are lower bounds (raise --visibility-margin).",
    }
    with open(os.path.join(out_dir, "visibility.json"), "w") as f:
        json.dump(report, f, indent=2)

    name_w = max((len(r["name"]) for r in records), default=4)
    lines = [
        f"Visibility report ({summary['n_parts']} parts) — match view, "
        f"{resolution[0]}x{resolution[1]}  (margin {margin}/side)",
        f"  {summary['n_fully_visible']} fully visible, {summary['n_partial']} "
        f"partial, {summary['n_occluded']} occluded, {summary['n_clipped']} clipped",
        "",
        f"{'idx':>3}  {'name':<{name_w}}  {'status':<8}  {'vis%':>5}  "
        f"{'bbox_visible (x0,y0,x1,y1)':<28}  occluded_by",
    ]
    for r in records:
        vf = r["visible_fraction"]
        vpct = f"{vf * 100:4.0f}%" if vf is not None else "   —"
        bb = ("(" + ",".join(str(v) for v in r["bbox_visible"]) + ")"
              if r["bbox_visible"] else "—")
        occ = ", ".join(f"{e['name']}({e['frac']*100:.0f}%)"
                        for e in r["occluded_by"]) or "—"
        clip = " CLIPPED" if r["clipped"] else ""
        lines.append(
            f"{r['index']:>3}  {r['name']:<{name_w}}  {r['status']:<8}  {vpct:>5}  "
            f"{bb:<28}  {occ}{clip}")
    with open(os.path.join(out_dir, "visibility.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
