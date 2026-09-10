"""crop view — how much of the posed object falls OUTSIDE the match frame.

silhouette.py only ever sees the already-cropped render, so it can't know what fell
outside. Here the harness renders a second, OVERSCAN pass — the same camera
rays, but a wider frame (same fx/fy, larger W/H, principal point re-centered) so
the object is captured in full — then compares the object's silhouette area
INSIDE the true match rectangle vs. the whole overscan. That ratio is the "how
much is cropped / do I need to zoom out" signal. Restores the match
resolution/intrinsics afterward.
"""

import json
import os

import bpy
import numpy as np

from rig import camera, imaging, render


def render_view(ctx):
    a = ctx.args
    scene = bpy.context.scene
    cam = render.place_match_camera(a, ctx.cam_obj, ctx.center, ctx.radius,
                                    ctx.frame_intr)
    intr = camera.effective_intrinsics(ctx.cam_obj, cam, scene)
    W, H = int(intr["width"]), int(intr["height"])
    m = max(0.0, float(a.crop_margin))
    over, (padx, pady) = camera.overscan_intrinsics(intr, m)
    OW, OH = int(over["width"]), int(over["height"])
    saved_pax = scene.render.pixel_aspect_x
    saved_pay = scene.render.pixel_aspect_y
    saved_rx, saved_ry = scene.render.resolution_x, scene.render.resolution_y
    # grow the film to the overscan.
    scene.render.resolution_x, scene.render.resolution_y = OW, OH
    camera.apply_intrinsics(ctx.cam_obj.data, over)
    over_path = os.path.join(a.out, "crop.png")
    render.render_to(over_path)

    # measure silhouette inside the true frame vs. the whole overscan (shared
    # inner-rect / per-edge / border-touch math with the visibility view).
    rgba = imaging.load_png_rgba(over_path)  # (OH, OW, 4) top-down
    obj = rgba[:, :, 3] >= 0.5
    stats = imaging.frame_clip_stats(obj, padx, pady, W, H)
    total = stats["total_px"]
    inside = stats["inside_px"]
    outside = total - inside
    visible_frac = round(inside / total, 4) if total else 1.0
    clipped_frac = round(outside / total, 4) if total else 0.0
    edges = stats["edges"]
    # Fully caught by the OVERSCAN? Only then is the measured bbox — and thus the
    # zoom-out factor — the true extent; otherwise it's a lower bound.
    fully_captured = (total > 0) and stats["fully_captured_in_overscan"]

    # a zoom-out factor that would bring the whole object into the true frame:
    # ratio of the object's overscan bbox to the true-frame size (>=1 => clipped).
    ys, xs = np.where(obj)
    if len(xs):
        bx0, bx1 = xs.min(), xs.max() + 1
        by0, by1 = ys.min(), ys.max() + 1
        need_x = (bx1 - bx0) / W
        need_y = (by1 - by0) / H
        suggested_zoom_out = round(max(1.0, need_x, need_y), 3)
    else:
        suggested_zoom_out = 1.0

    _draw_true_frame(over_path, padx, pady, W, H)  # outline the match frame

    manifest = {
        "view": "crop",
        "match_frame": [W, H],
        "overscan_frame": [OW, OH],
        "crop_margin": m,
        "visible_frac": visible_frac,      # object inside the match frame / total
        "clipped_frac": clipped_frac,      # object outside the match frame / total
        "edge_overflow": edges,            # per-side clipped fraction of total
        "clipped": clipped_frac > 0.0,
        "fully_captured_in_overscan": fully_captured,
        "suggested_zoom_out": suggested_zoom_out,
        "suggested_zoom_out_is_lower_bound": not fully_captured,
        "image": "crop.png",
        "_notes": "Object posed as in match. visible_frac<1 => the object is cut "
                  "off by the match frame; shrink the shared SCALE (and/or recenter "
                  "via the frame's pose translation) by about suggested_zoom_out to "
                  "fit. If fully_captured_in_overscan "
                  "is false, the object exceeds even the overscan, so zoom-out is a "
                  "LOWER BOUND — raise --crop-margin to measure the true extent.",
    }
    with open(os.path.join(a.out, "crop.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    _write_crop_txt(a.out, manifest)

    # restore the true match frame so later views/exports are unaffected.
    scene.render.resolution_x, scene.render.resolution_y = saved_rx, saved_ry
    camera.apply_intrinsics(ctx.cam_obj.data, intr)
    scene.render.pixel_aspect_x = saved_pax
    scene.render.pixel_aspect_y = saved_pay


def _draw_true_frame(png_path, padx, pady, W, H):
    """Outline the true match frame (inner rectangle) on the overscan PNG so a
    glance shows what's inside vs. cut off."""
    with imaging.edit_png_overlay(png_path) as top:  # top-down (y-down) array
        oh, ow = top.shape[:2]
        x0, x1 = padx, padx + W - 1
        y0, y1 = pady, pady + H - 1
        line = (1.0, 1.0, 1.0, 1.0)  # opaque white outline
        for x in (x0, x1):
            if 0 <= x < ow:
                top[max(0, y0):min(oh, y1 + 1), x] = line
        for y in (y0, y1):
            if 0 <= y < oh:
                top[y, max(0, x0):min(ow, x1 + 1)] = line


def _write_crop_txt(out_dir, manifest):
    e = manifest["edge_overflow"]
    lines = [
        f"Crop / framing report — match frame {manifest['match_frame'][0]}x"
        f"{manifest['match_frame'][1]}, overscan {manifest['overscan_frame'][0]}x"
        f"{manifest['overscan_frame'][1]} (margin {manifest['crop_margin']}/side).",
        "",
        f"  visible in frame : {manifest['visible_frac']*100:.1f}% of the object",
        f"  clipped by frame : {manifest['clipped_frac']*100:.1f}% of the object",
        f"  overflow per edge: left {e['left']*100:.1f}%  right {e['right']*100:.1f}%"
        f"  top {e['top']*100:.1f}%  bottom {e['bottom']*100:.1f}%",
        f"  suggested zoom-out (scale down) factor: {manifest['suggested_zoom_out']}x"
        + (" (LOWER BOUND)" if manifest["suggested_zoom_out_is_lower_bound"] else ""),
    ]
    if manifest["clipped"] and not manifest["fully_captured_in_overscan"]:
        lines.append("  NOTE: object exceeds the overscan too — the zoom-out is a "
                     "lower bound; raise --crop-margin for the true extent.")
    if not manifest["clipped"]:
        lines.append("  -> fully inside the frame; no zoom-out needed.")
    else:
        lines.append("  -> object is cut off; shrink the shared SCALE (and/or "
                     "recenter via the frame's pose translation).")
    with open(os.path.join(out_dir, "crop.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[render_wrapper] wrote {os.path.join(out_dir, 'crop.json')} + crop.txt "
          f"(visible {manifest['visible_frac']*100:.1f}%, "
          f"clipped {manifest['clipped_frac']*100:.1f}%)")
