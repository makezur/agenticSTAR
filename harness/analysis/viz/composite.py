"""
composite.py — build a single side-by-side panel for the judge step.

Runs in the 'artscript' micromamba env (NOT Blender's python), from harness/:
  micromamba run -n artscript python -m analysis.viz.composite \
      --source inputs/mug/image.png --render runs/mug/views/match.png \
      [--mask inputs/mug/mask.png] [--metrics-out runs/mug/mesh/metrics.json] \
      --out runs/mug/views/composite.png

Panel = [ SOURCE | RENDER | SILHOUETTE | MASKED SOURCE ]  — the SHARED comparison
row (`viz.rows`), the same builder, column order and colour key the candidate sheets
and sweep_sides use, so this panel cannot drift from theirs. Columns whose inputs are
absent are simply dropped: without --mask you get [ SOURCE | RENDER ].
  - MASKED SOURCE: the photo with background (and, with --hand-mask, the occluder)
    blanked to the SAME backdrop the render is presented on, for a 1:1 read. Last,
    so SOURCE / RENDER / SILHOUETTE — the compare-and-diff sequence — stay adjacent.
  - SILHOUETTE (needs --mask): the render silhouette overlaid directly on the source mask (raw
    1:1 — no normalization, because the camera is fixed and the object is built
    at true scale in front of it, so they align). AGREE / EXTRA / MISSING, plus
    MAYBE OCCLUDED and the ignored hand region with a --hand-mask. The colour key
    is drawn beside the title from OVERLAP_LEGEND — the ONE definition of the
    colours, so no caption can drift from the pixels. It is always ONE row, and the
    header bar is STACKED ABOVE each panel, at one height for the whole strip, so no
    render pixel is covered and the panels line up row-for-row (see
    lib/panels.label_row — --height sizes the renders, and the strip is one bar
    taller). The IoU is NOT
    stamped on the panel; it lives in --metrics-out and the console line.
The RGB colour agreement is reported as NUMBERS ONLY (color_score / rgb_mae in
--metrics-out); there is no colour-residual panel.
If depth inputs are given (--tracking + --render-depth, and --mask, which scopes the
scored region), OUR-DEPTH and a SIGNED DEPTH-RESIDUAL panel (blue=render in
front/too near, red=behind/too far, white=agree) are appended — the sign says
which way to move (see depth.py). Without --mask the depth panels are skipped
with a warning.

When --side-by-side-out is given, a second, focused comparison is also written:
  [ SOURCE | RENDER | MASKED SOURCE ]
i.e. the same row minus the SILHOUETTE column.

TRANSPARENCY. A render is RGBA and Blender leaves colour under alpha=0 (an
anti-aliased rim, ~1px), so a panel that reads the file with `load_bgr` keeps a faint
fringe that one presented through `renderer_settings.present` does not. Every panel
path here goes through `rows.from_paths` -> `present`, so a render looks the SAME
whichever tool drew it. --bg-mode picks the backdrop for the render and the masked
source alike; 'black' (the default) composites onto one opaque BACKDROP_BGR so
viewers see a consistent background.
"""

import argparse
import os

import cv2

from analysis.lib import panels, rasters
from analysis.lib.io import write_json
from analysis.scorers import depth, silhouette
from analysis.viz import depth_units, rows
from core import renderer_settings


# The overlap colours + their legend live in `lib.panels` (the one owner shared
# with the candidate sheets) and the row assembly in `viz.rows`. Re-exported
# here under their original names for readers and docs.
OVL_AGREE = panels.OVL_AGREE
OVL_EXTRA = panels.OVL_EXTRA
OVL_MISSING = panels.OVL_MISSING
OVL_WAIVED = panels.OVL_WAIVED
OVL_HAND = panels.OVL_HAND
OVERLAP_LEGEND = panels.OVERLAP_LEGEND
OVERLAP_CAPTION = panels.OVERLAP_CAPTION
raw_overlay_panel = panels.overlay_panel

# The judge strip's columns, in order. The full default row plus the two depth
# columns: `rows.build_row` drops whichever it has no inputs for, so this one tuple
# covers every combination of --mask / --tracking rather than branching per output.
COMPOSITE_COLUMNS = rows.DEFAULT_COLUMNS + ("depth", "depth_residual")


def side_by_side_panel(source_path, mask_path, render_path, hand_mask_path="",
                       hand_dilate=0.0, height=512, bg_mode="black",
                       render_sub=""):
    """Build a focused SOURCE | RENDER | MASKED SOURCE comparison strip.

    `render_sub` — optional second label line under RENDER naming WHICH candidate
    the strip shows (e.g. a sweep candidate's "rank 03"). Identity, not score:
    scores belong in the report and --metrics-out, not burned into a picture
    (see sweep_sides.row_sub).

    A thin wrapper over `rows.build_row` — the same builder the candidate sheets
    use — so this strip and a sheet row cannot drift apart. It is exactly the
    default row minus the SILHOUETTE column."""
    inputs = rows.from_paths(source_path, mask_path, render_path,
                             hand_mask_path=hand_mask_path,
                             hand_dilate=hand_dilate, bg_mode=bg_mode)
    return rows.build_row(
        inputs, columns=("source", "render", "masked_source"), height=height,
        render_sub=render_sub).image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--render", required=True)
    p.add_argument("--mask", default="")
    p.add_argument("--hand-mask", default="", help="binary HAND (occluder) mask "
                   "png; its region is shaded gray (don't-care) and iou_visible "
                   "headlines the silhouette panel")
    p.add_argument("--hand-dilate", type=float, default=0.0,
                   help="grow the hand ignore-region by this fraction of the "
                   "longer side (0 = the default: the hand mask exactly, as the "
                   "segmenter produced it). Dilating only ever REMOVES pixels "
                   "from scoring")
    p.add_argument("--out", required=True)
    p.add_argument("--side-by-side-out", default="",
                   help="also write a focused SOURCE | RENDER | MASKED SOURCE "
                        "comparison PNG; needs --mask")
    p.add_argument("--side-by-side-max-dimension", type=int, default=0,
                   help="longest-edge target for a derived side-by-side preview; "
                        "the canonical output stays native and previews never "
                        "drop below 0.5x (0 = no preview)")
    p.add_argument("--bg-mode", default="black",
                   choices=list(renderer_settings.BG_MODES),
                   help="backdrop the render and masked source are presented on, "
                        "in --out and --side-by-side-out alike: 'alpha' preserves "
                        "transparency; 'black' (default) uses solid black")
    p.add_argument("--metrics-out", default="", help="also write metrics JSON here")
    p.add_argument("--overlap-out", default="", help="also write the standalone "
                   f"silhouette-overlap panel ({OVERLAP_CAPTION}) as its OWN png "
                   "here — text-free and at native resolution. This is the "
                   "individual image the visual judge reads each iteration; needs "
                   "--mask (skipped with a warning otherwise)")
    p.add_argument("--depth-residual-out", default="", help="also write the standalone "
                   "SIGNED DEPTH-RESIDUAL panel (ours - observed gt; blue=render in "
                   "front/too near, red=behind/too far, white=agree) as its OWN png "
                   "here, so the judge sees WHERE depth is wrong AND which way to move; "
                   "needs the depth inputs (--tracking + --render-depth + --mask)")
    p.add_argument("--timeline-dir", default="", help="also archive the full "
                   "composite into <dir>/<frame>/<NNN>_composite.png (one entry per "
                   "step) for scrubbing the run's progression; needs --frame")
    p.add_argument("--frame", default="", help="frame name for the --timeline-dir "
                   "subfolder (e.g. 000040)")
    p.add_argument("--tag", default="", help="timeline entry name (e.g. iter03); "
                   "default is a zero-padded auto-increment per frame")
    # depth panels (optional; needs observed depth + the rendered depth.npy). When given,
    # a 5th OUR-DEPTH panel and a 6th DEPTH-RESIDUAL-vs-GT panel are appended.
    p.add_argument("--tracking", "--pi3x", dest="tracking", default="",
                   help="the capture's tracking dir (observed depth + cameras); "
                   "with --render-depth appends OUR-DEPTH + DEPTH-RESIDUAL(vs "
                   "observed gt) panels. --pi3x is the old name, still accepted")
    p.add_argument("--render-depth", default="", help="depth.npy from the 'depth' "
                   "view (float32); with --tracking adds the OUR-DEPTH + RESIDUAL "
                   "panels, ALONE adds the object-units DEPTH panel (monocular "
                   "— depth / the pose.json scale, no GT needed)")
    p.add_argument("--view", type=int, default=None,
                   help="tracking view index (else resolved from --source/keyframes)")
    p.add_argument("--conf-thr", type=float, default=None,
                   help="observed confidence below this is ignored (depth panels); "
                   "default resolves the run's depth_config.json floor, else 0.1")
    p.add_argument("--height", type=int, default=512)
    args = p.parse_args()

    h = args.height

    report = None
    clean_overlay = None  # native-res, text-free overlay — the judge's read (--overlap-out)
    if args.mask:
        src_mask = rasters.load_mask(args.mask)
        rnd_mask = rasters.render_silhouette(args.render)
        src_bgr = rasters.load_bgr(args.source)
        # Presented on the OPAQUE backdrop, always — not args.bg_mode. The RGB
        # residual needs 3 channels, and a score must not move because someone
        # asked for a transparent PNG; the metric scores the render's own pixels
        # (it is masked to the mask/render overlap regardless).
        rnd_bgr = rasters.load_render(args.render, "black")
        hand = rasters.load_mask(args.hand_mask) if args.hand_mask else None
        # One keep-region K = ¬hand at the source-mask grid, reused everywhere so
        # the picture matches the numbers (silhouette shading, IoU, RGB overlap).
        keep = (rasters.build_keep(hand, src_mask.shape, args.hand_dilate)
                if hand is not None else None)
        report = silhouette.silhouette_iou(src_mask, rnd_mask,
                                           source_bgr=src_bgr, render_bgr=rnd_bgr,
                                           hand_mask=hand, hand_dilate=args.hand_dilate)
        # The standalone overlay the judge reads (--overlap-out): built here at
        # NATIVE resolution and text-free. The strip's own SILHOUETTE column is
        # built by the shared row builder below, from the same constants; the
        # RGB colour signal stays in the NUMBERS only (color_score / rgb_mae /
        # overlap_frac in metrics_<frame>.json, from silhouette_iou above) — the
        # per-pixel residual heatmap panel was dropped as one panel too many.
        clean_overlay = panels.overlay_panel(src_mask, rnd_mask, keep=keep)

    # optional DEPTH panels (needs observed depth + rendered depth.npy): OUR-DEPTH
    # and DEPTH-RESIDUAL-vs-observed(gt) columns. depth.py owns the math AND the
    # words.
    depth_residual_panel = None  # standalone "where is depth wrong" image
    depth_parts = None
    if args.tracking and args.render_depth and args.mask:
        drep, dmaps, _, _ = depth.compute(
            args.tracking, args.render_depth, args.mask,
            source=args.source, view=args.view, conf_thr=args.conf_thr,
            hand_mask=args.hand_mask, hand_dilate=args.hand_dilate)
        depth_parts = depth.depth_panels(
            rasters.load_bgr(args.source), dmaps, drep, h)
        depth_residual_panel = depth_parts["residual"].image  # text-free, for the judge
    elif args.render_depth and not args.tracking:
        # MONOCULAR: no observed pointmap to compare against, but the render
        # still knows how far the object sits — draw OUR depth in OBJECT UNITS
        # (depth / the predicted scale s; depth_units owns the picture AND the
        # NEAR/FAR key). Same "depth" column of the standard row; there is no
        # residual column because there is nothing to disagree with.
        scale = depth.resolve_scale(None, args.render_depth)
        if scale is None:
            print("[composite] WARNING: --render-depth without --tracking needs a "
                  "pose.json scale for the object-units DEPTH panel; skipping")
        else:
            depth_parts = {"our": depth_units.object_units_panel(
                rasters.load_render_depth(args.render_depth), scale, height=h)}
    elif args.tracking:
        print("[composite] WARNING: depth panels need --tracking AND "
              "--render-depth AND --mask; skipping the depth panels")

    # The strip is the SHARED row (viz.rows) — the same builder, columns and colour
    # key the candidate sheets and sweep_sides use, so this panel cannot drift from
    # theirs. Columns whose inputs are absent (no --mask -> no masked source or
    # silhouette; no --tracking -> no depth) are dropped by `build_row`, which is why a
    # mask-less run still gets its SOURCE | RENDER pair.
    inputs = rows.from_paths(args.source, args.mask, args.render,
                             hand_mask_path=args.hand_mask,
                             hand_dilate=args.hand_dilate,
                             bg_mode=args.bg_mode, depth_panels=depth_parts)
    built = rows.build_row(inputs, columns=COMPOSITE_COLUMNS, height=h)
    out_img = built.image

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, out_img)
    print(f"[composite] wrote {args.out}  ({out_img.shape[1]}x{out_img.shape[0]})")

    if args.side_by_side_out:
        if not args.mask:
            print("[composite] WARNING: --side-by-side-out needs --mask; skipping "
                  "the focused comparison")
        else:
            comparison = side_by_side_panel(
                args.source, args.mask, args.render, args.hand_mask,
                args.hand_dilate, h, args.bg_mode)
            variants = panels.write_image_variants(
                args.side_by_side_out, comparison,
                preview_max_dimension=args.side_by_side_max_dimension,
                write_manifest=True)
            print(f"[composite] wrote {variants['original']}  "
                  f"({comparison.shape[1]}x{comparison.shape[0]})")
            if variants["preview"] != variants["original"]:
                print(f"[composite] wrote {variants['preview']} (preview)")

    # standalone overlap PNG — the individual image the visual judge reads.
    # Text-free and at NATIVE resolution (matches the other images the judge sees);
    # the caption + IoU live only in the composite strip / timeline above.
    if args.overlap_out:
        if clean_overlay is None:
            print("[composite] WARNING: --overlap-out needs --mask; skipping the "
                  "standalone overlap")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.overlap_out)),
                        exist_ok=True)
            cv2.imwrite(args.overlap_out, clean_overlay)
            print(f"[composite] wrote {args.overlap_out}  "
                  f"({clean_overlay.shape[1]}x{clean_overlay.shape[0]})")

    # standalone DEPTH-RESIDUAL PNG — the individual image showing WHERE depth
    # disagrees most (needs the depth panels, i.e. --tracking + --render-depth).
    if args.depth_residual_out:
        if depth_residual_panel is None:
            print("[composite] WARNING: --depth-residual-out needs --tracking + "
                  "--render-depth (+ --mask); skipping the standalone depth residual")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.depth_residual_out)),
                        exist_ok=True)
            cv2.imwrite(args.depth_residual_out, depth_residual_panel)
            print(f"[composite] wrote {args.depth_residual_out}  "
                  f"({depth_residual_panel.shape[1]}x{depth_residual_panel.shape[0]})")

    # debug timeline — archive this step's full composite under timeline/<frame>/.
    panels.timeline_save(out_img, args.timeline_dir, args.frame, "composite",
                         tag=args.tag)

    if report is not None:
        if args.metrics_out:
            write_json(args.metrics_out, report)
            print(f"[composite] wrote {args.metrics_out}")
        if report.get("iou_visible") is not None:
            print(f"[composite] visible silhouette IoU = {report['iou_visible']:.3f} "
                  f"(raw={report['iou_raw']:.3f})")
        else:
            print(f"[composite] raw silhouette IoU = {report['iou_raw']:.3f}")


if __name__ == "__main__":
    main()
