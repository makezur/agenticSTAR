"""
sweep_sides.py — post-factum SOURCE | RENDER | MASKED SOURCE side-by-sides for a
sweep/apply order's rendered candidates.

A sweep (or apply) run with --sweep-dump-topk already re-renders its winner and
the best candidate per grid cell at full match resolution (sweep_best.png,
sweep_top_NN.png) — the contact sheet an agent inspects. But judging a
candidate against the SOURCE image (the silhouette-trap check: right mask,
wrong facing) means eyeballing render and source in separate viewers. This tool
closes that gap AFTER the fact, with no Blender involved: it reads the order's
own report (sweep.json / apply.json — which records the frame, mask, hand mask,
and each ranked candidate's image), pairs every rendered candidate with the
source frame, and writes a labeled side-by-side strip per candidate:

    side_by_side_sweep_best.png
    side_by_side_sweep_top_00.png ...   (labeled "rank 00" — identity, NOT score:
                                         numbers stay in the report, see row_sub)

It works on any existing sweep dir — views/NNNN passes and the pool's
renders/<order-id>/ — because the reports are self-describing. The
only thing it cannot do is conjure renders that were never made: a sweep run
without --sweep-dump-topk has only sweep_best.png to pair.

`pool/panels.py` calls `write_sides` itself, so a pool order's strips exist before
its result is published and an agent never runs this by hand for the default four
columns. Reach for the CLI when you want something the
automatic pass does not give you: different columns, a different height, strips for
a run made outside the pool, or a re-panel of an old dir.

Runs in the 'artscript' env (NOT Blender's python), from harness/:

  micromamba run -n artscript python -m analysis.viz.sweep_sides \
      --render-dir RUN_DIR/spool/renders/refine-120 --run-dir RUN_DIR

--run-dir supplies the source frame via layout.json (frames_dir + the report's
`frame`); pass --source to override. Mask paths come from the report itself
(resolved against the repo root, matching how the sweep was invoked).

TWO HALVES, and only one of them is about drawing. The REPORT half — `find_report`,
`report_rows`, `row_paths`, `frames_index`, `is_object_report` — is the one
normalizer for every sweep-family report shape, imported by `candidate_sheet` and
the sweep engine; it answers "where are this order's pixels". The PANEL half is
a call to `analysis.viz.rows.build_row`, the builder shared with
`composite` and the sheets, so `--columns` here accepts the same column names
(add `overlap` for the silhouette check on a per-candidate strip).
"""

import argparse
import os

import cv2
import numpy as np

from analysis.lib.io import read_json
from analysis.lib import panels, rasters
from analysis.scorers import depth as depth_scorer
from analysis.viz import depth_units
from analysis.viz import rows as rows_lib
from core import captures, run_layout
from core.sweep_family import SWEEP_VIEWS

HARNESS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(HARNESS)


def _resolve(path, fallback=""):
    """A report path as-is, else against the repo root (sweep reports record
    mask paths relative to the CWD Blender ran from — usually the repo root),
    else the layout-derived `fallback`. Returns the first existing candidate,
    else the input (so the eventual imread error names the searched path)."""
    if not path:
        return fallback if fallback and os.path.isfile(fallback) else ""
    for c in (path, os.path.join(REPO, path), fallback):
        if c and os.path.isfile(c):
            return c
    return path


REPORT_STEMS = SWEEP_VIEWS


def find_report(render_dir):
    """(report dict, stem) for the dir's sweep/apply/oapply/oapply_all/osweep report.

    All five are self-describing reports over rendered candidates; the object-
    canonical ones (oapply/oapply_all/osweep) differ in that a candidate carries a
    per-FRAME image dict rather than one image, which `report_rows` normalizes away.
    Order matters only when a dir holds several: the single-frame verbs first, then
    the object ones in the order an agent is likeliest to have asked for."""
    for stem in REPORT_STEMS:
        path = os.path.join(render_dir, f"{stem}.json")
        report = read_json(path) if os.path.isfile(path) else None
        if isinstance(report, dict):
            return report, stem
    raise ValueError(
        f"no {' or '.join(f'{s}.json' for s in REPORT_STEMS)} in {render_dir}")


def is_object_report(report):
    """True for an oapply/osweep report — a candidate SET scored over frames.

    Detected by shape, not by filename, so a report found under any stem is
    handled correctly: object reports carry a top-level `frames` LIST and their
    ranked entries carry `images` (a per-frame dict)."""
    return isinstance(report.get("frames"), list) and any(
        isinstance(r.get("images"), dict) for r in report.get("ranked") or [])


def report_rows(report, drop_duplicate_winner=False):
    """Normalize any report to a flat list of ROW dicts, one per rendered image.

    This is the single shape every sheet builder consumes:

        {"image": str, "frame": str|None, "candidate_id": str,
         "display_rank": int|"best", "label": str|None, "score": float|None,
         "combined"/"iou_raw"/"iou_visible"/"mean_gate_iou"/...: passthrough}

    A sweep/apply report yields one row per rendered candidate, all sharing the
    report's single `frame`. An oapply/osweep report yields one row per
    (candidate, frame) pair, each carrying ITS OWN frame — which is why rows, not
    the sheet, must own the source image.

    `drop_duplicate_winner` skips the `best` row when the SAME winner is also a
    ranked panel (sweep/apply render it twice: *_best.png and the rank-0 panel).
    Sheets want that — one row per candidate, keyed by the stable rank — while
    per-candidate strips want a strip for every image actually on disk.
    """
    rows = []
    if is_object_report(report):
        for rec in report.get("ranked") or []:
            images = rec.get("images") or {}
            rank = int(rec.get("rank", len(rows)))
            label = rec.get("label") or f"rank-{rank:04d}"
            for frame, image in sorted(images.items()):
                rows.append({
                    **{k: v for k, v in rec.items() if k != "images"},
                    "image": image,
                    "frame": frame,
                    "candidate_id": f"{label}  {frame}",
                    "display_rank": rank,
                    "label": label,
                    # shared mode scores a candidate by its MEAN over frames;
                    # per-frame mode has no mean to report (the row IS one frame),
                    # so it carries that frame's own gate IoU.
                    "score": (rec.get("mean_gate_iou")
                              if rec.get("mean_gate_iou") is not None
                              else rec.get("gate_iou")),
                })
        return rows

    frame = report.get("frame")
    best = report.get("best") or {}
    ranked = report.get("ranked") or []
    # the winner's placement is rendered twice when dump_topk is on; a sheet
    # prefers the ranked copy, whose rank is the stable row identity.
    winner_also_ranked = drop_duplicate_winner and any(
        rec.get("image") and int(rec.get("rank", -1)) == 0 for rec in ranked)
    if best.get("image") and not winner_also_ranked:
        rows.append({**best, "image": best["image"], "frame": frame,
                     "candidate_id": str(best.get("candidate_id") or "best"),
                     "display_rank": "best", "label": None,
                     "score": best.get("combined")})
    seen = {r["image"] for r in rows}
    for rec in ranked:
        image = rec.get("image")
        if not image or image in seen:
            continue
        rank = int(rec.get("rank", len(rows)))
        rows.append({**rec, "image": image, "frame": frame,
                     "candidate_id": str(rec.get("candidate_id")
                                         or f"rank-{rank:04d}"),
                     "display_rank": rank, "label": None,
                     "score": rec.get("combined")})
        seen.add(image)
    return rows


def row_sub(row):
    """The RENDER panel's caption for one row — from `rows.render_caption`.

    The wording lives with the column that carries it; this only reads the row's
    identity out of the report shape it normalizes. The score stays in the report,
    where it can be sorted and recomputed — `report_rows` carries it for callers
    that want it as data.
    """
    return rows_lib.render_caption(row["display_rank"], row.get("label"))


def candidate_images(report):
    """[(image filename, label sub-line)] for every RENDERED candidate."""
    return [(row["image"], row_sub(row)) for row in report_rows(report)]


def _rendered_depth(render_dir, row, run_dir=""):
    """(depth, object scale) for a candidate, or None when unavailable."""
    name = row.get("depth")
    if not name:
        return None
    path = os.path.join(render_dir, name)
    if not os.path.isfile(path):
        print(f"[sweep-sides] skipping candidate depth {name}: file missing")
        return None
    scale_arg = (row.get("pose") or {}).get("scale")
    scale_path = (os.path.join(run_dir, "views", "depth.npy")
                  if run_dir else path)
    scale = depth_scorer.resolve_scale(scale_arg, scale_path)
    if scale is None:
        print(f"[sweep-sides] skipping candidate depth {name}: no pose scale")
        return None
    return rasters.load_render_depth(path), scale


def rendered_depth_range(items, run_dir=""):
    """One pooled object-unit [near, far] range across candidate rows.

    ``items`` is ``[(render_dir, row), ...]``. Each depth is divided by its
    reported object scale before pooling, so rolled sheets remain meaningful
    even if they contain orders with different scene-unit scales.
    """
    values = []
    for render_dir, row in items:
        loaded = _rendered_depth(render_dir, row, run_dir=run_dir)
        if loaded is None:
            continue
        depth, scale = loaded
        finite = depth[np.isfinite(depth)]
        if finite.size:
            values.append(finite / scale)
    return panels.pct_range(values) if values else None


def rendered_depth_panels(
        render_dir, row, height=0, run_dir="", depth_range=None):
    """Visualization-only OUR DEPTH panel for one rendered candidate.

    The report's optional ``depth`` field names a camera-Z ``.npy`` produced
    beside that candidate's RGB render. No observed/Pi3X depth is loaded here;
    the heatmap is rendered depth in object units using the run's shared scale.
    Missing/old artifacts return ``None`` so legacy panels remain valid.
    """
    loaded = _rendered_depth(render_dir, row, run_dir=run_dir)
    if loaded is None:
        return None
    depth, scale = loaded
    dmin, dmax = depth_range or (None, None)
    return {"our": depth_units.object_units_panel(
        depth, scale, dmin=dmin, dmax=dmax, height=height)}


def _layout(run_dir):
    if not run_dir:
        return {}
    return run_layout.load_run_layout(run_dir) or {}


def source_for(report, layout, source=""):
    """The source-frame path: --source wins; else layout.json's frames_dir +
    the report's `frame`."""
    if source:
        return source
    if not layout.get("frames_dir"):
        raise ValueError("need --run-dir (for layout.json frames_dir) or --source")
    frame = report.get("frame")
    if not frame:
        raise ValueError("report has no 'frame' — pass --source explicitly")
    return os.path.join(layout["frames_dir"], frame)


def row_paths(report, layout, frame, source="", frame_masks=None):
    """(source, mask, hand_mask) for ONE row's frame.

    Sweep/apply reports record the mask paths at the top level (one frame per
    report). Object reports carry a `frames` list whose entries may name their own
    mask, so `frame_masks` (built by `frames_index`) is consulted first. In every
    case layout.json is the authoritative fallback, since a report records mask
    paths relative to the CWD Blender ran from and a later reader may not share it.
    """
    per_frame = (frame_masks or {}).get(frame) or {}
    src = source or _frame_source(layout, frame) or source_for(
        report, layout, source=source)
    mask = _resolve(per_frame.get("mask") or report.get("mask") or "",
                    fallback=_mask_from_layout(layout, "masks_dir", frame))
    hand = _resolve(per_frame.get("hand_mask") or report.get("hand_mask") or "",
                    fallback=_mask_from_layout(layout, "hand_masks_dir", frame))
    return src, mask, hand


def frames_index(report):
    """{frame: entry} from an object report's `frames` list ({} otherwise)."""
    out = {}
    for entry in report.get("frames") or []:
        if isinstance(entry, dict) and entry.get("frame"):
            out[entry["frame"]] = entry
    return out


def _frame_source(layout, frame):
    """layout's frames_dir + this row's frame, when both are known."""
    if not frame or not layout.get("frames_dir"):
        return ""
    path = os.path.join(layout["frames_dir"], frame)
    return path if os.path.isfile(path) else ""


def _mask_from_layout(layout, key, frame):
    """layout's masks_dir/hand_masks_dir + the frame's stem, when both exist —
    the authoritative fallback: reports record mask paths relative to the CWD
    the sweep ran from, which a later reader may not share, while layout.json
    records the dirs absolutely."""
    d = layout.get(key)
    if not d or not frame:
        return ""
    return captures.mask_for(d, frame)


STRIP_COLUMNS = ("source", "render", "masked_source")


def write_sides(render_dir, run_dir="", source="", out_dir="", height=512,
                bg_mode="black", limit=0, columns=STRIP_COLUMNS,
                drop_duplicate_winner=False, hand_dilate=0.0,
                max_dimension=0, return_variants=False):
    """Write side_by_side_<image-stem>.png per rendered candidate; returns the
    list of written paths.

    `columns` defaults to the three-panel strip this tool has always written; any
    column in `viz.rows.COLUMNS` is accepted, so a caller wanting the silhouette
    overlap on a per-candidate strip asks for it here rather than reaching for
    the paginated sheet.

    `drop_duplicate_winner` skips the `best` row when the SAME candidate is also a
    ranked panel — sweep/apply render the winner twice (`<stem>_best.png` and the
    rank-0 panel), so a strip per IMAGE means judging one candidate twice. The CLI
    default stays False (a strip for every image actually on disk, which is what
    this tool has always promised); the pool passes True, because there a strip is
    evidence an agent reads rather than a file it asked for by name.

    `hand_dilate` is the occluder-mask dilation, defaulting to the exact mask —
    `candidate_sheet` passes its own small dilation, so a caller wanting a sheet
    row and a strip to agree pixel-for-pixel passes the same value here.

    `max_dimension` targets a derived preview's longest edge. The canonical strip
    remains native; `0` returns it directly and a preview never drops below half
    scale. `return_variants` exposes both paths and raster metadata to the pool.
    """
    render_dir = os.path.abspath(render_dir)
    columns = rows_lib.parse_columns(
        ",".join(columns) if not isinstance(columns, str) else columns,
        require=("render",), allowed=rows_lib.IMAGE_COLUMNS)
    report, _ = find_report(render_dir)
    rows = report_rows(report, drop_duplicate_winner=drop_duplicate_winner)
    depth_range = rendered_depth_range(
        [(render_dir, row) for row in rows], run_dir=run_dir)
    if limit:
        rows = rows[:max(1, int(limit))]
    if not rows:
        raise ValueError(
            f"{render_dir}: report lists no rendered candidate images — "
            "re-run the order with dump_topk (--sweep-dump-topk) to get "
            "per-candidate renders")
    layout = _layout(run_dir)
    frame_masks = frames_index(report)
    out_dir = os.path.abspath(out_dir or render_dir)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    variants_written = []
    for row in rows:
        image, sub = row["image"], row_sub(row)
        render = os.path.join(render_dir, image)
        if not os.path.isfile(render):
            print(f"[sweep-sides] skipping {image}: file missing")
            continue
        # per ROW: an object report pairs each candidate with its OWN frame.
        src, mask, hand = row_paths(report, layout, row.get("frame"),
                                    source=source, frame_masks=frame_masks)
        # the shared row builder — same columns and colours as a sheet row.
        strip = rows_lib.build_row(
            rows_lib.from_paths(src, mask, render, hand_mask_path=hand,
                                hand_dilate=hand_dilate, bg_mode=bg_mode,
                                depth_panels=rendered_depth_panels(
                                    render_dir, row, height=height,
                                    run_dir=run_dir,
                                    depth_range=depth_range)),
            columns=columns, height=height, render_sub=sub).image
        out = os.path.join(out_dir,
                           f"side_by_side_{os.path.splitext(image)[0]}.png")
        variants = panels.write_image_variants(
            out, strip, preview_max_dimension=max_dimension,
            write_manifest=True)
        variants_written.append(variants)
        written.append(variants["preview"])
        print(f"[sweep-sides] wrote {variants['original']}")
        if variants["preview"] != variants["original"]:
            print(f"[sweep-sides] wrote {variants['preview']} (preview)")
    if not written:
        raise ValueError(f"{render_dir}: none of the report's candidate images "
                         "exist on disk")
    return variants_written if return_variants else written


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--render-dir", required=True,
                   help="a sweep/apply output dir (holds sweep.json or "
                        "apply.json + the candidate PNGs)")
    p.add_argument("--run-dir", default="",
                   help="run dir whose layout.json names the source frames_dir")
    p.add_argument("--source", default="",
                   help="explicit source frame path (overrides --run-dir)")
    p.add_argument("--out-dir", default="",
                   help="where to write the strips (default: --render-dir)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--bg-mode", default="black",
                   help="backdrop for masked source + match (as composite.py)")
    p.add_argument("--limit", type=int, default=0,
                   help="only the first N candidates (0 = all rendered)")
    p.add_argument("--max-dimension", type=int, default=0,
                   help="longest-edge target for each derived preview; canonical "
                        "strips stay native and previews never drop below 0.5x "
                        "(0 = no preview)")
    p.add_argument("--dedupe-winner", action="store_true",
                   help="skip the *_best.png strip when the winner is also a "
                        "ranked panel (the same candidate rendered twice)")
    p.add_argument("--columns", default=",".join(STRIP_COLUMNS),
                   help="comma-separated columns from the shared row registry "
                        "(source, render, overlap, masked_source, depth; `match` and "
                        "`silhouette` are accepted aliases); drawn in the order "
                        "you name them; "
                        f"default: {','.join(STRIP_COLUMNS)}")
    args = p.parse_args()
    write_sides(args.render_dir, run_dir=args.run_dir, source=args.source,
                out_dir=args.out_dir, height=args.height, bg_mode=args.bg_mode,
                limit=args.limit, columns=args.columns,
                drop_duplicate_winner=args.dedupe_winner,
                max_dimension=args.max_dimension)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
