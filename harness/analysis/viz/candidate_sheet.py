"""Build paginated, auditable source-vs-candidate sheets for sweep/apply output.

The standalone default row is:

    SOURCE | RENDER | SILHOUETTE

Pool-built pose windows append DEPTH when the candidate report carries the
same-render depth artifact.

Each row is built by ``analysis.viz.rows`` — the SAME builder behind the
per-frame composite strip — so a column tuned there is tuned here, including the
silhouette overlap's colour key. This module owns only what is specific to a
SHEET: pagination, the per-candidate ID header and rim, and the manifest. Use
``--no-overlap`` or an explicit ``--columns`` list without ``overlap`` when a
narrower sheet is more useful.
"""

import argparse
import os

import cv2

from analysis.lib import panels
from analysis.lib.io import write_json
from analysis.viz import depth_units
from analysis.viz import frame_sheet
from analysis.viz import rows as rows_lib
from analysis.viz import sweep_sides
from core import renderer_settings
from core.visual_budget import check_budget


DEFAULT_COLUMNS = rows_lib.DEFAULT_COLUMNS
COLUMN_ALIASES = rows_lib.COLUMN_ALIASES
VALID_COLUMNS = rows_lib.VALID_COLUMNS
ROW_HEADER_HEIGHT = 40
PANEL_SEPARATOR_WIDTH = 8
PANEL_SEPARATOR_BGR = (110, 110, 110)
ROW_SEPARATOR_HEIGHT = 12
CANDIDATE_RIM_WIDTH = 6
CANDIDATE_RIM_BGR = (76, 90, 76)
# Compatibility re-export: raster policy lives with the shared panel helper.
MIN_DOWNSAMPLE_SCALE = panels.MIN_DOWNSAMPLE_SCALE
# The sheet has always shaded the hand region with a small dilation, unlike the
# composite strip's exact-mask default. Named rather than inlined so the
# difference is a stated choice instead of a stray literal — and PUBLIC, because
# `pool/panels.py` writes a sheet and a per-candidate strip for the same order and
# the two must shade the same pixels; a caller that had to guess 0.01 would be one
# edit away from two panels of the same candidate disagreeing about the occluder.
HAND_DILATE = 0.01


def parse_columns(value, no_overlap=False):
    """Validate a sheet's column list. Delegates to the registry in `viz.rows`,
    adding the sheet's own rule: a candidate row without RENDER shows nothing."""
    return rows_lib.parse_columns(
        value, drop=("overlap",) if no_overlap else (), require=("render",),
        allowed=rows_lib.IMAGE_COLUMNS)


def rendered_candidates(report):
    """Rendered report entries in displayed order with stable IDs.

    Delegates to `sweep_sides.report_rows`, the one normalizer that also handles
    oapply/osweep (a candidate there carries a per-FRAME image dict, so it yields
    one row per (candidate, frame) pair). A sheet shows one row per candidate, so
    the duplicate *_best.png copy of the rank-0 winner is dropped."""
    return sweep_sides.report_rows(report, drop_duplicate_winner=True)


class _FrameCache:
    """Per-frame source paths + decoded pixels, resolved once per distinct frame.

    Rows own their frame; this keeps the load cost at one per distinct frame
    rather than one per row. Only the SOURCE side is cached — the match render differs per row, so
    each row finishes its own `RowInputs` from this shared half."""

    def __init__(self, report, layout, source="", bg_mode="black"):
        self._report = report
        self._layout = layout
        self._source = source
        self._bg_mode = bg_mode
        self._frames = sweep_sides.frames_index(report)
        self._cache = {}

    def paths(self, frame):
        """(source, mask, hand_mask) for one frame; raises without a mask, since a
        candidate sheet's whole job is the source-vs-render comparison."""
        if frame in self._cache:
            return self._cache[frame]
        source_path, mask_path, hand_path = sweep_sides.row_paths(
            self._report, self._layout, frame, source=self._source,
            frame_masks=self._frames)
        if not mask_path:
            raise ValueError("candidate sheets require an object mask"
                             + (f" (frame {frame})" if frame else ""))
        self._cache[frame] = (source_path, mask_path, hand_path)
        return self._cache[frame]


def _row_header(width, channels, candidate_id):
    header = panels.solid(ROW_HEADER_HEIGHT, width, (15, 15, 15), channels)
    label_text = "CANDIDATE"
    left_text = str(candidate_id)
    right_text = str(candidate_id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = 0.38
    label_width = cv2.getTextSize(
        label_text, font, label_scale, 1)[0][0]
    left_x = 12 + label_width + 12
    right_scale = 0.70
    right_width = cv2.getTextSize(
        right_text, font, right_scale, 2)[0][0]
    left_scale = 0.70
    left_limit = max(1, width - right_width - left_x - 36)
    while left_scale > 0.45:
        left_width = cv2.getTextSize(
            left_text, font, left_scale, 2)[0][0]
        if left_width <= left_limit:
            break
        left_scale -= 0.05
    label_color = (145, 145, 145, 255) if channels == 4 else (145, 145, 145)
    cv2.putText(
        header, label_text, (12, 28), font, label_scale,
        label_color, 1, cv2.LINE_AA)
    id_color = (165, 200, 165, 255) if channels == 4 else (165, 200, 165)
    cv2.putText(
        header, left_text, (left_x, 30), font, left_scale,
        id_color, 2, cv2.LINE_AA)
    cv2.putText(
        header, right_text, (width - right_width - 12, 30),
        font, right_scale, id_color, 2, cv2.LINE_AA)
    line = (150, 150, 150, 255) if channels == 4 else (150, 150, 150)
    cv2.line(header, (0, ROW_HEADER_HEIGHT - 2),
             (width, ROW_HEADER_HEIGHT - 2), line, 2)
    return header


def _row(paths, render_path, rec, columns, height, bg_mode, render_dir,
         run_dir, depth_range):
    """One candidate row: the shared comparison strip under its ID header, rimmed.

    The panels themselves come from `rows.build_row`; what is added here is the
    sheet-specific chrome — the CANDIDATE header and the muted-green rim that lets
    a visual model group a row's panels without competing with the overlap colours.

    `height` is the height of the RENDERS, the same thing it means to
    `rows.build_row`, `composite` and `sweep_sides` — not of the finished row. The
    chrome is ADDED to it (each panel's header bar, then this row's ID header and
    rim), because every one of those stacks rather than overlays; nothing here may
    be subtracted out of the picture the reader came to look at.
    """
    if height <= 0:
        raise ValueError(f"--height must be positive; got {height}")
    source_path, mask_path, hand_path = paths
    inputs = rows_lib.from_paths(
        source_path, mask_path, render_path, hand_mask_path=hand_path,
        hand_dilate=HAND_DILATE, bg_mode=bg_mode,
        depth_panels=sweep_sides.rendered_depth_panels(
            render_dir, rec, height=height, run_dir=run_dir,
            depth_range=depth_range))
    built = rows_lib.build_row(
        inputs, columns=columns, height=height,
        sep_width=PANEL_SEPARATOR_WIDTH, sep_bgr=PANEL_SEPARATOR_BGR)
    body = built.image
    content = panels.vstack_rows(
        [_row_header(body.shape[1], body.shape[2], rec["candidate_id"]), body],
        sep_height=0)
    rim = (*CANDIDATE_RIM_BGR, 255) if content.shape[2] == 4 \
        else CANDIDATE_RIM_BGR
    return cv2.copyMakeBorder(
        content,
        CANDIDATE_RIM_WIDTH, CANDIDATE_RIM_WIDTH,
        CANDIDATE_RIM_WIDTH, CANDIDATE_RIM_WIDTH,
        cv2.BORDER_CONSTANT, value=rim), built


def write_sheets(render_dir, run_dir="", source="", out_dir="",
                 rows_per_page=3, columns=DEFAULT_COLUMNS, no_overlap=False,
                 height=360, bg_mode="black", budget=0, waive_budget=False,
                 max_dimension=0):
    """Write candidate_sheet_NNN.png pages and a JSON manifest.

    `render_dir` is one dir or a LIST of dirs; several orders roll onto one set of
    pages, rows keyed by order id, so an agent compares candidates across orders in
    one image. Returns ``(page_paths, manifest_path)``.
    """
    render_dirs = ([render_dir] if isinstance(render_dir, str)
                   else [str(d) for d in render_dir])
    if not render_dirs:
        raise ValueError("need at least one --render-dir")
    rows_per_page = int(rows_per_page)
    if rows_per_page < 1:
        raise ValueError("--rows-per-page must be at least 1")
    columns = parse_columns(",".join(columns) if not isinstance(columns, str)
                            else columns, no_overlap=no_overlap)
    layout = sweep_sides._layout(run_dir)

    multi = len(render_dirs) > 1
    records, sources, reports = [], {}, []
    for d in render_dirs:
        d = os.path.abspath(d)
        report, report_stem = sweep_sides.find_report(d)
        rows = rendered_candidates(report)
        if not rows:
            raise ValueError(
                f"{d}: report lists no rendered candidate images; re-run "
                "the order with dump_topk (--sweep-dump-topk)")
        order_id = os.path.basename(d)
        sources[d] = _FrameCache(report, layout, source=source,
                                 bg_mode=bg_mode)
        reports.append(os.path.join(d, f"{report_stem}.json"))
        for rec in rows:
            records.append({**rec, "_dir": d, "_order": order_id,
                            "candidate_id": (f"{order_id}  {rec['candidate_id']}"
                                             if multi else rec["candidate_id"])})

    check_budget(len(records), budget=budget, waived=waive_budget,
                 what=(f"{len(render_dirs)} orders" if multi
                       else os.path.basename(render_dirs[0])))
    depth_range = (sweep_sides.rendered_depth_range(
        [(rec["_dir"], rec) for rec in records], run_dir=run_dir)
        if "depth" in columns else None)

    out_dir = os.path.abspath(out_dir or render_dirs[0])
    os.makedirs(out_dir, exist_ok=True)
    page_paths = []
    original_page_paths = []
    page_rasters = []
    manifest_rows = []
    drawn_columns = list(columns)
    page_count = (len(records) + rows_per_page - 1) // rows_per_page
    for page_start in range(0, len(records), rows_per_page):
        page_records = records[page_start:page_start + rows_per_page]
        page_number = len(page_paths) + 1
        page_rows = []
        for row_number, rec in enumerate(page_records, start=1):
            render_path = os.path.join(rec["_dir"], rec["image"])
            if not os.path.isfile(render_path):
                raise FileNotFoundError(
                    f"candidate render does not exist: {render_path}")
            # each ROW resolves its own frame's source (cached per frame), which
            # is what lets one sheet mix frames — and oapply exist at all.
            paths = sources[rec["_dir"]].paths(rec.get("frame"))
            source_path, mask_path, _ = paths
            row_image, built = _row(
                paths, render_path, rec, columns, height, bg_mode, rec["_dir"],
                run_dir, depth_range)
            page_rows.append(row_image)
            drawn_columns = built.columns
            manifest_rows.append({
                "candidate_id": rec["candidate_id"],
                "image": rec["image"],
                "display_rank": rec["display_rank"],
                "page": page_number,
                "row": row_number,
                "frame": rec.get("frame"),
                "order": rec["_order"],
                "source": source_path,
                "mask": mask_path,
                **({"label": rec["label"]} if rec.get("label") else {}),
                "combined": rec.get("combined"),
                "iou_raw": rec.get("iou_raw"),
                "iou_visible": rec.get("iou_visible"),
                **({"mean_gate_iou": rec["mean_gate_iou"]}
                   if rec.get("mean_gate_iou") is not None else {}),
            })
        page = panels.vstack_rows(page_rows, sep_height=ROW_SEPARATOR_HEIGHT)
        if depth_range and "depth" in drawn_columns:
            header = frame_sheet._page_header(
                page.shape[1], page_number, page_count,
                page_start + 1, page_start + len(page_records), len(records),
                title="DEPTH (OBJECT UNITS)",
                ramp=depth_units.object_units_ramp(*depth_range))
            page = panels.vstack_rows([header, page], sep_height=0)
        page_path = os.path.join(
            out_dir, f"candidate_sheet_{page_number:03d}.png")
        variants = panels.write_image_variants(
            page_path, page, preview_max_dimension=max_dimension)
        page_paths.append(variants["preview"])
        original_page_paths.append(variants["original"])
        page_rasters.append({"page": page_number, **variants})
        print(f"[candidate-sheet] wrote {variants['original']}")
        if variants["preview"] != variants["original"]:
            print(f"[candidate-sheet] wrote {variants['preview']} (preview)")

    manifest = {
        "schema_version": 4,
        "report": reports[0],
        "reports": reports,
        # what the rows actually DREW: a column whose inputs were absent is
        # skipped per row rather than failing the page, so the manifest must
        # report the drawn set, not the requested one.
        "columns": drawn_columns,
        **({"depth_scale": {
            "units": "object_lengths",
            "range": [round(depth_range[0], 4), round(depth_range[1], 4)],
            "range_mode": "shared-2-98pct",
        }} if depth_range else {}),
        "rows_per_page": rows_per_page,
        "page_raster": {
            "max_dimension": max_dimension,
            "min_scale": panels.MIN_DOWNSAMPLE_SCALE,
            "pages": page_rasters,
        },
        # `pages` remains the default inspection list. Schema v4 adds the native
        # originals explicitly so an agent can escalate without rebuilding.
        "pages": page_paths,
        "original_pages": original_page_paths,
        "rows": manifest_rows,
    }
    manifest_path = os.path.join(out_dir, "candidate_sheet_manifest.json")
    write_json(manifest_path, manifest)
    print(f"[candidate-sheet] wrote {manifest_path}")
    return page_paths, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render-dir", required=True, action="append",
                        help="a sweep/apply/oapply/osweep output dir; repeat to "
                             "roll several orders onto one sheet")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--source", default="",
                        help="explicit source frame path (overrides --run-dir)")
    parser.add_argument("--out-dir", default="",
                        help="output directory (default: --render-dir)")
    parser.add_argument("--rows-per-page", type=int, default=3)
    parser.add_argument(
        "--columns", default=",".join(DEFAULT_COLUMNS),
        help="comma-separated columns; default: "
             f"{','.join(DEFAULT_COLUMNS)}")
    parser.add_argument(
        "--no-overlap", action="store_true",
        help=f"omit the silhouette overlap column ({panels.OVERLAP_CAPTION}) "
             "from the default or explicit column set")
    parser.add_argument("--height", type=int, default=360,
                        help="height of each RENDER in pixels; the row comes out "
                             "taller by its panel header bars, ID header and rim "
                             "(all of which stack above the pictures, never over "
                             "them)")
    parser.add_argument(
        "--max-dimension", type=int, default=0,
        help="longest-edge target for each derived preview; canonical pages stay "
             "native and previews never drop below 0.5x (0 = no preview)")
    parser.add_argument("--bg-mode", default="black",
                        choices=list(renderer_settings.BG_MODES))
    parser.add_argument("--budget", type=int, default=0,
                        help="max panel rows before refusing (0 = the default 100)")
    parser.add_argument("--waive-visual-budget", action="store_true",
                        help="build the sheet even past the budget")
    args = parser.parse_args()
    write_sheets(
        args.render_dir, run_dir=args.run_dir, source=args.source,
        out_dir=args.out_dir, rows_per_page=args.rows_per_page,
        columns=args.columns, no_overlap=args.no_overlap, height=args.height,
        bg_mode=args.bg_mode, budget=args.budget,
        waive_budget=args.waive_visual_budget,
        max_dimension=args.max_dimension)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
