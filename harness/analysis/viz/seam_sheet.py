"""seam_sheet — every SEAM on one sheet, each as a visual PAIR band.

A seam is a single temporal STEP (frame_a -> frame_b) whose two frames were posed
by DIFFERENT owners: two refiner windows, or a window and an already-committed
neighbor (analysis.temporal.seams.crossing_steps). Nobody saw both sides of it,
so it is the one place a cross-window disagreement — most often a ~180 deg basin
flip on one side only — can hide.

ONE BAND PER SEAM, ROLLED ONTO ONE SHEET (the analysis.viz.candidate_sheet shape:
a row per thing to judge, stacked, so the whole decision is one read):

    +=================================================================+
    | SEAMS  3 of 3   pass p1                                         |
    +-----------------------------------------------------------------+
    | 01  000040.jpg -> 000048.jpg   rot 171.4 deg   lid +2 deg       |
    |      | A 000040 | B 000048 |               <- SOURCE row        |
    |      |          |          |               <- COMMITTED row     |
    +-----------------------------------------------------------------+
    | 02  000088.jpg -> 000096.jpg   ...                              |
    +=================================================================+

THE PICTURES CARRY THE SHEET, NOT THE CAPTION. A band's note is one terse line
(analysis.temporal.seams.step_lines): the magnitudes, nothing restated that the
title or a column header already says, no unit legend repeated per band. The
sheet exists to be LOOKED at, and pixels spent on prose are pixels taken from the
frames. Nothing is lost by that — the full-precision numbers live in the temporal
report JSON, and this line plus every frame's source/render path lives in the
sheet manifest, so the caption is a pointer, not the record.

WHY ROLLED AND NOT ONE PAGE EACH. The reviewer's job is one verdict PER SEAM, and
the seams inform each other: two seams flipping the SAME way is one window in the
wrong basin (re-pose that window), while two flipping independently is two
separate steps to adjudicate. Separate pages hide that correlation. Seam counts
stay small — K contiguously tiling windows produce K-1 seams, plus one per border
against a committed frame outside every window — so all of them fit. `seams_per_page`
still bounds a sheet (and =1 gives the close-inspection page).

WHY THIS IS NOT frame_sheet. analysis.viz.frame_sheet answers "show me the
timeline": many frames, a uniform grid, paginated in temporal order. A seam is the
opposite shape — TWO frames compared against EACH OTHER, where the comparison is
the entire content. So the pair carries a bright rim and a wide gutter, and which
two frames are under review is unambiguous at a glance.

CONTEXT IS OPT-IN (`context=0` by default) for the same reason. Flanking frames
are drawn dim, but they still take column width: at a fixed sheet width, one frame
of context per side halves the pair's own pixels — exactly the resolution the basin
call depends on. Ask for context when the question is the local TREND (does the
motion continue past the seam?); leave it off when the question is the pair.

BOTH ROWS MATTER. The source row is the ground truth: the captured video is
always temporally smooth, so a jump there is real motion. The COMMITTED row is
the render of the pose actually written to pose.json (`match_<stem>.png` in a
pass dir) — and it is the only thing that shows which basin a pose landed in.
A large `rotation_deg` reads the same for a genuine half-turn and a spurious
flip; put the two renders next to the two photos and the ambiguity resolves by
eye. A frame with no render in the pass dir gets an explicit placeholder panel,
never a silently dropped tile.

This module DRAWS; it does not judge. It attaches no verdict and computes no
threshold. Each band's note line is the step's measured magnitudes: taken
verbatim from the caller when it has the steps in hand, or measured here from
`--pose-json` via analysis.temporal.seams.step_lines. The
numbers are printed so the eye has something to check them against — never so
that a number decides.

Runs in the 'artscript' env, from harness/:
  micromamba run -n artscript python -m analysis.viz.seam_sheet \
      --run-dir RUN_DIR --pass-dir RUN_DIR/iterations/NNNNNN/renders/NNNN \
      --pose-json RUN_DIR/mesh/pose.json \
      --seam 000040.jpg:000048.jpg [--seam ...] [--context 1]
"""

import argparse
import math
import os

import cv2
import numpy as np

from analysis.lib import panels, rasters
from analysis.lib.io import write_json
from analysis.viz import frame_sheet
from core import renderer_settings

COLUMN_HEADER_HEIGHT = 34
CAPTION_HEIGHT = 24
COLUMN_RIM_WIDTH = 4
SEAM_RIM_BGR = (196, 148, 72)        # the pair under review — bright
CONTEXT_RIM_BGR = (58, 58, 58)       # flanking context — dim
COLUMN_GAP = 10
PAIR_GAP = 22                        # a wider gutter splits A from B
PAGE_HEADER_HEIGHT = 46
BAND_TITLE_HEIGHT = 32
BAND_GAP_HEIGHT = 14                 # the rule between two seams' bands
BAND_GAP_BGR = (70, 70, 70)
NOTE_LINE_HEIGHT = 22
MISSING_RENDER_BGR = (26, 26, 32)
# the widest a band title can get ("01   000000.jpg  ->  000000.jpg"), so a
# one-line note can be placed after it without measuring every title.
_TITLE_WIDTH = 330


def parse_seams(specs):
    """``["a.jpg:b.jpg", ...]`` -> ``[{"from": "a.jpg", "to": "b.jpg"}, ...]``.

    Also accepts entries that are already ``{"from", "to"}`` dicts (so a caller
    can hand over seams_lib.crossing_steps output untouched, note lines and all).
    Order is preserved; duplicates are the caller's business."""
    out = []
    for spec in specs or []:
        if isinstance(spec, dict):
            if not spec.get("from") or not spec.get("to"):
                raise ValueError(f"seam needs 'from' and 'to': {spec!r}")
            out.append(spec)
            continue
        for token in str(spec).split(","):
            token = token.strip()
            if not token:
                continue
            parts = [p.strip() for p in token.split(":")]
            if len(parts) != 2 or not all(parts):
                raise ValueError(
                    f"seam {token!r} must be FROM:TO (e.g. 000040.jpg:000048.jpg)")
            out.append({"from": parts[0], "to": parts[1]})
    if not out:
        raise ValueError("no seams given")
    return out


def measure_notes(seams, pose_json):
    """Fill each seam's `note` with its measured magnitudes, from `pose_json`.

    A seam whose caller already supplied a note keeps it (a caller holding the
    steps can add who posed each side, which no pose.json can know).
    For everything else this diffs the committed sequence once and captions each
    band with analysis.temporal.seams.step_lines — so a hand-run sheet is not
    silently caption-less, and the numbers on it are the SAME ones the temporal
    report and the gates print.

    Still draws-only: a note is a measurement printed under a title. A seam whose
    step is not in the sequence (a frame absent from pose.json) simply keeps an
    empty note rather than failing the sheet — the pictures are the point."""
    from analysis.temporal import report as temporal_report             # noqa: PLC0415
    from analysis.temporal import seams as seams_lib                    # noqa: PLC0415

    # through analysis.temporal.report, like every other temporal read in the
    # tree: it carries the per-step records beside the per-frame table, so a
    # caption here quotes the same numbers the report prints rather than a second
    # computation of them.
    steps = {seams_lib.step_key(s): s
             for s in temporal_report.build(pose_json)["steps"]}
    out = []
    for seam in seams:
        if seam.get("note"):
            out.append(seam)
            continue
        step = steps.get((seam["from"], seam["to"]))
        out.append({**seam,
                    "note": seams_lib.step_lines(step) if step else []})
    return out


def resolve_pair(names, seam, context=0):
    """``(pair, context_before, context_after)`` frame names for one seam.

    Both seam frames must exist in `names` and be ADJACENT in it: a seam is one
    temporal step, and a sheet spanning a gap would invite comparing frames that
    never met in the residual. `context` frames on each side are included when
    available (clamped at the sequence ends, never invented)."""
    context = max(0, int(context))
    a, b = seam["from"], seam["to"]
    missing = [n for n in (a, b) if n not in names]
    if missing:
        raise ValueError(
            f"seam frame(s) absent from the sequence: {', '.join(missing)}")
    ia, ib = names.index(a), names.index(b)
    if ib - ia != 1:
        raise ValueError(
            f"seam {a} -> {b} is not one step: they sit {ib - ia} apart in the "
            "frame order (a seam is a single transition)")
    return ([a, b],
            names[max(0, ia - context):ia],
            names[ib + 1:ib + 1 + context])


def render_path(pass_dir, frame):
    """The committed-pose render for `frame` (``match_<stem>.png`` in a pass dir),
    or "" when there is no pass dir or the render was never produced."""
    if not pass_dir:
        return ""
    stem = os.path.splitext(os.path.basename(frame))[0]
    path = os.path.join(pass_dir, f"match_{stem}.png")
    return path if os.path.isfile(path) else ""


def _solid(height, width, color):
    return np.full((height, width, 3), color, dtype=np.uint8)


def _centered(image, height, width):
    """`image` scaled to `height` and centered in a `width` black panel."""
    scaled = panels.scale_to_height(image, height)
    if scaled.shape[1] > width:
        scaled = cv2.resize(scaled, (width, height), interpolation=cv2.INTER_AREA)
    panel = _solid(height, width, (0, 0, 0))
    x = (width - scaled.shape[1]) // 2
    panel[:, x:x + scaled.shape[1]] = scaled
    return panel


def _caption(text, width, color=(190, 190, 190)):
    strip = _solid(CAPTION_HEIGHT, width, (15, 15, 15))
    cv2.putText(strip, text, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                color, 1, cv2.LINE_AA)
    return strip


def _column_header(text, role, width, in_seam):
    head = _solid(COLUMN_HEADER_HEIGHT, width, (18, 18, 18))
    color = (235, 235, 235) if in_seam else (140, 140, 140)
    cv2.putText(head, text, (7, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2 if in_seam else 1, cv2.LINE_AA)
    role_width = cv2.getTextSize(role, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    cv2.putText(head, role, (width - role_width - 7, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (196, 148, 72) if in_seam else (110, 110, 110), 1, cv2.LINE_AA)
    return head


def _missing_render_panel(height, width):
    """An explicit "no render" panel. A seam is reviewed by comparing renders, so
    an absent one is INFORMATION (this frame was never applied in this pass) and
    must be visible, not a hole in the layout."""
    panel = _solid(height, width, MISSING_RENDER_BGR)
    text = "no committed render in this pass"
    scale = 0.46
    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    cv2.putText(panel, text, (max(6, (width - tw) // 2), height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (120, 120, 130), 1, cv2.LINE_AA)
    return panel


def _column(source_path, frame, role, in_seam, image_height, content_width,
            render_pathname="", with_render=False, bg_mode="black"):
    """One frame's column: header / SOURCE / (optional) COMMITTED render."""
    parts = [_column_header(frame, role, content_width, in_seam),
             _caption("SOURCE (captured video)", content_width),
             _centered(rasters.load_bgr(source_path), image_height,
                       content_width)]
    if with_render:
        parts.append(_caption("COMMITTED RENDER (pose.json)", content_width,
                              color=(196, 178, 140)))
        if render_pathname:
            bgr, alpha = rasters.load_rgba(render_pathname)
            shown = renderer_settings.present(bgr, alpha, bg_mode)
            if shown.ndim == 3 and shown.shape[2] == 4:
                shown = shown[:, :, :3]
            parts.append(_centered(shown, image_height, content_width))
        else:
            parts.append(_missing_render_panel(image_height, content_width))
    content = np.vstack(parts)
    return cv2.copyMakeBorder(
        content, COLUMN_RIM_WIDTH, COLUMN_RIM_WIDTH, COLUMN_RIM_WIDTH,
        COLUMN_RIM_WIDTH, cv2.BORDER_CONSTANT,
        value=SEAM_RIM_BGR if in_seam else CONTEXT_RIM_BGR)


def _page_header(width, page_number, page_count, first, last, total, pass_dir):
    """The sheet header: which seams are on this page, out of how many."""
    head = _solid(PAGE_HEADER_HEIGHT, width, (12, 12, 12))
    left = (f"SEAMS  {first:02d}-{last:02d} OF {total:02d}"
            + (f"   pass {os.path.basename(pass_dir)}" if pass_dir else
               "   (source only — no committed renders)"))
    right = f"PAGE {page_number:02d}/{page_count:02d}"
    cv2.putText(head, left, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (220, 220, 220), 2, cv2.LINE_AA)
    rw = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0]
    cv2.putText(head, right, (width - rw - 12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (160, 160, 160), 2, cv2.LINE_AA)
    return head


def _title_text(seam, ordinal):
    """A band's handle: its ordinal and its two endpoints. The ordinal numbers the
    verdicts the reviewer owes — one per band."""
    return f"{ordinal:02d}   {seam['from']}  ->  {seam['to']}"


def note_layout(seams, width):
    """``(inline, slots)`` — how the bands' notes are laid out on this sheet.

    ONE layout for the whole sheet, chosen from the longest note present, because
    bands must stay the same height to read as a column of equals. `inline` is True
    when every note is a single line that fits beside its title on the title row,
    which is the common case now that a note is one terse line — it costs the sheet
    no vertical space at all. Otherwise notes take `slots` lines below the title,
    padded to the same count for every band."""
    notes = [[str(line) for line in (seam.get("note") or [])] for seam in seams]
    slots = max((len(lines) for lines in notes), default=0)
    single = [lines[0] for lines in notes if len(lines) == 1]
    inline = slots <= 1 and len(single) == len([n for n in notes if n]) and all(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        <= width - _TITLE_WIDTH - 60 for line in single)
    return inline, (0 if inline else slots)


def _band_header(width, seam, ordinal, note, slots, inline):
    """One seam's caption strip: its title, and its measured numbers verbatim from
    the caller — beside the title when they fit on one line, below it otherwise.

    Descriptive: this states what was measured, never whether it is acceptable."""
    lines = [str(line) for line in (note or [])]
    height = BAND_TITLE_HEIGHT + (0 if inline else NOTE_LINE_HEIGHT * slots + 6)
    strip = _solid(height, width, (24, 24, 30))
    cv2.putText(strip, _title_text(seam, ordinal), (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (232, 232, 232), 2, cv2.LINE_AA)
    if inline:
        if lines:
            cv2.putText(strip, lines[0], (_TITLE_WIDTH + 24, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (196, 205, 215), 1,
                        cv2.LINE_AA)
        return strip
    for i, line in enumerate(lines):
        scale = 0.48
        while (scale > 0.32 and cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width - 60):
            scale -= 0.02
        cv2.putText(strip, line, (44, BAND_TITLE_HEIGHT + 16 + i * NOTE_LINE_HEIGHT),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (196, 205, 215), 1,
                    cv2.LINE_AA)
    return strip


def write_seam_sheets(seams, run_dir="", frames_dir="", pass_dir="", out_dir="",
                      context=0, tile_height=340, bg_mode="black",
                      seams_per_page=0, basename="seam", pose_json=None):
    """Roll every seam onto one sheet, one band each.

    Returns ``(page_paths, manifest_path)``.

    `seams` — ``[{"from", "to", "note": [lines]}]`` (see parse_seams); `note` is
    optional and printed verbatim in the band's caption.
    `pose_json` — a pose.json (dict or path) to MEASURE the missing notes from
    (see measure_notes). A caller holding the steps already passes notes instead.
    `pass_dir` — a pass directory holding ``match_<stem>.png`` committed renders.
    Omit it for a source-only sheet; include it whenever the question is which
    basin a pose landed in, since only the render shows that.
    `context` — frames of flanking context per side (dim rim), clamped at the ends.
    `tile_height` — the height of ONE image row (its caption included). It is NOT
    a budget split across rows: adding the committed-render row makes the band
    taller rather than shrinking the photos, because a seam is decided by looking
    closely at both.
    `seams_per_page` — 0 (default) puts every seam on ONE sheet, so correlated
    seams (several flipping the same way = one window in the wrong basin) are
    visible together. Set it to bound a very long sheet; 1 gives a page per seam
    for close inspection.
    """
    frames_dir, names, _ = frame_sheet.resolve_source(
        run_dir=run_dir, frames_dir=frames_dir)
    seams = parse_seams(seams)
    if pose_json is not None:
        seams = measure_notes(seams, pose_json)
    resolved = [resolve_pair(names, seam, context=context) for seam in seams]

    tile_height = int(tile_height)
    with_render = bool(pass_dir)
    if tile_height <= CAPTION_HEIGHT + 24:
        raise ValueError(
            f"--tile-height must exceed {CAPTION_HEIGHT + 24} pixels "
            "(one image row plus its caption)")
    image_height = tile_height - CAPTION_HEIGHT
    seams_per_page = int(seams_per_page) or len(seams)
    if seams_per_page < 1:
        raise ValueError("--seams-per-page must be at least 1 (0 = all on one)")

    if pass_dir:
        pass_dir = os.path.abspath(pass_dir)
        if not os.path.isdir(pass_dir):
            raise ValueError(f"pass dir does not exist: {pass_dir}")
    if not out_dir:
        out_dir = os.path.join(
            os.path.abspath(run_dir) if run_dir else frames_dir, "seam_sheets")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ONE geometry for every band on every page: same column width, same column
    # count, same caption height. Bands must be directly comparable — that is the
    # whole reason for rolling them together — so nothing may shift between them.
    widths = []
    for pair, before, after in resolved:
        for name in list(before) + list(pair) + list(after):
            image = rasters.load_bgr(os.path.join(frames_dir, name))
            widths.append(max(1, int(round(
                image.shape[1] * image_height / image.shape[0]))))
    content_width = max(widths)
    column_width = content_width + 2 * COLUMN_RIM_WIDTH
    max_columns = max(len(pair) + len(before) + len(after)
                      for pair, before, after in resolved)
    band_width = (max_columns * column_width
                  + (max_columns - 1) * max(COLUMN_GAP, PAIR_GAP))
    note_inline, note_slots = note_layout(seams, band_width)

    def band(seam, ordinal, pair, before, after):
        """One seam's band: caption + its row of columns. Also returns the
        manifest entries for the frames it drew."""
        columns, entries = [], []
        plan = ([(n, "context", False) for n in before]
                + [(pair[0], "A (from)", True), (pair[1], "B (to)", True)]
                + [(n, "context", False) for n in after])
        for name, role, in_seam in plan:
            rp = render_path(pass_dir, name) if with_render else ""
            columns.append(_column(
                os.path.join(frames_dir, name), name, role, in_seam,
                image_height, content_width, render_pathname=rp,
                with_render=with_render, bg_mode=bg_mode))
            entries.append({"frame_id": name, "role": role, "in_seam": in_seam,
                            "source_path": os.path.join(frames_dir, name),
                            "render_path": rp, "has_render": bool(rp)})

        gaps = [PAIR_GAP if entries[i]["in_seam"] and entries[i + 1]["in_seam"]
                else COLUMN_GAP for i in range(len(columns) - 1)]
        row = _solid(max(c.shape[0] for c in columns), band_width, (45, 45, 45))
        x = 0
        for i, column in enumerate(columns):
            row[0:column.shape[0], x:x + column_width] = column
            x += column_width + (gaps[i] if i < len(gaps) else 0)
        head = _band_header(band_width, seam, ordinal, seam.get("note"),
                            note_slots, note_inline)
        return np.vstack([head, row]), entries

    total = len(seams)
    page_count = math.ceil(total / seams_per_page)
    rule = _solid(BAND_GAP_HEIGHT, band_width, BAND_GAP_BGR)
    pages, manifest_seams, missing_renders = [], [], []
    for page_index in range(page_count):
        start = page_index * seams_per_page
        chunk = list(zip(seams, resolved))[start:start + seams_per_page]
        stack = [_page_header(band_width, page_index + 1, page_count,
                              start + 1, start + len(chunk), total, pass_dir)]
        for local, (seam, (pair, before, after)) in enumerate(chunk):
            ordinal = start + local + 1
            drawn, entries = band(seam, ordinal, pair, before, after)
            if local:
                stack.append(rule)
            stack.append(drawn)
            missing_renders += [e["frame_id"] for e in entries
                                if with_render and not e["has_render"]]
            manifest_seams.append({
                "ordinal": ordinal,
                "from": seam["from"], "to": seam["to"],
                "note": [str(line) for line in (seam.get("note") or [])],
                "page": page_index + 1, "frames": entries,
            })
        page = np.vstack(stack)
        page_path = os.path.join(out_dir, f"{basename}_{page_index + 1:03d}.png")
        if not cv2.imwrite(page_path, page):
            raise OSError(f"failed to write {page_path}")
        pages.append(page_path)
        print(f"[seam-sheet] wrote {page_path} ({len(chunk)} seam band(s))")
    for entry in manifest_seams:
        entry["page_path"] = pages[entry["page"] - 1]

    manifest = {
        "schema_version": 1,
        "frames_dir": frames_dir,
        "pass_dir": pass_dir,
        "context": max(0, int(context)),
        "tile_height": tile_height,
        "seams_per_page": seams_per_page,
        "with_committed_render": with_render,
        # named, not counted: a seam whose render is absent was never applied in
        # this pass, and the reviewer needs to know WHICH one.
        "frames_without_render": sorted(set(missing_renders)),
        "seam_count": total,
        "pages": pages,
        "seams": manifest_seams,
    }
    manifest_path = os.path.join(out_dir, f"{basename}_manifest.json")
    write_json(manifest_path, manifest)
    print(f"[seam-sheet] wrote {manifest_path}")
    return pages, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="roll every seam (a cross-window transition) onto one sheet, "
                    "one band each: the pair of source frames side by side and, "
                    "with --pass-dir, their committed-pose renders. Draws only — "
                    "no threshold and no verdict",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default="",
                        help="run whose layout.json supplies frames and order")
    parser.add_argument("--frames-dir", default="",
                        help="source frames directory; overrides the layout path")
    parser.add_argument("--pass-dir", default="",
                        help="pass dir with match_<stem>.png committed renders; "
                             "omit for a source-only sheet")
    parser.add_argument("--seam", action="append", default=[],
                        help="FROM:TO (repeatable, or a comma list); the two "
                             "frames must be one step apart")
    parser.add_argument("--pose-json", default="",
                        help="committed poses to caption each band's magnitudes "
                             "from (default: RUN_DIR/mesh/pose.json when it "
                             "exists); --no-notes draws the pictures bare")
    parser.add_argument("--no-notes", action="store_true",
                        help="draw the bands without the measured caption")
    parser.add_argument("--context", type=int, default=0,
                        help="flanking context frames per side (default: 0)")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--tile-height", type=int, default=340,
                        help="height of ONE image row (default: 340); the render "
                             "row makes the band taller, not the photos smaller")
    parser.add_argument("--seams-per-page", type=int, default=0,
                        help="0 (default) = every seam on one sheet, so "
                             "correlated seams are visible together; 1 = a page "
                             "per seam for close inspection")
    parser.add_argument("--bg-mode", default="black",
                        choices=list(renderer_settings.BG_MODES))
    parser.add_argument("--basename", default="seam")
    args = parser.parse_args()
    if not args.run_dir and not args.frames_dir:
        parser.error("one of --run-dir or --frames-dir is required")
    if not args.seam:
        parser.error("at least one --seam FROM:TO is required")
    pose_json = args.pose_json
    if not pose_json and args.run_dir and not args.no_notes:
        default_pose = os.path.join(args.run_dir, "mesh", "pose.json")
        pose_json = default_pose if os.path.exists(default_pose) else ""
    if args.no_notes:
        pose_json = ""
    if pose_json and not os.path.exists(pose_json):
        parser.error(f"--pose-json not found: {pose_json}")
    try:
        write_seam_sheets(
            args.seam, run_dir=args.run_dir, frames_dir=args.frames_dir,
            pass_dir=args.pass_dir, out_dir=args.out_dir, context=args.context,
            tile_height=args.tile_height, bg_mode=args.bg_mode,
            seams_per_page=args.seams_per_page, basename=args.basename,
            pose_json=pose_json or None)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
