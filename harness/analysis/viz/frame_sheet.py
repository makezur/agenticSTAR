"""Build paginated, temporally ordered source-frame sheets.

The tool reads a run's layout.json or an explicit frames directory, labels every
tile with its frame ID and sequence position, and writes as many pages as needed.
Use --frames for a subset (repeatable names, comma lists, globs, or an inclusive
``first-last`` range) and --every for optional overview sampling.

``--columns``/``--frames-per-page``/``--tile-height`` describe the LARGEST page
the sheet may occupy, not a grid every sheet must fill. A short selection is
repacked into that same footprint: the grid is rechosen and the tiles grow until
one budget edge binds, so a six-frame window sheet reads as a 3x2 of big tiles
instead of a 4x3 half full of black squares. Pages are also balanced, so the last
page is never the only sparse one. Growth stops at the source resolution, so
packing never upscales past native pixels, and never shrinks a tile below the
``--tile-height`` you asked for. Pass ``--no-pack`` for the literal fixed grid.
"""

import argparse
import glob
import json
import math
import os

import cv2
import numpy as np

from analysis import frames as frame_utils
from analysis.lib.io import write_json
from core import run_layout


HARNESS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(HARNESS)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

TILE_HEADER_HEIGHT = 42
TILE_RIM_WIDTH = 4
TILE_RIM_BGR = (72, 72, 72)
# highlighted tiles (a caller's "these are the frames under review") carry the
# same bright rim as seam_sheet's pair, so the mark reads the same everywhere.
HIGHLIGHT_RIM_BGR = (196, 148, 72)
CELL_GAP = 12
PAGE_HEADER_HEIGHT = 50
STACKED_KEY_HEADER_HEIGHT = 78
KEY_HEADER_STACK_WIDTH = 640


def _existing_dir(path):
    """Resolve a directory, including run layouts moved with the repository."""
    if not path:
        return ""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return path
    normalized = path.replace("\\", "/")
    marker = "/captures/"
    if marker in normalized:
        relocated = os.path.join(
            REPO, "captures", normalized.split(marker, 1)[1])
        if os.path.isdir(relocated):
            return relocated
    return ""


def _directory_frames(frames_dir):
    names = [
        name for name in os.listdir(frames_dir)
        if os.path.isfile(os.path.join(frames_dir, name))
        and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]
    return frame_utils.order_frames(names)


def resolve_source(run_dir="", frames_dir=""):
    """Return ``(frames_dir, ordered_frame_names, layout_or_empty)``."""
    layout = {}
    if run_dir:
        layout = run_layout.load_run_layout(run_dir) or {}
    candidates = [frames_dir, layout.get("frames_dir", "")]
    if layout.get("capture"):
        candidates.append(os.path.join(layout["capture"], "frames"))
    resolved_dir = next(
        (resolved for path in candidates
         if (resolved := _existing_dir(path))),
        "")
    if not resolved_dir:
        raise ValueError(
            "could not resolve a frames directory; pass --frames-dir or a "
            "--run-dir whose layout.json points to an existing capture")

    names = list(layout.get("frames") or []) if layout else []
    if not names:
        names = _directory_frames(resolved_dir)
    names = frame_utils.order_frames(os.path.basename(name) for name in names)
    missing = [
        name for name in names
        if not os.path.isfile(os.path.join(resolved_dir, name))
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} layout frames are missing from {resolved_dir}: "
            f"{preview}")
    if not names:
        raise ValueError(f"no source images found in {resolved_dir}")
    return resolved_dir, names, layout


def _selector_tokens(specs):
    return [
        token.strip()
        for spec in specs or []
        for token in str(spec).split(",")
        if token.strip()
    ]


def _name_lookup(names):
    lookup = {}
    for name in names:
        lookup[name] = name
        lookup[os.path.splitext(name)[0]] = name
    return lookup


def parse_steps(specs, names):
    """Resolve repeatable ``A:B`` temporal steps against ordered frame names."""
    lookup = _name_lookup(names)
    steps = []
    for spec in specs:
        parts = [part.strip() for part in str(spec).split(":") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"--step must be 'A:B', got {spec!r}")
        resolved = []
        for part in parts:
            name = lookup.get(os.path.basename(part))
            if not name:
                raise ValueError(
                    f"step endpoint {part!r} is not one of the run's frames; "
                    f"the layout has: {', '.join(names)}")
            resolved.append(name)
        a, b = resolved
        if a == b:
            raise ValueError(f"--step {spec!r} names the same frame twice")
        if names.index(a) > names.index(b):
            a, b = b, a
        steps.append((a, b))
    return steps


def select_frames(names, frames_dir, specs=(), every=1):
    """Select a temporal subset from ``names`` and return it in source order."""
    every = int(every)
    if every < 1:
        raise ValueError("--every must be at least 1")
    tokens = _selector_tokens(specs)
    if not tokens:
        selected = list(names)
    else:
        lookup = _name_lookup(names)
        selected_set = set()
        unknown = []
        for token in tokens:
            basename = os.path.basename(token)
            exact = lookup.get(basename)
            if exact:
                selected_set.add(exact)
                continue

            if any(char in token for char in "*?["):
                pattern = token if os.path.isabs(token) \
                    else os.path.join(frames_dir, token)
                hits = [
                    lookup.get(os.path.basename(path))
                    for path in sorted(glob.glob(pattern))
                ]
                hits = [hit for hit in hits if hit]
                if hits:
                    selected_set.update(hits)
                    continue

            if token.count("-") == 1:
                first, last = (part.strip() for part in token.split("-", 1))
                first = lookup.get(os.path.basename(first))
                last = lookup.get(os.path.basename(last))
                if first and last:
                    i0, i1 = names.index(first), names.index(last)
                    if i0 > i1:
                        raise ValueError(
                            f"frame range {token!r} runs backwards")
                    selected_set.update(names[i0:i1 + 1])
                    continue
            unknown.append(token)
        if unknown:
            raise ValueError(
                "unknown frame selectors: " + ", ".join(unknown))
        selected = [name for name in names if name in selected_set]
    selected = selected[::every]
    if not selected:
        raise ValueError("frame selection is empty")
    return selected


def _solid(height, width, color):
    return np.full((height, width, 3), color, dtype=np.uint8)


def balance_pages(total, frames_per_page):
    """Split ``total`` frames into near-equal pages of at most ``frames_per_page``.

    Chunking greedily leaves the remainder alone on the last page: 30 frames at
    12-up is 12/12/6, so the final sheet is half black. The page COUNT is what
    the budget fixes, so once it is known the frames spread evenly across it and
    every page is equally full to within one frame.
    """
    page_count = max(1, math.ceil(total / frames_per_page))
    base, extra = divmod(total, page_count)
    return [base + (index < extra) for index in range(page_count)]


def _grid_candidates(count, max_columns, max_rows):
    """Yield ``(columns, rows)`` grids that hold ``count`` tiles in the budget."""
    for columns in range(1, max_columns + 1):
        rows = math.ceil(count / columns)
        if rows <= max_rows:
            yield columns, rows


def plan_layout(count, aspect, columns, frames_per_page, tile_height,
                native_height=0, pack=True):
    """Choose the grid and tile height for ``count`` frames within the budget.

    The requested grid defines a maximum page footprint. ``pack`` reflows a short
    page inside that same footprint: every grid that fits is scored by the image
    area it can give each tile, and the winner is the one that shows the most
    pixels. ``aspect`` is width/height of the widest source frame, which is what
    makes the width budget bite -- wide frames run out of page width before they
    run out of height, so for them fewer columns is what buys resolution.

    Returns ``(columns, rows, tile_height)``. Tiles never shrink below the
    requested ``tile_height`` and never exceed ``native_height`` when given, so
    packing trades empty space for real pixels and stops at native resolution.
    """
    fixed = TILE_HEADER_HEIGHT + 2 * TILE_RIM_WIDTH
    max_rows = math.ceil(frames_per_page / columns)
    # The footprint the requested grid would have occupied: the ceiling to stay
    # inside. Width uses the widest frame so no tile is ever cropped.
    budget_image_h = tile_height - fixed
    budget_w = (columns * (round(budget_image_h * aspect) + 2 * TILE_RIM_WIDTH)
                + (columns - 1) * CELL_GAP)
    budget_h = max_rows * tile_height + (max_rows - 1) * CELL_GAP

    if not pack or count >= frames_per_page:
        return columns, max_rows, tile_height

    best = None
    for cand_cols, cand_rows in _grid_candidates(count, columns, max_rows):
        # Largest image height this grid can afford on each axis.
        by_height = (budget_h - (cand_rows - 1) * CELL_GAP) // cand_rows - fixed
        by_width = ((budget_w - (cand_cols - 1) * CELL_GAP) // cand_cols
                    - 2 * TILE_RIM_WIDTH) / aspect
        image_h = int(min(by_height, by_width))
        if native_height:
            image_h = min(image_h, native_height)
        # Never below what was asked for; a grid that cannot even hold the
        # requested size is not an improvement.
        image_h = max(image_h, budget_image_h)
        # Prefer more pixels. When two grids show the same size -- which happens
        # once growth is clamped at native resolution -- prefer the one with
        # fewer empty cells, since blank tiles are the thing being removed. Only
        # then prefer the wider, more timeline-like grid so order reads
        # left-to-right.
        score = (image_h, -(cand_cols * cand_rows - count), cand_cols)
        if best is None or score > best[0]:
            best = (score, cand_cols, cand_rows, image_h + fixed)
    if best is None:
        return columns, max_rows, tile_height
    return best[1], best[2], best[3]


def _fit_text(text, max_width, initial=0.72, minimum=0.38, thickness=2):
    scale = initial
    font = cv2.FONT_HERSHEY_SIMPLEX
    while scale > minimum:
        width = cv2.getTextSize(text, font, scale, thickness)[0][0]
        if width <= max_width:
            break
        scale -= 0.04
    return scale


def _tile(path, frame_id, sequence_index, total, image_height, content_width,
          highlight=False):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read frame: {path}")
    scale = image_height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    resized = cv2.resize(image, (width, image_height), interpolation=cv2.INTER_AREA)
    image_panel = _solid(image_height, content_width, (0, 0, 0))
    x = (content_width - width) // 2
    image_panel[:, x:x + width] = resized

    header = _solid(TILE_HEADER_HEIGHT, content_width, (15, 15, 15))
    ordinal = f"{sequence_index:03d}/{total:03d}"
    right_scale = 0.58
    right_width = cv2.getTextSize(
        ordinal, cv2.FONT_HERSHEY_SIMPLEX, right_scale, 1)[0][0]
    left_scale = _fit_text(
        frame_id, content_width - right_width - 34, initial=0.72)
    cv2.putText(
        header, frame_id, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
        left_scale, (205, 205, 205), 2, cv2.LINE_AA)
    cv2.putText(
        header, ordinal, (content_width - right_width - 8, 27),
        cv2.FONT_HERSHEY_SIMPLEX, right_scale,
        (180, 180, 180), 1, cv2.LINE_AA)
    content = np.vstack([header, image_panel])
    return cv2.copyMakeBorder(
        content,
        TILE_RIM_WIDTH, TILE_RIM_WIDTH, TILE_RIM_WIDTH, TILE_RIM_WIDTH,
        cv2.BORDER_CONSTANT,
        value=HIGHLIGHT_RIM_BGR if highlight else TILE_RIM_BGR)


def _page_header(width, page_number, page_count, first, last, total,
                 title="SOURCE FRAMES", ramp=None):
    stacked_key = ramp is not None and width < KEY_HEADER_STACK_WIDTH
    header_height = (
        STACKED_KEY_HEADER_HEIGHT if stacked_key else PAGE_HEADER_HEIGHT)
    header = _solid(header_height, width, (12, 12, 12))
    if ramp is not None:
        # A keyed sheet's useful header is the key. Tile ordinals already show
        # the covered range, and "PAGE 01/01" only steals room from the scale.
        left = title
        right = f"{page_number:02d}/{page_count:02d}" if page_count > 1 else ""
    else:
        left = f"{title}  {first:03d}-{last:03d} OF {total:03d}"
        right = f"PAGE {page_number:02d}/{page_count:02d}"
    right_width = cv2.getTextSize(
        right, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
    # a long title (the depth sheet's carries its scale) shrinks rather than
    # running under the page counter.
    left_scale = _fit_text(
        left, width - right_width - 36,
        minimum=0.30 if stacked_key else 0.38)
    title_y = 27 if stacked_key else 32
    cv2.putText(
        header, left, (12, title_y), cv2.FONT_HERSHEY_SIMPLEX,
        left_scale, (215, 215, 215), 2, cv2.LINE_AA)
    cv2.putText(
        header, right, (width - right_width - 12, title_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (165, 165, 165), 2, cv2.LINE_AA)
    if ramp:
        _header_ramp(
            header, left, left_scale, right_width, ramp,
            stacked=stacked_key)
    return header


def _header_ramp(
        header, left, left_scale, right_width, ramp, stacked=False):
    """A colour-key strip in the header's free middle, between the title and
    the page counter — the depth sheet's shared NEAR/FAR scale rides the
    header instead of buying its own band.

    ``ramp`` is ``(strip_bgr, labels)`` with ``labels`` a list of
    ``(fraction, text)`` ticks along the strip. The strip stretches into
    whatever the title and counter leave over (capped); when that leaves too
    little to read, interior ticks thin out first, and below 120px the whole
    key is dropped. Keyed sheets use a compact title so this threshold remains
    reachable even on narrow packed pages."""
    strip, labels = ramp
    width = header.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_w = cv2.getTextSize(left, font, left_scale, 2)[0][0]
    if stacked:
        x0, x1 = 12, width - 12
    else:
        x0 = 12 + title_w + 30
        x1 = width - right_width - 40
    bar_w = min(560, x1 - x0)
    if bar_w < 120:
        return
    bar_h = 16
    y1 = header.shape[0] - 6
    y0 = y1 - bar_h
    header[y0:y1, x0:x0 + bar_w] = cv2.resize(
        strip, (bar_w, bar_h), interpolation=cv2.INTER_AREA)
    if bar_w < 340 and len(labels) > 3:
        labels = [labels[0], labels[len(labels) // 2], labels[-1]]
    for frac, text in labels:
        x = int(round(x0 + frac * (bar_w - 1)))
        tw = cv2.getTextSize(text, font, 0.45, 1)[0][0]
        tx = min(max(x - tw // 2, x0), x0 + bar_w - tw)
        cv2.putText(header, text, (tx, y0 - 3), font, 0.45,
                    (235, 235, 235), 1, cv2.LINE_AA)
        if 0.0 < frac < 1.0:
            cv2.line(header, (x, y0), (x, y0 + 3),
                     (235, 235, 235), 1)


def write_sheets(run_dir="", frames_dir="", out_dir="", frame_specs=(),
                 every=1, frames_per_page=12, columns=4, tile_height=300,
                 pack=True, highlight_frames=(), title="SOURCE FRAMES",
                 stem="frame_sheet", ramp=None):
    """Write frame-sheet pages and return ``(page_paths, manifest_path)``.

    ``highlight_frames`` — frame names whose tiles carry the bright rim
    (seam_sheet's "under review" color) instead of the neutral one: a caller's
    way to say which of the sheet's frames are the SUBJECT — gap_sheet's step
    endpoints — without inventing a second sheet geometry. Purely visual; the
    manifest records each tile's ``highlight`` flag so the mark is auditable.

    ``title``/``stem``/``ramp`` let another sheet ride this exact grid under
    its own name — the depth sheet (analysis.viz.depth_units) pages its
    object-units heatmap tiles here with title "OUR DEPTH ...", stem
    "depth_sheet" and its shared NEAR/FAR colour key as ``ramp`` — so a
    second tile geometry never gets written. ``stem`` names the pages
    (``<stem>_NNN.png``) and the manifest (``<stem>_manifest.json``);
    ``ramp`` is ``(strip_bgr, [(fraction, text), ...])``, drawn in every
    page header's free middle (see ``_header_ramp``).
    """
    frames_dir, all_names, _ = resolve_source(
        run_dir=run_dir, frames_dir=frames_dir)
    selected = select_frames(
        all_names, frames_dir, specs=frame_specs, every=every)
    highlight = set(highlight_frames or ())
    missing_highlights = highlight - set(selected)
    if missing_highlights:
        raise ValueError(
            "highlight_frames not on this sheet: "
            + ", ".join(sorted(missing_highlights)))

    frames_per_page = int(frames_per_page)
    columns = int(columns)
    tile_height = int(tile_height)
    if frames_per_page < 1:
        raise ValueError("--frames-per-page must be at least 1")
    if columns < 1:
        raise ValueError("--columns must be at least 1")
    columns = min(columns, frames_per_page)
    fixed_height = TILE_HEADER_HEIGHT + 2 * TILE_RIM_WIDTH
    if tile_height <= fixed_height + 24:
        raise ValueError(
            f"--tile-height must exceed {fixed_height + 24} pixels")

    shapes = []
    for name in selected:
        image = cv2.imread(os.path.join(frames_dir, name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(
                f"could not read frame: {os.path.join(frames_dir, name)}")
        shapes.append(image.shape[:2])
    # The widest frame sets the shared tile width, so no source is ever cropped.
    aspect = max(width / height for height, width in shapes)
    native_height = max(height for height, _ in shapes)

    # Spread frames evenly first: the per-page fill is what the layout is packed
    # around, so balancing and packing compound instead of fighting.
    page_sizes = balance_pages(len(selected), frames_per_page)
    requested = (columns, math.ceil(frames_per_page / columns), tile_height)
    # One layout for every page, sized for the fullest, so pages stay directly
    # comparable and a frame does not change size when it lands on another page.
    columns, rows_per_page, tile_height = plan_layout(
        max(page_sizes), aspect, columns, frames_per_page, tile_height,
        native_height=native_height, pack=pack)
    packed = (columns, rows_per_page, tile_height) != requested
    image_height = tile_height - fixed_height

    content_width = max(1, int(round(image_height * aspect)))
    tile_width = content_width + 2 * TILE_RIM_WIDTH
    page_width = columns * tile_width + (columns - 1) * CELL_GAP
    page_body_height = (
        rows_per_page * tile_height + (rows_per_page - 1) * CELL_GAP)
    if packed:
        print(f"[frame-sheet] packed {len(selected)} frames into "
              f"{columns}x{rows_per_page} tiles at {tile_height}px "
              f"(requested {requested[0]}x{requested[1]} at {requested[2]}px)")

    if not out_dir:
        out_dir = os.path.join(
            os.path.abspath(run_dir) if run_dir else frames_dir,
            "frame_sheets")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    total = len(selected)
    page_count = len(page_sizes)
    pages = []
    manifest_frames = []
    blank = _solid(tile_height, tile_width, (18, 18, 18))
    start = 0
    for page_index, page_size in enumerate(page_sizes):
        page_names = selected[start:start + page_size]
        body = _solid(page_body_height, page_width, (45, 45, 45))
        for local_index, name in enumerate(page_names):
            row, column = divmod(local_index, columns)
            x = column * (tile_width + CELL_GAP)
            y = row * (tile_height + CELL_GAP)
            sequence_index = start + local_index + 1
            highlighted = name in highlight
            tile = _tile(
                os.path.join(frames_dir, name), name,
                sequence_index, total, image_height, content_width,
                highlight=highlighted)
            body[y:y + tile_height, x:x + tile_width] = tile
            manifest_frames.append({
                "frame_id": name,
                "path": os.path.join(frames_dir, name),
                "sequence_index": sequence_index,
                "page": page_index + 1,
                "row": row + 1,
                "column": column + 1,
                "highlight": highlighted,
            })
        # Keep unused cells explicit so the final page retains the same grid.
        for local_index in range(
                len(page_names), rows_per_page * columns):
            row, column = divmod(local_index, columns)
            x = column * (tile_width + CELL_GAP)
            y = row * (tile_height + CELL_GAP)
            body[y:y + tile_height, x:x + tile_width] = blank

        header = _page_header(
            page_width, page_index + 1, page_count,
            start + 1, start + len(page_names), total, title=title,
            ramp=ramp)
        page = np.vstack([header, body])
        page_path = os.path.join(
            out_dir, f"{stem}_{page_index + 1:03d}.png")
        if not cv2.imwrite(page_path, page):
            raise OSError(f"failed to write {page_path}")
        pages.append(page_path)
        print(f"[frame-sheet] wrote {page_path}")
        start += page_size

    manifest = {
        "schema_version": 1,
        "frames_dir": frames_dir,
        "selected_count": total,
        "source_count": len(all_names),
        "frames_per_page": frames_per_page,
        "columns": columns,
        "rows": rows_per_page,
        "tile_height": tile_height,
        "packed": packed,
        # What was asked for, so a reader can tell a packed sheet's geometry from
        # the budget it was packed into.
        "requested": {
            "columns": requested[0],
            "rows": requested[1],
            "tile_height": requested[2],
        },
        "page_sizes": page_sizes,
        "pages": pages,
        "frames": manifest_frames,
    }
    manifest_path = os.path.join(out_dir, f"{stem}_manifest.json")
    write_json(manifest_path, manifest)
    print(f"[frame-sheet] wrote {manifest_path}")
    return pages, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default="",
                        help="run whose layout.json supplies frames and order")
    parser.add_argument("--frames-dir", default="",
                        help="source frames directory; overrides layout path")
    parser.add_argument(
        "--frames", action="append", default=[],
        help="subset: repeatable frame names, comma lists, globs, or an inclusive "
             "first-last range; output remains in temporal order")
    parser.add_argument(
        "--every", type=int, default=1,
        help="keep every Nth frame after subset resolution (default: 1)")
    parser.add_argument("--out-dir", default="")
    parser.add_argument(
        "--frames-per-page", type=int, default=12,
        help="most frames on one page; also the page-footprint budget (default: 12)")
    parser.add_argument(
        "--columns", type=int, default=4,
        help="most columns; a short selection may use fewer, larger tiles "
             "(default: 4)")
    parser.add_argument(
        "--tile-height", type=int, default=300,
        help="minimum complete tile height; packing grows it to fill the "
             "budget (default: 300)")
    parser.add_argument(
        "--no-pack", dest="pack", action="store_false",
        help="keep the literal fixed grid, padding short pages with empty cells")
    args = parser.parse_args()
    if not args.run_dir and not args.frames_dir:
        parser.error("one of --run-dir or --frames-dir is required")
    write_sheets(
        run_dir=args.run_dir, frames_dir=args.frames_dir,
        out_dir=args.out_dir, frame_specs=args.frames, every=args.every,
        frames_per_page=args.frames_per_page, columns=args.columns,
        tile_height=args.tile_height, pack=args.pack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
