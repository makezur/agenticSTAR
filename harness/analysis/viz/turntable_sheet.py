"""Build one turntable sheet per ARTICULATION STATE from a render pass.

The turntable is the 3D shape gate: it renders a whole set of orbit views for
EVERY unique joint configuration, so a run with 8 observed states and the default
8-view set puts 64 loose PNGs in one pass dir. The gate is only useful if it is
actually read, and "open 64 files" is the instruction that turns a required check
into a skipped one. So the pass is laid out as sheets: ONE page per state, tiles
labelled with the direction they were shot from, the state and its source frame in
the page header, plus a manifest indexing every tile.

    RUN_DIR/views/<NNNN>/turntable_sheets/turntable_sheet_<state>.png
                                          turntable_sheet_manifest.json

THE TILES STAY ON DISK. A sheet answers "is this object coherent from every angle,
in this configuration" at a glance; it is not a replacement for the individual
renders, which is what `critic.py` feeds a VLM at full resolution and what you open
when a suspect region needs pixels. Both media, same renders — the sheet is the
index, not the archive.

WHAT IS SHARED, AND WITH WHOM. The (azimuth, elevation) grammar in the filenames is
`core.turntable_views`, the same module `views/turntable.py` writes them with — a
sheet that parsed a format the renderer merely happened to emit would be one edit
from silently finding no tiles. The chrome (header bars, solid fills, stacking with
separators) is `analysis.lib.panels`, so these pages look like the candidate sheets
and the composite strips rather than like a third convention. Renders are read
through `rasters.load_render`, which presents the transparent film on a chosen
backdrop — reading them as plain images would keep the ~1px colour fringe Blender
leaves under alpha=0.

Deliberately NOT `viz.rows`: that builder assembles a SOURCE-vs-RENDER comparison
row, and a turntable view has no source to compare against (it is a novel view, not
aligned to any photo). Forcing it through would mean a row of one column.

THE REFERENCE TILE. Each page opens with the SOURCE photo of the frame that
labelled the state (resolved through the run's layout.json via `--run-dir`, or an
explicit `--frames-dir`), at the same tile height as the renders. No render is
pixel-aligned to it — every other tile is a novel view — but it anchors the page:
what the object actually looks like in this configuration, next to what was built.
Absent (no run dir given, a 'rest' state with no frame, a missing file) the page is
simply all renders, as before.
"""

import argparse
import glob
import os

import cv2

from analysis.lib import panels, rasters
from analysis.lib.io import write_json
from analysis.viz import frame_sheet
from core import renderer_settings, run_layout, turntable_views


# Tile geometry. The tile is sized by its RENDER height, like every other panel in
# this package (`panels.label` stacks the bar ABOVE the image), so a tile comes out
# one bar taller than this number.
DEFAULT_TILE_HEIGHT = 300
DEFAULT_COLUMNS = 4

TILE_GAP = 10
TILE_SEPARATOR_BGR = (70, 70, 70)
PAGE_HEADER_HEIGHT = 52
PAGE_BG = (30, 30, 30)

# Bands, loosest-first, for grouping tiles into rows. A page reads top-to-bottom as
# "from above / level / from below", which is the order a person inspects a shape
# in — not the order the renderer happened to emit them in.
_BANDS = (
    ("FROM ABOVE", lambda el: el >= 45.0),
    ("LEVEL", lambda el: -15.0 < el < 45.0),
    ("FROM BELOW", lambda el: el <= -15.0),
)


def _band(elevation):
    for name, test in _BANDS:
        if test(elevation):
            return name
    return "LEVEL"


def find_tiles(pass_dir):
    """Every turntable render in `pass_dir`, grouped by state.

    Returns an ordered dict-like list of `(state, [(azimuth, elevation, path)])`.
    States come out in the order their labels sort, which for the usual frame-stem
    labels ('000100', '000140') is temporal order; a rigid object's single 'rest'
    sorts alone. Within a state, tiles are ordered by BAND then azimuth — the page's
    reading order, decided here so the manifest and the pixels agree.
    """
    pass_dir = os.path.abspath(pass_dir)
    found = {}
    for path in sorted(glob.glob(os.path.join(pass_dir,
                                              turntable_views.IMAGE_GLOB))):
        parsed = turntable_views.parse_image_name(os.path.basename(path))
        if parsed is None:                    # not one of ours; leave it alone
            continue
        state, azimuth, elevation = parsed
        found.setdefault(state, []).append((azimuth, elevation, path))
    band_order = {name: i for i, (name, _test) in enumerate(_BANDS)}
    return [
        (state, sorted(tiles, key=lambda t: (band_order[_band(t[1])], t[0])))
        for state, tiles in sorted(found.items())
    ]


def _tile(path, azimuth, elevation, tile_height, bg_mode):
    """One tile as an UNLABELLED `panels.Panel`: the render, the direction that
    titles it, and the band as its caption.

    Returned unlabelled because bar height is a property of the ROW, not the tile —
    `panels.label_row` measures whether the band caption can ride beside the title
    and, when it cannot, gives every tile in the row its own caption line. Labelling
    tiles one at a time (with `panels.label`) clipped the wider bands to
    "FROM ABOV" / "FROM BELO" at the tile edge, which is the exact failure the row
    builder exists to prevent."""
    render = rasters.load_render(path, bg_mode)
    image = panels.scale_to_height(render, int(tile_height))
    return panels.Panel(image, f"az{azimuth:03.0f} el{elevation:+03.0f}",
                        sub=_band(elevation))


def _resolve_frames_dir(run_dir, frames_dir):
    """The directory holding the SOURCE photos, or '' when unresolvable.

    Soft where `frame_sheet.resolve_source` raises: a reference tile is an anchor
    on a page whose real content is the renders, so "no photos found" degrades to
    the all-renders page rather than failing the sheet build."""
    if frames_dir:
        frames_dir = os.path.abspath(frames_dir)
        return frames_dir if os.path.isdir(frames_dir) else ""
    if not run_dir:
        return ""
    layout = run_layout.load_run_layout(run_dir) or {}
    candidate = layout.get("frames_dir", "")
    return candidate if candidate and os.path.isdir(candidate) else ""


# Source-photo extensions tried when the pass MANIFEST does not name the frame
# (the state label is the frame STEM; the photo on disk carries an extension).
_SOURCE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _source_path(frames_dir, state, frame):
    """The state's source photo on disk, or ''. `frame` (from the pass MANIFEST)
    wins when present; otherwise the stem is tried against the usual extensions —
    a 'rest' state has no photo and correctly resolves to nothing."""
    if not frames_dir:
        return ""
    names = [frame] if frame else [state + ext for ext in _SOURCE_EXTENSIONS]
    for name in names:
        path = os.path.join(frames_dir, name)
        if os.path.isfile(path):
            return path
    return ""


def _reference_tile(path, tile_height):
    """The source photo as the page's FIRST tile: what the object actually looks
    like in this configuration, beside what was built.

    A photo, so it is read with `load_bgr` (no alpha film to present) — and it is
    NOT pixel-aligned to any render on the page: every turntable view is a novel
    view. It anchors identity, not pose. Captioned by the file it shows (however
    it was resolved), so the anchor names its own provenance."""
    image = panels.scale_to_height(rasters.load_bgr(path), int(tile_height))
    return panels.Panel(image, "SOURCE", sub=os.path.basename(path))


def _page_header(width, channels, state, frame, index, total, n_views):
    """The page's identity: WHICH configuration this is, and where it came from.

    The state label IS a frame stem (the first frame exhibiting the configuration),
    so it is spelled out as such — a reader who sees a coherence failure here needs
    to know which photo to go back to.
    """
    header = panels.solid(PAGE_HEADER_HEIGHT, width, (15, 15, 15), channels)
    white = (225, 225, 225, 255) if channels == 4 else (225, 225, 225)
    dim = (165, 165, 165, 255) if channels == 4 else (165, 165, 165)
    left = f"TURNTABLE  STATE {state}"
    if frame:
        left += f"  (frame {frame})"
    right = f"STATE {index:02d}/{total:02d}   {n_views} VIEWS"
    cv2.putText(header, left, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                white, 2, cv2.LINE_AA)
    right_width = cv2.getTextSize(
        right, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0][0]
    cv2.putText(header, right, (max(12, width - right_width - 12), 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, dim, 2, cv2.LINE_AA)
    line = (150, 150, 150, 255) if channels == 4 else (150, 150, 150)
    cv2.line(header, (0, PAGE_HEADER_HEIGHT - 2),
             (width, PAGE_HEADER_HEIGHT - 2), line, 2)
    return header


def _grid(tiles, row_counts):
    """`panels.Panel` tiles labelled and stacked into rows of `row_counts` tiles.

    Labelled in ONE `label_row` call for the whole page rather than per row: the bar
    height it picks depends on whether any caption needs its own line, so labelling
    row by row would give a page whose first row (carrying "FROM ABOVE") has a taller
    bar than its second, and the tiles would stop lining up across rows. A page is
    one comparison, so it gets one bar height.

    Rows are then padded to a uniform width by `panels.vstack_rows`, so a shorter
    row left-aligns under the others rather than stretching to fill — a stretched
    tile would be a different scale from its neighbours, and comparing scales is
    the one thing this page is for.
    """
    labelled = panels.label_row(tiles)
    rows, start = [], 0
    for count in row_counts:
        rows.append(panels.hstack_panels(labelled[start:start + count],
                                         sep_width=TILE_GAP,
                                         sep_bgr=TILE_SEPARATOR_BGR))
        start += count
    return panels.vstack_rows(rows, sep_height=TILE_GAP,
                              sep_bgr=TILE_SEPARATOR_BGR)


def _state_frames(pass_dir):
    """state label -> the frame name it was labelled by, from the pass MANIFEST.

    Cosmetic: the label already IS the frame stem, so this only recovers the
    extension for the header ('000140' -> '000140.jpg'). Absent or unreadable
    manifest -> empty, and the header simply omits the frame.
    """
    from analysis.lib.io import read_json
    manifest = read_json(os.path.join(pass_dir, "MANIFEST.json")) or {}
    return {os.path.splitext(str(f))[0]: str(f)
            for f in (manifest.get("frames") or [])}


def write_sheets(pass_dir, out_dir="", columns=DEFAULT_COLUMNS,
                 tile_height=DEFAULT_TILE_HEIGHT, bg_mode="black",
                 run_dir="", frames_dir=""):
    """Write one sheet per articulation state. Returns `(page_paths, manifest_path)`.

    `run_dir`/`frames_dir` locate the SOURCE photos for the per-page reference
    tile; both optional, and a state whose photo cannot be found simply gets the
    all-renders page.

    Raises `ValueError` when the pass holds no turntable renders — the caller asked
    for sheets of something that was not rendered, and silently writing an empty
    manifest would read as "the gate is clean".
    """
    columns, tile_height = int(columns), int(tile_height)
    if columns < 1:
        raise ValueError("--columns must be at least 1")
    if tile_height < 1:
        raise ValueError("--tile-height must be positive")

    pass_dir = os.path.abspath(pass_dir)
    states = find_tiles(pass_dir)
    if not states:
        raise ValueError(
            f"no turntable renders in {pass_dir} (looked for "
            f"{turntable_views.IMAGE_GLOB}) — render the view first: "
            "harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views turntable")

    out_dir = os.path.abspath(out_dir or os.path.join(pass_dir, "turntable_sheets"))
    os.makedirs(out_dir, exist_ok=True)
    frames = _state_frames(pass_dir)
    resolved_frames_dir = _resolve_frames_dir(run_dir, frames_dir)

    page_paths, manifest_states = [], []
    for index, (state, tiles) in enumerate(states, start=1):
        built = [_tile(path, az, el, tile_height, bg_mode)
                 for az, el, path in tiles]
        source = _source_path(resolved_frames_dir, state, frames.get(state, ""))
        if source:
            built.insert(0, _reference_tile(source, tile_height))
        # `columns` is a BUDGET, not a grid to fill (the frame-sheet rule): rows
        # are balanced, so 9 tiles at 4 columns is 3/3/3, not 4/4/1 with a lone
        # tile in a row of black.
        row_counts = frame_sheet.balance_pages(len(built),
                                               min(columns, len(built)))
        body = _grid(built, row_counts)
        header = _page_header(body.shape[1], body.shape[2], state,
                              frames.get(state, ""), index, len(states),
                              len(tiles))
        page = panels.vstack_rows([header, body], sep_height=0)
        page_path = os.path.join(out_dir, f"turntable_sheet_{state}.png")
        if not cv2.imwrite(page_path, page):
            raise OSError(f"failed to write {page_path}")
        page_paths.append(page_path)
        print(f"[turntable-sheet] wrote {page_path}  "
              f"({len(tiles)} views of state {state})")
        # the reference tile, when present, is tile 1; renders follow it. The
        # manifest numbers what is ON the page, so a reader counting tiles lands
        # on the file the number names.
        offset = 2 if source else 1
        manifest_states.append({
            "state": state,
            "frame": frames.get(state, ""),
            "page": page_path,
            "source": source,
            "columns": max(row_counts),
            "rows": len(row_counts),
            "views": [
                {"azimuth": round(az, 2), "elevation": round(el, 2),
                 "band": _band(el), "image": os.path.basename(path),
                 "tile": i + offset}
                for i, (az, el, path) in enumerate(tiles)
            ],
        })

    manifest = {
        "schema_version": 1,
        "pass_dir": pass_dir,
        "bg_mode": bg_mode,
        "tile_height": tile_height,
        "state_count": len(manifest_states),
        # what a reader should open when a tile needs full resolution
        "image_glob": turntable_views.IMAGE_GLOB,
        "pages": page_paths,
        "states": manifest_states,
    }
    manifest_path = os.path.join(out_dir, "turntable_sheet_manifest.json")
    write_json(manifest_path, manifest)
    print(f"[turntable-sheet] wrote {manifest_path}")
    return page_paths, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pass-dir", required=True,
                        help="the views/<NNNN>/ pass dir render.sh printed")
    parser.add_argument("--out-dir", default="",
                        help="output dir (default: PASS_DIR/turntable_sheets)")
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument("--tile-height", type=int, default=DEFAULT_TILE_HEIGHT,
                        help="height of each RENDER in px; a tile comes out one "
                             "header bar taller (the bar stacks above, never over)")
    parser.add_argument("--bg-mode", default="black",
                        choices=list(renderer_settings.BG_MODES))
    parser.add_argument("--run-dir", default="",
                        help="run whose layout.json locates the source photos "
                             "for the per-page reference tile")
    parser.add_argument("--frames-dir", default="",
                        help="source photos directory; overrides the layout path")
    args = parser.parse_args()
    write_sheets(args.pass_dir, out_dir=args.out_dir, columns=args.columns,
                 tile_height=args.tile_height, bg_mode=args.bg_mode,
                 run_dir=args.run_dir, frames_dir=args.frames_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
