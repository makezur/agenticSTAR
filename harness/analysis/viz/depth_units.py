"""depth_units.py — OUR rendered depth in OBJECT UNITS: the panel column and
the sheet (GT-free). THE one entry point for monocular depth visuals.

The monocular gap. `analysis.scorers.depth` needs an observed Pi3X pointmap to
compare against, so a run whose depth_config.json says `backend: "none"` (the
monocular case) renders NO depth visual at all — depth simply goes dark. But the
render itself still knows exactly how far the object sits: the `depth` view's
`depth_<stem>.npy` is planar camera-space Z in SCENE units, and the run's
predicted shared SCALE `s` (pose.json `scale`, the object's longest dimension in
those same units) is the natural yardstick. depth / s is depth in OBJECT UNITS —
"how many object-lengths away" — meaningful with no GT at all, and comparable
across frames because the unit is the object itself, not an arbitrary scene
scale.

TWO MEDIA, one owner (this module):

  * the DEPTH COLUMN — `object_units_panel` returns a `panels.Panel` titled
    "DEPTH" whose header key says NEAR/FAR with the endpoint VALUES, from the
    same turbo LUT the pixels were painted with (the overlap column's rule:
    the key cannot drift from the panel). `composite.py` puts it in the
    standard row when it has a rendered depth but no Pi3X to score it against,
    so the monocular judge strip carries a depth read like the GT one does.
  * the DEPTH SHEET — the CLI. Collects a pass's (or spool's) depth renders,
    computes ONE shared [near, far] range over every frame's finite pixels
    (2–98 percentiles), and pages them through `frame_sheet`'s grid — the same
    packing, pagination, frame-ID tiles and manifest the source sheets use.
    The SHARED scale is the point: one colour = one distance on every page, so
    poses breathing in depth (the `radial` oscillation the temporal table
    warns about) read as tiles pulsing blue↔red across time.

Runs in the 'artscript' micromamba env (NOT Blender's python), from harness/:
  micromamba run -n artscript python -m analysis.viz.depth_units \
      --pass-dir VIEWS/NNNN            # a shape pass: its depth_*.npy
      [--scan DIR]                     # a spool renders root: */depth_*.npy
      [--depth D.npy ...]              # or explicit file(s)
      [--frames 000040-000100]         # subwindow (names, comma lists, ranges)
      [--scale S]     # default: pose.json walk-up (scorers.depth.resolve_scale)
      [--out-dir DIR] # default: <run or pass dir>/depth_sheets
      [--range MIN,MAX] [--tile-height H] [--columns N] [--frames-per-page N]

Writes depth_sheet_NNN.png pages (+ manifest) and depth_units.json (s, the
shared range, per-frame stats in object units) into --out-dir; the per-frame
heatmap tiles land in <out-dir>/tiles/<stem>.png.
"""

import argparse
import glob
import os
import re

import cv2
import numpy as np

from analysis.lib import panels
from analysis.lib.io import write_json
from analysis.scorers.depth import resolve_scale
from analysis.viz import frame_sheet


# --------------------------------------------------------------------------- #
# stats + range
# --------------------------------------------------------------------------- #
def units_stats(render_depth, scale):
    """Distance stats of the finite (object) pixels, in OBJECT UNITS.

    Returns {n_px, min, p02, median, p98, max} (values None when the render has
    no finite pixel — an empty frame stays reportable, not fatal)."""
    d = render_depth[np.isfinite(render_depth)] / scale
    if d.size == 0:
        return {"n_px": 0, "min": None, "p02": None, "median": None,
                "p98": None, "max": None}
    return {
        "n_px": int(d.size),
        "min": round(float(d.min()), 4),
        "p02": round(float(np.percentile(d, 2)), 4),
        "median": round(float(np.median(d)), 4),
        "p98": round(float(np.percentile(d, 98)), 4),
        "max": round(float(d.max()), 4),
    }


def shared_range(depths, scale):
    """One [dmin, dmax] (OBJECT units) for a whole batch: 2–98 percentiles over
    every frame's finite pixels pooled, so one colour = one distance across the
    clip. Falls back to [0, 1] when nothing is finite anywhere.

    The pooled-percentile kernel is `panels.pct_range` (the depth scorer's
    two-map range is the same call); the object-units part — dividing by the
    predicted scale `s` — is what belongs to this module."""
    return panels.pct_range([d[np.isfinite(d)] / scale for d in depths])


# --------------------------------------------------------------------------- #
# the picture
# --------------------------------------------------------------------------- #
def object_units_heat(render_depth, scale, dmin, dmax):
    """TURBO heatmap of depth/scale over [dmin, dmax] object units.

    Background (non-finite) pixels are black. No key is burned into the
    picture: the panel's rides its title line and the sheet page's rides the
    header, both from `panels.turbo_ramp_strip` — one ramp per medium, read
    once and large, instead of a mini-legend per tile."""
    return panels.depth_heat(render_depth, dmin, dmax, divisor=scale)


def object_units_panel(render_depth, scale, dmin=None, dmax=None, height=0):
    """The DEPTH column for the standard row: a clean heatmap under a "DEPTH"
    title whose header line carries the NEAR→FAR ramp with the endpoint
    VALUES (`panels.label`'s ramp_legend — in line with the title, like the
    overlap column's swatch key).

    No caption — the median/stats live in depth_units.json and the sheet, not
    burned into the pixels. Un-labeled, like scorers.depth's panels: only the
    row assembler knows the bar height. Range defaults to this frame's own
    2–98 pct (a composite is a single frame; the SHEET is the shared-scale
    medium). `height` > 0 pre-scales the picture so the row's own rescale is
    a no-op."""
    if dmin is None or dmax is None:
        dmin, dmax = shared_range([render_depth], scale)
    img = object_units_heat(render_depth, scale, dmin, dmax)
    if height and height != img.shape[0]:
        img = panels.scale_to_height(img, height)
    return panels.Panel(img, "DEPTH", ramp_legend=(
        panels.turbo_ramp_strip(), f"NEAR {dmin:.2f}", f"{dmax:.2f} FAR"))


def object_units_ramp(dmin, dmax):
    """The shared frame-sheet ramp used by every object-unit depth sheet."""
    return (
        panels.turbo_ramp_strip(width=560, height=16),
        [(0.0, f"NEAR {dmin:.2f}"),
         (0.5, f"{(dmin + dmax) / 2:.2f}"),
         (1.0, f"{dmax:.2f} FAR")],
    )


# --------------------------------------------------------------------------- #
# collection: which depth renders go on the sheet
# --------------------------------------------------------------------------- #
def _stem_of(path):
    """'depth_000040.npy' -> '000040'."""
    name = os.path.splitext(os.path.basename(path))[0]
    return name[len("depth_"):] if name.startswith("depth_") else name


def collect(pass_dir="", scan_dir="", depth_files=()):
    """{stem: path} of every depth render the flags name.

    `pass_dir` takes a shape pass's own depth_*.npy; `scan_dir` sweeps ONE
    level of subdirs too (a spool's renders/<order-id>/ layout); `depth_files`
    are explicit. A stem found twice keeps the NEWEST file — re-rendering a
    frame must win over its stale predecessor — and says so."""
    paths = []
    if pass_dir:
        paths += glob.glob(os.path.join(pass_dir, "depth_*.npy"))
    if scan_dir:
        paths += glob.glob(os.path.join(scan_dir, "depth_*.npy"))
        paths += glob.glob(os.path.join(scan_dir, "*", "depth_*.npy"))
    paths += [os.path.abspath(f) for f in depth_files]
    out = {}
    for path in sorted(paths):
        stem = _stem_of(path)
        if stem in out and out[stem] != path:
            newest = max(out[stem], path, key=os.path.getmtime)
            print(f"[depth_units] {stem}: two renders, keeping newest "
                  f"({os.path.relpath(newest, os.path.dirname(newest) or '.')})")
            out[stem] = newest
        else:
            out[stem] = path
    return out


def select_stems(stems, specs):
    """`--frames` selectors (names, comma lists, inclusive first-last ranges)
    resolved against the collected stems, via the frame sheet's own selector —
    the sheet subset grammar is ONE grammar, not two. Source-image extensions
    are shed from the tokens first, so the "000040.jpg-000100.jpg" a window
    brief pastes selects the stems the depth renders are named by."""
    if not specs:
        return list(stems)
    # deliberately UNANCHORED: one spec token can be a comma list or a range
    # ("000040.jpg-000100.jpg"), so the extension to shed is not only a trailing
    # one. Frame stems are digits, so there is nothing else here to eat.
    specs = [re.sub(r"\.(jpe?g|png|webp)", "", str(t), flags=re.IGNORECASE)
             for t in specs]
    names = [f"{s}.png" for s in stems]      # select_frames keys on filenames
    picked = frame_sheet.select_frames(names, frames_dir="", specs=specs)
    return [os.path.splitext(n)[0] for n in picked]


# --------------------------------------------------------------------------- #
def write_sheet(found, scale, out_dir, fixed_range=None, frames_per_page=12,
                columns=4, tile_height=300):
    """Tiles + pages + stats for the collected {stem: depth path}.

    Renders each frame's object-units heatmap (SHARED range, no per-tile key)
    into <out_dir>/tiles/<stem>.png and hands the tiles to
    `frame_sheet.write_sheets` — the packing, pagination, frame-ID headers
    and manifest are ITS, not reimplemented here — along with the shared
    NEAR/FAR ramp, which rides each page header's free middle (between the
    title and the page counter) rather than buying its own band: the same
    colour is the same distance on every page, and the key is read once,
    large, not squinted at per tile.

    Returns (pages, manifest_path, report_path)."""
    depths = {stem: np.load(path) for stem, path in found.items()}
    dmin, dmax = fixed_range or shared_range(list(depths.values()), scale)

    tiles_dir = os.path.join(out_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    report = {"scale": scale,
              "range": [round(dmin, 4), round(dmax, 4)],
              "range_mode": "fixed" if fixed_range else "shared-2-98pct",
              "frames": {}}
    for stem, depth in depths.items():
        img = object_units_heat(depth, scale, dmin, dmax)
        cv2.imwrite(os.path.join(tiles_dir, f"{stem}.png"), img)
        stats = units_stats(depth, scale)
        report["frames"][stem] = stats
        med = stats["median"]
        dist = (f"median {med:g} obj-units away" if med is not None
                else "NO rendered pixels")
        print(f"[depth_units] {stem}: {dist}  ({stats['n_px']} px)")

    pages, manifest = frame_sheet.write_sheets(
        frames_dir=tiles_dir, out_dir=out_dir,
        frames_per_page=frames_per_page, columns=columns,
        tile_height=tile_height, stem="depth_sheet",
        title="DEPTH (OBJECT UNITS)",
        ramp=object_units_ramp(dmin, dmax))
    report_path = os.path.join(out_dir, "depth_units.json")
    write_json(report_path, report)
    print(f"[depth_units] wrote {report_path}  (s={scale:g}, "
          f"{len(depths)} frame(s), range [{dmin:.2f},{dmax:.2f}])")
    return pages, manifest, report_path


def main():
    p = argparse.ArgumentParser(
        description="the depth sheet: OUR rendered depth in object units, one "
                    "shared NEAR/FAR scale across every frame (no GT needed)")
    p.add_argument("--pass-dir", default="",
                   help="render pass dir; its depth_*.npy go on the sheet")
    p.add_argument("--scan", default="",
                   help="also sweep DIR and DIR/*/ for depth_*.npy (a spool's "
                        "renders/<order-id>/ layout)")
    p.add_argument("--depth", nargs="*", default=[],
                   help="explicit depth .npy file(s)")
    p.add_argument("--frames", action="append", default=[],
                   help="subwindow: frame names/stems, comma lists, or an "
                        "inclusive first-last range (e.g. 000040-000100); the "
                        "SHARED scale is computed over the selection")
    p.add_argument("--scale", type=float, default=None,
                   help="predicted object scale s (scene units); default: read "
                        "`scale` from the run's pose.json next to the depth "
                        "files (same walk-up as scorers.depth)")
    p.add_argument("--out-dir", default="",
                   help="sheet pages + depth_units.json land here (default: "
                        "depth_sheets/ beside the depth files)")
    p.add_argument("--range", default="",
                   help="fixed MIN,MAX in object units (e.g. to compare runs); "
                        "default: shared 2-98 pct across the selection")
    p.add_argument("--frames-per-page", type=int, default=12)
    p.add_argument("--columns", type=int, default=4)
    p.add_argument("--tile-height", type=int, default=300,
                   help="minimum complete tile height; the sheet packs short "
                        "selections into bigger tiles (default: 300)")
    args = p.parse_args()

    found = collect(args.pass_dir, args.scan, args.depth)
    if not found:
        raise SystemExit(
            "no depth renders found (--pass-dir/--scan/--depth); the 'depth' "
            "view writes depth_<stem>.npy — render it first")
    stems = select_stems(sorted(found), args.frames)
    found = {s: found[s] for s in stems}

    first = next(iter(found.values()))
    scale = resolve_scale(args.scale, first)
    if scale is None:
        raise SystemExit(
            "no scale: pass --scale, or point at a pass dir whose pose.json "
            "declares one (object units are depth / s; without s there is no "
            "unit to draw in)")

    fixed = None
    if args.range:
        lo, hi = (float(v) for v in args.range.split(","))
        fixed = (lo, max(hi, lo + 1e-6))

    out_dir = os.path.abspath(args.out_dir or os.path.join(
        os.path.dirname(first), "depth_sheets"))
    write_sheet(found, scale, out_dir, fixed_range=fixed,
                frames_per_page=args.frames_per_page, columns=args.columns,
                tile_height=args.tile_height)


if __name__ == "__main__":
    main()
