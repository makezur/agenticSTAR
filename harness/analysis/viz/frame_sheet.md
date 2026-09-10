# `frame_sheet.py` - paginated initial source-frame review

Builds readable, temporally ordered source-frame sheets before reconstruction
starts. Every tile has a large frame ID, its sequence position, a neutral-gray
grouping rim, and clear spacing. The sheet uses grayscale UI so color attention
stays on the source frames. Long sequences are split into numbered pages
automatically.

## Run

```bash
micromamba run -n artscript env PYTHONPATH=harness \
  python -m analysis.viz.frame_sheet \
  --run-dir RUN_DIR
```

With defaults, the tool writes 12 frames per page in a four-column grid:

```
RUN_DIR/frame_sheets/frame_sheet_001.png
RUN_DIR/frame_sheets/frame_sheet_002.png
...
RUN_DIR/frame_sheets/frame_sheet_manifest.json
```

The manifest records every frame's page, row, column, source path, and sequence
index. View every generated page during initial decomposition. Open individual
source frames afterward when small geometry, text, articulation, or occlusion
details need full resolution.

## Subsets

`--frames` is repeatable and accepts exact names, comma lists, globs, and
inclusive ranges. Output always returns to temporal order:

```bash
# Explicit frames
... python -m analysis.viz.frame_sheet --run-dir RUN_DIR \
  --frames 000040.jpg,000100.jpg,000180.jpg

# Inclusive temporal segment
... python -m analysis.viz.frame_sheet --run-dir RUN_DIR \
  --frames 000080.jpg-000180.jpg

# Repeatable selectors and globs
... python -m analysis.viz.frame_sheet --run-dir RUN_DIR \
  --frames '0000[4-8]0.jpg' --frames 000200.jpg

# Optional overview sampling after subset resolution
... python -m analysis.viz.frame_sheet --run-dir RUN_DIR --every 3
```

Use `--frames-dir CAPTURE/frames` instead of `--run-dir` when no run exists.

## Layout options

The grid flags describe the LARGEST page a sheet may occupy, not a grid every
sheet has to fill.

- `--frames-per-page N`: most frames on one page; default `12`. Extra pages are
  automatic.
- `--columns N`: most columns; default `4`. A short sheet may use fewer.
- `--tile-height PX`: minimum complete tile height; default `300`. Packing grows
  it to use the space the empty cells were wasting.
- `--no-pack`: keep the literal fixed grid and pad short pages with empty cells.
- `--out-dir DIR`: output location; defaults to `RUN_DIR/frame_sheets`.
- `--every N`: retain every Nth selected frame; default `1` (all).

## Packing: short sheets get bigger frames, not black squares

Two things keep a short sheet from padding with black tiles, both inside the
footprint the flags describe:

- **The grid is rechosen.** Every grid that still fits the budget is scored by
  the image area it can give each tile, and the biggest wins. Six frames become
  a 3x2 of 387px tiles rather than a 4x3 of 300px tiles: no empty cells, ~35%
  taller frames, and a *shorter* page. The widest source frame sets the shared
  tile width, so the width budget genuinely binds — for wide frames, fewer
  columns is what buys resolution.
- **Pages are balanced.** The page count is what the budget fixes; once it is
  known the frames spread evenly over it, so the last sheet is not mostly empty.
  Every page is equally full to within one frame, and all pages keep one geometry so they stay directly
  comparable and a frame does not resize when it lands on another page.

Two guarantees keep this honest: a tile is never smaller than the
`--tile-height` you asked for, and never grows past the source frame's own
resolution, so packing trades empty space for real pixels and never upscales.
A full page is untouched. `--no-pack` reproduces the fixed-grid sheet
exactly.

The manifest records `columns`, `rows`, `tile_height`, `page_sizes`, a `packed`
flag, and the `requested` budget, so a sheet's geometry is always attributable.

Sampling is for an overview sheet, not permission to skip the omitted source
frames when the reconstruction task requires reviewing every provided frame.

## Not `seam_sheet`

Both paginate frames, and they answer different questions. This one shows the
TIMELINE — many frames, a uniform grid, temporal order — which is what you want
when reading the sequence for the first time, or writing a window's
`expected_motion` before any sweep exists.
[seam_sheet.md](seam_sheet.md) shows a PAIR: two frames compared against each
other, sources above their committed renders, because the comparison is the whole
content. Reach for it when the question is "did these two windows agree at their
boundary", where the pair needs every pixel it can get and a 12-up grid would
throw away exactly the resolution the basin call depends on.
