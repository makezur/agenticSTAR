# `candidate_sheet.py` - paginated sweep candidate review

Builds readable, labeled pages from the full-resolution candidates already
rendered by a `sweep`, `apply`, `oapply`, `oapply_all`, or `osweep` order. It does not launch
Blender or re-render.

**You rarely run this by hand.** The render pool builds a sheet for every
sweep-family order the moment its renders land (`pool/panels.py` -> this module),
so a finished order already has its pages. Run it yourself to re-panel with
different columns, or to roll several orders onto one set of pages.

Each default row contains:

```
SOURCE | RENDER | SILHOUETTE
```

The compare-and-diff sequence — the photo, the render of it, and where they
disagree. `MASKED SOURCE` (the same photo on the render's backdrop) is opt-in
via `--columns`: a supporting panel that repeats pixels two columns already
show, at a quarter of every row's width.

The row itself is built by `analysis/viz/rows.py`, the **one** row builder shared
with `composite.py` (per-frame judge strip) and `sweep_sides.py` (per-candidate
strips). A column tuned there is tuned in all three, so a sheet row and a
composite strip cannot show the same pixels with different colours or a different
colour key. This module owns only what is specific to a *sheet*: pagination, the
candidate-ID header and rim, and the manifest.

Every row begins with a single restrained header line: a small gray `CANDIDATE`
label followed by the primary ID in desaturated green. The compact ID is
repeated at the far right so identity remains visible when a wide sheet is
cropped around `RENDER` or `OVERLAP`. Column headers contain names only; numeric
IoU/combined scores remain in the JSON manifest rather than competing with the
images. A thin, dark, muted-green rim encloses the entire candidate row so a
visual model can group the header and all four panels without competing with
the overlap colors. The rim is structural only; it does not encode score or
rank.

`SILHOUETTE` is the raw 1:1 silhouette comparison, painted from the same `OVL_*`
constants as `composite.py`'s panel (they live in `analysis/lib/panels.py`):

- green: source and render agree
- yellow: render only, so rendered volume is extra or misplaced
- magenta: source mask only, so rendered volume is missing or misplaced
- blue: our render inside the hand region — present but NOT scored, not an error
- gray: ignored hand-mask region

The colour **key is drawn on the panel**, from those same constants, so it cannot
drift from the pixels. It replaces the panel title — the swatches already say what
the panel is, and the word "SILHOUETTE" was what cost the key its space.

The header bar is **stacked above** each panel, not drawn over it, so no render
pixel is ever hidden and `--height` sizes the *renders* (a row comes out one bar
taller than `height`). Two rules follow:

- **The bar is the same height for every panel in a row.** Its height is a property
  of the ROW, not of the panel: `panels.BAR_HEIGHT` (title/key line only), or
  `panels.CAPTIONED_BAR_HEIGHT` when some column's caption *cannot fit on its title
  line*, in which case every column reserves the caption line.
  `panels.label_row` works that out once and applies it to all of them. Equal
  renders under equal bars is what makes the same *y* the same row of the capture in
  `RENDER` as in `SOURCE` beside it; a bar sized per panel would push one column's
  picture down relative to its neighbour's.
- **A caption rides on the title line when it fits.** The band is charged to *every*
  column in the row, so buying one for the two words naming which candidate a panel
  shows costs 20px across the whole strip to say what fits in the gap already beside
  the title. So it is measured, exactly as the key is (`panels.sub_fits_title_line`):
  a `sweep_sides` strip's `RENDER  rank 00` is one line, and only the depth panels'
  range/MAE/bias lines are long enough to buy the row a second one. Sheets caption
  nothing at all.
- **No panel carries a score.** The caption says *which* candidate, never how good
  it is (`rows.render_caption` is the only place the wording is built, and takes no
  score). Numbers live in the manifest and the report, where they can be sorted,
  diffed and recomputed; burned into a picture they go stale and compete with the
  pixels the panel exists to show.
- **The key is always one row.** It is fitted to the width by shortening the words
  and then the type, never by wrapping: full wording (`MAYBE OCCLUDED`) at the
  largest font that fits, else `HAND` in its place, else no key at all on a panel
  too narrow for even that. Same colours in every case. The chosen entries and font
  are reported on `rows.RowResult.legend` / `.legend_font_scale`.

## Run

```bash
micromamba run -n artscript env PYTHONPATH=harness \
  python -m analysis.viz.candidate_sheet \
  --render-dir RUN_DIR/spool/renders/<order-id> \
  --run-dir RUN_DIR \
  --rows-per-page 3
```

The report in `--render-dir` must list full-resolution candidate images. Every
sweep-family order has at least one (the winner); ask for more with `dump_topk`
or `visuals: "all"`.

An object-centric (`oapply`/`oapply_all`/`osweep`) report scores every frame and lists a per-frame
image dict per candidate, so it yields **one row per (candidate, frame)** and
each row resolves its own frame's source and mask.

Useful options:

- `--rows-per-page N`: number of candidates on each page; default `3`.
- `--height PX`: height of each **render**; default `360`. The row comes out
  taller — the panel header bars, the candidate-ID header and the rim are all
  added above/around the pictures, never over them (see above).
- `--max-dimension PX`: target a derived preview's longest edge after assembly;
  `0` (the default) selects the native page directly. The preview preserves aspect
  ratio, never enlarges a page, and never drops below half of its native linear
  size: a dense page may therefore remain larger than the target. The canonical
  `candidate_sheet_NNN.png` always remains native; a reduced default is written
  as `candidate_sheet_NNN_preview.png`. The pool reads the
  same setting from
  `run_config.json`'s `pool.candidate_sheet_max_dimension`, which makes the
  automatic pages for a run comparable and records the choice with that run.
- `--columns LIST`: exact column order. Valid names are `source`, `render`,
  `overlap`, and `masked_source` (`silhouette` is accepted as an alias
  for `overlap`, `match` for `render`); `render` is required.
- `--no-overlap`: remove `overlap` from the default or explicit column set.
- `--out-dir DIR`: output directory; defaults to the first `--render-dir`.
- `--source PATH`: source image override when `--run-dir` is unavailable.
- `--render-dir` repeats: roll several orders onto one set of pages. Row IDs are
  then prefixed with the order id so a row still says where it came from.
- `--budget N` / `--waive-visual-budget`: the sheet refuses more than 100 rows
  (see `core/visual_budget.py`) unless waived; `--budget` tightens or loosens it.

Examples without the overlap:

```bash
# Simple opt-out while retaining the other defaults
... python -m analysis.viz.candidate_sheet ... --no-overlap

# Exact narrow layout
... python -m analysis.viz.candidate_sheet ... --columns source,render

# Two orders compared side by side on one set of pages
... python -m analysis.viz.candidate_sheet \
      --render-dir RUN_DIR/spool/renders/flip-000010 \
      --render-dir RUN_DIR/spool/renders/noflip-000010 \
      --run-dir RUN_DIR
```

## Outputs

- `candidate_sheet_001.png`, `candidate_sheet_002.png`, and so on: native pages.
- Optional `candidate_sheet_001_preview.png` derivatives when `--max-dimension`
  reduces a page.
- `candidate_sheet_manifest.json` (`schema_version: 4`), mapping every displayed
page and row to its report candidate ID, image, rank, scores, and the `frame`,
`order`, `source`, and `mask` that row was built from. `page_raster` records the
requested `max_dimension`, `min_scale`, and every page's preview/original paths,
sizes, and applied scale. `pages` is the default inspection list;
`original_pages` is the exact-raster escalation list. `reports` lists every report
that fed the sheet.

Select candidates by the printed ID or manifest, never by remembered page/row
position.
