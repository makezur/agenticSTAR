# `turntable_sheet.py` — one page per articulation state

Lays a render pass's turntable orbit views out as **one sheet per articulation
state**, so the 3D shape gate is something you read rather than a directory you
open file by file. A run with 8 observed states and the default 8-view set puts 64
loose PNGs in a pass dir; "open 64 files" is the instruction that turns a required
check into a skipped one.

Every tile is labelled with the **direction it was shot from** and grouped
**FROM ABOVE / LEVEL / FROM BELOW** (the order you inspect a shape in, not the
order the renderer emitted them). The page header names the state and the source
frame it was labelled by, so a coherence failure leads back to the photo.

When the source photos are reachable (`--run-dir` whose layout.json locates them,
or an explicit `--frames-dir`), each page **opens with the state's SOURCE photo**
as a reference tile — what the object actually looks like in this configuration,
beside what was built. It is an identity anchor, **not** a comparison column: no
render on the page is pixel-aligned to it (every turntable view is a novel view).
A state with no photo (a rigid `rest`, a missing file) simply gets the all-renders
page.

The shape pass builds these automatically after every render pass (passing the
run dir, so the reference tile is there) — normally you just
read `PASS_DIR/turntable_sheets/`. Run it by hand for an older pass, a different
tile size, or a pass rendered straight from `render.sh`.

## Run

```bash
micromamba run -n artscript env PYTHONPATH=harness \
  python -m analysis.viz.turntable_sheet \
  --pass-dir RUN_DIR/views/0007
```

Writes into the pass dir:

```
RUN_DIR/views/0007/turntable_sheets/turntable_sheet_000100.png
RUN_DIR/views/0007/turntable_sheets/turntable_sheet_000140.png
...
RUN_DIR/views/0007/turntable_sheets/turntable_sheet_manifest.json
```

One page per state, named by that state — so "show me the open-lid configuration"
is a filename, not a manifest lookup. A rigid object with no joints yields a single
`turntable_sheet_rest.png`.

## Options

- `--pass-dir DIR`: the `views/<NNNN>/` pass dir `render.sh` printed (required).
- `--out-dir DIR`: output location; defaults to `PASS_DIR/turntable_sheets`.
- `--columns N`: tiles per row; default `4`.
- `--tile-height PX`: height of each **render**; default `300`. A tile comes out one
  header bar taller — the bar is stacked above the picture, never over it (the
  alignment invariant in [`../lib/panels.py`](../lib/panels.py)).
- `--bg-mode alpha|black`: backdrop for the transparent renders; default `black`,
  matching what the VLM critic is fed.
- `--run-dir DIR`: run whose `layout.json` locates the source photos for the
  reference tile.
- `--frames-dir DIR`: source photos directory; overrides the layout path.

## The manifest

`turntable_sheet_manifest.json` indexes every tile: its state, source frame, page,
tile number, azimuth, elevation, band, and the image filename. That is what makes a
sheet **auditable** — a suspect tile leads back to a full-resolution PNG and a known
viewpoint, rather than to "the third picture on the second row".

## The tiles are not replaced

A sheet is an **index, not an archive**. The individual
`turntable_<state>_az<AAA>_el<±EE>.png` renders stay in the pass dir: that is what
[`critic.py`](../scorers/critic/critic.md) feeds a VLM at full resolution, and what
you open when a region needs pixels. The manifest records the glob
(`turntable_*_az*_el*.png`) for exactly that reason.

## Why this is not a `viz.rows` sheet

The [candidate sheets](candidate_sheet.md) and [composite](composite.md) strips are
built from `viz.rows`, which assembles a **SOURCE vs RENDER** comparison row. A
turntable view is a *novel* view — there is no photo taken from `az135 el-30` to
compare it against — so forcing it through that builder would mean a row of one
column. The **chrome** is still shared (`lib/panels.py` header bars, fills and
stacking), so these pages look like the rest of the family rather than a third
convention.

The view directions and the filename grammar live in
[`../../core/turntable_views.py`](../../core/turntable_views.py), shared with the
Blender-side [`views/turntable.py`](../../views/turntable.md) that writes them — a
sheet that parsed a format the renderer merely happened to emit would be one edit
away from silently finding no tiles.
