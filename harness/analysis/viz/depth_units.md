# `depth_units.py` - our depth in OBJECT UNITS: the DEPTH panel and the depth sheet

The one entry point for GT-free (monocular) depth visuals. The `depth` view's
`depth_<stem>.npy` is planar camera-space Z in scene units; divided by the
run's predicted shared SCALE `s` (pose.json `scale`) it becomes depth in
**object units** — "how many object-lengths away". No observed pointmap is
needed, so this works exactly where `analysis.scorers.depth` goes dark
(`depth_config.json` `backend: "none"`). Shape passes render the depth view
whether or not depth is scored — the `report` switch gates only the observed-depth scoring.
Its own switch is `render` (`--no-depth-render`), which is what turns the view —
and so everything on this page — off.

Two media, one owner:

* **The DEPTH column.** `object_units_panel` returns the standard row's depth
  panel: titled `DEPTH`, with the NEAR→FAR ramp and its endpoint values IN
  LINE with the title (`panels.label`'s ramp_legend, built from the same
  turbo LUT as the pixels — the overlap column's rule: the key cannot drift
  from the picture). The picture stays clean; no caption.
  `composite.py` draws it whenever it has `--render-depth` but no `--tracking`, so
  the monocular judge strip carries a depth read like the GT one does. Range
  is the frame's own 2–98 pct — a composite is one frame; the sheet is the
  shared-scale medium.

* **The depth sheet** (the CLI). Collects depth renders, computes ONE shared
  `[near, far]` range (2–98 pct over every selected frame's finite pixels
  pooled) and pages the heatmap tiles through `frame_sheet`'s grid — same
  packing, pagination, frame-ID headers and manifest. ONE colour key rides
  each page header's free middle, between the title and the page counter (a
  ramp with NEAR/FAR and five tick values in object units); the tiles carry
  no key, because at tile size a stamped ramp is decoration, not a key. One
  colour = one distance on every page, so poses breathing in depth (the
  temporal table's oscillating `radial`) read as tiles pulsing blue↔red
  across time.

## Run

```bash
micromamba run -n artscript env PYTHONPATH=harness \
  python -m analysis.viz.depth_units \
  --pass-dir RUN_DIR/views/NNNN          # a shape pass's depth_*.npy
```

or collect from a spool (one depth order per frame, e.g. at self-check):

```bash
  python -m analysis.viz.depth_units \
  --scan RUN_DIR/spool/renders --frames 000040-000100 \
  --out-dir WDIR/depth_sheets
```

Writes into `--out-dir` (default `depth_sheets/` beside the depth files):

```
depth_sheet_NNN.png        # the pages, the NEAR/FAR ramp in each header
depth_sheet_manifest.json  # frame_sheet's grid manifest
depth_units.json           # s, the range used, per-frame stats (object units)
tiles/<stem>.png           # the per-frame heatmaps (clean; the page carries the key)
```

Flags: `--frames` (names, comma lists, inclusive `first-last` ranges — the
frame sheet's own selector grammar; the shared range is computed over the
selection), `--scale S` (default: pose.json walk-up, same as
`scorers.depth.resolve_scale`), `--range MIN,MAX` (pin the scale, e.g. to
compare two runs), `--columns/--frames-per-page/--tile-height` (the frame
sheet's packing budget). A stem found twice (re-rendered frame) keeps the
newest file and says so.

## Reading it

* **median N obj-units away** (stdout + `depth_units.json`) is the headline
  number per frame; the header ramp gives the numeric NEAR/FAR key.
* With the shared range, the SAME colour on two tiles is the SAME distance.
  A smooth approach walks the colormap smoothly; a tile that jumps against
  its neighbours while the object's apparent size does not change is a pose
  breathing in depth — fix `tz`, don't explain the number.
* Black = no rendered geometry. An all-black tile with `n_px: 0` means that
  frame's pose left the frustum — a real finding, not a tool failure.
