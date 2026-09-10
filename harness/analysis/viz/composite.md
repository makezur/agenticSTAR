# `composite.py` — the visual judge panels

Builds the labeled panels for judging the match, emits the **standalone
silhouette-overlap image** the visual judge actually reads, writes a metrics JSON
(with a mask), and archives each step to a per-frame **timeline** for debugging.
Imports `analysis.scorers.silhouette` (silhouette + IoU) and
`analysis.scorers.depth` (depth panels); the shared panel drawing lives in
`analysis.lib.panels`, so there is no import cycle.

## Run (artscript env)
```bash
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.viz.composite \
    --source IMAGE --render PASS_DIR/match.png \
    --mask MASK [--hand-mask HAND_MASK] --metrics-out RUN_DIR/mesh/metrics.json \
    --overlap-out PASS_DIR/overlap.png \
    --timeline-dir RUN_DIR/timeline --frame FRAME --tag iter03 \
    [--tracking CAPTURE/tracking --render-depth PASS_DIR/depth.npy \
     --depth-residual-out PASS_DIR/depth_residual.png] \
    --out PASS_DIR/composite.png
```
(PASS_DIR = the `views/<NNNN>/` dir `render.sh` printed.)
`--mask` and `--metrics-out` are optional; `--height` controls panel height.
`--hand-mask` / `--hand-dilate` (default `0` — the hand mask exactly) mirror `silhouette.py` — the
written JSON is the same report, and the panel matches the numbers.

- `--overlap-out PATH` — also write the silhouette-overlap panel as its **own**
  image (needs `--mask`; skipped with a warning otherwise). This single, in-domain
  image is what the judge `Read`s each iteration — read it on its own, not the
  stacked composite.
- `--depth-residual-out PATH` — also write the **signed** depth-residual panel
  (`ours − observed gt`; blue=render in front/too near, red=behind/too far,
  white=agree) as its **own** image, so the judge sees WHERE depth is wrong AND
  which way to move it. Needs the depth inputs (`--tracking` + `--render-depth` +
  `--mask`); skipped with a warning otherwise.
- `--timeline-dir DIR` + `--frame NAME` [+ `--tag LABEL`] — also archive the full
  composite to `DIR/<frame>/<NNN>_composite.png` (one entry per step). `<NNN>` is
  `--tag` if given (e.g. `iter03`), else a zero-padded auto-increment per frame.
  Timeline folders are grouped by frame so you can scrub one frame's progression.
- `--render-depth depth.npy` ALONE (no `--tracking`, the monocular case) — append the
  **DEPTH** panel in OBJECT UNITS (depth / the pose.json scale `s`; no GT
  needed, `analysis.viz.depth_units` owns the picture and its NEAR/FAR key).
  There is no residual panel because there is nothing to disagree with.
- `--tracking DIR` + `--render-depth depth.npy` [+ `--view N` / `--conf-thr`] — append a
  5th **OUR-DEPTH** panel and a 6th **signed DEPTH-RESIDUAL** panel (`ours −
  observed(gt)`; blue=in front/too near, red=behind/too far, white=agree); needs a
  `--mask` too. `depth.py` owns the math.

## Outputs
- `composite.png` (`--out`) — the stacked judge panel (for humans/debugging):
  - **With a mask:** `[ source | render | silhouette (raw 1:1) ]`, plus
    `[ | our-depth | depth-residual ]` when `--tracking`/`--render-depth` are given.
    - **silhouette panel**: the render silhouette overlaid on the mask, with a
      colored legend drawn beside the title **from the same `OVL_*` constants
      that paint the pixels** (so no caption can drift from the picture) —
      **AGREE (green), EXTRA (yellow) = render only, MISSING (magenta) = mask
      only.** The two loud colors are exactly the two scored errors. With a
      `--hand-mask` the ignored (don't-care) hand region is a dim **gray** and
      our render *inside* it is lit **blue** = **MAYBE OCCLUDED** —
      present but unscorable (the object may well continue behind the hand, and
      we can't tell from this view), so it is neither penalized nor rewarded and
      is not read as an error. No IoU is
      stamped on the panel; the numbers live in `--metrics-out` and the console
      line, and the panel is for reading *where* the disagreement is.
    - **RGB colour agreement is numbers-only** (`color_score` / `rgb_mae` /
      `overlap_frac` in `--metrics-out`); there is no colour-residual panel.
    - **depth panels** (with `--tracking`/`--render-depth`): OUR-DEPTH (TURBO over the
      shared range) and the **signed** DEPTH-RESIDUAL `ours − observed(gt)` (blue =
      render in front/too near, red = behind/too far, white = agree — the sign
      says which way to move). Mirror the standalone `depth.py` panels.
  - **Without a mask:** just `[ source | render ]` (no silhouette panel, no IoU).
- `overlap.png` (`--overlap-out`) — the **standalone** silhouette-overlap image
  (same green/yellow/magenta/blue/gray coding, text-free and native-resolution).
  The judge's per-iteration read.
- `side_by_side.png` (`--side-by-side-out`) — the native focused comparison.
  With `--side-by-side-max-dimension`, `side_by_side_preview.png` is the default
  reduced read and `side_by_side.variants.json` records both paths and sizes.
  Escalate to the native file when the preview does not settle the detail.
- `depth_residual.png` (`--depth-residual-out`) — the **standalone** signed
  depth-residual image (`ours − observed gt`; blue=render in front/too near,
  red=behind/too far, white=agree, labelled with `depth_bias`/MAE/`depth_mae_canon`). Shows the judge where geometry sits at the wrong depth
  **and which way to move it**.
- `timeline/<frame>/<NNN>_composite.png` (`--timeline-dir`) — the debug archive.

`Read` the standalone `overlap.png` (plus the render + source images individually)
and judge with your own vision alongside the numbers.
