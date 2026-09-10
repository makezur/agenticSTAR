# `depth.py` — depth agreement: rendered depth vs the observed pointmap

**The geometric signal silhouette IoU can't give you.** IoU scores one 2D
silhouette and is **depth-blind** — a part floating toward the camera overlaps the
same outline as one welded on. This tool adds the third dimension: it compares the
harness's rendered depth (the [`depth` view](../../views/depth.md) → `depth.npy`,
planar camera-space Z) against the depth **observed** for the same view (the Z
channel of the capture's per-frame camera-frame pointmap, `depth/<stem>.npz`),
inside the object, weighted by the per-pixel confidence.

## When to run
- Render + score depth **every iteration** an observed-depth capture exists. It is
  cheap — the `depth` view rides the same `render.sh` launch as
  `match`/`turntable` (see [`../views/depth.md`](../../views/depth.md)) — so there is
  no reason to skip it on any pass.
- It is a **strong-but-noisy guide, not a gate.** A large `depth_mae_canon`
  is a prompt to check the **pose** (usually translation-Z)
  **before** iterating on shape, but it is weighed against IoU and the VLM critic
  (observed depth is noisy) and does **not** by itself block finalize.

## Frames & scale (important)
- The harness **camera 0** (identity extrinsic, OpenCV convention) IS the frame
  the observed per-view `local` pointmap lives in.
- The observed pointmap has **no absolute metric scale** (an arbitrary but
  self-consistent gauge — Pi3X's, for that backend). We adopt it as ground truth:
  the canonical mesh is unit and the frame's pose +
  shared `SCALE` carry it into observed units, so rendered depth compares to it
  **raw** — no scale solving. Get the shared `SCALE` + the frame's pose
  translation-Z right and the raw depth matches. This is what generalises to video
  (one unit mesh + per-keyframe poses).

## Run (artscript env)
```bash
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.scorers.depth \
    --tracking     CAPTURE/tracking              \
    --render-depth PASS_DIR/depth_000040.npy     \
    --mask         MASK.png                      \
    --source       IMAGE.png                     \
    [--view N | --frame-name 000040.jpg]         \
    [--hand-mask HAND_MASK] [--conf-thr 0.1]     \
    [--timeline-dir RUN_DIR/timeline --frame 000040 --tag iter03] \
    --out PASS_DIR/depth_000040.json --panel PASS_DIR/composite_depth.png
```
(PASS_DIR = the `views/<NNNN>/` dir `render.sh` printed.)
The tracking view is chosen by `--view`, or by matching `--frame-name` / the
`--source` basename to `keyframes.json` (the view↔frame map). This is a **reported
diagnostic, not a gate** — there is no pass/fail exit code (mirrors `color_score`).

`--timeline-dir DIR` [+ `--frame NAME` (default `--frame-name`) + `--tag LABEL`]
also archives the depth panel to `DIR/<frame>/<NNN>_depth.png` — one entry per step,
the same per-frame timeline `composite.py` writes to (`_depth` vs `_composite`
suffixes coexist), so the depth progression scrubs beside the composite. (You can
also get the depth residual folded straight into the composite panel via
`composite.py --tracking --render-depth`.)

## Scored region
`object mask ∩ rendered geometry ∩ observed keep ∩ (conf ≥ --conf-thr)`, minus the
hand region when `--hand-mask` is given. Weight `w = conf` there, 0 elsewhere.
- **`--conf-thr`** (default **0.1**): confidence below this — or `keep`==False
  — is ignored ("can't be trusted, don't penalise"). Pixels that survive are
  weighted by `conf`, so a 0.2-confidence pixel counts less than a 0.8 one.
- **`--hand-mask`** / **`--hand-dilate`**: where a hand occludes the object, the
  sensor sees the **hand's** surface (nearer), not the object continuing behind it, so
  scoring depth there is meaningless. This waives the hand region (reuses
  `rasters.build_keep`, same as `silhouette.py`'s `iou_visible`); `hand_coverage`
  reports the waived fraction.

The default `--conf-thr` (and the two depth flip switches below) come from the
per-run `RUN_DIR/depth_config.json`, resolved by walking up from the render/out
path (`core.depth_config.resolve_conf_thr`).

## Depth switches — `depth_config.json` `cost` / `report` / `render`
`depth_config.json` carries three optional booleans (all default **true**; absent
field / missing file behaves as before). They are **independent** and none of them
affects the per-frame **camera seeds** or `measure_depth` — those read the
pointmap directly:

- **`report`** — this scorer + the depth panels. `false` makes `shape_pass` skip
  depth **scoring** (no `depth_<stem>.json`, no residual images), so every
  downstream reporter goes quiet on depth. It does **not** stop the depth render.
  `shape_pass`'s `--depth-report`/`--no-depth-report` overrides it per pass.
- **`render`** — the depth **view** (`depth_<stem>.npy`), which this scorer
  consumes but does not own: it also feeds the GT-free object-units visuals
  ([depth_units.md](../viz/depth_units.md)), so it survives `report: false`.
  `false` drops the view entirely. `--depth-render`/`--no-depth-render` per pass.
- **`cost`** — the `sweep`'s depth **supervision** penalty (see
  [sweep.md](../../views/sweeps/sweep.md)). `false` forces `depth_weight 0` there.

Scaffold them off with `run.sh --no-depth-report` / `--no-depth-render` /
`--no-depth-cost`.

## Key fields
- **`depth_mae` / `depth_rmse`** — conf-weighted absolute / RMS depth error in
  observed units. **The headline.** Lower is better.
- **`depth_mae_canon`** — the same RAW `depth_mae` expressed in
  **canonical units** (divided by the object's longest dimension — the shared
  `SCALE`, auto-read from the run's `pose.json` next to `--render-depth`, or
  passed via `--scale`). The natural
  metric prior: `0.23` means "off by a quarter of the object" whether it's a
  microwave or a pair of scissors. **A large value on any frame is a strong-but-noisy
  guide that the pose (usually translation-Z) may be off — investigate before
  iterating on shape, but weigh it against IoU and the critic, since observed depth
  is noisy. It is a guide, not a gate.**
- **`depth_bias` / `depth_bias_canon`** — the **signed** mean residual (`render −
  observed`), in observed units / canonical units. `depth_mae` says *how far off*;
  `depth_bias` says *which way*: **`+` = render sits BEHIND / too far** (pull it
  in → smaller `SCALE` or `−translation-Z`), **`−` = render IN FRONT / too near**
  (the opposite). This is the number the signed residual panel visualises
  pixel-by-pixel, so one look tells you the knob direction without a one-off
  script. (A near-zero bias with a high MAE = the error cancels across the object
  → it's shape/pose, not a uniform depth offset.)

  There is deliberately **no per-view scale fit** here: a fitted scale ASSUMES the
  pose is right, so a pose error masquerades as "scale" and the fitted residual
  explains it away.
  The guide signal is the RAW `depth_mae` / `depth_mae_canon` (strong but noisy —
  weigh it against IoU and the critic, don't gate on it); `depth_bias` carries
  the direction.
- **`point_l2_mae` / `point_l2_rmse`** — secondary: full **XYZ** pointmap error
  (unproject the rendered depth through Pi3X K, compare 3-D points). Catches
  lateral (X/Y) drift that depth-along-the-axis alone misses.
- `depth_coverage` — fraction of the scorable object mask our render actually
  fills; `mean_conf_used` — mean Pi3X confidence over the scored pixels;
  `n_scored_px`, `view`, `frame_name`.

## Panel — `--panel`
`[ source | our depth | pi3x depth | residual ]`. Both depth maps are
TURBO-colormapped over a **shared** range so they're directly comparable. The
residual is **signed** (`ours − pi3x`) on a diverging map so it shows which *way*
depth is off, not just how much:
- **blue** = render is **in front of** Pi3X (**too near**) → push it back
  (`+translation-Z` for a `+Z`-forward camera, or larger `SCALE`);
- **red** = render is **behind** Pi3X (**too far**) → pull it in (`−translation-Z`,
  or smaller `SCALE`);
- **white** = agreement; **black** = outside the scored region (incl. waived hand).

Intensity toward blue/red grows with `|error|` (saturating at the same range the
depth panels share), so a uniform tint = a uniform depth offset (turn the knob),
a mix of blue *and* red = shape/pose error (no single knob fixes it). The signed
mean is reported as `depth_bias`.

## Standalone residual — `--residual-out`
`--residual-out PATH` writes just that **signed** DEPTH-RESIDUAL panel (`ours −
observed gt`; blue=render in front/too near, red=behind/too far, white=agree, labelled
with the `depth_bias`/MAE/`depth_mae_canon`) as its **own**
image — the individual, in-domain image the
visual judge `Read`s to see WHERE depth is most wrong AND which way to move (turns
a guess-iteration into zero; mirrors `composite.py --overlap-out` for the
silhouette). `composite.py` can emit the same image directly via
`--depth-residual-out`.
