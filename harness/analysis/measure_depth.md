# `measure_depth.py` — measure tentative per-frame object poses off the observed pointmap

**A per-frame pose *prior*, not a pose *score*.** [`depth.py`](scorers/depth.md)
grades a pose you already committed to; this tool hands you one to start from — for
**every** frame, not just the reference. The capture ships per-pixel camera-frame
3-D points (`depth/<stem>.npz`) plus the per-view cameras, so the object's placement
can be *read* out of the cloud instead of guessed by eye:

- object **center in frame k** → `FRAMES[k]["pose"]["translation"]` (measured per frame)
- Pi3X **relative camera pose** `rot(inv(c2w[k]) @ c2w[ref])` →
  `FRAMES[k]["pose"]["quaternion"]` (rotation seeded from the camera motion; the
  **reference frame is identity**)
- oriented-box **extents** (fused across frames) → the shared **`SCALE`** (canonical
  longest ≈ 1) **and** the canonical **W:H:D proportions** for `build()`
- Pi3X **intrinsics** (rescaled to the mask resolution) → the exact `--intrinsics`
  string to render with

## Per-frame pose model
Anchor the **reference** frame's object rotation at **identity** and read every
*other* frame's rotation off the known Pi3X camera motion: the object sits still in
the world while the camera orbits, so its rotation in frame k's camera is exactly the
relative camera pose `rot(inv(c2w[k]) @ c2w[ref])` — the *same* transform the render
side uses to auto-seed non-moved frames (`transforms.reseat_pose`). **Translation is
measured independently per frame** (the object's centroid in that view), so a frame
where the object was re-gripped still gets a correct translation even though its
propagated rotation may be stale. This is a rough **init everywhere**; refine rotation
with the `sweep` and verify with [`depth.py`](scorers/depth.md).

## Frames & scale
The harness renders every frame from a fixed **camera 0** (identity extrinsic, OpenCV
convention) and **moves the object** into frame k's camera; Pi3X's per-view `local`
pointmap lives in that same **frame-k camera frame**, and we adopt Pi3X's (arbitrary
but self-consistent) scale as truth — same contract as [`depth.py`](scorers/depth.md).
So each frame's `center` is already in the units its
`FRAMES[k]["pose"]["translation"]` expects: paste it straight in, no conversion.

## Run — single view (default, artscript env)
```bash
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.measure_depth \
    --tracking CAPTURE/tracking                         \
    --mask   MASK.png                               \
    [--hand-mask HAND.png] [--hand-dilate 0.0]      \
    [--view N | --frame-name 000040.jpg | --source IMAGE.png] \
    [--conf-thr 0.1] [--min-points 50]              \
    --out PASS_DIR/measure_depth.json
```
Single-view emits the measurement fields below (translation/scale/proportions/
intrinsics) for one view, **without** a camera-motion rotation (that needs the
reference view — use multi-frame mode for the per-frame `seed_pose`). The tracking view is
chosen by `--view`, or by matching `--frame-name` / the `--source` basename against
`keyframes.json` (shared with `depth.py` via `resolve_view`). This is a **reported
estimate, not a gate** — no pass/fail exit code.

## Multi-frame mode (opt-in) — the per-frame seed
Passing **`--run-dir`** *or* **`--frames`** measures **every keyframe on its own
mask**, emits a paste-ready per-frame pose + a `pose_snippet` (pose-only fragments), and
fuses the per-frame extents into one **shared base shape**. Two ways to invoke:

```bash
# (a) from a run's layout.json (written by run.sh for multiview runs)
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.measure_depth \
    --run-dir runs/<name> [--out runs/<name>/measure_multi.json] [--per-frame-out]

# (b) explicit — no run dir
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.measure_depth \
    --tracking CAPTURE/tracking --frames "000000.jpg,000070.jpg" \
    --masks-dir MASKS_DIR [--hand-masks-dir HAND_DIR] --ref-frame 000000.jpg
```
Explicit flags **win over** `layout.json`; `--ref-frame` defaults to the first of
`--frames`. Don't combine a multi-frame trigger with `--mask`/`--view` (they pick a
single view). Per-frame masks/hand-masks are resolved by basename:
`<masks-dir>/<stem>.png`, `<hand-masks-dir>/<stem>.png`. Default `--out` is
`RUN_DIR/measure_multi.json` when `--run-dir` is given, else print-only.

**Output** — one combined JSON:
- **`per_frame`** — one report per frame (the single-view schema below) plus a
  `status` (`ok` / `too_few_points` / `skipped_no_mask` / `error`) and, when OK, a
  **`seed_pose`** `{quaternion, translation}` (the per-frame pose; ref = identity).
- **`pose_snippet`** — paste-ready per-frame **pose fragments** (printed at the end of
  the run): one `"pose": {quaternion, translation}` block per frame. Copy each INTO the
  matching `scene.py` `FRAMES` entry — it is deliberately **not** a full `FRAMES = {...}`
  assignment, so pasting it **never touches a frame's `moved`/`joints`** (a wholesale
  block paste would drop them, silently flipping a hand-set `moved=True` back to False —
  which this pose-only tool must never do).
- **`base`** — the fused shared shape: **`scale`** = median of per-frame longest
  extents (+ `scale_spread`); **`canonical_ratios`** = `[1.0, median(mid), max(short)]`
  — the **short/depth axis is the MAX across views** because a single front-surface
  view underestimates depth (a *lower bound*; widen toward
  `canonical_ratios_spread.short.max` if depth reads thin); **`intrinsics_for_render`**
  comes from the reference frame.

This tool measures **pose only** and **never decides `moved`** — that call is the
agent's own visual read of the source frames (see AGENT_TASK.md golden rule 6;
`aggregate.py` cross-checks committed `moved` flags afterward). A frame's `seed_pose`
**rotation** is a camera-motion guess a genuine re-grip may invalidate; the
**translation stays measured (correct)**, so keep it and re-orient with the `sweep`. If
the evidence says *articulation*, model it with a `JOINT` (the base pose stays put) rather
than moving the base translation.

SCALE and proportions are invariant to rigid motion, so moved/articulated frames are
**kept** in the fuse (median stays robust).

## Fitted region
`object mask ∩ observed keep ∩ (conf ≥ --conf-thr) ∩ finite Z`, minus the hand region when
`--hand-mask` is given. The object mask and observed grid are aligned on the **mask
resolution** (the observed grid resampled up to it, `K` rescaled to match — reuses
`depth_obs.resample_to`), so `intrinsics_for_render` is correct for renders at that
resolution.
- **`--conf-thr`** (default **0.1**): confidence below this — or `keep`==False —
  is dropped; surviving points weight the centroid by `conf`.
- **`--hand-mask`** / **`--hand-dilate`**: where a hand occludes the object, Pi3X sees
  the **hand's** (nearer) surface, not the object — so those points would drag the
  center/extents off. This excludes them (reuses `rasters.build_keep`, same as
  `silhouette.py`/`depth.py`).
- **`--min-points`** (default **50**): if fewer confident object points survive, it
  emits an `error` field (bad mask/view/threshold) instead of a bogus fit.

## Key fields
- **`seed_pose`** (multi-frame) = `{quaternion, translation}`, the paste-ready per-frame
  pose. `quaternion` = `rot(inv(c2w[k]) @ c2w[ref])` (ref = `(1,0,0,0)`); `translation`
  = the measured centroid.
- **`suggested_translation`** / **`center`** = the (conf-weighted) centroid in frame k
  → `FRAMES[k]["pose"]["translation"]`.
- **`suggested_scale`** = the **longest** OBB extent → the shared top-level `SCALE`
  (canonical longest ≈ 1, so this maps unit → Pi3X units; scale is shared across
  frames, never per-frame).
- **`canonical_ratios`** = `[1.0, mid/long, short/long]` → proportion the parts in
  `build()` (longest : mid : shortest).
- **`obb_axes`** — 3×3, rows = unit principal axes in the frame-k camera frame, longest
  first; **`obb_extents`** — full width along each. A *hint* for which way the box
  points (rotation itself comes from the camera motion, not these axes).
- **`aabb_extents`** — plain axis-aligned extents; sanity fallback.
- **`intrinsics_for_render`** — `"fx,fy,cx,cy,W,H"`, ready for `render.sh --intrinsics`
  (format matches `rig/camera.py::parse_intrinsics_str`).
- `n_points`, `mean_conf`, `view`, `frame_name`.

## rotation from camera motion, NOT from the PCA axes
The per-frame rotation is the **camera-motion** rotation `rot(inv(c2w[k]) @ c2w[ref])`
(pure numpy `rig.lie.matrix_to_quat`), anchored at identity on the reference frame. We
deliberately do **not** derive rotation from the PCA/OBB axes: turning axes on a partial
front-surface cloud into an object rotation is fragile (the visible surface biases the
axes) and the axis→rotation convention is easy to get wrong. `obb_axes` is only a
which-way-does-it-point hint. Poses are authored as scalar-first **quaternions** (see
`scene_example.py`); the `sweep` view also emits paste-ready quaternion poses.

## Recommended flow
1. Run multi-frame (`--run-dir` / `--frames`) at the start → **paste each `pose_snippet`
   fragment INTO its `scene.py` `FRAMES` entry** (a measured translation + camera-motion
   rotation for every frame; leave `moved`/`joints` as authored), set the shared `SCALE`
   from `base.scale`, proportion `build()` by
   `base.canonical_ratios`, and render with `--tracking` (each frame's K applied
   automatically) or `--intrinsics <base.intrinsics_for_render>`.
2. Refine each frame's rotation: eyeball the `match` view, or run a wide-range `sweep`
   with `--sweep-dump-topk` and pick the orientation from the `sweep_top_*.png`
   contact sheet. If the images show a **re-grip**, set `moved=True` and
   re-orient; if they show **articulation**, add a `JOINT` and leave the base pose.
3. **Verify with [`depth.py`](scorers/depth.md)** per frame — aim for a small `depth_mae_canon`
   and a near-zero `depth_bias_canon` (translation-Z / scale already
   right); iterate rotation, then fine-tune.
