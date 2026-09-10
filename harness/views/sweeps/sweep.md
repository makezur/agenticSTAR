# `sweep` view — "which pose and/or joint states best fit the mask (and depth)?"

**Answers:** "given a roughly right configuration, what exact camera-frame pose
**and/or** articulation state maximizes the silhouette IoU for this frame **without
sitting at the wrong depth**?"

`sweep` is the **SEARCH** verb: it scores many candidates against a mask and ranks
them. Its imperative sibling is the [`apply`](apply.md) view — *"tilt 90°, boom,
render, what's the IoU?"*: it composes ONE directional order onto the frame's current
pose and scores just that one config against the mask (a single-candidate sweep, same
IoU/depth report), with a paste-ready block. Reach for `apply` when you already know
the move you want to check; reach for `sweep` when you want the tool to find the move
that best fits the photo.

`sweep` is the SINGLE unified search view. It renders the REAL built geometry at
each candidate through the fixed camera 0, scores the alpha-silhouette IoU against
the object mask (minus an observed-depth penalty, with `--tracking`), and writes the best
**paste-ready** pose and/or joint block. It searches a **unified DOF vector** — any
mix of:

- **pose "order" DOFs** — the six camera-frame increments composed onto the frame's
  current pose: `roll`/`yaw`/`pitch` (deg) orbit the object about a pivot, `dpx`/`dpy`
  shift it in the image plane (fractions of frame width/height) and `tz` nudges it in
  depth (a fraction of the object's depth). What each one means, what a POSITIVE value
  looks like on screen, and the directional preset tokens are the verified table in
  [`../../../conventions/DIRECTIONS.md`](../../../conventions/DIRECTIONS.md) — read it
  before you sign a `roll`/`yaw`/`pitch` order or a directional preset. Implementation:
  [`../../rig/lie.py`](../../rig/lie.py) `apply_order`.

  Rotations orbit the **pivot** — the object's bbox centre at the frame's own joint
  states, resolved once per run, frozen across candidates and refine passes, and
  recorded in `sweep.json` under `pivot`. The exact update formula is in
  [`../../../conventions/POSE.md`](../../../conventions/POSE.md) §3.

- **joint DOFs** — the **absolute** state (revolute: degrees; prismatic: canonical
  units) of any joint your `scene.py` declares in `JOINTS`, fed through the
  harness's forward kinematics. Clamped to the joint's `limit`.

> **The two kinds measure from different origins.** A pose DOF is a *delta* on the
> base pose, so `0` means "unchanged"; a joint DOF is an *absolute* state, so `0`
> means "articulate this joint flat" — a real move. The practical consequence:
> `0` is only a no-op for a pose DOF. A joint that is in the space but in no range
> holds at the **frame's declared state** (and the report lists it under `Held:`,
> not `Swept joints:`), because pinning it at `0` would silently straighten it and
> score a differently-articulated object than the frame declares.

Name only pose DOFs for a **pose fit**; name only joints for an **articulation
fit**; name **both** (or `--sweep-space both`) for a **coupled fit** on a moved
frame where the pose and a joint trade off (fitting one un-fits the other). The
space is inferred from the DOFs you name (override with `--sweep-space`).

The winning `quaternion` + `translation` paste into `FRAMES[<frame>]["pose"]` and
joint states into `FRAMES[<frame>]["joints"]`. The order is purely **rigid** —
there is no scale DOF, because `SCALE` is global (one number for the built object),
so a per-frame verb must not search it; to resize, edit `SCALE` in `scene.py`. It
**cannot fix shape** — it only re-poses/articulates finished
geometry. If the **turntable** shows a wrong/detached/under-modelled/misproportioned
part, or the **VLM critic** flags a `shape` fix (any severity), edit `build()`
FIRST: sweeping against wrong geometry just finds the pose that best paints the wrong
silhouette and entrenches the error under a deceptively high IoU.

## Search methods and basins

The default `--sweep-strategy de` runs a contained port of SciPy's bounded
`best1bin` differential evolution over all named DOFs jointly. It runs independent
seeded restarts and keeps one winner from each; Powell polishing of each restart's
best result is opt-in.
Use `dof:min,max` bounds; a legacy third grid-step value is accepted and ignored.
The main controls are:

- `--sweep-topk` (default 6 for DE) — restart count and number of finalists.
- `--sweep-de-max-evals` (default 3600) — total DE evaluation cap, divided as
  evenly as possible across the restarts.
- `--sweep-de-popsize` — population multiplier per active DOF.
- `--sweep-de-mutation` — mutation `F`, either fixed (`0.7`) or dithered
  (`0.5,1.0`).
- `--sweep-de-recombination` — crossover probability `CR`.
- `--sweep-polish {none,powell}` (default `none`) and
  `--sweep-polish-max-evals` — optional total local-finish budget, also divided
  across the restarts.

Restart `i` uses seed `--sweep-de-seed + i`. The winners are score-sorted into
`ranked`, and every completed restart winner is rendered as a panel (unless
`--sweep-visuals none` is explicit). A timeout may finish fewer restarts, in which
case the report contains the winners that actually completed.

The **basin** is the combination of the initial configuration and local bounds:

1. `--sweep-pose-start` (or the frame pose) is `P0`.
2. Pose bounds are residual orders composed onto `P0`; joint bounds are absolute.
3. DE searches only inside those bounds. It does not infer or cross an excluded
   180-degree flip.

Thus a basin is similar to a sweep with an initial point, but the initial point is
the coordinate reference and a seeded candidate, not a direction supplied to the
optimizer. If two facings are plausible, run them as separate pose starts (or first
inspect explicit flip hypotheses), optimize each locally, and compare their visual
finalists. This keeps semantic flip selection with the agent while the solver handles
continuous fitting.

DE records every restart's seed and budget, separate `de` and `polish` stages,
optimizer settings, generation-best scores, termination reason, and evaluation
counts in `sweep.json`. Since its samples are continuous, it does not fabricate a
lattice or dense 2-D score field.

### Grid strategy

`--sweep-strategy grid` uses a full Cartesian product over the named DOFs,
coarse-to-fine with `--sweep-refine` passes. This is the **explicit dense-map
tool**: name exactly two DOFs with `steps>=2` to get a full 2-D score field.

The tool renders the **2-D score field** (an ASCII heat-grid
in `sweep.txt`, the full matrix in `sweep.json`) so you can SEE a coupled valley the
per-axis marginals hide. Cost is `∏steps`, so keep the DOF count and per-DOF step
counts modest (the pre-flight ETA auto-caps a runaway grid); to explore a coupled
pose↔joint trade-off, name both axes and read the 2-D field.

### What refine actually does

`--sweep-refine N` is a **coarse→fine zoom around the single best cell**, run as a
loop over the SAME grid machinery (`engine.run`) — not a separate search. Each pass:
score the current grid → take the best-scoring candidate → re-center every swept
DOF's window on that candidate's value → shrink each half-span by
`--sweep-refine-shrink` (`0.5` = halve) → score the new grid. The **step count per
DOF is unchanged**, so every pass costs the same as the coarse pass and doubles (at
`shrink 0.5`) the resolution. Worked example, `shrink 0.5`, winner at 0 each time:

```
pass 0 (coarse):  yaw -180..+180   pitch -60..+60
pass 1 (refine):  yaw  -90..+90    pitch -30..+30
pass 2 (refine):  yaw  -45..+45    pitch -15..+15
```

Refinement stays in the **fixed vector space** (`planner.refine_ranges`) rather than
re-basing the pose on the winner: every pass's candidates carry the exact composed
matrix, so the pasted best always reproduces its scored render.

**All passes land in ONE flat candidate list — but the passes are on the record.**
`scored` accumulates across passes and `ranked` is that list sorted by combined score,
so a report's `n_candidates` is the total over all passes (with `refine 1` it is ~2×
the coarse grid product — that multiple is the passes, not a bug). What each pass
scored is recorded, so the fine lattice is reconstructable from disk rather than
merely inferrable from an off-grid value:

- `passes` (`sweep.json`) — one entry per pass: `{pass, kind, n, ranges}`, plus
  `center` (the DOF vector the window re-centered on) and `shrink` for a refine pass.
  `ranges` is **that pass's** band, so `passes[-1].ranges` is the grid the winner
  actually came from. `n` is what was really scored, so a pass cut short by the
  timeout is visible rather than implied by a short list.
- `pass` on every `ranked` entry (and on `best`) — which lattice that candidate came
  from. This is what lets a viewer draw the coarse grid with the refined window nested
  inside it, and tell a coarse cell from a refined one.
- top-level `ranges` is still the **coarse** band the agent ordered (== `passes[0].ranges`);
  `n_passes` counts the refine passes only. A sharded coarse pass records `slice`,
  which is why its `n` is below its lattice.
- `sweep.txt` gets a one-line `SEARCH PASSES` summary (counts + spans per pass) only
  when refine actually ran.

`metrics.cell_key` deliberately does **not** clamp a value into the ordered band for
exactly this reason: a refined winner legitimately sits outside the coarse lattice, so
clamping would file it in a cell it never occupied. The pass records are what make
that honest instead of merely unexplained.

The object-frame verbs record the same thing in the shape their search has:
`osweep`/`oapply` in **per-frame** mode put a `passes` list on **each** `frames[]`
entry (each frame refines into its own window), while **shared** mode has one basin
and so one top-level `passes`. See [osweep.md](osweep.md).

**Refine is direction-agnostic — it does NOT respect a directional band.**
`refine_ranges` only re-centers and shrinks; it has no memory that the window was
one-sided. With the recipe-A style `--sweep-angle-preset pitch:standard_topcloser`
(which parses to the one-sided band `0..+30`), at `shrink 0.4`:

| winner lands | refine pass 1 window | consequence |
|---|---|---|
| mid-band (`+15`) | `+9.0..+21.0` | stays inside the ordered band |
| far edge (`+30`) | `+24.0..+36.0` | **leaves the band** — follows the gradient past your guess |
| near edge (`0`) | `-6.0..+6.0` | **re-symmetrizes** — half the budget goes to the direction you aimed away from |

An edge winner is COMMON on a directional sweep (aiming the band at the error is the
point), so both rows happen. Leaving the band can be useful — it tracks
an error larger than you estimated — but a directional order is **not** a hard
constraint on where refine looks, and the near-edge case silently spends half the
remaining budget on the excluded side. If you need the band respected, order the
range explicitly and use `--sweep-refine 0`, then re-order a fresh band.

**Refine is single-basin by construction.** It re-centers on the one best cell, so
after pass 1 a genuine second basin (a flip twin at `yaw 180`, say) is outside the
window forever — it gets no fine candidates and is only ever represented by its
COARSE cell. Refine polishes a basin; it does not choose between basins. Use a wide
coarse grid with `--sweep-refine 0` to CHOOSE a basin by eye, or `oapply 'flips'` for
the discrete symmetry hypotheses, and refine only once the basin is settled.

A `--sweep-grid-slice` shard disables refine entirely (`engine.py`), as does an
explicit `--sweep-candidates` list — there is no basin to refine into.

**With `--tracking`, ranking is DEPTH-AWARE.** Silhouette IoU alone is depth-blind: a
nearer+smaller pose projects the same outline, so a family of depth-wrong poses ties
on IoU. With `--tracking`, every scoring render also writes its Z pass (same render
call) and candidates are ranked by

    combined = gate IoU − depth_weight × min(depth_canon, 1)

where `depth_canon` is the candidate's **RAW** conf-weighted depth error
`|render_Z − observed_Z|` in **canonical units** (fraction of the object's longest
dimension), so `0.23` reads as "off by a quarter of the object" for any object. Raw
on purpose: a fitted scale assumes the pose is right, so a pose error
masquerades as "scale". A candidate with no scorable depth pixels takes the full
penalty. The legend prints a blunt **`depth loss: HIGH/LOW/UNSCORED`** verdict for
the winner (HIGH = error > 0.10 of the object's size → a strong-but-noisy hint that
the pose/translation-Z, or a joint state, is off — confirm against the photo / IoU
/ critic; observed depth is a guide, not a gate).

**Turning depth supervision off.** The per-run `RUN_DIR/depth_config.json` flip
switch `"cost": false` forces `depth_weight 0` here (pure-IoU ranking), regardless
of `--sweep-depth-weight` — the sweep resolves it by the same walk-up it uses for
`conf_thr`. Scaffold it off with `run.sh --no-depth-cost`. This is independent of
the depth-`report` switch and does **not** affect the tracking camera seeds; see
[depth.md](../../analysis/scorers/depth.md). **Depth-off is the right lever for
"no depth" — NOT dropping `--tracking`.** `--tracking` supplies the real per-frame camera
K *and* the depth; dropping it to silence depth would also throw away the intrinsics
and score every candidate under the wrong projection.

**Intrinsics are auto-resolved when `--tracking` is omitted.** If you launch a sweep
without `--tracking` (or `--intrinsics`/`--camera-json`), the harness walks up from the
output dir to `RUN_DIR/layout.json` and uses its capture's `tracking/` dir, printing
`[render_wrapper] auto-resolved --tracking from layout.json: …`. So a manual sweep in a
run that has tracking still scores under the real camera. Only when NO layout tracking dir is
found does it fall back to the iPhone-13 placeholder K (with the loud warning). Pass
`--no-auto-tracking` to force that placeholder fallback.

## Grid landscape stats — `dof_stats`

Grid and explicit-candidate reports can include `dof_stats` (JSON) plus a
`sweep.txt` table describing per-DOF score projections. DE reports deliberately omit
it: adaptive continuous samples do not form a per-axis landscape, and comparing the
best value with the sampled minimum/maximum is not a requested-boundary test. Read
DE through its restart finalists, scores, images, and optimizer telemetry instead.

It stays in ONE Blender process (persistent mesh/BVH), and candidates are scored
by a **direct GPU raster** (`rig/raster.py`): the evaluated meshes are drawn once
per candidate into an offscreen framebuffer with a flat depth-tested shader —
pixel-identical to the EEVEE alpha≥0.5 silhouette (verified byte-equal ranked
lists, with and without depth supervision) at ~0.4 ms/candidate vs ~33 ms for a
full `bpy.ops.render.render`. Depth supervision reads camera-Z straight off the framebuffer's
depth attachment (no compositor EXR rider). If the GPU raster can't initialize
(no GL context even after a warmup render) the loop falls back to the EEVEE
path automatically; `SWEEP_GPU_RASTER=0` forces that fallback for A/B checks.
`--sweep-quality` trades render resolution + grid density for speed. The loop
holds up at HIGH scoring resolutions too: the mask readback is a 1-channel
byte (not 4-channel float — 16x less transfer, same thresholded mask) and the
IoU is `imaging.GridIoU` einsum dot products, so even the largest scoring
grid (quality 1.0 — long side at the 512px cap) runs ~1.6 ms/candidate
(~0.6 raster + ~0.55 IoU) — all still byte-equal to the EEVEE reference.

## Run
```bash
# pose fit — DE in the default local basin:
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views sweep \
    --match-res IMAGE --sweep-mask MASK --sweep-quality 0.5

# explicit DE bounds (camera-frame order increments around the base pose):
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views sweep \
    --match-res IMAGE --sweep-mask MASK \
    --sweep-ranges 'roll:-17,17;dpx:-0.05,0.05;dpy:-0.05,0.05;tz:-0.1,0.1'

# ARTICULATION fit — search one joint's state against the mask (name the joint):
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views sweep \
    --match-res IMAGE --sweep-mask MASK --sweep-ranges 'door_hinge:-90,12'

# COUPLED pose+joint (DE changes yaw and hinge jointly):
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views sweep \
    --tracking CAPTURE/tracking --frames a.jpg,b.jpg --ref-frame a.jpg --match-res IMAGE \
    --sweep-mask MASK \
    --sweep-ranges 'yaw:-17,17;door_hinge:-80,12'

# DIRECTION + SEARCH in one call: apply the critic's ~90° turn, then run local DE
# (--sweep-start-shift carries joints too, unlike --sweep-pose-start):
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views sweep \
    --match-res IMAGE --sweep-mask MASK \
    --sweep-start-shift 'yaw:90' --sweep-angle-preset yaw:standard

# DENSE 2-DOF MAP of a tricky coupled valley (full 5x5 field + ASCII heat-grid):
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views sweep \
    --match-res IMAGE --sweep-mask MASK --sweep-strategy grid --sweep-refine 0 \
    --sweep-ranges 'yaw:-17,17,5;door_hinge:-80,12,5'

# occluding hand: score iou_visible over the non-hand region (the gate):
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views sweep \
    --match-res IMAGE --sweep-mask MASK --sweep-hand-mask HAND_MASK
```

Every flag's `--help` (see [`../../rig/args.py`](../../rig/args.py)) is the ground
truth. Key ones:
- `--sweep-mask` (required) — the object mask candidates are scored against.
- `--sweep-space {pose,joint,both}` — override the auto-inferred space; also sets the
  empty-`--sweep-ranges` default box (pose = 6-DOF pose box; joint = all sweepable
  joints; both = coarse `yaw`+`tz` × all joints).
- `--sweep-strategy {de,grid}` — DE is the default; use grid for dense fields.
- `--sweep-ranges` — per-DOF `dof:min,max` for DE or `dof:min,max,steps` for grid,
  `;`-separated. `dof` is a pose
  order (`roll,yaw,pitch,dpx,dpy,tz`) OR a declared joint name; **mix freely**.
  Joint spans are clipped to the joint's `limit`. Omitted DOFs held. Empty = the
  space's default box (grid density is scaled by `--sweep-quality`).
- `--sweep-angle-preset` — per-axis SIZE preset for the empty-`--sweep-ranges`
  default box: `;`-separated `axis:preset` where `axis` ∈ `roll,yaw,pitch` and
  `preset` ∈ `tiny`/`standard`/`large`/`huge`. Size the default search band by
  name instead of typing degrees:

  | preset | half-span | full band |
  |---|---|---|
  | `tiny` | ±5° | a last-mile touch-up around a good pose |
  | `standard` | ±15° | the default (local refinement) |
  | `large` | ±30° | a wider band for a weak prior |
  | `huge` | ±45° | the widest named band; beyond it, spell a range |

  **Prefer a DIRECTIONAL preset.** Append a direction and the band becomes
  ONE-SIDED, spending the whole `2·half` budget on the side you actually saw
  (same candidate count, nothing wasted on the wrong side, and an underestimated
  defect still lands inside): `roll:standard_cw` / `roll:standard_ccw`,
  `yaw:standard_leftcloser` / `_rightcloser`, `pitch:standard_topcloser` /
  `_bottomcloser` — the tokens name WHICH EDGE swings toward the camera (signs
  and the on-screen reading of every axis are in
  [`../../../conventions/DIRECTIONS.md`](../../../conventions/DIRECTIONS.md)).
  So `--sweep-angle-preset pitch:standard_topcloser` searches `pitch ∈ [0°,+30°]`.
  A **symmetric** (bare) preset is generally NOT what you want except `tiny`
  (a genuine sign-agnostic touch-up): from a photo you can almost always see
  WHICH WAY the pose is off, so say it and don't spend half your grid searching
  the direction you already ruled out.

  Overrides ONLY the named angle axes; unnamed rotation axes keep the default
  standard span. Other DOFs retain local defaults. Explicit ranges override
  presets — a full-swing search (unknown facing / plausible flip) is an explicit
  DE range like `--sweep-ranges 'yaw:-180,180'`, or an explicit grid strategy with
  `--sweep-ranges 'yaw:-180,180,9'` for a coarse facing comparison. E.g.
  `--sweep-angle-preset
  yaw:large_leftcloser;pitch:tiny`. (Note: in
  `--sweep-space both` the empty pose box is only `yaw`+`tz`, so a `roll`/`pitch`
  preset there has no default-box entry to widen — presets pair with a pose fit.)
- `--sweep-joints` — comma-list restricting the DEFAULT (empty-ranges) joint sweep.
- `--sweep-quality` — speed↔fidelity, clamped `(0.05,1]`; scoring res = quality ×
  match-res (floored ~96px), and grid strategy's default density.
- `--sweep-refine` / `--sweep-refine-shrink` — grid-only coarse-to-fine passes; see
  [What refine actually does](#what-refine-actually-does) for the window arithmetic
  and the two edge behaviours (it can leave the band you ordered).
- `--sweep-topk` (strategy default) — for DE, run this many independent restarts,
  list one winner per completed restart in score order, and render all of those
  finalists (default 6). For grid or explicit candidates, list this many candidates
  in plain score order (default 10); listing and panels remain independent there.
  `0` selects the strategy default.
- `--sweep-dump-topk` (0) — re-render N candidates to `sweep_top_00.png…`, taking
  the **best candidate in each distinct cell of the grid you ordered** (cell =
  `round((v - lo) / step)` per swept DOF, from your own `ranges`, so there is no
  tolerance constant to tune and it is stable under `--sweep-refine`). Never accepts
  anything out of score order, so panel `k` is simply the best candidate in the
  `k`-th distinct cell. If fewer cells are occupied than you asked for you get
  **fewer panels, possibly one** — that is the truth about the sweep (refine
  converging into a single cell, say), and better evidence than filler from
  elsewhere in the band. Each entry's `rank` is its position in the full score
  ranking. This flag controls grid/explicit-candidate panels; DE panels its restart
  winners instead.
- `--sweep-visuals` (`auto`) / `--waive-visual-budget` — how many candidates become
  images at all. For DE, `auto` and `all` both render every restart winner. For
  grid/candidates, `auto` honours `--sweep-dump-topk` and always renders at least
  the winner, while `all` panels **every** scored candidate in score order. Panel
  sets are capped by the 100-panel visual budget (`core/visual_budget.py`) unless
  waived. `none` scores only; `sweep_best.png` is still written, it is the view's
  own output. Under the render pool these are also order keys, and the pool turns
  the images into a candidate sheet automatically (see
  [pool/README](../../pool/README.md)).
- `--sweep-depth-weight` (default: the depth backend's `weight` from `depth_config.json`, else 0.1) — observed-depth penalty weight (active only with `--tracking`,
  and overridden to 0 when `depth_config.json` has `"cost": false`). Note: pool
  orders that want depth to actually bite commonly state a heavier 0.5.
- `--sweep-hand-mask` / `--sweep-hand-dilate` — waive an occluder; `iou_visible`
  becomes the gate. `--sweep-hand-dilate` defaults to **0** (the hand mask
  exactly), which is `silhouette.py`'s default too — the gate you sweep on and the
  gate that re-verifies the winner agree unless you change one.
- `--sweep-pose-start` — JSON pose to center the search on (default: the frame's pose);
  use it to select the local DE basin or the center of a grid search.
- `--sweep-start-shift` — **direction + search in one call.** A directional order (same
  grammar as [`apply`](apply.md): `;`-separated `dof:value`, pose orders +
  absolute joint states) composed onto the base **before** searching, so the solver
  searches a window AROUND the shifted pose. Pair with `--sweep-angle-preset` /
  `--sweep-ranges` for the band: `--sweep-start-shift yaw:90 --sweep-angle-preset
  yaw:standard` searches ±15° around a 90° turn. Unlike `--sweep-pose-start` (placement
  only) it **carries joints**. A pose center shifts the search window; a joint ALSO
  named in `--sweep-ranges` keeps its explicit band (the center is ignored for that
  joint — pose DOFs are relative, joints absolute). Mutually exclusive with
  `--sweep-candidates`; composes after `--sweep-pose-start`.
- `--sweep-candidates` — explicit candidate JSON text or a JSON file; bypasses
  grid + refine. The value is a list (or `{ "candidates": [...] }`) whose entries
  are `{ "candidate_id", "hypothesis_id" or "hypothesis_ids", "pose"?,
  "joints"? }`. Missing pose/joints inherit the current frame. Candidate and
  hypothesis IDs survive into `sweep.json`, so a caller that gathered candidates
  from several hypothesis sweeps can rescore them all uniformly in one order.
- `--sweep-grid-slice i/K` — score only shard `i` of `K` (parallel sharding; disables
  refine).
- `--sweep-timeout SECONDS` — wall-clock budget; on timeout the best-so-far is still
  ranked, re-rendered, and written (flagged `timed_out`). A pre-flight ETA auto-caps
  a runaway grid at 10 min when no timeout is set. Depth (`--tracking`) is ~4×/candidate.

Scoring renders at **1 AA sample**, long side capped at **512px**, reading the mask
back off the offscreen framebuffer as a 1-channel byte (`fb.read_color`); only the
EEVEE fallback goes through `foreach_get` on a render result. The winner is always
re-rendered at full match-res +
samples for `sweep_best.png` and re-verified by `silhouette.py`.

## Outputs (in the pass dir `render.sh` prints)
- `sweep_best.png` — the winning configuration re-rendered at **full match res**.
- `sweep_top_00.png … sweep_top_<N-1>.png` — for DE, every completed restart
  winner at full res, sorted by score (`00` = the numeric winner). For grid or
  explicit candidates, `--sweep-dump-topk N` selects the best candidate per ordered
  grid cell, or `--sweep-visuals all` renders every scored candidate. These are the
  object alone. When depth rendering is enabled, each panelled candidate also gets
  a same-render `*_depth.npy`; this adds compositor/EXR readback but not a second
  Blender render, and the pool displays it as OUR DEPTH in object units. To
  judge one against the SOURCE frame (the silhouette-trap check) you need a panel.
  **Under the render pool you get both kinds for free**: the manager writes the
  paginated `candidate_sheet_NNN.png` *and* one `side_by_side_<image>.png` per
  candidate before publishing the result, so nothing here needs a post-hoc tool
  (see [pool/README](../../pool/README.md) § panelled). Run imperatively, build them
  yourself — no re-render needed:
  `micromamba run -n artscript env PYTHONPATH=harness python -m
  analysis.viz.candidate_sheet --render-dir PASS_DIR --run-dir RUN_DIR`
  (works on any past sweep/apply/oapply/oapply_all/osweep dir;
  `analysis.viz.sweep_sides` writes one strip per image instead of a paginated
  sheet).
- `sweep.txt` — human table: per-candidate `combined`/`iou_raw`/`iou_vis`/`dp_canon` +
  the swept pose order and/or joint states, the grid-only DOF-landscape table and
  2-D score field, the `depth loss` verdict, and **`BEST — paste into
  FRAMES[<frame>]`** blocks (a `"pose"` block iff pose DOFs were swept; a `"joints"`
  block naming EVERY declared joint whenever the scene articulates — a FRAMES joints
  dict missing one straightens it, since FK reads an absent state as 0.0). No `SCALE`
  line is ever emitted — the order is
  rigid and there is no scale DOF (see above).
- `sweep.json` — machine manifest: `space`, `method` (`"grid"`, or `"candidates"` when
  `--apply`-style explicit candidates were scored instead of a lattice, or
  `"grid_sharded"` in a merged parent written by a shard merge — see § Parallel sharding),
  `center` (the
  raw `--sweep-start-shift` order, if any — note `pose_start` then reflects the SHIFTED base),
  `swept_dofs`,
  `held_joints` (the legend's "Held:" line — every candidate's own `joints` is
  already the full state, so nothing reconstructs one from this),
  `pose_start`,
  `pivot` — the frozen orbit point the recorded `roll`/`yaw`/`pitch` actually turned
  about: `centre` (camera frame), `depth_z` (its +Z — what `tz` is a fraction of),
  `canonical_centre` (the canonical point it was carried from), `joint_states` (the
  articulation it was measured at) and `source` (`"fk_aabb"` here; the object-frame
  views record `"canonical_origin"`, since that grammar right-multiplies and has no
  pivot). Recorded because an `order` without its pivot does not determine a pose, and
  because it makes any future change to the pivot's DEFINITION visible on old runs
  rather than silently retroactive.
  `best` (pose block and/or joints block + IoUs +
  `depth_mae`/`depth_canon`/`combined` + image), `ranked` top-K, `gate_field`,
  `depth_weight`, `dof_stats` (grid/candidate per-DOF landscape; omitted for DE),
  `score_fields` (every
  dense score matrix the grid can project — keyed `"yaw|pitch"` for a DOF PAIR and
  `"yaw"` for one DOF's marginal, both the same object: `dofs`, `a_values`, `b_values`,
  `scores`; each cell is the BEST combined score over the DOFs it collapses, `null`
  where nothing was scored), `ranges` (the COARSE
  band), `passes` (what each pass actually scored — see
  [What refine actually does](#what-refine-actually-does)), `n_passes` (refine passes
  only), `sweep_resolution`, `status`/`timed_out`, and `panels` (`asked` / `rendered` /
  `visuals` / `selection` / `cells_occupied` — so a shortfall is on the record). A
  merged sharded parent adds `cells_from_listed_only: true` there: the merge can only
  bin the slice winners its shards already rendered, so `cells_occupied` counts the
  listed union rather than every scored candidate.
  For DE, `ranked` is one winner from every completed restart in **score** order;
  each carries its `restart`, `seed`, and panel `image`. For grid/candidates,
  `ranked` is the score-ordered top-K plus any candidate that got a panel. Every
  entry carries `rank` (the one name a candidate has), `pass`, and an `image` when
  rendered.

  `rank` is a candidate's NAME, not its line number: because the listing is gappy
  (topk cuts it, then panelled candidates from below the cut are appended keeping
  their true rank), the 11th entry may well be `rank: 13`. `sweep.txt` prints both —
  a `row` column (line number) and a `rank` column (this number) — and `rank` is the
  one to quote in notes and to feed
  [reseed](../../pool/reseed.md) as `sweep.json#rank=13`.

## The IoU here is a fast PROXY
The per-candidate IoU is computed in-process with numpy (mirroring
[silhouette.py](../../analysis/scorers/silhouette.md)'s `iou_raw` / `iou_visible`) at the low
sweep resolution — typically within ~0.005 of the authoritative score. **Always
re-verify the winner**: paste `best` into the frame, render `match,depth`, and run
`silhouette.py` on `match.png` + `depth.py` on `depth.npy`. IoU is depth-blind — check
the `turntable` too, because a sweep can shove parts to the wrong depth to win a
silhouette. When observed depth exists, run WITH `--tracking` so depth is in the ranking, and
never paste a BEST whose `depth loss` reads `HIGH`.

The sweep restores the base pose + render resolution/intrinsics afterward, so it
does **not** affect `match.png`, the turntable, `pose.json`, or the exported GLB
(you can request `--views sweep,match,turntable` in one launch).

## Coupled valley — map it with a 2-DOF grid

When two axes **trade off** — the pathological case is `dx` alone worse AND `dy`
alone worse but `dx+dy` best — the per-axis marginals hide the diagonal optimum.
Order a dense 2-DOF `grid` over exactly those two axes (`steps≥2` on both, e.g.
`'yaw:-17,17,5;door_hinge:-80,12,5'`) and read the **2-D score field** (the ASCII
heat-grid in `sweep.txt`, the full matrix in `sweep.json`): the diagonal ridge is
legible directly. Then `--sweep-pose-start` that winning pose and refine.

## Workflow — auto-tune a frame

Use `sweep` when a configuration is roughly right but the gate IoU is stuck below
target and you'd otherwise hand-tune the numbers. It only re-poses / articulates the
finished object — so if the **turntable** or the **critic** says a part is wrong,
missing, under-modelled, or misproportioned, that's a **shape** error; fix `build()`
first (no pose, however swept, can fix geometry).

1. **Read** the frame's current `pose`/`joints` and the last `iou_raw` (and the
   critic's tagged fixes — if any are `shape`, fix `build()` before sweeping).
2. **Pick the space + DOFs.** Name the pose orders and/or joints you want to move.
   If the pose↔joint are coupled on a moved frame, name both — a grid over the two
   maps the valley (order a dense 2-DOF grid to SEE it).
3. **Pick quality + step counts.** Keep the DOF count and per-DOF `steps` modest so
   the product `∏steps` stays cheap. Low quality (0.2–0.3) when far, high (0.6–1.0)
   when close.
4. **Run** (see the examples). With `--tracking`, ranking is depth-aware.
5. **Read** `sweep.json` `best` + `sweep_best.png`. For DE, compare the restart
   finalists and their images; do not infer axis status or bound pressure from an
   adaptive sample trace.
6. **Paste** `best` into `FRAMES[<frame>]["pose"]` / `["joints"]` — `sweep.txt` has
   ready blocks. (Nothing to paste into `SCALE`: the sweep never searches scale.)
7. **Verify (do not skip):** render `match,turntable`, run
   [`silhouette.py`](../../analysis/scorers/silhouette.md) on `match.png`; confirm the gate IoU
   improved AND the turntable still shows a coherent articulated assembly.

## Parallel sharding (optional — a large coarse grid only)
The single-process run already scores the whole grid cheaply (persistent BVH); shard
only when a coarse grid is genuinely large. Launch K background processes over
disjoint `--sweep-grid-slice i/K` stride slices, each to its own out dir, merge the
per-shard `sweep.json`s (pick the global-best across all `ranked` lists), then run
refinement single-process (`--sweep-pose-start` that pose, `--sweep-refine 2`). Keep K
small (2–4): each process pays Blender startup + `build()`, and EEVEE processes
contend for one GPU.

Merging shards is now always something you do yourself, and there is one trap in it
you must handle: every shard writes its OWN `sweep_top_00.png`, so the names collide
across shards, and a filename inherited into the parent's report names a file the
parent's directory does not have. Re-bin the shards' rendered candidates across the
whole lattice, copy the survivors under parent-unique names, and drop `image` from any
row you cannot back with a file — a report that names a missing render breaks every
sheet builder downstream (`analysis/viz/sweep_sides.report_rows`).
