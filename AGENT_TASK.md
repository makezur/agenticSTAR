# Agent Task - Procedural 3D Reconstruction from Multiple Views
You are a 3D reconstruction agent. Given one or more RGB frames of one object,
their object masks, optional hand-occlusion masks, and known per-frame cameras,
write a Blender Python (`bpy`) script that:

1. builds the object from primitives in a canonical rest pose;
2. estimates its per-frame rotation and translation with one shared scale; and
3. declares articulated joints and their per-frame states.

Iterate against headless renders until every frame reads as the same object in
the observed configuration.

## Deliverables
- `RUN_DIR/scene.py`: declarative geometry, materials, shared `SCALE`,
  `REFERENCE_FRAME`, shared `JOINTS`, and per-frame `FRAMES`.
- `RUN_DIR/mesh/object.glb`: canonical, rest-pose, watertight visual mesh with
  individually named parts.
- `RUN_DIR/mesh/pose.json`: shared scale, per-frame object poses and joint
  states, plus the shared joint definitions.
- Iteration evidence and final reports described under **Finalization lock**.

The GLB and `pose.json` together form an articulated visual rig. Collision
meshes, swept-clearance checks, and self-collision validation are downstream.

## Inputs
The launch prompt supplies:

- `CAPTURE`: the run's capture directory — the capture dir
  `tools/make_capture.py` produces (`datasets/common/capture.py`). Its
  `tracking/` subdir carries the known per-frame cameras (intrinsics/poses +
  `keyframes.json`); its optional `depth/` subdir carries per-frame observed
  pointmaps from the configured depth backend (per the manifest's `depth`
  field; `none` = masks-only run).
- `FRAMES`: comma-separated frame names; the first is the reference frame.
- `FRAMES_DIR`, `MASKS_DIR`, `HAND_MASKS_DIR`: multi-frame source, object-mask,
  and optional hand-mask directories (subdirs of the capture).
- `RUN_DIR`: the current run directory.

A hand mask marks an occluder, not object geometry. Pass it to scoring tools as
`--hand-mask`; that region is don't-care. Ask for any missing required input
before proceeding.

## Working conventions (always in force)
- **Reuse existing tools.** Build a tool only for a genuine gap, implement it
  minimally, and record the gap in `RUN_DIR/wish.md`. Always leave `wish.md`,
  even when the existing tools were sufficient.
- **Reconstruct fresh.** Do not read, copy, or seed from another `runs/`
  directory.
- **Respect the selected depth backend.** The launch prompt's `DEPTH_BACKEND`
  names it; depth scoring reads its confidence floor from
  `RUN_DIR/depth_config.json`. Do not pass `--conf-thr` unless deliberately
  overriding it.
- **Treat co-located docs as canonical.** Tool flags come from `--help`; tool
  behavior comes from its adjacent Markdown file. This task states reasoning,
  ownership, workflow, and gates rather than duplicating those manuals.

## Golden rules
1. **Honor the ownership boundary.** You own `scene.py`: geometry, materials,
   `SCALE`, `REFERENCE_FRAME`, `JOINTS`, and `FRAMES`. The harness owns cameras,
   lighting, GPU/render settings, pose application, forward kinematics, and
   export. Never create cameras/lights or set `bpy.context.scene.render.*`.
2. **Decompose from all frames.** Read every provided frame and mask before
   building. `build()` must contain the union of parts visible in any frame,
   including interiors or components hidden in the reference view. Missing a
   revealed part is a failed decomposition.
3. **Build canonical, once.** In `build()`, use the object's natural orientation,
   `+Z` up, centered at the origin, longest dimension roughly 1. Ignore cameras.
   Build every part in its neutral rest state. Per-frame pose and articulation
   create the observations. Follow [POSE.md](conventions/POSE.md).
4. **Use the simplest sufficient primitive model.** Use cubes, cylinders,
   spheres, cones, tori, convex hulls, transforms, exact booleans, bevel,
   solidify, and mirror. Import shared `box`, `convex_hull`, `set_color`, and
   `apply_boolean` helpers instead of redefining them. Increase geometric
   fidelity when the evidence demands it. Read the complete helper and geometry
   contract in [harness/shapes/README.md](harness/shapes/README.md).
5. **Match shape and color.** Establish shape, proportions, and placement first,
   then assign materials to every part. The judged render is full RGB;
   unmaterialed parts receive only a neutral fallback.
6. **Use one shared scale and honest per-frame poses.**
   `FRAMES` has one entry for every observed frame.

   - `SCALE` is the object's frame-invariant physical size, expressed once as a
     single uniform scalar. Anisotropic `(sx, sy, sz)` scale and per-frame scale
     are forbidden. Encode axis proportions in canonical `build()` geometry;
     non-uniform scale does not commute with rotation.
   - `REFERENCE_FRAME` is the easiest gauge frame; author its `pose` directly.
   - A pose contains a scalar-first unit quaternion `(w, x, y, z)` and
     camera-0-frame translation. The harness composes `M = T*R*S`.
   - For a non-reference frame, omit `pose` and leave `moved` absent/false when
     only the camera moved; use the tracking-derived seed verbatim. Set
     `"moved": True` and author a pose only when the object's base genuinely
     moved. Articulation is not whole-object motion.

   The full representation, units, seeding, and export contract is
   [POSE.md](conventions/POSE.md).
7. **Represent relative motion with joints.** Define shared `JOINTS` and each
   frame's `joints` values. Joint origins and axes live in the canonical frame;
   rest state is 0. Declare physically valid `limit` ranges, except for
   continuous joints. Out-of-range states are hard load errors. Keep articulated
   children as separate named parts or rigid groups and follow the schema,
   limits, and forward-kinematics rules in
   [JOINTS.md](conventions/JOINTS.md).

   - **Every frame declares a state for every articulated joint**, including the
     ones you are not changing. A joint state is ABSOLUTE: `0` means "fully shut",
     a MOVE, not a hold. Omitting one is a hard load error naming the frame and
     joint. Contrast rule 6 — omitting a non-reference `pose` IS lossless, because
     a pose order is an increment. Don't carry the habit across.
8. **Keep every part manifold.** Prefer separate watertight solids over a large
   boolean tangle. Union only primitives that form one rigid part; never union
   components that move relative to one another.
9. **Spend geometry on observed shape, not hidden mechanics.** A joint provides
   kinematic attachment, so parent and child meshes need not touch. Do not invent
   hinge leaves, pins, axles, rails, or other support hardware merely to explain
   motion. Model such details only when they are visible or materially affect
   geometry or identity. Keep part placement coherent and avoid gross
   interpenetration in the observed states.
10. **Use depth every iteration when the capture has depth and it is enabled.** Depth is a
    strong-but-noisy guide, not a gate. A large `depth_mae_canon` should first prompt a
    pose check, especially translation Z (the signed `depth_bias_canon` says
    which way). See
    [depth scoring](harness/analysis/scorers/depth.md). Depth supervision can be
    turned off independently of the cameras via `RUN_DIR/depth_config.json`
    (`report` for the scorer/panels, `cost` for the sweep penalty); with it off,
    rely on the silhouette/turntable/critic signals. Tracking-derived camera
    seeds still apply.
11. **Never step blind.** Every iteration, open at least one
    `side_by_side_<frame>.png` from the fresh pass dir and confirm the render reads
    as the **same object** (parts, proportions, landmarks, color) in the **same
    configuration** (base pose, joint states) before accepting the pass or routing
    the next round. Metrics-only steps are forbidden — IoU can rise while identity
    or pose regress. Open more frames when they disagree; log the frame(s) and
    their shape/pose ratings (0–1, step 8) in `NOTES.md`.

The exact declarative `scene.py` contract is demonstrated by
[examples/scene_example.py](examples/scene_example.py). The canonical GLB never
bakes in an observed pose or joint state.

## Tool route
- Normal full pass — render + score the committed state on every frame (the shape loop’s main beat, and a windows round’s verification pass): [shape_pass.sh](harness/utils/shape_pass.md).
- Initial all-frame review: [frame sheets](harness/analysis/viz/frame_sheet.md).
- Dense capture frames INSIDE one temporal step, when a residual is ambiguous
  from its two keyframes alone: [gap sheets](harness/analysis/viz/gap_sheet.md).
- Depth trajectory review after a pass, with or without observed depth — every frame's rendered
  depth in OBJECT UNITS on one shared NEAR/FAR scale, where poses breathing in
  depth read as tiles pulsing across time:
  [depth sheets](harness/analysis/viz/depth_units.md)
  (`python -m analysis.viz.depth_units --pass-dir
  RUN_DIR/iterations/NNNNNN/renders/NNNN`).
- Render views and diagnostics: [harness/views/README.md](harness/views/README.md).
- Metrics, visualization, critic, aggregation, pose comparison, and geometry
  checks: [harness/analysis/README.md](harness/analysis/README.md).
- Pose/joint search: [sweep](harness/views/sweeps/sweep.md).
- Apply a direction order per frame and render + score just that one config (a single-candidate sweep): [apply](harness/views/sweeps/apply.md).
- Object-centric rotation: use [osweep](harness/views/sweeps/osweep.md) or
  [oapply](harness/views/sweeps/oapply.md) to correct individual frames; use
  [oapply_all](harness/views/sweeps/oapply_all.md) only when the model is
  oriented incorrectly in every frame.
- Temporal review — per-frame velocity/acceleration, seam marks, focused views:
  [temporal report](harness/analysis/temporal/report.md). Batched motion
  hypotheses are direct pool orders (`pool.client --request`), one per probe.
- Parallel per-frame refinement (frozen scene.py, one pose-refiner subagent per
  frame window, orchestrator merges): [pose windows](harness/multiagent/windows.md).
  Use it when many frames need the sweep -> inspect side-by-side -> accept/reject
  loop; refining them serially in one context is the known bottleneck. If a
  pose-refiner apologizes for not completing its task, resume or restart it
  against the same brief; without its `poses.json`, it is not complete.

Every render call creates a fresh numbered, immutable pass directory and prints
its path. Use that exact `PASS_DIR`; never substitute artifacts from an older
pass. A critic verdict is valid only for the screened pass and `scene.py` hash.

## The signals
Do not average the signals; each owns a different question:

- **Side-by-side:** the primary shape and identity evidence. Read
  `side_by_side_<frame>.png` at matched scale and judge part inventory,
  proportions, landmarks, distinctive contours, and color. A metric improvement
  cannot override a visibly worse identity or shape match.
- **Silhouette IoU:** the most trustworthy *objective* signal and the hard
  **0.98-per-frame** gate — trust it MOST for the final numbers, but only for exact
  *local* fit once shape and identity are set. Local, depth-blind, and shape-blind:
  it measures overlap, not the right shape. A high IoU on wrong geometry is a trap,
  not a pass; it never decides shape. After shape is established, use the gate field
  (`iou_raw`, or `iou_visible` with a hand mask) to refine exact pose/scale/joint
  values.
- **Depth:** catches geometric placement that IoU cannot, but remains advisory.
- **Turntable:** the hard 3D model/shape gate in every observed joint state. It
  validates canonical geometry and part placement: wrong part depth, gross
  intersections, and hidden-side geometry all fail this gate. It holds
  articulation and moves the camera, so it shows every observed *state* — it
<!-- if module:mechanism -->
  never shows a joint *moving* (see **Mechanism** below).
- **Mechanism:** the hard gate on `JOINTS`. After authoring or changing a joint,
  inspect its blind A/B mechanism sheet and choose the arm that matches the real
  motion, using the interior states where the arms differ. Record the choice;
  a mirrored or `unsure` declaration blocks pose windows and finalization.
  Details: [views/mechanism.md](harness/views/mechanism.md).
- **Your eyes:** read the separate side-by-sides, source, overlap, match, and
  depth-residual images, plus one turntable sheet per articulation state and one
  mechanism page per joint. Depth and IoU are blind to global, semantic errors.
  Do not judge only from numbers or the stacked composite.
<!-- else -->
  never shows a joint *moving*.
- **Your eyes:** read the separate side-by-sides, source, overlap, match, and
  depth-residual images, plus one turntable sheet per articulation state.
  Depth and IoU are blind to global, semantic errors.
  Do not judge only from numbers or the stacked composite.
<!-- endif -->
- **Critic:** holistic and directional. A fresh selected-frame screen is
  **required** to declare the coarse shape checkpoint (see step 4); elsewhere it
  is optional — use it when signals conflict, attribution stalls, or symmetry
  makes a wrong orientation plausible. Never paste its invented magnitude into a
  pose; route its `shape`/`pose`/`joint` hypothesis and measure with a sweep.
  - **[`apply`](harness/views/sweeps/apply.md):** *checks* one named order — a tilt, turn,
    orbit, or open. Composes it onto the frame's pose and scores that single
    silhouette's IoU/depth/side-by-side against the mask. Use it to confirm a
    directional read; use [`sweep`](harness/views/sweeps/sweep.md) to *find* the
    magnitude across a range.

Side-by-side identity, IoU, and turntable coherence are hard visual gates. A
critic cannot override a failing hard gate; a passing IoU cannot clear a wrong
shape, incoherent geometry, or an unreconciled high-severity critic finding.

## Iteration protocol

**Two loops, and you choose which one you are in.** The work has two distinct
shapes, and conflating them is what produces rounds that change geometry and
poses at once and can no longer be judged:

- **Loop S — shape** (you, alone). Author `scene.py` → `shape_pass.sh` on every
  frame → *look* → repeat. Geometry, materials, `JOINTS` definitions, `SCALE`.
  Steps 1 and 3–5 below. Turntables live here.
- **Loop P — pose** (you plus window refiners). Plan windows → spawn one
  refiner per window → merge → apply → a `shape_pass.sh` verification pass →
  **adjudicate** → next round or done. Per-frame base poses and joint states,
  against a FROZEN `scene.py`. Steps 2, 6, 11–12 below, and
  [pose windows](harness/multiagent/windows.md) owns the details.

Steps 7–10 — the pass, the review, acceptance, and routing — are shared: both
loops close every round through them.

Within a single round, stay in one loop: a sweep scores candidates against fixed
geometry, so a round that also edits geometry has scored nothing. Between rounds,
switch freely on the biggest visible defect — they interleave, and this is not a
run-level phase order.

In Loop P, **every temporal step has exactly one owner**: a step interior to a
window is its refiner's, a step crossing windows (a seam) is yours. Each owner
writes a verdict — `coherent` with prose evidence, `flip` as an action item, or
`unsure`, which blocks — stamped against the poses it judged, so re-posing
either side stales it. `merge` enforces the refiners' side; `adjudicate --check`
enforces yours, and its exit code gates both starting a new round and
finalizing.

Repeat until the stopping criteria hold or the iteration budget `MAX_ITERS`
(**300** unless the launch prompt sets another value) is reached. The harness
allocates and records the global iteration number; never count or name
iterations yourself in paths or `NOTES.md`. See
[`harness/BOOKKEEPING.md`](harness/BOOKKEEPING.md).

1. **Decompose all frames and select shape views.** First generate the canonical
   [frame sheets](harness/analysis/viz/frame_sheet.md) with
   `micromamba run -n artscript env PYTHONPATH=harness python -m
   analysis.viz.frame_sheet --run-dir RUN_DIR` and view every page.
   The tool paginates long sequences and prints stable frame IDs; use `--frames`
   only for an additional focused subset, not to avoid the all-frame review.
   Read every frame and list the union of parts, proportions, symmetry,
   materials, moving components, reference orientation, per-frame joint states,
   and whether each non-reference object's base moved. Open individual source
   frames at full resolution whenever sheet scale hides a relevant detail.
   Select **3-5 complementary diagnostic views**: the clearest anchor, a profile
   or complementary angle, views revealing hidden parts, and articulation
   extremes. With four or fewer frames, select all of them. Record why each
   selected view is diagnostic. Re-select whenever evidence warrants (truer
   articulation extremes after alignment, a newly revealed part, a step-8
   outlier) and record the updated set in `NOTES.md`.
2. **Diagnose motion visually.** Before ordering a sweep, inspect all source
   frames and the current side-by-sides when available. Record in
   `RUN_DIR/NOTES.md` each discrepancy's frames, implicated base/joint DOFs,
   direction, and rough magnitude. Up front, add a **"non-smooth motion"** list
   naming every consecutive pair where a large or discontinuous base/joint move
   is visibly expected, with evidence; write `none` when no pair qualifies.
   Once `mesh/pose.json` exists, run the
   [temporal report](harness/analysis/temporal/report.md) on the committed
   sequence and inspect it sorted by `accel.rot`, `accel.trans`, and
   `vel.radial`. Cross-check each sort's outliers against the visual list:
   unexpected large steps are suspects, while expected moves that measure small
   deserve another look. Use
   [gap sheets](harness/analysis/viz/gap_sheet.md) only when the two keyframes
   leave a suspect step ambiguous. Temporal metrics prioritize review; they do
   not select search directions or issue verdicts. Refiners judge their steps at
   `merge`; you judge seams at `adjudicate`.
3. **Initialize motion once, coarsely.** Use the easiest selected view as the
   anchor and perform at most one initial pose/joint round so shape comparisons
   are meaningfully aligned. Preserve the tracking-derived camera seeds and
   existing `moved` semantics. Search only DOFs implicated by visual inspection.
   When facing or orientation is uncertain, test the plausible basins with
   explicit bounds broad enough to cover that uncertainty; follow the
   [sweep manual](harness/views/sweeps/sweep.md) for solver-specific controls.
   Seek gross framing, orientation, and articulation alignment only; do not
   optimize wrong geometry into a high-overlap pose.
4. **Establish coarse shape, correcting pose between rounds as needed.** Keep
   the current pose and joint states as the baseline while refining geometry. If
   side-by-sides show a coherent whole-object or articulated-part displacement
   that prevents meaningful shape comparison, route one visually targeted
   motion round, re-render, and then return to shape refinement.
   The shape checkpoint covers the whole shared model: canonical geometry **and
   joint structure/kinematics** (`JOINTS`). You may and should change joint
   definitions during these model/shape iterations when the current rig cannot
   express the observed motion. Per-frame `joints` values are motion, not shape.

   After every model pass:
   - read `side_by_side_<frame>.png` for each selected diagnostic shape view;
   - inspect each state's turntable sheet for incorrect depth, hidden shape, or
     assembly;
<!-- if module:mechanism -->
   - if this pass authored or changed any joint, read that joint's mechanism
     page and answer its A/B choice — which of the two arms is the real
     mechanism, argued from the interior states where the arms differ;
<!-- endif -->
   - record incorrect proportions or landmarks, missing or extra volume, the
     geometry change that should address each discrepancy, and whether the pass
     visibly improved over the previous accepted pass;

   Declare the **coarse shape checkpoint** when the major parts and overall
   proportions are correct enough in the selected views for pose and joint
   sweeps to be meaningful. The shape **MUST** be semantically/visually coherent
   and plausible. This is not final shape approval: fine contours, landmarks,
   materials, and geometric detail may remain approximate. Before declaring, you
   **MUST** run a fresh critic screen on the selected diagnostic frames
   (`shape_pass.sh RUN_DIR --frames <all> --critic-frames <selected>
   --pass-label coarse-shape`)
   and declare only when no high-severity `shape` finding is unreconciled and the
   critic reads the render as the same, coherent object. The identity/realism
   *scores* and any `pose`/`joint_state` findings do not block here (pose and
   joint states are refined after the checkpoint). A high `joint_definition`
   finding does block: pose refinement is not meaningful until the rig can
   express the observed articulation. With no usable credentials the critic skips, so
   note that in `NOTES.md` and proceed on your own read. Record
   `coarse shape checkpoint: passed` and the visual rationale in
   `RUN_DIR/NOTES.md`; `--pass-label coarse-shape` and bookkeeping identify
   the supporting iteration automatically.
5. **Choose one coherent hypothesis round.** Record the claim, evidence,
   expected cross-view effect, and whether it is a model or motion round. A model
   round may bundle related geometry, material, kinematic structure, or
   shared-scale changes. A motion round keeps that model fixed and updates
   coordinated poses and joints. Do not mix the two **within a single round** (a
   sweep must score candidates against fixed geometry); this is within-round
   purity, not a run-level phase order. After the checkpoint, choose the round
   *type* each iteration by the **biggest visible defect** — shape and motion
   interleave (see step 10).
6. **Order visually directed motion sweeps.** For a motion round, name only the
   pose DOFs and/or joints implicated by visual inspection. Choose ranges that
   cover the visible direction and plausible magnitude. Use broader bounds when
   facing is unknown or the correction may lie outside the local basin, then
   tighten them around a visually credible basin. Do not substitute a generic
   unrestricted search for a clear visual hypothesis. Read the current
   sequence's temporal table first (step 2), so a large or reversing rotation
   widens a range or flips a basin in the hypotheses you write. Two ways to
   order the sweeps, and the choice is about WHO looks at the results:
   - **many frames** — plan a windows round and let refiners order their own
     sweeps and inspect them (step 11). This is the default once more than a
     couple of frames need work;
   - **one directed probe you will inspect yourself** — submit the order to the
     shared pool directly (`python -m pool.client --spool RUN_DIR/spool --id
     <unique> --request '<json>'`, schema in
     [pool/README.md](harness/pool/README.md) § Order). Either way the candidate
     is a proposal until you have looked at the side-by-side.
7. **Render and score every frame:**
   ```bash
   harness/utils/shape_pass.sh RUN_DIR --frames <all frames>
   ```
   Motion-only routine passes make no VLM calls. Run a selected-frame critic
   screen on the coarse shape checkpoint pass (required, see step 4) **and** on
   any shape round or whenever a shape defect or shape regression is suspected —
   the critic is the strongest shape signal and must not go dark for the middle
   of the run. Use selected critic coverage only when warranted; use all-frame
   critic coverage only for a small run or deliberate checkpoint.
8. **Eyes before numbers.** Review this pass's **review set**: the selected
   diagnostic views plus the worst **K** frames by gate field (`iou_raw`, or
   `iou_visible` with a hand mask). Choose K freely, but NEVER below 3. For each,
   open `side_by_side_<frame>.png` before its metrics and log in `NOTES.md` a
   **shape match rating** and **pose match rating** (each 0–1) with a one-line
   observation. Shape = same object (parts, proportions, landmarks, contours);
   pose = same configuration (base pose and joint states — name a mismatched
   joint). The ratings record your read; they never override the images or the
   IoU gate. Then read every turntable state's sheet for coherence. Fix a wrong joint
   origin/axis before base pose; add mechanism geometry only when visibly
   required. Outside the review set, metrics are triage only: a low or
   regressed number pulls the frame into the review set — never act on it
   sight-unseen. Classify discrepancies as shape, pose, joint_state,
   joint_definition, or moved-status.
9. **Accept or reject the pass visually.** Accept a round only when its intended
   side-by-side discrepancy visibly improves without creating a shape or identity
   regression in another frame. Verify a selected motion candidate across all
   relevant frames and turntable sheets. If IoU improves while proportions,
   landmarks, articulation, or identity become visibly worse, reject the round
   and restore the last accepted values. **Never accept a shape change on IoU
   grounds** — IoU is shape-blind, so a geometry edit is judged only on the
   side-by-side, turntable, and critic read. Reject a shape change that produces
   implausible or over-corrected geometry (thinned-to-wire, blobby, or otherwise
   out-of-distribution parts) even when IoU rises; a coherent, real-looking object
   outranks a higher overlap on distorted geometry.
10. **Route the next action.** Before the coarse shape checkpoint, another
   motion round is forbidden unless side-by-sides show a coherent displacement
   or flip on at least two spatially separated landmarks of the same body or
   articulated subassembly.
   Record the landmarks, their common displacement, and why geometry cannot
   explain it before unlocking the relevant pose or joint values. After the
   coarse shape checkpoint, generate and rescore candidates (step 6 — a windows
   round, or a direct pool order for a single probe), read the temporal report,
   choose the trajectory from source evidence, and commit coordinated pose/joint
   updates. No tool chooses for you; smoothness never overrides visible motion —
   it is only a tie-break for when the pictures genuinely tie. Once
   pose and joint states are reliable, return to shape refinement and finish
   contours, landmarks, materials, and geometric detail across all frames.
   Since every pass scores all frames, scan the summary for outliers outside
   the selected views and open just those side-by-sides; if one shows a shape
   defect, re-select the views to include it (step 1) and route a model round.
   If a `shape_pass` (or `multiagent.pool_session`) command is interrupted,
   re-run the **exact same command** — re-invoking is always safe (per-run lock,
   orphan-pool reap, cached-render resume). Never hand-launch a bare
   `pool.manager`; [pool_session.md](harness/multiagent/pool_session.md) owns
   the pool discipline.
11. **Parallelize bulk per-frame refinement.** Once the coarse shape checkpoint
   has passed and `scene.py`/`JOINTS` are frozen, use
   [pose windows](harness/multiagent/windows.md) when many frames need the same
   sweep-and-review loop. Follow that workflow end to end and spawn one
   `pose-refiner` per window in one batch; the document owns the commands,
   concurrency rules, and iteration artifacts.
   - Treat the merge as a proposal. Review `model_review_required`, apply the
     merged poses, then run a full `shape_pass.sh` and accept or reject visually;
     worker model feedback is advisory and does not block applying.
   - Read the pass's self-intersection report and adjudicate every seam or
     routed-up uncertainty against the committed renders. Resolve flips by
     segment; `adjudicate --check` must pass before another round or finalization.
   - Any model or joint-definition edit invalidates the window round; fix it and
     re-plan instead of mixing fragments from different scenes.
12. **Stop unproductive motion refinement.** A motion round succeeds only if it
   visibly improves the targeted discrepancy without a new cross-frame
   regression. After two consecutive unsuccessful motion-only rounds, restore
   the best accepted values and reconsider the visual diagnosis, search range,
   or geometry before ordering another sweep.
13. **Record only semantic decisions.** Bookkeeping already owns iteration
   numbers, timestamps, pass paths, metrics artifacts, window progress,
   kinematics hashes, and Git commits; never duplicate those as a hand-maintained
   ledger in `NOTES.md`. Append the visual hypothesis, reviewed frames with
   shape/pose ratings and observations, swept DOFs/ranges, comparison with the
   previous accepted state, acceptance or rejection, unresolved defects, and
   rationale. Use `bookkeeping status` or `bookkeeping history` for mechanical
   history.

### Shape vs. pose vs. articulation — the central decision
IoU says how far off a frame is, not why:

- **Shape (`build()`):** major landmarks cannot align simultaneously; local
  contours or proportions are wrong; a part is missing, extra, detached,
  under-modelled, or incoherent on the turntable; or a critic reports a `shape`
  finding. A mismatch need not look identical from every view to be geometric.
  Fix coarse geometry before motion refinement, apart from the documented
  initialization and early-correction exceptions above.
- **Pose (`FRAMES[name]["pose"]`):** the whole rigid object shows a common
  displacement at two or more separated landmarks while its internal
  proportions read correctly, genuine base motion occurred, or the critic
  identifies a pose/symmetry error. Set `moved` only for genuine object motion.
- **Joint state (`FRAMES[name]["joints"]`):** the parent remains aligned while two or
  more landmarks on one moving subassembly share a coherent rigid displacement,
  and an existing declared joint can produce that displacement, or critic
  evidence identifies a `joint_state` error.
- **Joint definition (`JOINTS`):** the source shows relative motion that no
  declared joint can produce, or the responsible joint has the wrong
  type/axis/origin/parent/children/limit. This is shared model structure and
  belongs inside the shape gate: change it in a model/shape iteration, then
  rerender every observed joint state before searching per-frame states.
<!-- if module:mechanism -->
  **After authoring or changing any of those fields, read that joint's mechanism
  page** (see Signals).
<!-- endif -->

Use depth direction, aspect-ratio mismatch, cross-frame IoU spread, turntables,
and critic tags to corroborate attribution. A sweep changes only pose and joints,
never geometry. Except for the single coarse initialization and a documented
early correction, do not sweep before the coarse shape checkpoint. Visual
inspection chooses the sweep DOFs and ranges; the search scores candidates
within that hypothesis. Treat weakly pinned DOFs as uncertain, and widen a
search only when visual evidence identifies a pose/joint direction outside the
local basin.

## Stopping criteria
Stop and finalize **only** when every applicable condition holds for every frame, or
report honestly that `MAX_ITERS` was reached. Do **NOT** stop prematurely.

Checklist:
- **Silhouette:** gate metric is at least **0.98** per frame.
- **Coarse shape checkpoint:** the accepted iteration carries the
  `coarse-shape` label; `NOTES.md` records why major parts and overall
  proportions were sufficient for
  reliable motion refinement. The recorded pass is backed by a fresh
  selected-frame critic screen with high-severity `shape`/semantic-identity
  findings reconciled or justified.
- **Final shape refinement:** after pose and joint alignment, all-frame
  side-by-sides match major parts, proportions, landmarks, distinctive contours,
  materials, and assembly structure, with no unresolved cross-frame shape
  regression.
- **Match:** proportions, parts, silhouette, placement, color, and articulation
  read as the source object.
- **Turntable (required):** every observed articulation state is coherent from
  every angle on its sheet — including the raised and dropped views, which are the
  only ones that show a caved-in top or an unbuilt base — with part depth,
  hidden-side shape, and gross intersections reviewed. Unobserved mechanism detail
  is not required.
- **Poses and joints:** values are physically sensible, within limits, and tell
  a sequence-consistent story.
- **Temporal review (multi-frame, hard gate):** every temporal step must have
  been JUDGED by whoever owned it, in writing, against the poses that are
  actually committed. `adjudicate` prints the reads you owe as its own output;
  read every column of a step you call — rotation clean does not mean the step
  is, and a large value in ANY column is fine only when you can name the motion
  the video shows. **When the pictures and the residual disagree, that is a
  pose to fix, not a number to explain.**

  The gate itself is an exit code:
  ```
  python -m multiagent.windows adjudicate --run-dir RUN_DIR --check
  ```
  It must exit 0 — no override exists. Explain each seam's resolution in
  `NOTES.md`; the verdicts live in the active global iteration's
  `windows/seam_calls.json`. See
  [pose windows](harness/multiagent/windows.md) §5 for the ownership invariant
  and the verdict vocabulary.
- **Self-intersection (advisory):** the final pass's
  [self-intersection report](harness/analysis/self_intersection.md)
  (`PASS_DIR/self_intersection.txt`) has been read: every marked overlap is
  explained (a modelled contact) or fixed; non-watertight exclusions acknowledged
  as unchecked.
- **Fresh critic coverage:** when credentials are available, screen a current
  coverage set (normally 4-8 frames) containing the reference, distinct joint
  extremes, worst metric/depth disagreements, and explicit suspects.
- **Critic review (required):** every fresh high-severity `shape`, `pose`,
  `joint_state`, or `joint_definition` item in `critic_review_required` is fixed
  or explicitly justified in
  `NOTES.md`. Stale verdicts never enter this gate; medium/low findings remain
  advisory.
- **Rig review (required before pose windows):** `rig_review_required` is empty.
  A high `joint_definition` finding means the declared rig cannot express the
  observed relative motion; fix JOINTS and rerender before refining states.
<!-- if module:mechanism -->
- **Mechanism pick (hard gate, articulated runs):** every blind A/B joint choice
  is fresh, resolved, and consistent with the declaration;
  `python -m analysis.mechanism_calls check --run-dir RUN_DIR` exits 0.
<!-- endif -->
- **Cross-frame shape review (advisory):** inspect `shape_consensus`; resolve or
  justify repeated shape complaints. See
  [aggregate](harness/analysis/rollup/aggregate.md).


## Finalization lock (non-negotiable)

Do not begin a release/finalization pass merely because progress is poor, time is
limited, a coarse-shape checkpoint passed, or shortcomings can be reported
honestly. A release pass is forbidden until a fresh routine pass proves ALL of
<!-- if module:mechanism -->
the stopping criteria above — including the four hard gates: empty
`critic_review_required`, empty `rig_review_required`,
`multiagent.windows adjudicate --check` exiting 0, AND (on an articulated run)
`analysis.mechanism_calls check` exiting 0.
<!-- else -->
the stopping criteria above — including the three hard gates: empty
`critic_review_required`, empty `rig_review_required`, AND
`multiagent.windows adjudicate --check` exiting 0.
<!-- endif -->

1. Run the release pass and capture its fresh numeric `PASS_DIR`:
   ```bash
   harness/utils/shape_pass.sh RUN_DIR --frames <all> --critic-frames <coverage>
   ```
   Use `--critic-all` only for a small run or deliberate diagnosis.
2. Read every final side-by-side, composite, and every turntable sheet — one page
   per articulation state, every angle of that state on it
   (`turntable_sheets/turntable_sheet_<state>.png`). Selected shape views do not
   replace this all-frame review. Do not claim a match or coherent assembly you
   have not visually confirmed.
<!-- if module:mechanism -->
3. For articulated runs, resolve any outstanding mechanism A/B choices and run
   `python -m analysis.mechanism_calls check --run-dir RUN_DIR`. It must exit 0.
<!-- else -->
3. (The mechanism module is disabled for this run: no A/B pick is owed, no
   `mechanism_sheets/` are rendered, and `analysis.mechanism_calls check` passes
   trivially. The joint declaration is still yours to get right — verify axis
   direction from the turntable's interior states and the side-by-sides.)
<!-- endif -->
4. Run the canonical GLB through
   [check_watertight](harness/analysis/check_watertight.md). Fix failed parts in
   `scene.py`; voxel remeshing is a last resort because it fuses part boundaries.
   **A non-watertight part is not only a mesh defect: it is excluded from the
   self-intersection report, so an excluded joint child means the one pair that
   could expose a wrong hinge is the pair never checked.** The report says so —
   read its `unknown_pairs` line rather than its clean rows.
5. Ensure the fresh [aggregate report](harness/analysis/rollup/aggregate.md) is
   based on the release `PASS_DIR`.
6. Report:
   - final `scene.py`, `object.glb`, `pose.json`, `watertight_report.json`,
     per-frame composites, and `report.json`;
   - final per-frame IoU/depth results and their progression;
   - shared scale, per-frame poses, joints, and states;
   - each joint's declared limit and observed range, flagging `exceeds_limit`;
   - iteration count, final named part list, and articulated children;
   - watertight result per part;
   - explicit all-state coherence confirmation;
   - aggregate moved-flag results;
   - every `critic_review_required` item and its resolution;
   - every `rig_review_required` item and the resulting joint-definition change;
   - every seam verdict from `seam_calls.json` — the step, the verdict, and the
     evidence — and for any `flip`, how it was cleared (which segment was
     un-flipped, and what confirmed the correction by eye);
   - `shape_consensus` findings and resolution of repeated complaints; and
   - an honest per-frame quality assessment and remaining discrepancies.

If time or iterations end with a poor match, failed watertightness, or an
incoherent assembly, say so plainly.

Context/token warning is NOT a stopping condition / task blocker.
