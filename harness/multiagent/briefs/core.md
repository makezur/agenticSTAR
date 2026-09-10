<!--# CORE TEMPLATE — rendered by multiagent.windows._write_brief into each -->
<!--# RUN_DIR/iterations/NNNNNN/windows/<wid>/brief.md. Placeholders are filled -->
<!--# by string.Template; <!--IF flag--> ... <!--ELSE--> ... <!--ENDIF--> blocks -->
<!--# switch on per-window facts (known_issues, neighbors, guidance). Lines -->
<!--# starting with `<!--#` are template comments and are stripped. -->
<!--# -->
<!--# THE ONLY brief template. There is no per-round-type variant: anything -->
<!--# that changes round to round — "the basin is settled, don't re-search -->
<!--# it", "apply this un-flip first and confirm it" — is ORCHESTRATOR -->
<!--# GUIDANCE, free text passed with --guidance and rendered below. What -->
<!--# verifies a round is the orchestrator reading the temporal report over -->
<!--# the merged sequence at `adjudicate`. -->
# Pose-refinement window ${wid}

## Contract (hard rules)
- `scene.py` is FROZEN at sha1 `${scene_sha1}`. Never edit
  `scene.py`, `build()`, the `JOINTS` DEFINITIONS (each joint's
  type/axis/limit/child), or `SCALE`. You refine each frame's POSE:
  its base orientation/placement AND its PER-FRAME JOINT STATE (how
  open each hinge is in THIS frame). That per-frame joint state IS
  part of the pose you estimate — it is NOT part of the frozen JOINTS
  definitions, so setting it is YOUR job (see recipe C for open/
  closing frames where the base pose and joint state trade off).
- Maintain resumable work in `${progress}` and publish the completed
  fragment to `${frag}` only after temporal self-check. Write no other
  results outside `${wdir}/`. Never touch other windows or `mesh/`.
- Nothing is pinned: every frame listed below is yours to refine.
  The committed poses of your window's outside NEIGHBORS are given
  below as ADVISORY context only — the trajectory should flow
  smoothly through your window, but when the source image
  contradicts a neighbor's committed pose (e.g. it sits in a flipped
  basin), trust the image, not the neighbor. Seam jumps are checked
  after the merge. (If this round is a RECONCILIATION — the
  orchestrator has already settled a basin and wants it kept — the
  guidance section below says so, and it overrides this bullet: polish
  within the settled basin and REPORT a disagreement instead of
  re-flipping.)
- A numeric sweep winner is a CANDIDATE, not an answer: on this object
  silhouette-IoU winners are frequently wrong-facing ("silhouette
  traps"). Before accepting any pose, render and LOOK at the
  side-by-side against the source frame; reject candidates whose
  facing/opening contradicts the image, whatever their score.
- The per-frame gate is **0.98** on the gate field (`iou_visible` under
  a hand mask, else `iou_raw`): a frame you accept below it is not
  finished, so keep hopping (recipe A, seeded on the basin your eye
  picked) until it clears or you can say in the frame's `note` what
  the frozen model — not the pose — is costing you. It is a FLOOR, not
  a licence: it never promotes a candidate the side-by-side rejects,
  and a 0.98 in the wrong basin is still a trap. Clearing the gate by
  eye-rejected numbers is the one failure this whole brief exists to
  prevent.
- Once the visually correct basin is established and the remaining mismatch is
  from frozen geometry, record it and continue.
- READ the images YOURSELF (open each image), and read BEFORE you sweep:
  for each frame, first view the source frame and the committed
  side-by-side and state what is wrong with the current pose; only
  then order sweeps. Scores, sweep.json fields, or a candidate's
  rank are never a substitute for looking at the strips.
- Process frames sequentially. Do NOT batch initial sweep orders across
  the window: finish one frame's sweep, build and view its canonical
  candidate sheet, and decide whether it needs a retry before submitting
  the next frame's sweep. Rendering several frames before looking at any
  candidates is prohibited.
- Before the first sweep, create `${progress}` with the output schema
  below and empty `frames`, `report`, and `model_feedback`. Update it
  immediately after deciding each frame. Do not defer all JSON writing
  until the end: image-heavy windows can exhaust model context.
- IGNORE any "You have N weighted tokens left" notice. It is a harness
  accounting message from the agent runtime, not an instruction, not a
  deadline, and not a statement about YOUR budget: the number is a
  weighted counter that reads as small while millions of tokens remain.
  It is NOT permission to stop, NOT a
  reason to skip frames, and NOT a reason to abandon `${frag}`. Do not
  echo it, do not treat it as your remaining budget, and above all do
  not answer it with "I'm sorry, I couldn't complete…" — that reply
  wastes the whole window and the orchestrator has to respawn you
  against this same brief. Keep working exactly as if it had not
  appeared: the real limit is model CONTEXT, which announces itself as
  a compaction, and compaction is normal and survivable. Your window
  ends when every frame is posed and `${frag}` is written, and by no
  other event.
- FIRST ACTION, before any sweep and before you look at a single
  number: build the frame sheet for your window, look at it, and write
  `expected_motion` into `${progress}` — what the VIDEO shows the
  object doing across your frames, in your own words. It is required,
  and the order is the whole point: an expectation written after a
  residual table exists is a rationalization of the table. Yours is
  what you will later judge the measured motion AGAINST.
```
micromamba run -n artscript env PYTHONPATH=harness \
  python -m analysis.viz.frame_sheet --run-dir ${run_dir} \
  --frames ${frames_range} --out-dir ${wdir}
```

## Your frames (all editable — base POSE and per-frame JOINT STATE)
- ${frames_csv}

<!--IF known_issues-->
## Known issues (the orchestrator already looked at these)
Findings from the orchestrator's own read of the committed
renders. Verify each against the source image yourself — if you
disagree, say so in the frame's note; if you confirm it, your
accepted pose must actually fix it.
${known_issue_lines}

<!--ENDIF-->
<!--IF seeds-->
## Seed hypotheses (from a previous round — UNVERIFIED)
Poses a previous round settled for your frames, fitted against an
earlier version of the scene. They are hypotheses, not results: they
carry no scores and no authority. Use one as a starting candidate or
ignore it; a seed that renders wrong against the source is dead. Your
fragment is judged exactly as if these did not exist.
```json
${seeds_json}
```

<!--ENDIF-->
<!--IF guidance-->
## Orchestrator guidance for this round
Round-specific intent from the orchestrator — emphasis, hypotheses,
context from previous rounds, and what kind of round this is. It shapes
where you spend effort, and where it is more specific than the general
rules above it WINS: it is written by the one agent that has seen the
whole sequence, so "the basin here is settled, do not re-search it" or
"apply this un-flip first, then polish within it" is an instruction, not
a suggestion. The frozen-scene and fragment-only rules of the contract
are the exception — those never bend.
${guidance}

<!--ENDIF-->
## Neighbor context (ADVISORY — outside your window, not yours to set)
<!--IF neighbors-->
```json
${neighbors_json}
```
<!--ELSE-->
(none — this window is the whole sequence)
<!--ENDIF-->

## Inputs
- source frames: `${frames_dir}`
- object masks:  `${masks_dir}`
${hand_masks_line}
- committed trajectory: `${pose_json_path}`

## Rendering (shared pool — do NOT launch your own Blender pool)
A render pool serves the spool at `${spool}`. Submit sweep/
match orders and wait for results:
```
micromamba run -n artscript env PYTHONPATH=harness \
  python -m pool.client --spool ${spool} \
  --id ${wid}-<frame-stem>-<attempt> \
  --request '<request JSON>' --timeout 900
```
Order IDs are immutable output identities. NEVER reuse an ID already
present in this spool, even for a retry: the client rejects reuse so an
old result or render directory cannot be mistaken for the new order.
Request JSON mirrors pool/serve.py's schema; results land in
`<spool>/renders/<order-id>/` — the report (sweep.json / sweep.txt),
the winner's own render (sweep_best.png), and the panels the pool
builds for you: candidate_sheet_NNN.png pages plus one
side_by_side_<image>.png per candidate. Pick the RECIPE that matches
the frame's situation (all share the envelope
`{"frame": "<frame>.jpg", "views": "sweep",
"provenance": ${provenance_json}, "sweep": {...}}`).
Keep that `provenance` object on every order. It does not affect rendering;
it makes the iteration and window survive spool archival and copied results.
Masks: the pool fills BOTH scoring masks from the run's layout when
your order doesn't state them — `sweep.mask` (the object mask, which a
sweep REQUIRES) and `sweep.hand_mask`, so occluded pixels are waived
and the sweep gates on `iou_visible` (check sweep.json's `gate_field`).
The recipes below still spell `mask` out, and stating it is always
fine. To opt out of the hand waiver on a frame, set `"hand_mask": ""`
INSIDE the `sweep` object (scoring keys never go at the request top
level — the pool rejects unknown top-level keys).

### Orientation grammar — two frames, never mixed
Describe what you SEE in CAMERA-ALIGNED terms — the same vocabulary the
Known-issues findings speak, and exactly the `ranges` DOFs of a `sweep`
order, so a visual read translates directly into a search:
- `roll`  — in-plane spin about the view axis (+ = CLOCKWISE on screen);
- `yaw`   — orbit about camera-up (+ = the object's LEFT edge swings
  CLOSER to the camera, showing more of its left side);
- `pitch` — orbit about camera-right (+ = the object's TOP edge swings
  CLOSER, tipping the top toward the camera);
- `dpx` / `dpy` — image-plane shift right / down (fraction of frame
  width / height);
- `tz`    — closer / farther (fraction of the object's depth);
- `<joint>` — the ABSOLUTE joint state (revolute joints: degrees),
  clipped to the declared limit.
You MUST reason in DEGREES and a DIRECTION, not vague words: you are the
one who orders the search, so your estimate is what sizes AND aims it.
For each frame, state a rough magnitude AND which way per axis
("~20 deg yaw, left edge closer; ~5 deg roll cw; top pitched ~10 deg
away") BEFORE choosing a recipe; the estimate maps directly onto the
span you order:
- up to ~5 deg  -> a `tiny` band (±5) — last-mile touch-up, sign no
  longer matters;
- ~5–15 deg     -> a `standard` DIRECTIONAL band — recipe A's default;
- ~15–30 deg    -> a `large` directional band;
- ~30–45 deg    -> a `huge` directional band, or `center` the estimated
  move and search a directional band around it;
- ~90 / ~180 deg (a quarter-turn / a flip) -> not a band at all: a
  DISCRETE basin problem — recipe B.
PREFER a DIRECTIONAL preset over a symmetric one (except `tiny`): you
saw which way the pose is off, so spend the whole budget there. A
directional preset is `axis:PRESET_dir` — `roll:standard_cw` /
`_ccw`, `yaw:standard_leftcloser` / `_rightcloser`,
`pitch:standard_topcloser` / `_bottomcloser` (the token names which
EDGE swings toward the camera; the verified table is a few lines down,
just before the recipes). It searches
one-sided, e.g. `pitch:standard_topcloser` -> `pitch ∈ [0,+30]`, same
render count as symmetric ±15 but none wasted on the direction you
ruled out.
A sweep whose window doesn't cover your estimate just re-finds the base
pose and looks "stuck"; a symmetric full-swing grid where a directional
band would do wastes renders and invites silhouette traps. Say the
number AND the direction, size the band to cover it with margin, and
let refine polish the rest.
There is NO scale DOF: `SCALE` is global and frozen for a motion
round, so the grammar rejects `sigma` outright. Sweep `tz` when the
apparent SIZE is off — in a monocular view size and depth trade off.

The OBJECT-CANONICAL grammar is a DIFFERENT frame: `rx`/`ry`/`rz`
rotate about the object's OWN axes (+X right, +Y front, +Z up) as one
rotation vector, and `flip:x`/`flip:y`/`flip:z`/`flips` name discrete
180-deg flips about them — the language of the object-centric
`osweep`/`oapply` views. These are NOT the camera-frame `yaw`/`pitch`
(they only coincide for an upright object; the parser redirects you if
you mix the two grammars).

READ THE TOOL DOCS before your first order — the recipes below are
starting points, not the full grammar:
- `${directions_md}` —
  what a POSITIVE `roll`/`yaw`/`pitch` (and each directional preset) does
  on screen. Pasted in full immediately below;
- `${sweep_md}` —
  camera-frame DOF semantics, the `angle_preset` sizes (tiny ±5 /
  standard ±15 / large ±30 / huge ±45 deg) and their DIRECTIONAL
  one-sided variants, and `center` (compose a directional move, then
  search around it);
- `${osweep_md}` +
  `${oapply_md}` — object-canonical
  `rx/ry/rz`, the `flips` panel, `preset` (`z:quarters` / `z:full` — one axis
  plus a range of angles, no triples to compute), and what to paste. Both are
  PER-FRAME: every frame gets its own rotation, so a one-frame order is just an
  ordinary single fit. (`${oapply_all_md}`
  applies ONE shared rotation to every frame — a build-orientation fix, not a
  per-frame refinement; you almost never want it inside a window.)
- `${pool_md}` § Order — the full
  pool request schema (which keys go inside `sweep`/`oapply`/`osweep`,
  never top-level);
- `${report_md}` — the
  temporal table you must read before writing a step verdict: what
  each column (rotation / translation / radial, velocity AND
  acceleration, per joint) is telling you. Its reading chapter is
  pasted at the end of this brief (## Reading the temporal table), so
  you need not open it.

### The VERIFIED screen-direction convention
From `${directions_md}`. Trust it over any sign you remember.

${directions_body}

### The recipes

**A. Local descent** — facing already correct, error is a small
roll/pitch/framing residual. When you saw WHICH WAY each axis is off,
aim the band: leave `ranges` empty and name a DIRECTIONAL
`angle_preset` per axis so the whole budget is spent on your side
(here: top tips toward the camera, spins clockwise):
```json
{"strategy": "grid", "mask": "<masks_dir>/<frame>.png",
 "angle_preset": "pitch:standard_topcloser;roll:standard_cw",
 "refine": 2, "refine_shrink": 0.4, "dump_topk": 6}
```
When you truly can't call the direction (rare), an explicit symmetric
range is the fallback. Either way ALWAYS include a refine stage (a
5-step grid alone bottoms out at ~8 deg cells and will look "stuck";
never submit a local fit with `"refine": 0`):
```json
{"strategy": "grid", "mask": "<masks_dir>/<frame>.png",
 "ranges": "roll:-17,17,5;yaw:-17,17,5;pitch:-17,17,5",
 "refine": 2, "refine_shrink": 0.4, "dump_topk": 6}
```

**B. Basin / facing search** — wrong face suspected (flipped basin,
suspect neighbor context). Three escalating scopes; take them in
order, because each is far cheaper than the next and symmetry
usually surrenders to the first.

**B1. NAME the discrete candidates (`oapply`).** An object-centric
flip panel renders identity + the three 180-deg flips about the
object's own axes, scoped to just this frame — four legible renders
instead of a 245-cell grid:
```json
{"frame": "<frame>.jpg", "views": "oapply",
 "oapply": {"oapply": "flips", "frames": "<frame>.jpg"}}
```
Flip twins routinely TIE on IoU — decide by eye from the renders
(facing, branding, hinge side), never by score. A custom candidate
set works too (e.g. "identity|flip:z|rz:90"), and when the question
is "which QUARTER-TURN about the up axis?" name the angle range
instead of listing it: `{"preset": "z:quarters"}` (0 included, so
you also learn whether turning at all helped).

**B2. SEARCH the object's own axes (`osweep`).** When you can say
the frame is turned about one of the OBJECT's axes but not by how
much — "it faces backwards-ish", "it is lying on its side" — the
question is a canonical rotation with an unknown magnitude, and
`osweep` searches exactly that: a grid over `rx`/`ry`/`rz`,
right-multiplied onto the frame's own pose, per frame,
independently. A camera-frame `yaw` cannot express it ("turn it
about its own up-axis" is only a yaw for an upright object), which
is why the named flips of B1 and this are the same grammar and the
camera-frame grid below is not.
```json
{"frame": "<frame>.jpg", "views": "osweep",
 "osweep": {"preset": "z:full", "frames": "<frame>.jpg"}}
```
The settings go in an `osweep` object, NOT in `sweep` — the
object-centric views have their own key, and the masks come from the
run layout. Its keys are exactly `preset` / `ranges` /
`angle_preset` / `frames` (+ `visuals`); anything else is rejected.
`preset` names the angle set so you never hand-compute a triple:
`z:flip` (0/180), `z:quarters`, `z:octants`, `z:full` (whole
circle) — every set includes 0, so the no-op competes and you learn
whether turning helped at all. `ranges` is the explicit form
(`"rz:-180,180,9"`, or one-sided `"rz:0,60,7"` once you know the
sign) and beats `preset`. There are NO directional preset tokens
here on purpose: a
positive canonical rotation has no fixed on-screen sense, so a
screen-relative word would be a lie in some frame. Read the
per-frame rotation column and the renders, not the mean.

**B3. The camera-frame full swing.** Only if no canonical rotation
explains the facing. Coarse, no refine — you are choosing a basin by
EYE from the contact sheet, not polishing a number:
```json
{"strategy": "grid", "mask": "<masks_dir>/<frame>.png", "quality": 0.25,
 "ranges": "roll:-180,180,7;yaw:-180,180,7;pitch:-90,90,5",
 "refine": 0, "dump_topk": 24}
```
Whichever settles it, follow up with recipe A seeded on the visually
right basin.

**C. Coupled pose + joint** — open/closing frames where base
orientation and the joint state trade off against the silhouette.
Sweep them TOGETHER (`"space": "both"`), moderate spans, refine on:
```json
{"strategy": "grid", "mask": "<masks_dir>/<frame>.png", "space": "both",
 "ranges": "roll:-30,30,5;yaw:-30,30,5;<joint>:<lo>,<hi>,7",
 "refine": 1, "refine_shrink": 0.4, "dump_topk": 6}
```

**D. Placement only** — pose and joint visually right, object is
shifted / too near / too far. Image-plane + depth, orientation held:
```json
{"strategy": "grid", "mask": "<masks_dir>/<frame>.png",
 "ranges": "dpx:-0.08,0.08,7;dpy:-0.08,0.08,7;tz:-0.15,0.15,7",
 "refine": 1, "refine_shrink": 0.4, "dump_topk": 6}
```

Retention: ordinary local, coupled, and placement sweeps keep at most
six candidates (dump_topk 6, as the recipes state) so the canonical
review fits one readable page; only a wide basin search may use the
larger count its recipe states.

Flat-score caveat: on hand-held/near-symmetric frames the IoU
surface is often FLAT — hundreds of candidates within ~0.01. There a
higher number is noise, not signal: choose between near-ties by the
side-by-side (facing, opening, landmarks), or keep the committed
pose and say so in the note.
EVERY PANEL IS ALREADY BUILT when the result lands — do NOT run a
panel tool as a routine step. The pool writes both media into
`<spool>/renders/<order-id>/`, from the same four columns:
- `candidate_sheet_NNN.png` + `candidate_sheet_manifest.json` —
  several candidates per page: WHICH of these is right;
- `side_by_side_<image>.png` — one candidate across the full width:
  is THIS one right, at the size where facing and branding read.
The result's `panels` block names the default inspection files
(`pages`, `strips`) and their native counterparts (`original_pages`,
`original_strips`). With a configured preview cap the defaults end in
`_preview.png`; otherwise each default and original path is the same.
`strips_available` appears when more candidates were rendered than
got a strip; the sheet still has every one of them.

Columns are `source,render,overlap`; overlap is
green=agreement, yellow=render only (extra), magenta=source-mask only
(missing), blue=our render waived under the hand, gray=ignored hand.
Select candidates only by the IDs printed on the rows and in the
manifest, never by remembering page/row position.

Re-panel by hand ONLY for something the automatic pass does not give
you — a different column set, a bigger height, or several orders
rolled onto one page:
```
micromamba run -n artscript env PYTHONPATH=harness \
  python -m analysis.viz.candidate_sheet \
  --render-dir <spool>/renders/<order-id> --run-dir ${run_dir} \
  --rows-per-page <N>
```
(`analysis.viz.sweep_sides` is the same for per-candidate strips.)
Choose `--rows-per-page` so landmarks remain readable; `--no-overlap`
omits the overlap, `--columns` sets an exact subset/order, and RENDER
is mandatory (`match` is accepted as its alias).

Legibility beats compactness: facing, opening, and whether branding
reads forward or mirrored vanish in dense sheets. Cap a sheet at six
candidates and aim for tiles at least ~360 px tall; split larger sets
across pages. Review existing sheets before opening individual
renders. Start with the `pages`/`strips` preview path. If facing,
branding, a seam, or chirality remains unresolved, open the matching
`original_pages`/`original_strips` file at native image size. When
facing/chirality is unresolved between near-IoU twins, compare their
native candidate renders directly with the source; forward-reading
branding breaks the tie, not score.

Do NOT create or relabel candidate sheets with ad-hoc PIL, ImageMagick,
shell, or notebook code when the harness tool supports the required
sheet. Use an ad-hoc fallback only if a required input/presentation is
unsupported or the tool fails. Before doing so, state exactly what is
missing or failed; preserve report candidate IDs/ranks verbatim and
write an auditable row-to-candidate mapping.

## Per-frame loop
1. Open the SOURCE frame, then the committed side-by-side, and
   state the defect in camera-aligned DEGREES per axis (see the
   orientation grammar) — that estimate sizes the band you order.
   Classify: small residual -> recipe A; wrong facing / a ~90/~180
   discrete turn -> B (then A); hinge/base trade-off -> C;
   framing/depth only -> D.
2. Open the order's candidate sheet, then the
   side_by_side_*.png strips it points at — both are already in the
   order dir, nothing to build. Compare each candidate against the
   source frame (facing, opening, occlusion). Reject silhouette traps.
3. If rejected, sweep the alternative basin explicitly (recipe B) or
   keep the committed pose.
4. To go again on the SAME frame, descend from the candidate you
   picked — do NOT retype its numbers. Send a `seed` naming that
   candidate and the pool fills the placement and the FULL joint state
   for you at claim time:
```json
{"id": "<new-unique-id>", "views": "sweep", "frame": "<frame>.jpg",
 "seed": "<spool>/renders/<prev-order-id>/sweep.json#rank=<N>",
 "provenance": ${provenance_json},
 "sweep": {"strategy": "grid", "ranges": "yaw:-6,6,7",
           "refine": 1, "dump_topk": 6}}
```
   This is how you fit 6 DOFs on an affordable grid: sweep two or three
   DOFs densely, pick by eye, then sweep the next few STARTING FROM YOUR
   PICK. Each order's `sweep.txt` ends with a ready-made **NEXT HOP**
   block — copy it and fill in the rank.
   - `<N>` is the **`rank`** column, not the `row` column. The listing is
     gappy (topk cuts it, then panelled candidates from below the cut are
     appended keeping their true rank), so line 10 is often rank 13.
     `#best` and `#candidate_id=<id>` also resolve.
   - It will usually NOT be rank 0. You are overruling the numeric winner
     on purpose whenever the strips say so; `seed` exists so that costs
     you nothing.
   - `seed` carries the pose AND every joint state, together. Writing
     `sweep.pose_start` by hand does NOT: placement and articulation live in
     different keys, so a hand-written hop that forgets `joints` silently
     reverts the joint to the frame's committed state while the pose
     carries — and the render still looks plausible. Use `seed`.
   - An explicit `sweep.pose_start`, `pose`, or `joints` beside a `seed` WINS, so
     you can override one half and keep the other. (The seed's placement
     lands in `sweep.pose_start` for a sweep/apply order, top-level `pose`
     otherwise — every view is seedable.)
   - A `seed` that cannot resolve FAILS the order with the reason. It
     never falls back to the committed pose.
   - A seed is FRAME-SCOPED: seeding frame N from frame M's record is
     refused, because that is usually a copy-paste slip. To do it on
     PURPOSE — "start this frame from the neighbour I just fitted", which
     is often a much better grid centre than the frame's committed pose —
     add `"seed_cross_frame": true` beside the `seed`. The pose lands
     VERBATIM: the camera moved between the two frames and nothing
     compensates for that, so widen the grid to cover the step (and
     expect to spend a hop on placement, recipe D).
   - You never state `scale` — not in a `seed`, not in a hand-written
     `sweep.pose_start`. The shared `SCALE` is the object's and the engine
     applies it; a pose start that states a DIFFERENT scale is refused.
   - Stop hopping when a hop no longer changes what you see. For DE, compare
     the restart finalist renders directly; the harness deliberately provides
     no per-DOF landscape or boundary status from its adaptive samples.
   - Caveat: descent handles ONE DOF group at a time, so it cannot escape
     a COUPLED valley (base orientation trading off against a hinge). For
     those, sweep them together with recipe C rather than alternating
     hops.
5. Record the accepted pose AND its per-frame joint state(s) in the
   progress file immediately, with a confidence and a one-line note
   (what you rejected and why). A frame whose hinge/lid opening was wrong
   is not fixed
   until its `joints` state is set, not just its base pose.
   A progress entry is itself a `seed` source — `"seed":
   "${progress}#<frame>.jpg"` resumes the descent from what you last
   accepted, which is also how a replacement agent picks up mid-frame
   rather than starting the frame over. So write after EVERY hop you
   accept, not only when the frame is finished.

## Temporal self-check and your STEP CALLS (before you finish)
Once your fragment has the poses you believe are right, run:
```
micromamba run -n artscript env PYTHONPATH=harness \
  python -m multiagent.windows selfcheck --run-dir ${run_dir} --window ${wid}
```
It overlays YOUR poses on the committed trajectory and prints the SAME
temporal table the orchestrator reads after the merge, scoped to your
window plus its committed neighbors — so a non-smooth step or a seam
flip surfaces now, while you can still re-pose. It also STAMPS a call
sheet into `${progress}`: one `step_calls` record per interior step of
your window (${n_steps} of them), with the measured numbers and a
`pose_hash` filled in and the verdict blank.

**READ EVERY COLUMN of that table, not just the rotation.** No tool
picks the important one out for you — deliberately, because there is no
threshold that could. A step can be clean in `rot` and still be wrong:
- `rot` — orientation change. An isolated spike is the signature of a
  pose in the mirrored basin, not the object whipping around;
- `trans` / `radial` — how far it moved, and the signed part of that
  along the line of sight. DEPTH is what a single camera constrains
  WORST, so pose error surfaces here: an oscillating `radial` column
  while the object's apparent SIZE in the frames never changes is your
  poses breathing in depth, not motion. Fix it with recipe D (`tz`);
- the ACCELERATION columns — where the motion CHANGED. One mis-posed
  frame between two good neighbors is a velocity out-and-back, which
  lands as a spike exactly on that frame;
- each joint's state/vel/accel — articulation in its own units.
The full guide is pasted at the end of this brief (## Reading the
temporal table).

**Then LOOK at your depth trajectory — the DEPTH SHEET is required.**
The `radial` column is depth as numbers; this is depth as pictures, and
a single camera constrains depth WORST, so it gets its own look. Render
your accepted poses' depth (one order per frame of your window, seeded
from your fragment so it is YOUR pose, not the committed one):
```
micromamba run -n artscript env PYTHONPATH=harness \
  python -m pool.client --spool ${spool} \
  --id ${wid}-depth-<frame-stem> --timeout 900 --request \
  '{"frame": "<frame>.jpg", "views": "depth", "seed": "${progress}#<frame>.jpg"}'
```
then sheet them on ONE shared NEAR/FAR scale and OPEN it:
```
micromamba run -n artscript env PYTHONPATH=harness \
  python -m analysis.viz.depth_units --scan ${spool}/renders \
  --frames ${frames_range} --out-dir ${wdir}/depth_sheets
```
One colour = one distance on EVERY tile (the header names the scale).
The read: across your window the object should walk the colormap the
way the video shows it moving. A tile that jumps blue↔red against its
neighbors while the object's apparent SIZE in the source frames never
changes is a pose breathing in depth — that is recipe D (`tz`), a pose
to fix, not a number to explain. An all-black tile means that pose left
the frustum entirely. Name the sheet in the affected steps' `evidence`.

**You must call every one of your ${n_steps} interior steps.** For each,
re-open the SOURCE frames either side and your accepted renders, then
write one of:
- `coherent` — plus `evidence` in prose that names what the pictures
  SHOW: the visible motion ("lid swings up, hinge line clears the
  rim"), or `static`, or "the two basins tie here; I kept the side
  continuous with the neighbors". A LARGE number is perfectly coherent
  when the video shows the motion — a real throw of the lid is not a
  defect, and "it's big" is not an argument by itself. Compare against
  the `expected_motion` you wrote before sweeping: if the measured
  motion matches what you predicted from the frame sheet, say so.
- `unsure` — you genuinely cannot explain what you see. This ROUTES UP
  to the orchestrator and does NOT block your merge, so use it instead
  of inventing an explanation. Say what you tried and what confused you.

When a step is AMBIGUOUS from its two endpoints — you are about to
write `unsure`, or an expected large move measured small, or a spike
your frame sheet did not predict — first inspect its 6-DOF pose sheet:
```
micromamba run -n artscript env PYTHONPATH=harness python -m \
  analysis.viz.pose_pair_sheet --run-dir ${run_dir} --step FRAME_A:FRAME_B
```
If the motion itself is still unclear, sheet the frames BETWEEN the keyframes:
```
micromamba run -n artscript env PYTHONPATH=harness \
  python -m analysis.viz.gap_sheet --run-dir ${run_dir} \
  --step FRAME_A:FRAME_B --out-dir ${wdir}/gap_sheets
```
The dense frames often turn an `unsure` into a confident `coherent`
("the lid really does slam in those 10 frames") or a confident re-pose.
This is an escalation, not a routine step: do not build one per step —
the endpoints plus your `expected_motion` settle most steps. If you use
one, name its sheet in the call's `evidence`.

You may NOT write `flip` on your own interior step: inside your own
window you can re-pose the frame, so a flip you spotted is a flip you
FIX. (`flip` is a verdict only the orchestrator writes, at seams.)
**When the pictures and the residual disagree, that is a pose to fix,
not a number to explain.** If a step is only explicable by "the pose is
wrong", re-pose it and re-run selfcheck rather than writing prose about
it.

Never type a `pose_hash` yourself — selfcheck computes it from the poses
you actually have. It exists so that re-posing a frame invalidates the
verdicts about it: re-run selfcheck after ANY re-pose and the affected
calls come back blank for you to redo. The merge refuses a call whose
hash no longer matches, so a verdict can never outlive its evidence.

A step marked as touching a neighbor OUTSIDE your window is a SEAM: it
is not yours to call (the orchestrator owns it), but judge it against
the image anyway and do not blindly match the neighbor — the neighbor
may itself be in a wrong basin. If you believe it is, say so in
`report`.

Copy the finished `step_calls` from `${progress}` into `${frag}`.

## Committed starting poses
```json
${committed_json}
```

## Output fragment schema
Write this structure incrementally to `${progress}`. After self-check,
write the completed structure to `${frag}`.
```json
${schema_json}
```
Omit frames you did not improve (they keep the committed pose).
`expected_motion` and `step_calls` are REQUIRED — the merge refuses a
fragment that is missing either, or whose step calls are incomplete,
stale, or out of vocabulary. Everything else about them is above.
Write the quaternion the sweep gives you as-is — do NOT hand-normalize
it. The harness normalizes on read (six-decimal JSON drift is harmless)
and prints a loud WARNING only if a quaternion's norm is far enough from
1 to be genuinely malformed (e.g. a rotation vector pasted into the
(w,x,y,z) slot); if you see that, fix the value, don't normalize it.

## Reporting what you cannot fix (optional `report`)
You are doing the visual inspection anyway, so you will sometimes SEE a
problem that is not yours to fix. Do NOT try to fix it and do NOT bury it
in a final message that gets lost among the other windows — put it in the
fragment's top-level `report` list and the merge hands it straight to the
orchestrator (merge_report.json `window_reports`). Use it for, e.g.:
- a NEIGHBOR (outside your window) that looks itself flipped/wrong, so
  your seam step is real and the fix belongs on the neighbor's side;
- a bad input: a blurred/mislabeled source frame, or a mask that leaks
  or clips the object, that no pose can satisfy.
Name the `frame` when the observation is about one (a neighbor you don't
own is fine — it's flagged, not rejected). This is separate from a
frame's per-frame `notes`, which record what YOU rejected and why in a
pose you DID set.

## Frozen-model feedback (required `model_feedback`)
Your fragment MUST contain a top-level `model_feedback` list; write `[]`
when the frozen model is sufficient. Do not hide model defects in frame
notes or the advisory `report`. Add one structured item when no base pose
or declared joint state can explain the source:
```json
{"kind": "missing_joint|wrong_joint|geometry|scale|input",
 "severity": "high|med|low", "frames": ["<frame>.jpg"],
 "part": "<part>", "observation": "<visible evidence>",
 "required_motion": "<relative motion or geometric correction>"}
```
`missing_joint` means rigidly attached parts visibly move relative to
each other but no declared joint can produce that motion. `wrong_joint`
means a declared joint has the wrong axis/origin/type/children/limit.
Use `geometry` or `scale` only when pose and declared joint searches
cannot explain the mismatch. Model findings are ADVISORY: they never block
pose application; the orchestrator reviews them and decides whether to
fix/replan before accepting the merge. `input` findings are advisory too.

## Finishing
The only deliverable is `${frag}` in the schema above, and it is not
finished until every interior step carries a fresh verdict. The order
is: pose every frame -> selfcheck -> re-pose anything the table exposes
-> selfcheck again -> write the verdicts -> copy to `${frag}`. Keep
your final message to a three-to-six-line summary: frames updated,
frames kept, traps rejected, any step you called `unsure` and why, and
anything reported upward — everything of substance belongs in the
fragment, not the message.

## Reading the temporal table
Pasted from `${report_md}` so you need not open it. This is what turns
`selfcheck`'s table into a verdict: what each column measures, which
ones a pose defect hides in, and the two things the numbers cannot tell
you themselves. Nothing in the table is thresholded or flagged — a small
number is not a clearance and a large one is not a defect, which is why
reading it is YOUR job and not a tool's.

${report_reading_body}
