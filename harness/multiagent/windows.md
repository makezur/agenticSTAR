# multiagent.windows — parallel per-window pose refinement with subagents

## Why

**The bottleneck is visual judgment, not rendering.** On a near-symmetric object
the sweep's numeric IoU winners are frequently **silhouette traps**: right
silhouette, wrong facing or basin — a branded cover scoring the same as the ruled
interior behind it. So every candidate has to be *looked at* against its source
frame before it can be committed, and that looking is the scarce resource. Blind
parallel sweeps cannot be trusted no matter how many you run.

Refining a whole trajectory that way is inherently serial in one context: sweep a
frame, open the contact sheet and side-by-side, accept or reject, patch
`scene.py`, verify, next frame. Hundreds of image inspections at a fixed cadence,
and a context that fills up long before the frames run out.

So parallelize the *judgment*, not only the renders: cut the trajectory into
frame **windows** and refine them concurrently, one subagent per window — each in
a fresh small context, each running exactly that loop for its 5–7 frames.

## The contract that makes windows composable

1. **Frozen scene.** `build()` + `JOINTS` + `SCALE` (i.e. all of `scene.py`)
   are frozen at the plan's `scene_sha1`. Refiners estimate each frame's base
   pose AND its per-frame joint *state* — the state is part of the pose, NOT
   part of the frozen `JOINTS` definitions.
   They never edit `scene.py`. The merge refuses to run if `scene.py` changed:
   every fragment carries the plan's `scene_sha1`, and a mismatch is fatal
   rather than a warning — poses fitted against different geometry are not
   poses of the same object.
2. **No pins — non-overlapping windows, seams checked after merge.** Every
   frame is owned by exactly one window and every frame is editable. With
   `scene.py` (and `SCALE`) frozen there is no continuous gauge freedom: each
   frame's pose is determined by its own image evidence up to a *discrete*
   silhouette-basin choice, so the only cross-window failure mode is a basin
   flip at a seam. A pinned boundary would just promote a committed pose —
   itself possibly flipped — into a hard constraint and confuse the refiner.
   Instead the brief shows the committed poses of the window's outside
   neighbors as **advisory** context ("the trajectory should flow smoothly
   through your window; if the image contradicts a neighbor, trust the
   image"), and seam flips are caught by the post-merge temporal gate (§5).
3. **Fragments, not edits.** A refiner writes
   `iterations/NNNNNN/windows/<wid>/poses.json` (pose
   + optional joints + confidence + note per frame, plus the two required
   temporal fields — `expected_motion` and a `step_calls` verdict on every
   interior step; see §5). It never edits
   `scene.py`, `mesh/`, or another window's files. The orchestrator merges and
   is the only writer of the shared state — this is what removes parallel-write
   conflicts entirely. The orchestrator's write is itself mechanical:
   `multiagent.windows apply` regenerates the `FRAMES` literal from the merged
   trajectory (after apply every frame carries an authored pose — previously
   seeded frames are baked to their resolved poses — and comments inside the
   old FRAMES dict are not preserved; keep annotations in NOTES.md). A fragment
   may also carry an optional top-level `report` list: free-text observations
   the refiner made while inspecting but *cannot fix itself* — a neighbor that
   is itself flipped, or a bad source frame/mask. `merge` collects every
   window's `report` into `merge_report.json`'s
   `window_reports` (tagged with the window id, out-of-window frames flagged),
   so a refiner's visual judgment reaches the orchestrator instead of dying in
   an ephemeral final message. Advisory only — a report never fails the merge.
   Every fragment also carries required structured `model_feedback` (`[]` when
   none): `missing_joint`, `wrong_joint`, `geometry`, `scale`, or `input`, with
   severity, frames, part, visible observation, and required relative motion.
   This is how a refiner reports that no permitted base pose or declared joint
   state can explain what it sees. `merge` collects it separately; high-severity
   non-input findings populate the advisory `model_review_required` queue, which
   remains visible through apply and verification but does not block either.
   Every `plan` opens a new global pose iteration and writes its workspace under
   `iterations/NNNNNN/windows/`. Fragments, scratch images, merge/apply
   products, and `scene_before_apply.py` remain with that iteration. A second
   plan is refused until the active iteration is completed or explicitly
   aborted.
4. **One shared render pool, started by `multiagent.pool_session`.** All
   refiners submit orders to a single pool spool (`pool.client --request
   '<json>'`), and nobody starts their own. One pool = one GPU tenant: N
   private pools of resident EEVEE Blenders oversubscribe the GPU. Start it with
   `python -m multiagent.pool_session --run-dir RUN_DIR`, which takes the per-run
   coordinator lock and the per-GPU host lock, reaps any orphan a killed
   predecessor left, and records the pool's process group so a later invocation
   can clean up after it. Do not hand-launch `pool.manager &`: a bare pool holds
   no locks, so a concurrent pass or another run on the same GPU can start a
   second pool.
   Reasoning parallelizes; rendering stays a single right-sized pool
   (`pool.manager.clamp_workers`). The pool fills
   `sweep.hand_mask` from the run layout's `hand_masks_dir` automatically
   (default-ON, per frame), so refiner sweeps gate on `iou_visible` on
   hand-held frames without restating the path; `"hand_mask": ""` inside the
   `sweep` object is the explicit opt-out, and misplaced top-level scoring
   keys are rejected rather than silently dropped (see pool/README.md § Order).
   An order id is immutable for the lifetime of that spool because it names
   `orders/`, `claimed/`, `results/`, and `renders/` artifacts. Use ids such as
   `w02-000160-a1`, and increment the attempt suffix for every retry. The client
   rejects reuse instead of returning an old result or mixing render files.
   `apply` also requires exactly one top-level `FRAMES` assignment. Duplicate
   assignments are rejected instead of silently rewriting the last one.
5. **Every step has exactly one owner, and its owner writes a verdict.** This
   is the invariant the whole temporal layer serves:

   > A step INTERIOR to a window is the refiner's; a step CROSSING ownership
   > (a seam) is the orchestrator's. Every owner writes a verdict on their
   > steps, in a checkable artifact, stamped with a pose hash so re-posing
   > either side stales the verdict mechanically.

   Three verdicts, everywhere, and **no thresholds anywhere**:
   `coherent` (with prose evidence naming the visible motion, or "static", or
   "the basins tie here; kept the side continuous with the neighbors" — the
   smoothness preference lives INSIDE this verdict as a tie-break for when the
   eye reports a tie, never as a number), `flip` (an action item feeding
   segment reconciliation below — and not a verdict a refiner may write on its
   own interior step, since inside its own window it can re-pose the frame),
   and `unsure` (**blocks**; there is no escape flag).

   A large residual is therefore not a defect and a small one is not a
   clearance: a wild jump the video shows gets `coherent` with the motion
   named, and a quiet seam nobody looked at is as unreviewed as a violent one.
   **When the pictures and the residual disagree, that is a pose to fix, not a
   number to explain.**

   No-thresholds has a second consequence that is easy to lose: **no tool here
   picks out which of the report's columns mattered.** `selfcheck`'s call sheet
   and `adjudicate`'s queue name the steps and print the whole table; they do
   not print a headline number per step. That is not terseness — a step shown
   with one number beside it has been answered by whoever chose the number, and
   the reader takes the rest as decoration. Rotation, translation, radial and
   each joint, velocity AND
   acceleration, are all on the table; which of them explains a given step is
   the judgement being delegated. `analysis/temporal/report.md` § "Reading the
   table" is the guide, and it is inlined into every brief for that reason.

   Refiners call their interior steps at `selfcheck` (which stamps the sheet)
   and ship them in the fragment; `merge` validates them, computes
   `seam_steps` by ownership, and routes any refiner `unsure` up so blocking
   happens at exactly one place. The orchestrator then calls the seams at
   `adjudicate` — with the committed-pose renders in front of it (`adjudicate`
   DRAWS the sheet from its `--pass-dir`; both basins look smooth from the source
   video alone, so only the renders show which side is flipped) — and resolves
   any confirmed `flip` with the reconciliation round below.

   **The sheet is drawn, not recommended,** for the same reason the tables are
   printed: a step that needs a second command sometimes does not happen. OPEN
   the page `adjudicate` names and cite its manifest in `evidence`.

   **The two gates are `adjudicate --check`'s exit code**, at the only two
   moments where an unjudged step would become permanent: `plan` refuses a new
   round while the newest one has an open seam (a new round re-poses frames,
   which stales the verdicts the open seam was waiting for — the question
   becomes unanswerable rather than answered), and finalizing requires it
   passing. Nothing lives in `aggregate.py`: this is process bookkeeping, and
   it belongs with the process owner.

   `plan` also refuses a round whose rig has an unanswered or failing
   **mechanism pick** (`analysis.mechanism_calls check`): a wrong joint
   declaration — a mirrored `axis` above all — invalidates every fragment the
   round would produce, and the refiners would chase it with pose sweeps that
   can never fix it. March verdicts are hash-stamped against the declaration,
   so a plan over an unchanged rig re-asks for nothing. A run whose
   `modules.json` has the `mechanism` module off skips this refusal (the gate
   itself reports `disabled`); see `core/modules.py`.

## Flip reconciliation — un-flip a segment, polish only if needed

**Trigger:** a seam you called `flip` — looked at with `seam_sheet --pass-dir`
renders and confirmed against the source frames. A post-merge `seam_steps`
entry is the common case, but a flip surfaced by routine pose-diff triage or at
finalize resolves the same way. The smoothness tie-break does the deciding:
when the two basins tie by eye, un-flip the minority segment toward its
neighbors; only a distinctive feature clearly on the wrong side justifies
keeping a flip. "The object is symmetric" is never a valid excuse to leave the
jump in — under a symmetric hypothesis, commit the smooth trajectory that
corresponds to the real motion.

**The unit is the segment, and the call is the ORCHESTRATOR's.** A flip puts
a whole contiguous segment in the mirrored basin: an interior segment fires
*two* seam flags, one ending at a trajectory end fires one. Fixing seams
independently double-flips the window between them, and the diagnosis needs
the global picture (both seams, the residual table, the reference frame's
basin — the reference is authored against the image, so its side is ground
truth). A refiner can't make this call: from inside the segment the flipped
basin looks self-consistent — exactly why round one missed it.

```
# 1. DIAGNOSE: cluster confirmed flips into a segment — the contiguous frames
#    between a confirmed-flip seam and the next one (or a trajectory end).

# 2. CHECK the shared un-flip on JUST the segment — oapply_all, because the
#    whole point is ONE rotation for the segment (per-frame `oapply` would let
#    each frame pick its own and tell you nothing about the seam). Flip twins tie
#    on IoU: read the panel renders, not the scores.
PYTHONPATH=harness python -m bookkeeping run \
    --run-dir RUN_DIR --kind pose --record-only -- bash -c \
    'harness/render.sh "$HARNESS_RUN_DIR/scene.py" \
        "$HARNESS_ITERATION_DIR/renders" --views oapply_all \
        --frames "<segment frames>" --match-res IMAGE --masks-dir MASKS_DIR \
        --tracking CAPTURE/tracking --oapply flips'

# 3. COMMIT the paste blocks + full shape_pass.sh pass (the flip must land
#    BEFORE any window round is planned — plan gates on committed state).
#    If the pass holds the gates, STOP: a near-exactly-symmetric object
#    often needs no polish.

# 4. Only if the flip dipped IoU: a segment-scoped round to polish the
#    residual (± 1–2 margin frames; --known-issues = the per-frame dips
#    from step 3), then the standard close (merge, apply, verification,
#    temporal gate). PIN THE BASIN IN --guidance: round one's "trust the
#    image over the neighbor" invites flipping straight BACK, and on a
#    symmetric object the mirrored twin often SCORES higher.
... python -m multiagent.windows plan --run-dir RUN_DIR \
        --frames <first>-<last> --frames-per-window 4 \
        --known-issues findings.json \
        --guidance 'the basin here is SETTLED — I un-flipped this segment
                    and confirmed the label reads lower-right. Polish
                    within it: no full-swing grids, no flip panels. If the
                    image contradicts the basin, keep the committed pose
                    and say so in `report` — do not re-flip.'
```

**Delegating the whole un-flip.** Steps 2–4 above are the orchestrator doing it
inline, which is right for a small segment. For a large one — or several at once —
hand the segment to a refiner: state the chosen rotation, the evidence that
decided it, and the order of operations in `--guidance`, and it runs the
`oapply_all` check itself, confirms by eye against the source frames, commits the
corrected poses as its fragment, then polishes within that basin.
```
... python -m multiagent.windows plan --run-dir RUN_DIR \
        --frames <first>-<last> --frames-per-window 4 \
        --guidance 'UN-FLIP: this segment sits in the mirrored basin.
                    1) check the shared correction with an oapply_all
                       "flips" panel over your frames — ONE rotation for the
                       segment, not per-frame;
                    2) confirm by eye against the source: the label must
                       read lower-right, the handle left. Flip twins TIE
                       on IoU, so the score cannot confirm it;
                    3) commit the corrected poses, then polish the residual
                       within that basin.
                    If your eyes say the correction is wrong, keep the
                    committed poses and say so in `report` — never search a
                    third basin.'
```

**Why this is guidance and not a flag.** These two rounds differ from round one
in exactly one bit — whether a basin search is allowed — plus the procedure for
the un-flip. That was three brief-template overlays (`refine`/`polish`/`unflip`)
selected by `--brief`, on the theory that a round type differs in RULES and the
merge would validate against them. **That validation never existed, and it
should not:** a refiner crossing a basin is sometimes RIGHT (the un-flip call was
wrong — which the guidance above explicitly asks it to report), so a merge-time
threshold would reject the correct answer as a contract violation. What verifies a
round is you, reading the temporal report over the merged sequence at
`adjudicate`, with the renders in front of you. The cost of free text is that a
forgotten pin means a wasted round; the gate catches it, because a re-flipped
segment fires the same seams it fired the first time.

## Workflow

```
# 0. Once: a full shape_pass.sh pass exists (windows refine an EXISTING trajectory)
#    and ONE shared pool is up. Start it with pool_session, not a bare
#    pool.manager: it takes the per-run and per-GPU locks, reaps any orphan
#    from a killed predecessor, and records the process group so it can be
#    cleaned up later. Leave it in the foreground for the whole round —
#    Ctrl-C tears the pool down and releases both locks.
micromamba run -n artscript env PYTHONPATH=harness \
    python -m multiagent.pool_session --run-dir RUN_DIR --workers 3
#    (--check answers "is a pool already up for this run?" without starting one)

# 1. Plan windows (opens a global iteration and writes windows/plan.json + briefs).
#    PASS DOWN WHAT YOU ALREADY KNOW: if you have looked at the committed
#    side-by-sides (you should have — e.g. picking which frames need a round
#    at all), put each per-frame finding in a known-issues file instead of
#    hoping the refiner rediscovers it. One line per frame, stating what is
#    wrong with the COMMITTED pose in image terms, e.g.
#      {"000050.jpg": "source is fully closed; committed render is tent-open
#                      at hinge=0.60 — search a closed basin, not local",
#       "000150.jpg": "committed render is 180 deg flipped: label should
#                      read lower-right, handle left"}
... python -m multiagent.windows plan --run-dir RUN_DIR --frames-per-window 6 \
        --known-issues findings.json \
        --review-artifact pose_review_all.png
#    `plan` snapshots these planning inputs under the iteration it opens:
#      iterations/NNNNNN/windows/known_issues.json
#      iterations/NNNNNN/windows/review/<filename>
#    and records them in plan.json. Use temporary/source paths for the command;
#    do not leave pose-round review artifacts loose in RUN_DIR.
#    Reusing history is NEVER implicit. Request it when an older trajectory is
#    useful after a shape/kinematics change:
... python -m multiagent.windows plan --run-dir RUN_DIR \
        --seed-from-iteration previous
#    A specific six-digit iteration may replace `previous`. The generated
#    seed_reconciliation.json records every carried, clamped, dropped, or
#    review-required joint. These are starting hypotheses, never accepted poses.

# 2. Spawn one pose-refiner subagent per window,
#    prompt: "Follow RUN_DIR/iterations/NNNNNN/windows/w01/brief.md".
#    Spawn the round as
#    ONE batch — the windows are independent by construction (frozen scene,
#    per-window fragments), so there is never a reason to run them
#    one-at-a-time. See "Round hygiene" below.
#    Keep the spawn prompt to pointer + round goal; per-frame facts belong in
#    the brief's Known-issues section (via --known-issues), where they
#    survive respawns and reach whichever agent actually does the work.

# 3. Track / merge. The merge validates the TEMPORAL half of the fragment
#    contract too: every window owes `expected_motion` (written from the frame
#    sheet before its first sweep) and a `step_calls` verdict on every INTERIOR
#    step, each stamped with a pose hash. Missing, stale (a frame moved after
#    the call), out-of-vocabulary, or a `flip` a refiner should have fixed
#    itself = a merge violation. A refiner's `unsure` does NOT block: it is
#    routed up into `merge_report.json` `routed_up`, joining the seam queue.
#    A worker saying it could not complete is NOT completion. If it exits
#    without its `poses.json`, resume/restart that worker against the same
#    brief; `progress.json` is its checkpoint. Repeat until the fragment exists.
... python -m multiagent.windows status --run-dir RUN_DIR
... python -m multiagent.windows merge  --run-dir RUN_DIR

# 4. Apply the merge to scene.py mechanically (never hand-paste):
... python -m multiagent.windows apply --run-dir RUN_DIR
#    Review any model_review_required findings, but they do not block apply:
#    worker model feedback is evidence for the verification pass, not a
#    mechanical proof that the merged poses are unusable.
#    (gates on the frozen scene_sha1; rewrites the FRAMES literal from
#    iterations/NNNNNN/windows/merged_poses.json; backup and review copy stay
#    in that same windows directory). Then run a
#    lightweight composite pass — seam verdicts need the COMMITTED renders,
#    which is why adjudication comes after apply:
harness/utils/composite_pass.sh RUN_DIR

# 5. ADJUDICATE: call every step no refiner owned. This prints the forced
#    temporal reads as its OWN output — the SAME table ordered by each quantity
#    a pose defect hides in, then the seams-only view:
#      accel.rot    where a flip announces itself (a spurious flip is a big
#                   rotation that REVERSES, a real turn one that CONTINUES);
#      accel.trans  placement jitter — one mis-posed frame between two good
#                   neighbors is a velocity out-and-back spiking on that frame;
#      vel.radial   depth breathing — an oscillating radial column while the
#                   apparent SIZE never changes is the poses moving, not the
#                   object. Monocular depth is the worst-constrained DOF, and
#                   none of this is visible in a rotation column.
#    Those are orders to read in, not findings: nothing is thresholded, so a
#    small number is not a clearance. It also writes
#    iterations/NNNNNN/windows/seam_calls.json pre-stamped for every seam plus
#    everything `routed_up`, and DRAWS that round's sheet under
#    iterations/NNNNNN/windows/seam_sheets/ (+ manifest), naming the page.
... python -m multiagent.windows adjudicate --run-dir RUN_DIR \
        --pass-dir RUN_DIR/iterations/NNNNNN/renders/NNNN
#    OPEN the page it names (source frames AND committed renders — pass
#    --pass-dir or the sheet cannot answer the basin question at all), write a
#    `verdict` + `evidence` citing the manifest. When a seam's residual is
#    still ambiguous from the seam sheet alone — the renders answer the BASIN
#    question, not whether the motion itself is plausible — sheet the capture's
#    own dense frames inside that step (`python -m analysis.viz.gap_sheet
#    --run-dir RUN_DIR --step A:B`, see analysis/viz/gap_sheet.md) and cite it
#    too. An escalation for the steps that need it, not a per-seam ritual.
#    Then gate on it:
... python -m multiagent.windows adjudicate --run-dir RUN_DIR --check
#    --check exits non-zero until every seam is called, fresh, and not
#    `unsure`. THAT EXIT CODE IS THE GATE: `plan` refuses a new round while it
#    fails, and finalizing requires it passing. No override exists — an
#    `unsure` you can flag past is not a blocker. Resolve any `flip` with the
#    flip-reconciliation round above. Bare `--check` knows no --pass-dir, so it
#    only READS the record: provenance is written by the run that looked.
```

## Round hygiene (concurrency cap)

The failure this section exists for is a **silent fallback to serial**: a
6-window batch spawn hits the agent runtime's concurrent-subagent cap (with
stale agents from the previous round still holding slots), every retry fails
with a thread/agent-limit error, and the orchestrator "recovers" by running the
round one window at a time. That costs hours and gives up the entire point of
the workflow while still paying its overhead. Rules:

- **Release every window agent after `merge`, before planning the next round.**
  In runtimes where a finished subagent holds a slot until explicitly closed,
  leaked agents from round N are why round N+1's batch spawn fails.
- **A thread/agent-limit error means release stale agents and retry the whole
  batch.** It is never a reason to serialize the round — serial windows give
  up the entire point of this workflow while still paying its overhead.
- **Check the cap against the plan.** If `plan` produced more windows than the
  runtime allows concurrently (minus one slot for the orchestrator itself when
  nested), either raise the runtime's cap or split the spawn into two
  deliberate batches — don't fire N spawns and let the tail fail.

## Why not a big batch numeric sweep / N pools?

- Parallelizing *renders* was never the hard part. A tool that only produced
  numeric candidates in bulk has no consumer, because the silhouette-trap
  problem lives strictly *after* the numbers — a candidate is worthless until
  someone LOOKS at it. Windows parallelize the visual accept/reject step, which
  is the scarce one. A directed numeric probe the orchestrator will inspect
  itself is one pool order, not a coordinator.
- N pools would each pay G× resident Blender and contend for the single GPU —
  the crash mode this repo just debugged. The file-spool front door is already
  multi-client-safe (atomic order claim), so N refiners share one pool for free.
- Fragments + orchestrator-merge keeps `scene.py` single-writer, so no locking,
  no merge conflicts, and every change lands through the existing
  verification + temporal gates.

## Files

- `harness/multiagent/windows.py` — the CLI front door (analysis env, no bpy),
  plus the two commands that are about windows *as such*: `plan` (cutting them)
  and `adjudicate` (the steps that fall BETWEEN them), with `status`. Every
  command is reachable as `multiagent.windows <cmd>` and importable from this
  module; the other three live with the artifact they are about, so a reader
  looking for what a fragment means finds the command beside the machinery it
  uses instead of in a file of its own:
  - `fragments.py` — what a refiner ships and how it becomes the trajectory:
    the fragment reader, its validators (`expected_motion`, `step_calls`,
    `model_feedback`), the echo check, the overlay `merge` and `selfcheck` share
    (so a pre-commit check sees exactly what the merge will write), and **both
    of those commands** — they consume the same fragments through the same
    overlay, which is why a refiner's pre-commit table and the merge's are the
    same numbers by construction;
  - `scene_source.py` — reading and rewriting `scene.py`'s FRAMES: the `ast`
    checks that refuse any file shape a line-splice rewrite could get wrong, the
    validation that the rewrite reproduces the merge, and **`apply`**, the one
    command that writes the shared scene;
  - `briefs.py` — rendering the one document a refiner reads: core.md (the ONE
    template) + the per-window facts, with DIRECTIONS.md's sign convention and
    report.md's table-reading chapter inlined (the brief is a refiner's only
    instruction document, so a link is a dead end);
  - `calls.py` / `freshness.py` — the verdict vocabulary and the staleness hash;
  - `workspace.py` — the global iteration's preserved `windows/` workspace.

  Every temporal number comes from `analysis.temporal.report` — one producer, so
  a refiner's table, the merge's seam list, and the orchestrator's read cannot
  be different computations of the same thing. And **no command summarizes which
  of that table's columns mattered**: `selfcheck`'s call sheet and
  `adjudicate`'s queue name the STEPS and print the whole table, because the
  report is deliberately threshold-free and picking the significant column is
  the judgement being delegated, not a service to render. `adjudicate` prints it
  sorted by `accel.rot`, `accel.trans`, and `vel.radial` in turn (see
  `FORCED_READS`) so no single class of defect is the only one the gate can
  see — a basin flip lives in the first, placement jitter and depth breathing in
  the others.
- `harness/multiagent/pool_session.py` +
  [pool_session.md](pool_session.md) — the round's ONE shared render pool,
  started under the per-run and per-GPU locks with its process group recorded.
  Replaces hand-launching `pool.manager &`, which held no locks.
- `harness/multiagent/calls.py` — the temporal-verdict vocabulary
  (`coherent` / `flip` / `unsure`), the "every step has exactly one owner and
  one call" bookkeeping, and the validation both scopes share: a refiner's
  `step_calls` over its interior steps and (step 3) the orchestrator's
  `seam_calls` over the seams are the SAME artifact at two ownership scopes.
  No magnitude is consulted anywhere in it.
- `harness/multiagent/freshness.py` — `pose_hash`: a verdict is stamped against
  both frames' poses, both frames' joint states, and the shared scale, so
  re-posing (or rescaling) stales it mechanically instead of leaving prose that
  still reads fine. See conventions/state_json.md §9.
- the pose-refiner subagent bootstrap — runtime-specific: `.codex/agents/pose-refiner.toml`
  for Codex, `.claude/agents/pose-refiner.md` for Claude Code; it only points the
  refiner at its brief.
  The generated `iterations/NNNNNN/windows/<wid>/brief.md` is the SINGLE self-contained
  instruction document a refiner reads: `_write_brief` renders
  [briefs/core.md](briefs/core.md) — the ONE template — with the run-specific
  facts and this round's `--guidance`, so there is no second doc to fetch (and no
  relative path to break). Editing a rule means editing that template; only
  values/flags live in `windows.py`. There are no per-round-type template
  variants: see "Why this is guidance and not a flag" above.
- `RUN_DIR/iterations/NNNNNN/windows/` — one preserved pose round containing `plan.json`,
  `<wid>/brief.md`, `<wid>/progress.json` (the refiner's resumable draft, where
  `selfcheck` stamps the blank call sheet), `<wid>/poses.json` (refiner-written),
  `merged_poses.json`, `merge_report.json`, `paste_frames.py` (merge-written),
  `scene_before_apply.py` + `apply_report.json` (apply-written),
  `seed_reconciliation.json` (when history was explicitly requested), and
  `seam_calls.json` (adjudicate-written: the orchestrator's verdicts, and what
  both gates read) beside the `seam_sheets/` it was called against.

  **The sheet lives in the round, not the run.** `analysis.viz.seam_sheet`
  defaults to one fixed `RUN_DIR/seam_sheets/seam_001.png` — fine ad hoc, but
  every round overwrote the last, so a run kept only its final sheet while seven
  rounds of `evidence` still cited "the seam sheet". A verdict is hash-stamped so
  re-posing stales it; the picture behind it has to be as durable. `adjudicate`
  draws `iterations/NNNNNN/windows/seam_sheets/round_seams_001.png` plus its
  manifest and records
  the manifest on `seam_calls.json`. Its own basename, so going in close on one
  pair (a single `--seam`, or `--seams-per-page 1`) cannot leave the round's
  evidence a one-band crop of what was judged — that redraw writes to
  `seam_sheets/look/`.
