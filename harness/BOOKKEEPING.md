# Unified iteration bookkeeping

Every reconstruction run has one numeric iteration sequence:

```text
RUN_DIR/
  .git/                         private source/pose history
  .harness/bookkeeping.json     current iteration pointer
  iterations/
    000001/
      iteration.json
      scene.py
      pose_start.json
      pose.json
      renders/0001/
      windows/w01/
      composites/
        000010.png
        000020.png
      pose_pair_sheets/
```

The harness owns the numbers. Labels such as `coarse-composite` are metadata,
never directory names.

`pose_start.json` records the trajectory at iteration open; `pose.json` records
the verified result. These are compact history snapshots containing scale,
reference frame, canonical joint definitions, and per-frame object pose, joint
state, and moved status. Fixed camera tracking, intrinsics, part inventory, and
descriptive fields remain in the render pass's complete `pose.json` instead of
being duplicated in every iteration.

`iteration.json` stores `start_kinematics_hash` and, after verification,
`kinematics_hash`, but not the joint definitions themselves. Historical
reconciliation reads the corresponding pose snapshot to distinguish compatible
joints, limit-only changes, and structurally changed mechanisms.

`shape_pass.sh` opens a shape iteration, writes its renders under that iteration,
then commits `scene.py`, `mesh/pose.json`, `NOTES.md`, and the compact iteration
snapshots automatically. Completion also promotes every frame's final panel
into that iteration's flat `composites/` directory. Shape and pose iterations
cannot complete without the full composite set from `layout.json`. If a pose
iteration is open, the shape pass attaches as its verification pass instead;
composites remain unpromoted until a passing `adjudicate --check` closes and
commits it.

Promoted composites are hard-linked when possible (copied otherwise) and are
not stored as Git blobs. The enclosing iteration identifies the scene and pose
snapshot, metadata, and corresponding Git history.

Iteration-scoped diagnostics use the same owner. In particular,
`analysis.viz.pose_pair_sheet` defaults to the active iteration's
`pose_pair_sheets/` directory and rejects destinations outside that iteration.

Inspect or recover a run with:

```bash
PYTHONPATH=harness python -m bookkeeping status --run-dir RUN_DIR
PYTHONPATH=harness python -m bookkeeping history --run-dir RUN_DIR
git -C RUN_DIR log --oneline
```

For pose iterations, `bookkeeping status` also derives each window's
`pending`/`partial`/`complete` state from its surviving `progress.json` and
`poses.json`. It reports completed frame names, the recovery artifact path,
last activity time, and counts of step calls, model feedback, and reports.
The draft remains the full recovery log even when the worker dies and no
iteration commit was reached.

`history` lists every complete, open, and aborted iteration with labels,
timestamps, kinematics hashes, composite counts, and window progress. Neither
command requires the agent to maintain iteration numbers in `NOTES.md`.

Historical pose reuse is always explicit:

```bash
PYTHONPATH=harness python -m multiagent.windows plan \
  --run-dir RUN_DIR --seed-from-iteration previous
```

A number may replace `previous`. Base poses become unverified hypotheses.
Unchanged joint states carry, limit-only changes clamp, removed joints drop,
and added, unknown, or structurally changed joints use the current committed
state and are flagged for review. Decisions are written to
`windows/seed_reconciliation.json` and the generated window briefs.

The main agent can also produce a reconciled trajectory without starting pose
subagents:

```bash
PYTHONPATH=harness python -m bookkeeping reseed-pose \
  --run-dir RUN_DIR --from-iteration previous
```

This writes `.harness/pose-reseed.json` for inspection and does not change
`scene.py`. Add `--apply` to mechanically rewrite `FRAMES` through the same
validated, atomic rewriter used by window apply. Add `--frames 000080.jpg` for
one frame, a comma list for several frames, or an inclusive range such as
`--frames 000080.jpg-000130.jpg`. Unselected frames retain their current
committed poses. Reseeding never commits or verifies the result; run a normal
shape pass afterward.

An interrupted command leaves its iteration open. Rerunning the same normal
command attaches to it. If rendering finished but scoring did not, reuse those
renders and regenerate the composites without paying for Blender again:

```bash
harness/utils/shape_pass.sh RUN_DIR \
  --existing-pass RUN_DIR/iterations/NNNNNN/renders/NNNN
```

The recovery path uses the normal `analysis.viz.composite` pipeline and then the
normal bookkeeping completion gate. Abandon work explicitly:

```bash
PYTHONPATH=harness python -m bookkeeping abort \
  --run-dir RUN_DIR --reason "why this work is being abandoned"
```

Aborting updates bookkeeping metadata but does not delete render passes, window
workspaces, partial composites, logs, or failure evidence. Those remain under
the numbered iteration for diagnosis or reuse. Only the flat `composites/`
promotion is withheld.

Custom render-producing commands can be wrapped with `bookkeeping run`. The
command receives `HARNESS_RUN_DIR`, `HARNESS_ITERATION`, and
`HARNESS_ITERATION_DIR`; it must place a pass containing `scene_snapshot.py`
and `pose.json` below that directory.

```bash
PYTHONPATH=harness python -m bookkeeping run \
  --run-dir RUN_DIR --kind custom --label diagnostic -- COMMAND ...
```

Use `--record-only` for a diagnostic pass that belongs to an already-open
iteration but must not close it. Select the active iteration's kind and write
the render below the directory supplied to the command:

```bash
PYTHONPATH=harness python -m bookkeeping run \
  --run-dir RUN_DIR --kind pose --record-only -- bash -c \
  'harness/render.sh "$HARNESS_RUN_DIR/scene.py" \
      "$HARNESS_ITERATION_DIR/renders" ...'
```

The command must create a new pass; an older pass in the iteration is never
claimed as the command's output. `render.sh` rejects the legacy
`RUN_DIR/views` destination when the run has unified bookkeeping.

Collection exports `run-history.bundle`, so Git history survives bare and asset
archives without shipping the internal `.git/` directory.

The implementation lives in `harness/bookkeeping/`: `ledger.py` owns numeric
iterations, locking, artifact promotion, and private Git commits;
`pose_history.py` owns history selection and joint reconciliation; `cli.py`
contains only the command-line adapter. `multiagent.windows` calls these
services and does not maintain a second bookkeeping or reconciliation path.
