# `aggregate.py` — multi-frame roll-up + cross-frame checks

Collects the per-frame scores (from `silhouette.py` and `depth.py`) and the run's
`pose.json` into one `report.json` + a human `report.txt`, and adds cross-frame
checks (moved-flag sanity,
which needs **more than one frame**) plus a per-joint valid-range check.

## Run

```bash
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.rollup.aggregate \
    --run-dir RUN_DIR --pass 0012
# Or select the exact directory directly:
#   --run-dir RUN_DIR --views-dir RUN_DIR/views/0012 --out .../report.json
```

Pass directories are immutable. With one pass, `--run-dir` alone selects it; with multiple passes, aggregation fails until `--pass ID` or `--views-dir PATH` identifies one exactly. `--pass` accepts numeric IDs only. The selected pass must contain its own `pose.json`; use `--pose PATH` explicitly only for a legacy pass.

It reads, for each frame in the selected pass's `pose.json`:
- `<views-dir>/metrics_<stem>.json` — the silhouette IoU (optional),
- `<views-dir>/depth_<stem>.json` — the depth agreement (optional),
- `critic_<stem>.json` — the optional VLM screen, read as current evidence only
  when it exists in `<views-dir>`. An older file from another pass is never loaded
  into scores, findings, or finalize review because it describes older geometry or
  poses. The report records `critic_status` as `fresh`, `stale`, or `not_screened`;
  a stale row includes `critic_stale_source` for historical traceability, while
  `critic_source` and critic findings remain empty. Fresh outputs written by
  `shape_pass.sh` also record the screened pass and `scene_sha1`.

where `<stem>` is the frame name without extension (e.g. `000080` for
`000080.jpg`). So the workflow is: render all frames, then per frame write
`metrics_<stem>.json` (`composite.py --metrics-out`, which runs `silhouette.py` for
you) and `depth_<stem>.json` (`depth.py --out`), then run `aggregate.py`.
`harness/utils/shape_pass.sh` does all of this in one command.

## Output

A per-frame table (IoU, the raw canonical depth error `depth_mae_canon`
(`dcanon`; the error as a fraction of the object's longest dimension), the
signed `depth_bias_canon` (`dbias`; `+` = render too far, `−` = too near), joint
states, `moved` flag). Depth is a strong-but-noisy guide — weigh it against IoU
and the critic, don't gate on it; a large `dcanon` is a likely pose
(translation-Z, `dbias` says which way) or shape discrepancy to investigate
before iterating on shape — do **NOT** reach for the shared `SCALE` to paper
over it.

Then the cross-frame checks:

- **moved-flag sanity** — a frame declared `moved=False` uses its camera-seeded
  base pose verbatim, so all cross-frame change should be camera + joints. If such
  a frame still scores a low IoU (`iou < 0.75`) **or** a high depth error
  (`depth_mae_canon > 0.10`), `aggregate.py` surfaces it:
  either the object really did move (set `"moved": True` and author a pose) or a
  joint state is wrong. This is a **cross-check** on the agent's per-frame call,
  not the decision — the agent still makes the observational call (did the object
  move?) from the images.

- **joint valid-range** — for each revolute/prismatic joint, the **observed**
  state range across frames (`observed_min`/`observed_max`) vs its **declared**
  `limit`. Verdicts: `within` (all states inside the limit), `no_limit` (states
  seen but no `limit` — add one so impossible configs can't ship), `no_state`
  (joint never actuated), and `exceeds_limit` (a state fell outside the limit).
  An **authored** scene can never reach `exceeds_limit` — the harness rejects an
  out-of-range state at load — so it flags a `pose.json` produced by something
  OTHER than the authoring gate (e.g. a video optimizer): widen the `limit` if
  that config is real, or fix the state. This closes the **predict → correct**
  loop on joint limits: declare a range in `scene.py`, see how the observed states
  sit against it, and tighten it toward the true travel.

- **FINALIZE REVIEW REQUIRED (`critic_review_required`)** — every **high-severity**
  fresh current-pass discrepancy tagged **`joint_state`**,
  **`joint_definition`**, **`pose`**, or **`shape`**,
  flattened across frames. A passing IoU (or a clean depth guide) does **not** clear
  any of them, for three distinct reasons:
  - `joint_state`/`pose` are what the numeric signals are **blind** to: silhouette IoU can't
    see a small part hiding *behind* another when over-rotated (a joint), nor a
    near-symmetric whole-object mis-orientation (a pose), and depth barely moves for a
    small part. Reconcile via a `sweep` or a frame pose/joint edit.
  - `joint_definition` means no declared joint state can produce the visible
    articulation. Fix JOINTS and rerender; a state sweep cannot create a missing
    DOF. Legacy `joint` findings normalize to `joint_state`.
  - `shape` is *visible* to IoU, but a pose `sweep` **cannot** fix geometry and a high
    IoU can sit on wrong geometry (it is depth-blind — a nearer+smaller pose paints the
    same outline). So a passing IoU never licenses skipping a high `shape` fix: fix it
    in `build()`.

  Each must be **reconciled** before finalizing — acted on, or explicitly justified in
  `NOTES.md`. `med`/`low` fixes (any tag) stay in the soft critic summary (surfaced,
  not gated — IoU + the turntable already constrain most geometry).

- **RIG REVIEW REQUIRED (`rig_review_required`)** — the high-severity
  `joint_definition` subset of the critic gate, called out separately so it must
  be empty before pose windows begin.

- **shape findings (`shape_consensus`)** — a **cross-frame collation of fresh
  critic `shape` complaints**, all screened frames in one place. At 20–50 frames
  you can't re-read N
  `critic_<stem>.json` to notice that many frames complain about the same part, so this
  gathers **every** `shape` discrepancy (all severities) into one flat, frame-ordered list —
  `{frame, part, severity, note}` per finding, plus `n_findings`/`n_critiqued` for context.
  It is a **report, not a verdict**: there is deliberately
  - **no threshold / no `build_edit` call** — no magic fraction decides that geometry is
    wrong; you read the list and make the geometry call yourself. A `shape` complaint
    repeated across many frames is a strong hint the fix belongs in `build()` (a pose
    `sweep` can't touch geometry), but that's your judgement, not the tool's.
  - **no canonical-part normalisation** — the critic emits `part` as free VLM text and does
    **not** know the `build()`/`pose.json["parts"]` routing, so the finding reports its
    `part` **verbatim**; mapping it onto a canonical name would be a guess. Group by eye.

  (The report key stays `shape_consensus` for stability, but the value is now this plain
  collation.) High-severity `shape` fixes are **also** in `critic_review_required` above —
  that per-frame high-only gate is the finalize authority for shape; this list is the
  cross-frame surfacing (all severities), advisory.

The scale/moved/joint-range checks and the shape-findings collation are **advisory reads**
that point you at the right fix (pose depth vs. scale vs. joint state vs. object motion vs.
joint limit vs. a repeated geometry complaint). The finalize gates are the strong
ones: empty `critic_review_required` and `rig_review_required` here,
`multiagent.windows adjudicate --check` (every seam's temporal call made) and, on
an articulated run with the mechanism module on, `analysis.mechanism_calls check`.
Treat an unreconciled high-severity fresh pose/joint-state/joint-definition/shape
fix, or a step whose temporal call you have not made, as a reason **not** to
finalize.
