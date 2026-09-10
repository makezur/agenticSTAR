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
and the source images, don't gate on it; a large `dcanon` is a likely pose
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

The scale/moved/joint-range checks are **advisory reads** that point you at the
right fix (pose depth vs. scale vs. joint state vs. object motion vs. joint
limit). The finalize gates live elsewhere: `multiagent.windows adjudicate --check`
(every seam's temporal call made) and, on an articulated run with the mechanism
module on, `analysis.mechanism_calls check`. Treat a step whose temporal call you
have not made as a reason **not** to finalize.
