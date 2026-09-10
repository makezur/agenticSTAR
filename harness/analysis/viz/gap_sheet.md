# `gap_sheet.py` — the capture's own frames INSIDE one temporal step

Shows what the VIDEO did between two of a run's keyframes. A run samples its
capture (a layout listing every 10th frame is typical), so one step of the
[temporal report](../temporal/report.md) sums many unseen capture frames into a
single number. The capture directory holds EVERY frame; this tool sheets the
dense stretch `A..B` of them so a step's residual can be judged against the
motion that actually happened, not inferred from its two endpoints.

## When to reach for it — and when not to

This is an ESCALATION, not a routine step. The endpoints plus your
`expected_motion` settle most steps; build a gap sheet only when they cannot:

- a step you are about to call `unsure` — the dense frames often turn it into a
  confident `coherent` ("the lid really does slam in those 10 frames") or a
  confident re-pose;
- an expected large move that measured SMALL, or a measured spike your frame
  sheet did not predict;
- the orchestrator at `adjudicate`, when a seam's residual is not explained by
  the seam sheet's basin view.

Do NOT build one per step as a matter of course: it is more pages to read, and
a sheet nobody needed is attention taken from the frames that did need it.

## Run

```bash
micromamba run -n artscript env PYTHONPATH=harness \
  python -m analysis.viz.gap_sheet \
  --run-dir RUN_DIR --step 000100.jpg:000110.jpg
```

Writes one sheet dir per `--step` (repeatable):

```
RUN_DIR/gap_sheets/000100_000110/frame_sheet_001.png
RUN_DIR/gap_sheets/000100_000110/frame_sheet_manifest.json
```

The two ENDPOINT tiles — the run's own keyframes, the step being judged — carry
the bright rim seam_sheet uses for its pair under review; the in-between frames
keep the neutral rim, so "what the run committed" and "what only the camera saw"
never blur. The manifest is frame_sheet's (each tile carries a `highlight`
flag), plus a `step` block recording `from`/`to`, `keyframes_adjacent`,
`capture_count`, `shown_count`, `stride`, and `dropped` — so the evidence says
what was strided out.

## Options

- `--step A:B` (required, repeatable): a temporal step between two of the run's
  keyframes. Endpoints accept names or bare stems, in either order. Two
  keyframes that are NOT adjacent in the run's order warn loudly (`!! NOT ONE
  STEP`) and proceed — the sheet then spans several report rows and reads as a
  segment overview, not a step.
- `--every N`: show every Nth capture frame (endpoints always kept). Default:
  the gap is strided to fit ONE page and the chosen stride and dropped count
  are printed — never silently sampled.
- `--out-dir DIR`: root for the per-step dirs; defaults to `RUN_DIR/gap_sheets`.
- `--frames-per-page` / `--columns` / `--tile-height`: the frame_sheet page
  budget, identical semantics ([frame_sheet.md](frame_sheet.md)).

## Not `frame_sheet`, not `seam_sheet`

All three paginate frames; they answer different questions.

- [frame_sheet](frame_sheet.md) is the TIMELINE of the run's own keyframes —
  what you read before any sweep, and what `expected_motion` is written from.
- [seam_sheet](seam_sheet.md) is a PAIR judged against its COMMITTED RENDERS —
  the basin question, which only renders can answer.
- gap_sheet is SOURCE ONLY, dense, inside one step — the plausibility question.
  The in-between frames have no poses and therefore no renders; there is
  nothing to compare, only motion to witness.

## No verdicts

Like everything in the temporal layer, this tool draws and does not judge: no
threshold, no highlight of a "suspicious" frame. The sheet is the evidence a
`coherent` / `unsure` / `flip` call cites (name it in the call's `evidence`),
never the call itself.
