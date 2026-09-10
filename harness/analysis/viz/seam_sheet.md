# `seam_sheet.py` — the picture a temporal verdict is made against

A **seam** is a single temporal step (`frame_a -> frame_b`) whose two frames were
posed by **different owners**: two refiner windows, or a window and an
already-committed neighbor. Nobody saw both sides of it, so it is the one place a
cross-window disagreement — most often a ~180° basin flip on one side only — can
hide. This tool draws one band per seam so that disagreement is visible.

It exists because the numbers cannot settle the question. `rotation_deg` reads
~171° for a genuine half-turn and for a spurious flip alike; the source video is
always temporally smooth, so a jump *there* is real motion. Only the **committed
renders** show which basin a pose actually landed in. Put the two photos and the
two renders side by side and the ambiguity resolves by eye.

This module **draws only**: no threshold, no verdict, no selection. The seams come
from `analysis.temporal.seams.crossing_steps` (ownership alone), each band's
caption from `analysis.temporal.seams.step_lines` — measured off `--pose-json`
when the caller does not supply it — and the verdict from you, see
[`analysis.temporal.report`](../temporal/report.md).

## Run

```bash
micromamba run -n artscript env PYTHONPATH=harness \
  python -m analysis.viz.seam_sheet \
  --run-dir RUN_DIR --pass-dir RUN_DIR/views/0033 \
  --seam 000280.jpg:000290.jpg --seam 000320.jpg:000330.jpg
# -> SEAMS 01-02 OF 02, each band captioned "rot 78.0 deg  trans 0.114 of size
#    lid_hinge -7.5 deg (-7% of range)  gap 10"
```

`--seam FROM:TO` is repeatable and also accepts a comma list. The two frames must
be **one step apart** in the run's sequence; a gap is a caller error, not a wider
sheet. Use `--frames-dir CAPTURE/frames` instead of `--run-dir` when no run
`layout.json` exists.

With `--run-dir` the band captions come from `RUN_DIR/mesh/pose.json` for free —
the same magnitudes the temporal report and the gates print, so the eye has
something to check against. Point `--pose-json` elsewhere to caption against a
different trajectory, or pass `--no-notes` for bare pictures.

You rarely have to type the seam list:

- the [aggregate](../rollup/aggregate.md) report's TEMPORAL REVIEW REQUIRED block
  prints a ready-made command naming every unresolved pair, pointed at that
  report's own pass.
- `multiagent.windows merge`/`selfcheck` name the seams themselves (`seam_steps` in
  their reports) — pass those pairs as `--seam FROM:TO`.

## Layout

One band per seam, all of them rolled onto one sheet:

```
+=================================================================+
| SEAMS  3 of 3   pass 0033                                       |
+-----------------------------------------------------------------+
| 01  000280.jpg -> 000290.jpg   w01 | w02   rot 78.0 deg  ...    |
|      | A 000280 | B 000290 |               <- SOURCE row        |
|      |          |          |               <- COMMITTED row     |
+-----------------------------------------------------------------+
| 02  000320.jpg -> 000330.jpg   ...                              |
+=================================================================+
```

**Rolled, not one page each,** because the seams inform each other: two seams
flipping the *same* way is one window in the wrong basin (re-pose that window),
while two flipping independently is two separate calls. Separate pages hide that
correlation, and the counts are small — K windows give K−1 seams.

**The pictures carry the sheet, not the caption.** A band's note is one terse
line: the magnitudes, nothing the title or a column header already says, no unit
legend repeated per band. Nothing is lost — full precision lives in the temporal
report JSON, and that same line plus every frame's source/render path lives in the
manifest.

**The pair is marked, not merely adjacent.** A bright rim and a wide gutter make
the two frames under review unambiguous — this is not a timeline sheet (that is
[frame_sheet](frame_sheet.md)); the comparison *is* the content.

A frame with no `match_<stem>.png` in the pass dir gets an explicit placeholder
panel and is **named** in the manifest's `frames_without_render` — an absent
render is information (that frame was never applied in this pass), not a tile to
drop silently.

## Options

- `--pass-dir DIR`: pass containing `match_<stem>.png` committed renders. Omit for
  a source-only sheet, but then the sheet cannot answer the basin question at all.
- `--pose-json PATH` / `--no-notes`: where the measured band captions come from
  (default `RUN_DIR/mesh/pose.json`), and how to turn them off.
- `--context N`: flanking frames per side, drawn dim. **Default `0` — opt-in.** At
  a fixed sheet width one frame of context per side halves the pair's own pixels,
  which is exactly the resolution the basin call depends on. Ask for context when
  the question is the local *trend* (does the motion continue past the seam?);
  leave it off when the question is the pair.
- `--tile-height PX`: height of **one** image row; default `340`. The render row
  makes the band taller rather than shrinking the photos.
- `--seams-per-page N`: `0` (default) puts every seam on one sheet; `1` gives a
  page per seam for close inspection.
- `--out-dir DIR`, `--basename NAME`, `--bg-mode MODE`.

## Manifest

`<basename>_manifest.json` records `frames_dir`, `pass_dir`, `context`,
`tile_height`, `seams_per_page`, `with_committed_render`, `seam_count`, `pages`,
the named `frames_without_render`, and per seam: `ordinal`, `from`, `to`, `note`,
`page`, `page_path`, and one entry per drawn frame (`frame_id`, `in_seam`,
`has_render`, and its source/render paths). It is the auditable record of what was
looked at — cite it when you record the verdict's `evidence`.
