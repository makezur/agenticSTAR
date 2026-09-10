# `crop` view — "is the object cut off by the frame?"

**Answers:** "did my object spill outside the frame? do I need to zoom out?"

The `match` render only shows what's inside the frame, so you can't tell from it
how much of the object spilled outside. The `crop` view renders a second
**overscan** pass (same camera, wider frame) that captures the whole object, then
measures how much of it falls outside the real match frame — the signal for when
your shared `SCALE` is too big or the frame's pose is off-center.

## Run
```bash
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views crop \
    --match-res IMAGE --crop-margin 0.6
```
- `--crop-margin` — overscan per side (fraction of the match frame). Raise it if
  the object is clipped even in the overscan.

## Outputs (in the pass dir `render.sh` prints)
Each `render.sh` call writes into a fresh `RUN_DIR/views/<NNNN>/` pass dir and
PRINTS that dir; read this run's outputs from the printed dir. A view not rendered
this pass is absent from it (no stale leftovers).
- `crop.png` — the overscan render with the **true match frame outlined in
  white** (see at a glance what's inside vs. cut off).
- `crop.txt` / `crop.json` — `visible_frac` (object inside the frame),
  `clipped_frac` (object outside), per-edge overflow (`left/right/top/bottom`),
  and `suggested_zoom_out` — shrink the shared `SCALE` **down** by ~this factor to
  fit the whole object in frame (1.0 = already fully visible).

## Judge against common sense
If the object should be **fully visible** but `clipped_frac` is high, your
placement is too big/off-center — scale down by `suggested_zoom_out` and/or
recenter. If the source photo genuinely crops the object, some clipping is
expected. If `fully_captured_in_overscan` is false, the object exceeds even the
overscan (zoom-out is a lower bound) — raise `--crop-margin`. The crop view
restores the match frame, so it does **not** affect `match.png`, the turntable,
or the GLB.
