# `visibility` view — "which bits are visible, where do they land, what's in front?"

**Answers:** "this part is missing/misplaced — is it hidden behind another part,
and where did it actually go?"

The `ids` view tells you which pixels each part owns and flags a part as
`OCCLUDED` only when it has *zero* visible pixels. The `visibility` view goes
further: for **every** part it reports where it lands in the image (both its
visible extent and its full/unoccluded extent), **how much of it is visible**,
**which parts occlude it**, and whether it's **clipped by the frame**.

It renders the composite ID map plus one solo pass per part (others hidden), so
it costs N+1 fast renders. Run it **on demand** when a part is missing/misplaced,
not every iteration.

## Run
```bash
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views visibility \
    --match-res IMAGE --visibility-margin 0.1
```
- `--visibility-margin` — overscan per side so a part spilling off-frame is still
  captured (and its clipping measured). Use `0` to skip the overscan (occlusion
  only, no off-frame detection).

## Outputs (in the pass dir `render.sh` prints)
Each `render.sh` call writes into a fresh `RUN_DIR/views/<NNNN>/` pass dir and
PRINTS that dir; read this run's outputs from the printed dir. A view not rendered
this pass is absent from it (no stale leftovers).
- `visibility.png` — the match render with a **solid** box at each part's visible
  extent and a **dashed** box at its full/unoccluded extent, colored by the ID
  palette (dashed-only = a fully occluded part, drawn where it *would* land).
- `visibility.txt` — human legend: `idx  name  status  vis%  bbox_visible  occluded_by`.
- `visibility.json` — per part: `status` (`VISIBLE`/`PARTIAL`/`OCCLUDED`),
  `visible_fraction`, `bbox_visible`/`bbox_full`, `centroid_*`, `occluded_by`
  (the parts front-most over it, by pixel share), and per-part `clip`/`clipped`;
  plus a `summary`.

`status` is occlusion-only — framing is the separate `clipped` field.
