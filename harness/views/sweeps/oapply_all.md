# `oapply_all` view — "flip the whole object 180° everywhere. Render. What are the IoUs?"

**Answers:** "apply THIS one shared canonical re-orientation — a rotation vector
`rx/ry/rz` in the object's own frame — to the object across ALL frames, show me
every render, and tell me how each scores."

`oapply_all` is the **whole-run imperative** object-centric verb: it right-multiplies a
single canonical rotation onto **every** frame's pose (`M_k' = M_k · R_extra`), renders
each frame at full match-res, scores each against its mask, and writes **paste-ready**
per-frame pose blocks. You read the per-frame IoU table and the renders and decide "yep,
that un-flipped it" — no search.

**This is the verb for a BUILD-ORIENTATION error**: the object came out mis-oriented, so
it reads wrong in *every* frame, and the fix is ONE decision committed once. When the
frames disagree with each other instead — a window seam where some frames sit in a
flipped basin — the honest verb is the **per-frame** [`oapply`](oapply.md), which lets
each frame pick its own basin, or [`osweep`](osweep.md) to search per frame.

There is deliberately **no whole-run *searched* mode** (no `osweep_all`): a shared
rotation ranked by MEAN gate IoU can be mediocre in every frame — the same `rz:+30`
scores 0.80 in one view and 0.54 in another. So the whole-run path is imperative only:
you name a candidate set (or an `--opreset` range), every candidate is rendered for
every frame, and you commit by LOOKING. For a continuous whole-run question use
`--opreset 'z:full'`; for a discrete one, `--oapply 'flips'`.

Under the hood `oapply_all` **is** an `osweep` of exactly the candidates you name (no
grid, no refine) in *shared* mode, so the report, per-frame IoU/depth numbers, and paste
blocks are identical in form — just named `oapply_all.*`.

## The order — a SHARED canonical rotation (or a named set of them)

Each candidate takes **one absolute value per canonical DOF** `rx`/`ry`/`rz` (contrast
`--osweep-ranges`' `min,max,steps`). The DOFs, the right-multiplied semantics, and the
shared `--opreset` spelling are in [object_rotations.md](object_rotations.md) — read that
once for all three object verbs. What is specific here: the **SAME** rotation is applied
to every frame, so one decision commits for the whole run — and it turns about the
**canonical origin** (`M_k' = M_k · R_extra`, translation untouched), not the per-frame
verbs' FK-AABB pivot: one shared right-factor is exactly the correction a rotated
`build()` would be, and is what lets seeded frames inherit the reference's paste.

`--oapply` also accepts a **candidate set** — `'|'` separates candidates, plus
sugar for the symmetry checks that motivate most of these calls:

- `'rz:180'` — ONE custom candidate (the original grammar);
- `'flip:x' / 'flip:y' / 'flip:z'` — a 180° flip about a canonical axis
  (`rx`/`ry`/`rz` = 180);
- `'identity'` — the no-flip baseline;
- `'flips'` — the default **symmetry panel**: expands to
  `identity|flip:x|flip:y|flip:z`;
- `'flip:z|rz:90;ry:12'` — a custom set, freely mixed. Duplicate
  rotations are dropped (first label wins); an unknown DOF/axis fails loud.

`'flips'` is the symmetry panel; `--opreset` is the general form for **one axis over a
range of angles** (`'z:quarters'` asks which quarter-turn is right for the whole run,
`'z:full'` asks the continuous version as a panel). Both flags together score their
**union**.

## Run
```bash
# the symmetry panel: identity + all three 180° flips, across all frames:
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views oapply_all \
    --frames <all> --ref-frame <ref> --match-res IMAGE \
    --masks-dir MASKS_DIR --hand-masks-dir HAND_MASKS_DIR --tracking CAPTURE/tracking \
    --oapply flips

# a critic-directed nod + front-axis roll on the canonical object:
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views oapply_all \
    --frames <all> --match-res IMAGE --masks-dir MASKS_DIR \
    --oapply 'rx:17;ry:-9'

# the continuous whole-run question, as a panel you can look at (0 competes, so
# you also learn whether turning at all helped):
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views oapply_all \
    --frames <all> --match-res IMAGE --masks-dir MASKS_DIR \
    --opreset 'z:full'
```

`--oapply` **or** `--opreset` (at least one **required**) — the shared order.
`--oapply` takes `';'`-separated `dof:value` terms inside one candidate,
`'|'`-separated candidates, or the `flips`/`flip:x|y|z`/`identity` sugar above;
`--opreset` names an axis plus a range of angles. Both given = their union. Runs
against **all** `--frames`.

`oapply_all` **reuses `osweep`/`sweep`'s scoring flags** (same interface): every flag's
`--help` (see [`../../rig/args.py`](../../rig/args.py)) is the ground truth.
- `--masks-dir` (**required**) — per-frame object masks `<frame-stem>.png`.
- `--hand-masks-dir` — per-frame occluder masks; a frame with one gates on
  `iou_visible`.
- `--sweep-depth-weight` (with `--tracking`) — per-frame Pi3X depth penalty, same formula
  as `sweep`.
- `--sweep-quality` — scoring-render resolution (winners re-render at full match-res).

## Outputs (in the pass dir `render.sh` prints)
- `oapply_all_best_<frame>.png` — the applied configuration re-rendered at **full match
  res**, one per frame.
- `oapply_all.txt` — the per-frame IoU table (gate / iou_raw / iou_vis / dp_canon, with
  an `auth` column), the shared rotation (`rx/ry/rz` + quaternion), the **mean** +
  **min** gate IoU, and **paste-ready `"pose"` blocks for each AUTHORED frame**.
- `oapply_all.json` — machine form mirroring `osweep.json`: `view:"oapply_all"`,
  `mode:"shared"`, `shared_rotation`, `mean_gate_iou`, `min_gate_iou`, the per-frame
  `frames` block (each entry carrying its corrected `pose` and the `joints` state it
  was rendered at, so one entry is a complete reseed record), `status`, and a top-level
  `passes` — one basin, so one list (contrast
  `osweep`'s per-frame lists). For an applied SET that is a single
  `kind:"candidates"` entry with an empty `ranges`: a discrete set has no lattice and
  is never refined.
- `oapply_all_poses.json` — the fragment for `multiagent.windows` merge/apply.

## What to paste, panels, non-destructiveness

Shared contract — paste each **authored** frame's block (reference + any `moved`/authored
frame); **seeded frames need no edit**, they inherit the same `R_extra` via the camera
re-seed, and `oapply_all.txt` lists which. Scoping with `--frames` to a SUBSET turns
inheritance off. The panel/visual budget, the proxy-IoU caveat, the restore guarantee, and
when to reach for a per-frame verb instead are the shared ones. All of it:
[object_rotations.md](object_rotations.md).
