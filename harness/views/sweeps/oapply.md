# `oapply` view — "try these turns on each of these frames. Which one does each want?"

**Answers:** "apply these named canonical re-orientations — rotation vectors `rx/ry/rz`
in the object's own frame — to EACH of these frames independently, show me every render,
and tell me which one each frame chose."

`oapply` is the **per-frame imperative** object-centric verb, and `apply`'s object-centric
twin. Where [`apply`](apply.md) composes ONE camera-frame order onto ONE frame, `oapply`
applies a named candidate SET to each of N frames and lets **every frame pick its own
winner** by its own gate IoU. Semantically it is **N independent `apply` runs** that
happen to share one Blender process and one candidate set: nothing is aggregated, nothing
is inherited, no rotation is shared.

**Reach for it when the frames disagree with each other**: a window seam where some
frames sit in a flipped basin, "this one came out facing backwards". The report's
per-frame **rotation column** then tells you whether they converged on the same turn or
split. When the OBJECT was built mis-oriented — wrong in *every* frame, one decision to
commit — the verb is [`oapply_all`](oapply_all.md) instead. Its searched sibling is
[`osweep`](osweep.md) (same per-frame semantics, a grid instead of named turns).

## The order — a candidate SET, tried on every frame

Each candidate takes **one absolute value per canonical DOF** `rx`/`ry`/`rz` (contrast
`--osweep-ranges`' `min,max,steps`). The DOFs, the right-multiplied semantics (per
frame, about that frame's own FK-AABB centre: `M_k' = M_k · T(c_k)·R_extra·T(−c_k)`),
and the shared `--opreset` spelling are all in
[object_rotations.md](object_rotations.md) — read that once for all three object verbs.

`--oapply` names the candidate set — `'|'` separates candidates, plus sugar for the
symmetry checks that motivate most calls:

- `'rz:180'` — ONE custom candidate;
- `'flip:x' / 'flip:y' / 'flip:z'` — a 180° flip about a canonical axis
  (`rx`/`ry`/`rz` = 180);
- `'identity'` — the no-flip baseline;
- `'flips'` — the default **symmetry panel**: expands to
  `identity|flip:x|flip:y|flip:z`;
- `'flip:z|rz:90;ry:12'` — a custom set, freely mixed. Duplicate rotations are dropped
  (first label wins); an unknown DOF/axis fails loud.

`'flips'` is the symmetry panel; `--opreset` is the general form for **one axis over a
range of angles**, so `--opreset 'z:quarters'` answers *"which quarter-turn does each of
these frames want?"* in one order. Both flags together score their **union**.

## Run
```bash
# does each frame in this window want a flip? (each answers for itself):
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views oapply \
    --frames <window frames> --ref-frame <ref> --match-res IMAGE \
    --masks-dir MASKS_DIR --hand-masks-dir HAND_MASKS_DIR --tracking CAPTURE/tracking \
    --oapply flips

# which quarter-turn about the up axis does each frame want? (0 competes, so you
# also learn whether turning at all helped):
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views oapply \
    --frames <window frames> --match-res IMAGE --masks-dir MASKS_DIR \
    --opreset 'z:quarters'
```

`--oapply` **or** `--opreset` (at least one **required**). `--oapply` takes
`';'`-separated `dof:value` terms inside one candidate, `'|'`-separated candidates, or
the `flips`/`flip:x|y|z`/`identity` sugar above; `--opreset` names an axis plus a range
of angles. Both given = their union. Pair with `--frames` to scope the check to the
frames you are actually asking about — a per-frame verb costs candidates × frames
renders.

`oapply` **reuses `osweep`/`sweep`'s scoring flags** (same interface): every flag's
`--help` (see [`../../rig/args.py`](../../rig/args.py)) is the ground truth.
- `--masks-dir` (**required**) — per-frame object masks `<frame-stem>.png`.
- `--hand-masks-dir` — per-frame occluder masks; a frame with one gates on
  `iou_visible`.
- `--sweep-depth-weight` (with `--tracking`) — per-frame Pi3X depth penalty, same formula as
  `sweep`.
- `--sweep-quality` — scoring-render resolution (winners re-render at full match-res).

## Outputs (in the pass dir `render.sh` prints)
- `oapply_best_<frame>.png` — each frame's own winner re-rendered at **full match res**.
- `oapply.txt` — the per-frame table with a **rotation column** (each frame's chosen
  `rx/ry/rz`), gate / iou_raw / iou_vis / dp_canon, the mean + min gate IoU over the
  per-frame winners, and a paste-ready `"pose"` block for **every** frame.
- `oapply.json` — machine form: `view:"oapply"`, `mode:"per_frame"`, no
  `shared_rotation`; each `frames[]` entry carries its own `rotation` and `pose`, plus
  the `joints` state that frame was rendered at (these verbs search rotation, not
  articulation, but a report is a reseed source and a pose without its articulation is
  half a record); the
  flat `ranked` list carries one entry per panelled (frame, candidate) pair. Each
  `frames[]` entry also carries `passes` — for a named candidate SET that is one
  `kind:"candidates"` entry with an empty `ranges` (a discrete set has no lattice and
  is never refined); it is only interesting on a refined band sweep, see
  [osweep.md](osweep.md).
- `oapply_poses.json` — the fragment for `multiagent.windows` merge/apply, carrying each
  frame's own rotation.

## What to paste, panels, non-destructiveness

Per-frame contract — **every** scored frame is authored, nothing is inherited; read the
rotation column before pasting, and prefer merging `oapply_poses.json` via `multiagent.windows`
over N hand edits. The panel/visual budget, the proxy-IoU caveat, and the restore
guarantee are the shared ones. All of it:
[object_rotations.md](object_rotations.md).
