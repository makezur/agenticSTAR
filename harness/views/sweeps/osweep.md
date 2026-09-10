# `osweep` view — "which canonical rotation does each frame want?"

**Answers:** "a frame reads wrong in a way I can only name in the object's own axes
(flipped, on its side, facing backwards). What rotation of the object's own canonical
frame best fits it — for each of these frames, independently?"

`osweep` is the **object-centric** sibling of [`sweep`](sweep.md). Where `sweep`
searches **camera-frame** pose increments **left-multiplied** onto one frame's pose,
`osweep` searches a rotation in the object's own **canonical** frame
**right-multiplied** onto it:

```
M_k' = M_k · T(c_k) · R_extra · T(−c_k)     per frame k, each frame's own R_extra
```

about `c_k`, the frame's own **FK-AABB centre** — the same material point `sweep`
orbits ([lib/pivot.py](lib/pivot.py); maths in
[conventions/POSE.md](../../../conventions/POSE.md), "the right update"). Because
`R_extra` acts in the object's canonical frame *before* the frame's own rotation
`R_k`, it expresses what a camera-frame yaw cannot: "turn it about its own up-axis"
is not a yaw except for an upright object. The object re-orients about its own
measured centre, holding its place in the camera even for an off-centre build —
translation picks up only the pivot correction `s·R_k(I − R_extra)c_k` (zero for a
canonically-centred object), and (uniform) scale is untouched.

**It is PER FRAME.** Every frame searches the same grid **independently**, keeps its
own argmax by its own gate IoU, and refines its own window. Semantically an `osweep`
over N frames is **N independent `sweep` runs** that happen to share one Blender
process and one candidate grid: nothing is aggregated, nothing is inherited, no
rotation is shared. What you get back is a per-frame rotation column that tells you
whether the frames agreed.

When the OBJECT was built mis-oriented — wrong in *every* frame, one decision to commit
— the verb is [`oapply_all`](oapply_all.md) (`--oapply 'flips'` or
`--opreset 'z:full'`) instead. There is deliberately **no whole-run *searched* mode**
(no `osweep_all`): ranking a shared rotation by MEAN gate IoU can pick one that is
mediocre everywhere — the same `rz:+30` scores 0.80 in one view and 0.54 in another
(see [DIRECTIONS.md](../../../conventions/DIRECTIONS.md)) — so a shared rotation is
chosen imperatively, from a panel you look at.

Its imperative sibling is the per-frame [`oapply`](oapply.md) — *"try these turns on
each frame; which one does each want?"*: named rotations instead of a search. Reach for
`oapply` when you already know the candidates to check; reach for `osweep` to **search**.

## The knobs — a canonical rotation, per frame

`osweep` searches a grid over three **canonical object-frame** rotation DOFs
`rx`/`ry`/`rz` (degrees). The DOFs, the rotation-vector semantics, the reason there are
no directional preset tokens here, and the shared `--opreset` spelling are all in
[object_rotations.md](object_rotations.md) — read that once for all three object verbs.

The grid is a Cartesian product over the named DOFs (coarse→fine with
`--sweep-refine`, each frame refining around its OWN coarse winner). A DOF you don't
name is held at 0 (identity on that axis). `--opreset` is the flag to reach for first:
it names an angle set instead of making you hand-compute a `min,max,steps` triple, and
`--osweep-ranges` is the explicit numeric form that overrides it.

## It scores every frame — and each frame ranks for itself

`osweep` renders **every** frame at each candidate, but the ranking is **per frame**:
each frame keeps the candidate with its own best gate IoU. There is no cross-frame mean
in the decision — a mean is what would let a rotation that helps most frames but wrecks
one win. Each frame is scored against its own mask from `--masks-dir`
(`<frame-stem>.png`), with the usual per-frame `iou_visible` gate when a
`--hand-masks-dir` mask is present. (The report still prints the **mean** and **min**
gate IoU *over the per-frame winners* — a summary of how the run went, not the ranking
key.)

**With `--tracking`, ranking is depth-aware** exactly as in `sweep`: each frame ranks by its
own `combined = gate IoU − depth_weight · min(depth_canon, 1)`.

## Run
```bash
# search a full-swing up-axis turn for each frame in a window (near-symmetric objects:
# read the top-k — a 0 and a ±180 may tie), coarse, depth-aware:
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views osweep \
    --frames <window frames> --ref-frame <ref> --match-res IMAGE \
    --masks-dir MASKS_DIR --hand-masks-dir HAND_MASKS_DIR --tracking CAPTURE/tracking \
    --osweep-ranges 'rz:-180,180,7' --sweep-quality 0.3

# default box (standard preset, ±15° on all three canonical axes), then refine:
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views osweep \
    --frames <window frames> --match-res IMAGE --masks-dir MASKS_DIR \
    --sweep-refine 2

# widen one axis by name (no numbers to compute):
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views osweep \
    --frames <all> --match-res IMAGE --masks-dir MASKS_DIR \
    --osweep-angle-preset 'rz:large;ry:tiny'

# the same full swing as the first example, but as a NAMED angle set — no triple
# to compute, and 0 is included so you see whether turning helped at all:
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views osweep \
    --frames <all> --match-res IMAGE --masks-dir MASKS_DIR \
    --opreset 'z:full' --sweep-quality 0.3

# RE-LOCALISE — tracking lost big time, no axis suspected: identity + 64
# Haar-uniform rotations (baked seed, fully deterministic), each frame refining
# from its own coarse winner. Rotation only: assumes placement is roughly right.
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views osweep \
    --frames <lost frames> --match-res IMAGE --masks-dir MASKS_DIR \
    --opreset 'so3:64' --sweep-refine 2 --sweep-quality 0.3
```

Every flag's `--help` (see [`../../rig/args.py`](../../rig/args.py)) is the ground
truth. Key ones:
- `--masks-dir` (**required**) — dir of per-frame object masks `<frame-stem>.png` (the
  harness-wide convention). Frames without a mask file are skipped (with a warning).
- `--hand-masks-dir` — optional per-frame occluder masks `<frame-stem>.png`; a frame
  with one gates on `iou_visible` over the non-hand region (mirrors `silhouette.py`).
- `--osweep-ranges` — per-DOF grid `dof:min,max,steps`, `;`-separated. `dof` in
  `rx,ry,rz` (degrees). Omitted DOFs held at 0. Empty = a default box (the `standard`
  preset, ±15°, on all three axes) whose step counts scale with `--sweep-quality`.
- `--opreset` — one axis + a RANGE OF ANGLES (`'z:quarters'`), or the axis-free
  re-localise draw `'so3:N'` (see [object_rotations.md](object_rotations.md#so3n--re-localise-when-no-axis-is-suspected)).
  Shared by all three object-centric views. Beats `--osweep-angle-preset`, loses to
  `--osweep-ranges`.
- `--osweep-angle-preset` — per-axis SIZE preset for the empty-`--osweep-ranges` box:
  `;`-separated `axis:preset`, `axis` ∈ `rx,ry,rz`, `preset` ∈
  `tiny`(±5°)/`standard`(±15°, the default)/`large`(±30°)/`huge`(±45°). These
  presets are **symmetric** (no directional tokens — a canonical rotation has no
  fixed on-screen sense). A symmetric preset larger than `tiny` is generally NOT
  what you want once you know which way the object is turned: spell an explicit
  ONE-SIDED range instead (e.g. `--osweep-ranges 'rz:0,60,7'` to search only the
  positive quarter), so you don't spend half the grid on the direction you ruled
  out. Explicit ranges win (a full swing is `'rz:-180,180,9'`).
- Reused from `sweep` verbatim: `--sweep-quality`, `--sweep-refine`,
  `--sweep-refine-shrink`, `--sweep-depth-weight`, `--sweep-hand-dilate`,
  `--sweep-timeout`, `--tracking`, `--frames`, `--ref-frame`, `--match-res`.

**Intrinsics when `--tracking` is omitted.** Like `sweep`, if no `--tracking`
(or `--intrinsics`/`--camera-json`) is given, the harness auto-resolves the run's
Pi3X dir from `RUN_DIR/layout.json` (walk-up from the output dir), so every frame is
scored under its real camera K. `--no-auto-tracking` forces the iPhone-13 placeholder.
This also holds when the object-centric views run through the resident pool (`pool/`): the
worker resolves the full per-frame `FrameSpec` list so each frame carries its own
Pi3X K, matching the one-shot `render.sh` path.

## Outputs (in the pass dir `render.sh` prints)
- `osweep_best_<frame>.png` — each frame's own winning rotation re-rendered at **full
  match-res**, one image per frame.
- `osweep.txt` — legend, the **per-frame table** with a **rotation column** (each
  frame's chosen `rx/ry/rz`) alongside gate / iou_raw / iou_vis / dp_canon, the mean +
  min gate IoU over the per-frame winners, and a **paste-ready `"pose"` block for every
  frame** (see below).
- `osweep.json` — machine manifest: `view`, `mode:"per_frame"` (no `shared_rotation`),
  `mean_gate_iou`, `min_gate_iou`, a per-frame `frames` block (each with its own
  `rotation`, gate IoU, iou_raw/visible, depth_canon, corrected `pose`, `authored`,
  `n_candidates`, `passes`, best image), a flat `ranked` list of the panelled
  (frame, candidate) pairs, `depth_weight`, `status`.

  **`passes` is PER FRAME here**, not top-level: with `--sweep-refine` each frame
  re-centers on its **own** argmax, so the windows diverge after pass 1 and one shared
  list would describe none of them. Each entry is `{pass, kind, n, ranges}` (+ `center`
  / `shrink` on a refine pass), its `ranges` being **that pass's** lattice — so the fine
  grid a frame's winner came from is reconstructable from disk, and the frame's `n`s sum
  to its `n_candidates` (which is why that count exceeds the coarse grid product). In
  **shared** mode (`oapply_all`) there is one basin, so `passes` is one top-level list.
  `osweep.txt` says refine ran in one line and points here for the N bands. Same
  mechanism and same field names as `sweep` — see
  [sweep.md](sweep.md#what-refine-actually-does).
- `osweep_poses.json` — the fragment for `multiagent.windows` merge/apply, carrying each
  frame's own rotation.

## What to paste, panels, non-destructiveness

Per-frame contract — **every** scored frame is authored, nothing is inherited (mirrors
`measure_depth`'s per-frame snippet); read the rotation column before pasting, and prefer
merging `osweep_poses.json` via `multiagent.windows` over N hand edits. The panel/visual budget,
the proxy-IoU caveat, the restore guarantee, the near-symmetric-tie trap, and when a
`sweep`/`oapply_all` is the better scope are the shared ones. All of it:
[object_rotations.md](object_rotations.md).
