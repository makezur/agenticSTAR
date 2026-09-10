# Canonical object rotations — the contract shared by `osweep` / `oapply` / `oapply_all`

The three object-centric verbs ask the same question at different scopes, so they share
one rotation spec, one panel/visual contract, and one non-destructiveness guarantee.
Those live here once; each verb's own doc covers only what is different about it.

| verb | scope | who picks the winner |
|---|---|---|
| [`osweep`](osweep.md) | searched grid, per frame | each frame, for itself |
| [`oapply`](oapply.md) | named candidate set, per frame | each frame, for itself |
| [`oapply_all`](oapply_all.md) | named candidate set, ONE shared rotation | the run (mean over frames) |

---

## The canonical DOFs

Per [POSE.md](../../../conventions/POSE.md) the canonical object is +Z up, centred, longest
dimension ≈ 1 (see [`../../rig/lie.py`](../../rig/lie.py) `canonical_rotation`):

| DOF | axis | meaning |
|---|---|---|
| `rx` | canonical +X (right) | nod the object forward/back, deg |
| `ry` | canonical +Y (front) | roll about the front-facing axis, deg |
| `rz` | canonical +Z (up) | turn the object about its up axis, deg |

`(rx, ry, rz)` is a **rotation vector** (axis = direction, angle = norm): a multi-axis
candidate (`'rz:180;ry:12'`) is ONE rotation about the tilted axis, never a sequence.

These are **NOT** the camera-frame `yaw`/`pitch` of [`apply`](apply.md) — different frame;
the parser redirects you if you use those spellings. The rotation is **right**-multiplied
onto the frame's own pose, so it acts in the object's own frame; any DOF you don't name
is 0. **Where it pivots differs by verb** (the maths:
[conventions/POSE.md](../../../conventions/POSE.md), "the right update"):

- `osweep` / `oapply` turn about **that frame's FK-AABB centre** (the same material
  point the camera-frame `sweep` orbits, measured at the frame's articulation —
  [lib/pivot.py](lib/pivot.py)): `M_k' = M_k · T(c_k) · R_extra · T(−c_k)`. The
  object spins in place on screen even when the canonical build sits off-centre;
  `t` picks up only the pivot correction `s·R@(I − R_extra)@c_k`, and each frame's
  report records its pivot.
- `oapply_all` turns about the **canonical origin** (`M_k' = M_k · R_extra`,
  translation untouched) **deliberately**: one shared right-factor is exactly the
  correction a rotated `build()` would be, and is what lets seeded frames inherit
  the reference's paste. For a canonically-centred build the two pivots coincide.

(Uniform) scale is untouched in every verb.

A positive canonical rotation has **no fixed on-screen sense** — it depends on each
frame's pose, unlike the camera-frame DOFs whose signs are tabulated in
[DIRECTIONS.md](../../../conventions/DIRECTIONS.md). The same `rz:+30` can move a landmark
by up to 169° differently between two frames. That is exactly why these verbs re-render
every candidate for you to eyeball, and why there are **no directional suffixes** here
(unlike `--sweep-angle-preset`'s `standard_cw`): a screen-relative word would be a lie in
some frame. Signed *numbers* are fine — the positive sense is fixed in the object's frame.

## `--opreset` — one axis, a range of angles (reach for this first)

Most object-orientation questions are *"which turn about ONE axis is right?"*, and
spelling that as numbers means hand-computing a `min,max,steps` triple that lands on the
angles you meant. `--opreset` names the angle set instead, and **the same spec means the
same rotations in all three verbs** — whether you search it or apply it.

| `--opreset` | angles about that axis |
|---|---|
| `z:flip` | 0, 180 |
| `z:quarters` | 0, 90, 180, 270 |
| `z:octants` | every 45° (8 angles) |
| `z:full` | the whole circle, `[-180, 180]`, step count from `--sweep-quality` |
| `z:tiny` / `standard` / `large` / `huge` | the symmetric ±5°/±15°/±30°/±45° **band** (same table as `--osweep-angle-preset`) |
| `z:0,90,180` | exactly those angles |
| `so3:N` | identity + N **Haar-uniform random rotations** of the whole of SO(3) — no axis at all; **`osweep` only**, see below |

- `axis` is `x`/`y`/`z` (`rx`/`ry`/`rz` also accepted) — the object's **own** axes above.
- **Every turn set includes 0**, so the no-op baseline competes on the same table: you
  learn not just which turn scored best but whether turning at all helped (the same
  reason `'flips'` expands to `identity|flip:x|flip:y|flip:z`).
- The numeric form is **always an angle list, never `min,max,steps`** — that shape belongs
  to `--osweep-ranges`. `--opreset 'z:0,90,180'` is 3 rotations; `--osweep-ranges
  'rz:0,90,180'` is 180 candidates spanning a quarter turn. The value count is never used
  to guess which you meant.
- Two axes give the **Cartesian product**: `'z:quarters;x:tiny'`.
- `--sweep-refine` shrinks **band** presets (`tiny`…`huge`, `full`) only. A discrete turn
  set is not refined — there is no basin between 90° and 180° — and the winning turn is
  held at its own angle while other axes refine around it.
- **Precedence:** `--osweep-ranges` > `--opreset` > `--osweep-angle-preset`. Explicit
  numbers win because they are the least ambiguous statement of intent; the angle preset
  is last because it only ever *sized* the empty-ranges default box.

Given **both** an explicit candidate set (`--oapply`) and `--opreset`, their **union** is
scored — silently dropping a candidate you named while still announcing a winner would be
dishonest. Angles become **labelled candidates** (`identity`, `rz+090`, `rz+180`, …)
zero-padded so a directory listing and the ranked table sort by angle.

### `so3:N` — re-localise when no axis is suspected

`so3:N` is the move when tracking is lost **big time**: the orientation is wrong in a way
you cannot name about any axis. It scores the identity plus N rotations sampled
**uniformly w.r.t. the Haar measure** on SO(3) (unit quaternions from 4-D Gaussians — no
Euler-angle pole-piling), so no orientation is favoured and N controls granularity alone.

- **It must stand alone** — `'so3:64;x:tiny'` is an error. A Cartesian product of Haar
  samples with an axis lattice is not Haar-uniform. If you *know* the lost axis, don't use
  this: `'z:full'` covers the circle in ~7 candidates where `so3:64` spends 65.
- **Deterministic, no seed to pass.** The default seed is baked into the code
  (`planner_canon.SO3_SEED`), so `so3:64` names the *same* 64 rotations in every run, on
  every machine — and `so3:128`'s first 64 are exactly `so3:64`'s (one stream, sequential
  draws), so raising granularity extends the set instead of reshuffling what you already
  judged.
- **Unhappy with the draw? Re-roll in the spec: `so3:64@7`.** A different seed is a fresh
  *independent* Haar sample of the same size — the right escalation when every basin of
  the default draw judged wrong, and cheaper evidence than `so3:128` (whose first half
  you already scored). Every report's `passes[0].so3` records the exact `{n, seed}`, so
  any draw is reproducible from disk. (Changing the baked default: see the `SO3_SEED`
  comment in `planner_canon.py`.)
- **Pair it with `--sweep-refine`** (1–2 passes). At n=64 the nearest sample is typically
  a few tens of degrees from the truth; the refine pass opens a rotation-vector window of
  that size around each frame's winner and descends. Without refine you get the nearest
  basin, not the pose.
- **Rotation only, by construction.** The rotation turns about the frame's FK-AABB
  centre, so the object spins in place and cannot move — even for an off-centre
  build — and the camera never moves; which is exactly why this is safe to fire
  blindly. (About the canonical *origin*, an off-centre build would translate with
  every candidate and the IoU would rank translation error, not rotation.) It
  **assumes placement is roughly right**: silhouette IoU cannot rank rotations whose
  render misses the mask entirely. If translation is also lost, seed a rough
  placement first (the request's top-level `pose`, e.g. from the mask centroid),
  then `so3:N`, then a translation-only `sweep` if needed.
- **`osweep` only.** The imperative verbs (`oapply`/`oapply_all`) refuse it: they render
  *named* rotations for you to look at, and a random draw has no names — it is a search
  grid. (Shared mode over a blind draw would be the whole-run *searched* mode this family
  deliberately does not have — see [osweep.md](osweep.md).) No lattice → panels fall back
  to score order.

To build intuition for what a canonical rotation does before spending a run on it, render
a few named rotations with `oapply` and look at them.

## Panels and the visual budget

A set of at most `core/visual_budget.MAX_SET_RENDERS` candidates (8 today) re-renders
**every** candidate for every frame (`<verb>_<label>_<frame>.png`), not just the winner —
flip twins routinely tie on IoU, and the point of a flip panel is that you LOOK instead of
trusting the score. The winner additionally answers to `<verb>_best_<frame>.png` either
way.

The 100-panel visual budget is checked **ONCE for the whole order** (candidates × frames,
summed over frames), not per frame — ten frames of fifty candidates is five hundred
images, and the budget says so. `--sweep-visuals all` forces the per-candidate renders for
a bigger set too; `--sweep-visuals none` suppresses them and leaves only the winners.
Cost scales with frames, so scope with `--frames` to the frames you are actually asking
about.

To judge the panel against the source frames, build a candidate sheet — one row per
(candidate, frame), each resolving its own frame's source and mask:

```bash
micromamba run -n artscript env PYTHONPATH=harness python -m \
  analysis.viz.candidate_sheet --render-dir PASS_DIR --run-dir RUN_DIR
```

Under the render pool this is built for you before the order's result is published.

## What to paste

Both contracts leave `moved` / `joints` untouched, and both prefer the
`<verb>_poses.json` fragment (merged via [`multiagent.windows`](../../multiagent/windows.md),
single-writer) over hand-pasting — see [state_json.md](../../../conventions/state_json.md) for
the fragment shape.

**Per-frame (`osweep`, `oapply`) — EVERY frame, no inheritance.** Paste each scored
frame's `"pose"` block into its `FRAMES[<frame>]["pose"]`. There is no "paste the
reference and the rest inherit": the frames chose different rotations, which is the point,
so nothing can be inherited. **Read the rotation column before you paste** — if every
frame picked the same rotation, that is evidence the error is a build-orientation one,
so commit it once with [`oapply_all`](oapply_all.md) instead of N identical edits. If
they split, the split is the finding.

**Shared (`oapply_all`) — authored frames only; seeded frames inherit.** Paste each
**authored** frame's block (reference + any `moved`/authored frame). **Seeded frames need
no edit** — once the reference is pasted they inherit the same `R_extra` via the camera
re-seed, and the report lists which frames those are. Scoping with `--frames` to a SUBSET
turns inheritance off: every scored frame is then authored, and the report warns if the
reference frame is inside the subset, since pasting it would leak the rotation to
out-of-subset frames.

## Non-destructive — like every view

Every object verb snapshots and restores the base poses, world matrices, and
camera/render/scoring state, so a co-requested `match`/`turntable`/`depth` renders the
**original** poses and `pose.json` / the exported GLB are unaffected. The durable channel
is the paste block (or the fragment).

## The IoU here is a fast PROXY

Scoring is low-res numpy, exactly as in `sweep`/`apply` — good enough to rank, not to
certify. Re-verify a config you keep: paste the blocks, render `match` for all frames,
and run [`silhouette.py`](../../analysis/scorers/silhouette.md).

## They can't fix shape

These verbs only re-orient finished geometry. If a part is wrong, detached, or
under-modelled, edit `build()` first. And pick the right scope: a camera-frame drift
("the whole thing sits too far left, tipped a bit") is a [`sweep`](sweep.md) /
[`apply`](apply.md) job; one frame in a flipped basin is `osweep`/`oapply`; the SAME
re-orientation everywhere (a build-orientation error, a systematic gauge flip) is
`oapply_all`. **Near-symmetric axes tie** — a symmetric object can score the same at
`rz=0` and `rz=±180`, so read the top-K and the side-by-sides; the silhouette alone won't
break the tie. **Scale is not a knob here, or anywhere**: `SCALE` is the shared top-level
number for the built object and no per-frame verb searches it.
