# The maths of temporal residuals

Every temporal number in the repo — the pairwise residual, the per-step velocity,
the interior-frame acceleration, the radial split, the joint deltas — is derived
here, once. The code that implements each piece is named beside it; the consumer
doc for the table these numbers land in is [report.md](report.md).

Nothing in this file is a threshold or a verdict. §7 is why.

---

## 1. Parametrisation and the lift

An object pose is `(s, R_i, t_i)` with **`s` shared across all frames**
(POSE.md §3), acting canonical → that frame's camera:

```
p_cam = s R_i p + t_i
```

Each frame may also carry an externally-provided **camera pose** `(R_cam_i,
t_cam_i)` — a 4×4 `camera_c2w` from whatever tracker the run used. Composing:

```
p_world = s (R_cam_i R_i) p + (R_cam_i t_i + t_cam_i)
```

Reparametrise the translation as `t_i = s τ_i`, so **`τ` is in object units**
(fractions of the object's longest canonical dimension). Two per-frame quantities
fall out, and they are the basis of everything below:

```
W_i = R_cam_i R_i        the object's world ORIENTATION
u_i = R_cam_i τ_i        the camera->object offset, world AXES, object UNITS
```

`pose_diff.lift_placement` computes exactly this pair for one frame. The shared
`s` enters **exactly once**, there. It is the only per-frame *world* quantity in
the repo, which is what makes differentiating twice possible at all (§3) — a
purely pairwise API cannot express an acceleration.

### Why `t_cam` is dropped from `u`

`u_i` deliberately excludes the camera's own translation. Two reasons; the first
is a correctness argument, not a simplification:

- **It is in a different gauge.** The tracker's translations are in the tracker's
  own units, which need not agree with the object's shared `s`. Adding
  `t_cam_j − t_cam_i` into a residual and then dividing by `s` divides half the
  sum by a scale it has nothing to do with.
- **No re-pose can change it.** `t_cam` is fixed input, so excluding it makes the
  number depend on exactly what a refiner controls.

**The visible consequence, stated plainly: a static object filmed by a
TRANSLATING camera reports non-zero translation.** That is the honest price of a
gauge-free number. A purely *rotating* camera reads 0 on both halves.

## 2. The step residual (`pose_diff.placement_residual`)

Between frames *i* and *j*:

```
rotation      R_rel = W_i^T W_j  =  R_i^T (R_cam_i)^T R_cam_j R_j
              reported as the geodesic angle |log(R_rel)| in degrees + a unit axis

translation   Δu = u_j − u_i
              reported in object units (|Δu|) AND scene units (s |Δu|, metric —
              comparable with a depth residual, which is measured in those units)
```

### Rotation: body (right) increment, not spatial (left)

`W_i^T W_j`, not `W_j W_i^T`. The **angle is identical either way** (the two are
conjugate), so a wrong convention would be invisible in the headline number; only
the axis is at stake. Body is right because it is what an increment applied *to*
frame *i*'s pose is expressed in, and body increments **telescope**:

```
(W_i^T W_j)(W_j^T W_k) = W_i^T W_k
```

The caveat that comes with it: **axes at different *i* live in different frames**
and are not comparable with each other. Magnitudes are.

### Properties of the angle

- `[0, 180]`, and **double-cover robust**: `q` and `−q` give the identical angle.
- **It surfaces flips.** A physical half-turn reads ≈180° — the *maximum* — never
  ≈0. The angle is deliberately not flip-invariant.
- **One ambiguity, and it is not in the poses.** A genuine half-turn and a
  spurious jump into the opposite basin of a near-symmetric object produce the
  *same* number; only the video separates them. This is the fact §7 rests on.

Translation is a plain difference of two `u` sharing world axes, so consecutive
translation deltas **do** add up to the whole displacement.

## 3. The derivative ladder (`analysis.temporal.derivatives`)

Over the per-frame track of §1 states:

```
u_i                                   position       object units, world axes
v_i = u_i − u_{i−1}                   velocity       per STEP, indexed at the
                                                     later frame
a_i = v_{i+1} − v_i                   acceleration   per INTERIOR frame
    = u_{i+1} − 2 u_i + u_{i−1}

Ω_i = W_{i−1}^T W_i                   angular velocity increment (§2's residual)
α_i = log(Ω_i^T Ω_{i+1})              angular acceleration — a RATIO
```

**The first difference IS the pairwise residual**: `v_i` is precisely the step's
`translation_canon` and `|log Ω_i|` its `rotation_deg`, so the velocity layer
*reads them off the step record* rather than recomputing — one implementation per
quantity, and a producer and consumer cannot disagree about a number.

### Why `α` is a ratio, not a difference

`ω_{i+1} − ω_i` is not the log of any rotation: the two increments turn frames
one step apart, so their log vectors live in **different tangent spaces**, and
subtracting them subtracts coordinate triples. Both forms agree on the easy
readings — uniform motion about a fixed axis is 0 either way, and a single-axis
change gives the same number — which is exactly what made the wrong one easy to
keep. They diverge once the increments are about *different* axes: 80° about Z
then 80° about X is a **108°** ratio and a **113°** "difference".

### The fold — read `α` beside the two legs, never alone

Being a real rotation costs boundedness: `|α|` is a geodesic angle in `[0, 180]`,
so it *peaks* at a half-turn and a larger change folds back down:

| `|ω_in|`, sense | `|ω_out|`, sense | `|α|` | what happened |
|---|---|---|---|
| 170°, one way | 170°, reversed | 20° | a 340° reversal, folded past the peak |
| 10° | 30°, same way | 20° | a mild ramp |
| 170° | 170°, same way | 0° | fast but perfectly uniform |
| 90° | 90°, reversed | 180° | a reversal, right at the peak |

The two 20°s are the collision: identical `α`, wildly different motions. What
separates them is the legs' own per-step `rotation_deg` — 170 per step versus 10
and 30 — which is why the report prints the rotation velocity beside the angular
acceleration. **A small `α` therefore does not mean "smooth here";** a large one
always means the motion changed hardest there.

Translation's acceleration is a plain vector difference in one shared basis: no
wrap, no peak, no caveat.

## 4. The radial split

`u_i` points from camera *i*'s centre to the object, so its unit vector is the
**radial** direction there, and each derivative is projected onto it:

```
û_i = u_i / |u_i|
radial velocity        û_i · v_i     signed: > 0 = moving AWAY from the camera
radial acceleration    û_i · a_i     signed: > 0 = accelerating away
```

Read it against the row's own magnitude: a dot product alone is ≈0 *both* when
the motion is purely across the view *and* when there was no motion at all;
`|v_i|` separates those. The across-view part is deliberately **not** a separate
field — it is `sqrt(|v|² − radial²)`, the same fact twice.

A radial value is `None` when `u_i` has no direction (the object exactly at the
camera centre) — never a fabricated 0.

## 5. Joints

Joint states are **absolute**, in native units (degrees revolute, canonical units
prismatic), and frame-invariant — the camera never enters. The same ladder, per
joint:

```
delta_i  = state_i − state_{i−1}      per step        (pose_diff.joint_deltas)
change_i = delta_{i+1} − delta_i      per interior frame
net_i    = delta_i + delta_{i+1}      where it ended up net of the two steps
```

Plain subtraction is sound here because joint values are scalars in one shared
unit — none of §3's tangent-space trouble. An out-and-back (one frame in the
wrong basin between two agreeing neighbours) is `net ≈ 0` with both legs loud.

**No percent-of-travel field**, deliberately: `delta / (limit[1] − limit[0])` is
one division from fields already on the record, and one obvious name
(`percent_of_range`) fitted two different quantities — the travel a step
*consumed* versus the position a joint *sits at* — which can agree by coincidence
or mean opposite things (0% = "did not move" vs "pinned at the limit"). Both
operands ship; the consumer forms the ratio and names it.

## 6. `pose_frame` — which frame of reference a number lives in

- **`oriented`** — every frame carries a camera pose. Rotation is genuinely
  world; translation is camera-centred but world-*oriented* (§1). The two halves
  are not in the same frame — which is why the tag is not called `world`.
- **`camera`** — no frame carries one. The lift runs with `R_cam = I` (same
  algebra, one code path), so both halves are camera-relative and **conflate the
  object's motion with the camera's**: a static object filmed by a 70° orbit
  reads 70°, not 0°. The radial split is undefined.

Never compare a number from one against a number from the other. That is why
**partial camera coverage raises** — a sequence must have a camera pose on every
frame or on none (`sequence.diff_sequence`), and a pair with exactly one raises
in `pose_diff.pair_residual`. There is no "mixed" report to caveat.

## 7. What is deliberately NOT computed

- **No thresholds.** §2's one ambiguity is fatal to any cutoff: the thing that
  separates a genuine half-turn from a basin flip is not in the poses, it is in
  the video. A fixed cutoff fails in both directions — fires on genuinely fast
  motion, stays quiet on slow-but-impossible drift. So a large value means **"go
  look at this frame"**, never "this is wrong": the tooling measures, the agent
  that can see the frames judges.
- **No self-calibrating statistics** (median/MAD over the sequence). Over a
  5-frame refiner window that is four samples, one of which may be the flip you
  are hunting: noise dressed as rigor.
- **No division by `frame_gap`.** Dividing a step by its gap invents a smooth
  per-frame velocity the pair never demonstrated, and quietly makes a huge jump
  across a wide gap look comparable to a small one across a narrow gap. The gap
  is *reported* beside every number instead; "velocity" everywhere in this
  package means **per-step delta**, not per-unit-time.
- **No angular-velocity vector.** `axis × angle` in degrees is not a Lie-algebra
  element, and the axis half is body-frame — not comparable across steps (§2). The
  scalar `rotation_deg` is the angular speed; the increment *matrix* is what the
  acceleration composes.
- **No tangential field, no percent-of-travel** — §4 and §5.
