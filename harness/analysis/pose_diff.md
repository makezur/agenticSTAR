# `pose_diff` — the residual between two poses of one object

Given two frames of the same articulated object, `pose_diff` reports how far it
moved: a base-pose residual plus per-joint deltas. It is the bottom of the analysis
stack — `analysis.temporal` (per-step residuals along a sequence, the
velocity/acceleration ladder, the per-frame report) is all
built on this one file, so the conventions below are the conventions of every
temporal number in the repo.

It measures and reports. It has no thresholds and returns no verdict.

```bash
micromamba run -n artscript env PYTHONPATH=harness \
  python -m analysis.pose_diff --pose-json RUN/mesh/pose.json \
    --frame-a 000020.jpg --frame-b 000030.jpg
```

## 1. The parametrisation

An object pose is `(s, R_i, t_i)` with **s shared across all frames**
(POSE.md §3), acting canonical → that frame's camera:

```
p_cam = s R_i p + t_i
```

Each frame may also carry an externally-provided **camera pose**
`(R_cam_i, t_cam_i)` — a 4×4 `camera_c2w`, from whatever tracker the run used. In
world coordinates:

```
p_world = s (R_cam_i R_i) p + (R_cam_i t_i + t_cam_i)
```

Reparametrise the translation as `t_i = s tau_i`, so **`tau` is in object units**.
Two quantities fall out, and they are the whole basis of this module:

```
W_i = R_cam_i R_i           the object's world ORIENTATION
u_i = R_cam_i tau_i         the camera->object vector, world AXES, object UNITS
```

`lift_placement` computes exactly this pair for one frame. It is the seam
everything else is built from — and the only **per-frame** world quantity in the
repo, which is what makes velocity and acceleration expressible at all
(`analysis.temporal.derivatives`); a purely pairwise API cannot differentiate twice.

### Why the camera translation is dropped

`u_i` deliberately excludes `t_cam_i`. Two reasons, and the first is a correctness
argument, not a simplification:

- **It is in a different gauge.** The tracker's translations are in the tracker's
  own units, which need not agree with the object's shared `s`. Adding
  `t_cam_j - t_cam_i` into the residual and then dividing the sum by `s` divides
  half of it by a scale it has nothing to do with. That is what the old
  `world_base_residual` did.
- **No re-pose can change it.** `t_cam` is fixed input. Excluding it makes the
  residual depend on exactly what a refiner controls.

**The visible consequence, stated plainly: a static object filmed by a
TRANSLATING camera reports non-zero translation.** That is the honest price of a
gauge-free number, not a bug to patch by adding `t_cam` back. A purely *rotating*
camera reads 0, and rotation reads 0 in both cases.

## 2. The residual

```
rotation     W_i^T W_j        reported as a geodesic ANGLE in degrees + a unit axis
translation  u_j - u_i        reported in object units AND multiplied by s
```

**This is the residual in every setup.** It does not change with the run, the
object, or what supervision was available. What varies is only whether a camera
pose was available (§3), and that is named on every result by `pose_frame`.

### The two halves are in different frames — read the tag, not the name

- **Rotation is genuinely world.** `W_i` is a true world orientation.
- **Translation is camera-centred but world-oriented.** `u_i` points from camera
  *i*'s centre to the object, expressed in world axes.

This is why the tag is **`oriented`** and not `world`: calling it `world` would
imply the camera is out of the translation, and it is not.

### Why body (right), not spatial (left)

The rotation residual is the **body-frame** increment `W_i^T W_j`, not the spatial
`W_j W_i^T`. The **angle is identical either way** — the two are conjugate — so a
wrong convention is invisible in the headline number and shows up only in the axis.

Body is right here because it is what an increment applied *to* frame *i*'s pose is
expressed in, and because body increments **telescope** right-to-left:

```
(W_i^T W_j)(W_j^T W_k) = W_i^T W_k
```

**The caveat that comes with it:** axes at different *i* live in *different*
frames, so two steps' axes are not comparable with each other. Magnitudes are.

Translation is a plain difference of two `u`, which share world axes, so
consecutive translation deltas **do** add to the whole displacement.

## 3. `pose_frame` — always tagged

- **`oriented`** — both frames carry a camera pose. Rotation in world frame,
  translation world-oriented about each camera. This is the honest read.
- **`camera`** — a camera pose is missing, so both halves are taken in camera-*k*
  coordinates and **conflate the object's motion with the camera's**. A static
  object filmed by a 70° orbit reads 70°, not 0°.

Same algebra either way (the camera branch is the same code with `R_cam = I`), so
the only difference is what was composed first. Never compare an `oriented` number
with a `camera` one — which is why the tag is on every step, why a pair with
exactly one camera pose **raises** rather than falling back, and why
`analysis.temporal.sequence` raises on partial camera coverage over a sequence
(all frames or none; there is no "mixed" report to caveat).

Joint deltas are frame-invariant and identical either way.

## 4. Units — both are reported

| field | unit | comparable with |
|---|---|---|
| `translation_dist_canon`, `translation_canon` | object units (fractions of the longest canonical dimension) | other runs, other objects |
| `translation_dist` | scene units (metric, `= s ×` the canon distance) | a depth residual, which is measured in the same units |
| `rotation_deg` | degrees, `[0, 180]` | anything — it is already scale-free |
| `offset_canon_a`, `offset_canon_b` | object units | each other; the states the derivative ladder is built from |
| joint `delta` | native (degrees revolute, canonical units prismatic) | that joint |
| joint `limit` | native, the declared bounds | `delta`/`state` — form the fraction you need yourself |

Emitting both translation units is the point: the canonical one is what makes "0.5"
mean *half the object's longest dimension* regardless of the object, and the metric
one is what lets a translation residual be weighed against a depth error without
re-deriving it. Note the canonical figure is a fraction **of** `s`, so resizing the
object changes it even though no pose moved.

The shared SCALE enters **exactly once**, inside `lift_placement`.

## 5. Trust the rotation angle

`rotation_deg` is `|so3_log(W_i^T W_j)|` in degrees. It is the right scalar for "how
much did it rotate": one number instead of three coupled euler deltas, and
bi-invariant, so it does not depend on a decomposition order you would otherwise
have to pick.

- **Double-cover robust.** `q` and `−q` are the same rotation and give the identical
  angle, so a quaternion-sign artifact can never fool it.
- **It surfaces flips, it never hides them.** A physical half-turn reads ≈**180°**,
  which is the *maximum* — not ≈0. The angle is deliberately not flip-invariant.
- **Its only limit is interpretation.** A genuine half-turn and a spurious jump into
  the opposite basin of a near-symmetric object produce the *same* number. Nothing
  in the poses separates them; only the video does.

That last point is why nothing downstream thresholds this number. A large value
means **"go look at this"** — never "distrust the math", never "this is wrong". The
verdict is recorded by an agent that looked at the frames
([report.md](temporal/report.md) §8 is the loop).

## 6. API

- **`lift_placement(object_pose, camera_c2w, scale)`** → `(R_world, offset_canon)`
  — §1's pair, for one frame. Reads only the camera's **rotation**.
- **`camera_rotation(camera_c2w)`** → the camera's rotation block, SVD-projected to
  the nearest proper rotation (mirroring `lie.pose_from_matrix`), so an
  externally-provided matrix that drifted off-orthonormal cannot leak a
  non-rotation or a reflection into a residual.
- **`pair_residual(a, b, scale, joints)`** → §2, where each of `a`/`b` is
  `(object_pose, joint_states, camera_c2w)`. Returns
  `{pose_frame, pose, joints}`, **one shape whatever the inputs**.
- **`diff_frames(pose_json, a, b)`** → the whole thing off a loaded `pose.json` (or
  a path), pulling the shared `scale` + `joint_defs` from the file. Returns
  `{frame_a, frame_b, scale, pose_frame, pose, joints}`.
- **`frame_track(pose_json, order)`** → the per-frame series
  `{frame, offset_canon, orientation_world_deg, joints}`, which
  `analysis.temporal.derivatives` differences into velocity and acceleration. A
  frame with no camera pose gets `None`, so a caller sees the gap rather than
  differencing an invented placement. `orientation_world_deg` is `log(W_i)` in
  degrees — a *reporting* unit; convert to radians before exponentiating.
- **`joint_deltas(states_a, states_b, joints)`** → the one home for per-joint change
  math. A joint absent from a state map is REST (0.0) per JOINTS.md §3 — states are
  absolute, so an omitted joint does not mean "unchanged".
- **`joint_states(states, joints)`** → each joint's **absolute** state with its
  `type` and declared `limit` beside it — the per-frame counterpart to
  `joint_deltas`' per-step change. Neither reports a percent: "where it sits"
  (`(state − lo) / span`) and "how much travel a step consumed" (`delta / span`)
  are different quantities one obvious name fits equally well, so both operands
  ship and the caller forms the ratio it means.
- **`pose_to_lie`** → one `object_pose` dict → a `lie.Pose`, from the authoritative
  `quaternion` + `translation`, **never** a stored `matrix4x4` (state_json.md §1: that
  derived cache went stale and won over the quaternion — the phantom-flip bug, closed
  structurally in `lie.pose_from_object_pose`, which cannot read one).
- **`load_pose_json` / `frame_entry`** → re-exported from
  [`lib/pose_read`](lib/pose_read.py), which owns the document-level **ingest** and
  its state_json.md gates (§1 both pose fields present, §2 `joints` never nested,
  §8 a missing rotation raises instead of reading as identity). Reading lives there
  rather than here so every reader is gated — `temporal.sequence`,
  `temporal.report` and `multiagent.windows` — not just the one that happens to diff.

### What is not here

**No rotation vector.** `rotation_vector_deg` / `rotation_vector_world_deg` are
gone. Axis-times-angle is the same information as `rotation_axis` + `rotation_deg`
multiplied together, and in *degrees* it is not a usable Lie-algebra element — so it
was a third field that could only ever disagree with the two that define it. The one
consumer rebuilt it from the axis and the angle anyway.

**No derivatives.** Velocity, acceleration and their radial component are
differences of `frame_track`, not of a pair, so they live in
`analysis.temporal.derivatives`.

Pure numpy + stdlib on `rig.lie`; no `bpy`/`mathutils`/`scipy`, so it runs in both
Blender's bundled python and the `artscript` env.
