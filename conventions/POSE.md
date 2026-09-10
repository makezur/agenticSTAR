# POSE.md — pose, frames, scale & units

The **single place** for how this repo represents an object's placement. For how
tools **serialize** a frame's state (pose + joints) to JSON — key names, precision,
container shapes — see [state_json.md](state_json.md). If a
convention here disagrees with the code, the code wins — the source of truth is
[harness/rig/lie.py](../harness/rig/lie.py) (pure-numpy pose core) and
[harness/rig/transforms.py](../harness/rig/transforms.py) (the Blender-facing layer
that composes and applies pose). This doc is the human-readable contract those
files implement. For articulation (joints), see [JOINTS.md](JOINTS.md).

You author pose in `scene.py`; the harness **composes and applies** it. You never
move the camera — every frame renders from the fixed camera 0, and you place the
object in front of it.

---

## 1. The two frames

Everything lives in one of two coordinate frames. Getting them straight is 90% of
posing correctly.

### Canonical object frame — where you `build()`
`build()` creates geometry in the object's **own** frame, with no camera
awareness:

- **+Z = up**
- **centered at the origin**
- **longest dimension ≈ 1 unit**
- built in the **rest** (un-actuated) configuration — door closed, drawer shut

This is the shared base representation: the canonical mesh is what ships in the
GLB, and both pose and joints are declared relative to it. (Echoed in
`pose.json` as `"canonical": "+Z up, centered at origin, longest dimension ~= 1"`.)

> **These are a CONVENTION, not a checked invariant.** `core/scene_contract.py`
> validates the *source* of `scene.py` (required names, right shapes); nothing
> measures the built mesh, so a `build()` whose geometry sits off to one side — or
> spans 3 units — passes every check and renders fine. So code that needs the object's
> centre must **derive it from the geometry, never assume the canonical origin** (an
> off-centre object orbited about `c_canon = 0` swings right out of frame). Both
> sweep grammars derive it — the camera-frame pivot and the per-frame canonical
> pivot are the same FK-AABB measurement
> ([harness/core/centre_calculation.py](../harness/core/centre_calculation.py),
> measured in [lib/pivot.py](../harness/views/sweeps/lib/pivot.py)); only
> `oapply_all`'s shared rotation still turns about the canonical origin, by
> design (see "The osweep/oapply order" below).

### Camera-0 frame — where you pose
The fixed "match" camera sits at the world origin with an identity extrinsic, in
the **OpenCV/SfM** convention:

- **+X = image right**
- **−Y = image up**  (world +Y points **down** in the image)
- **+Z = into the scene** (straight ahead of the camera, the view direction)

These axes are locked in code as `Z_CAM = (0,0,1)`, `UP_CAM = (0,-1,0)`,
`RIGHT_CAM = (1,0,0)` in [lie.py](../harness/rig/lie.py) and `CAM0_UP = (0,-1,0)` in
[camera.py](../harness/rig/camera.py); `pose.json` records
`"frame_convention": "camera0_opencv (+X right, -Y up, +Z into scene)"`.

A per-frame pose therefore has to **orient** the canonical +Z ("up") to run along
world −Y and face the camera, and **translate** the object to positive Z (a few
units ahead) plus an X/Y shift to where it sits in the photo.

---

## 2. Pose representation

A per-frame `pose` is a rotation + a translation (no scale — scale is shared, see
§3):

```python
"pose": {
    "quaternion":  (w, x, y, z),      # scalar-first UNIT quaternion
    "translation": (tx, ty, tz),      # camera-0 frame, scene units
}
```

- **Rotation = a scalar-first unit quaternion `(w, x, y, z)`**, an *active*
  rotation of a point. This layout matches `mathutils.Quaternion` and three.js, so
  it is consistent across the whole stack (render, sweep). It is the
  authoring form and what the `sweep` / `measure_depth` tools paste.

---

## 3. Composition & scale — `M = T · R · S`

The harness composes a single canonical→camera-0 matrix per frame:

```
M = T · R · S            i.e.   p_cam = s · R @ p_canon + t
```

**Scale first, then rotate, then translate** ([transforms.py](../harness/rig/transforms.py)
`compose_pose`; [lie.py](../harness/rig/lie.py) `Pose` / `pose_to_matrix`). Reading
that out:

1. **`S` (scale)** sizes the ~unit canonical object up to its **physical size**.
2. **`R` (rotate)** orients it into the camera-0 frame.
3. **`t` (translate)** places it. **`t` is in the camera-0 frame, in scene
   units, and is applied *after* scaling** — so it is independent of `S`. You put
   the object a few units ahead of the camera along +Z (e.g. `tz ≈ 2.2`) and shift
   in X/Y to center it in the photo.

### `t = s · τ` — translation in object units

Because `t` is applied after scaling, it is in **scene units**, which makes it
incomparable across objects of different size. Reparametrising it against the shared
SCALE

```
t_i = s · τ_i          so   τ_i = t_i / s
```

puts the translation in **object units** — fractions of the object's longest
canonical dimension — so `τ = 0.5` means "half an object length" whatever the
object. This is not a second representation: nothing serializes `τ`, and `t`
stays the authoritative field. It is the unit every *analysis* number is reported
in, and the reason a residual is comparable between two runs at all
([analysis/pose_diff.md](../harness/analysis/pose_diff.md) §1, which composes it
with a per-frame camera rotation into `u_i = R_cam_i τ_i`).

Note the direction of the dependency: `τ` is a fraction **of** `s`, so changing the
shared SCALE changes every canonical number even though no per-frame pose moved.

### The sweep/apply order — the left update

The camera-frame order (`sweep`/`apply`, [lie.py](../harness/rig/lie.py)
`apply_order`) composes onto a frame's pose as

```
p_new = s · dR · R @ (p − c)  +  s · R @ c  +  t  +  dT
```

`c` is the pivot (the object's canonical FK-AABB centre,
[centre_calculation.py](../harness/core/centre_calculation.py)), `dR` the
roll/yaw/pitch increment, `dT` the dpx/dpy/tz nudge — so the object spins about
its own centre with framing preserved, and `s` passes through untouched.

### The osweep/oapply order — the right update

The canonical-frame order (`osweep`/`oapply`, [lie.py](../harness/rig/lie.py)
`right_reorient`) composes onto a frame's pose as

```
p_new = s · R · dR @ (p − c)  +  s · R @ c  +  t
```

i.e. `M' = M · T(c) · dR · T(−c)`: `R_new = R·dR`, `t_new = t + s·R@(I − dR)@c`,
`s` untouched. It mirrors the left update with the delta on the other side: left
`dR` reads in **camera** axes, right `dR` in the object's **canonical** axes, and
both fix the same FK-AABB centre `c` — a canonical `dR` about `c` **is** the
camera-frame rotation `R·dR·Rᵀ` about the transported pivot `s·R@c + t` (the two
forms are verified to agree numerically). With `c = 0` the
correction vanishes and `t` is untouched; only `oapply_all` keeps that, by design
([object_rotations.md](../harness/views/sweeps/object_rotations.md)).

### SCALE is one shared object property
`SCALE` is one **uniform numeric scalar**, declared **once** and reused by every
frame. Anisotropic `(sx, sy, sz)` scale is rejected: non-uniform scale does not
commute with rotation and makes pose composition ambiguous. Encode axis
proportions directly in canonical `build()` geometry. The object does not change
size as the camera moves, so scale is not a per-frame knob — per-frame `pose`
dicts carry **no** scale, and scale drift across frames is not even expressible.
Tune `SCALE` so the silhouette fills the frame like the source (absolute size
matters — the primary gate is a raw 1:1 silhouette IoU).

---

## 4. Per-frame model

`FRAMES` has one entry per observed frame:

- **`REFERENCE_FRAME`** — the frame easiest to reconstruct. It anchors the gauge:
  its camera is the identity extrinsic. **Author its `pose` directly** — omitting it
  is a **hard error at load**, not a default.

  A pose *increment* has an identity (§3), but this is an absolute **placement** and
  every other frame is seeded *from* it — a conjured gauge would silently become the
  run's ground truth and ship into `pose.json` as the estimate. Same defect class as
  defaulting a joint state ([JOINTS.md §3](JOINTS.md)).
- **Non-reference frames** — with Pi3X, the camera motion between frames is known,
  so the harness **auto-seeds** each non-reference frame's pose from the reference
  via that relative camera motion ([transforms.py](../harness/rig/transforms.py)
  `reseat_pose`, which reads back a quaternion + translation and keeps scale
  shared). **Omit `pose`** on these frames to accept the seed; you usually only set
  their joint states.
- **`"moved": True`** — set this (and author a `pose`) **only** when the *object
  itself* was physically moved between frames (picked up, slid). Default
  (absent/False) means the object did not move, so any visible change must be a
  joint actuating (see [JOINTS.md](JOINTS.md)), not a fake whole-object re-pose. An
  authored `pose` always overrides the seed; `moved` is the semantic label that
  records *why*.

Use `--debug-project x,y,z` to check where a canonical point lands in pixels under
the composed pose.

---

## 5. What ships — `pose.json`

Per frame, [transforms.py](../harness/rig/transforms.py) `frame_pose_to_dict`
serializes the base pose (this function is the serialization source of truth):

| field | role |
|---|---|
| `quaternion` `[w,x,y,z]` | **authoritative** rotation |
| `translation` `[tx,ty,tz]` | camera-0 frame, scene units |
| `scale` | per-frame echo of the shared `SCALE` |

These three fields fully determine the pose; no matrix or euler echo is stored.
Every consumer that needs the 4x4 derives it fresh
(`lie.object_pose_matrix`), making a stale-matrix bug unrepresentable.
`pose.json` also carries the top-level `frame_convention` and `canonical`
strings above, the shared `scale`, the `reference_frame`, and the joint
definitions (see [JOINTS.md](JOINTS.md)). The full container shapes, and the rule
for when a tool writes `object_pose` vs `pose`, are in
[state_json.md](state_json.md).

---

## 6. Worked example (the mug)

The canonical mug stands upright (its cylinder axis is already local +Z = canonical
up). To face camera 0, canonical +Z must map to world −Y (image up). A **+90°
rotation about X** does exactly that:

```python
FRAMES = {
    "frame0": {                                            # the reference
        "pose": {
            "quaternion":  (0.707107, 0.707107, 0.0, 0.0), # +90° about X: +Z -> -Y
            "translation": (0.0, 0.0, 2.2),                # 2.2 units ahead of cam 0
        },
    },
}
```

`(cos45°, sin45°, 0, 0) = (0.7071, 0.7071, 0, 0)` is the +90°-about-X quaternion.
See [examples/scene_example.py](../examples/scene_example.py) for the full runnable
scene.

---

## Source of truth

- Pose math (quaternion algebra, `Pose`, exp/log, sweep increments):
  [harness/rig/lie.py](../harness/rig/lie.py) — pure numpy, tested against a scipy
  oracle.
- Composition / application / serialization (`compose_pose`, `reseat_pose`,
  `frame_pose_to_dict`): [harness/rig/transforms.py](../harness/rig/transforms.py).
- Camera-0 frame + Pi3X per-frame cameras: [harness/rig/camera.py](../harness/rig/camera.py).
- The task that uses all of this: [AGENT_TASK.md](../AGENT_TASK.md).
