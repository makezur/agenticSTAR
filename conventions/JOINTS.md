# JOINTS.md — articulation convention

The **single place** for how this repo represents articulation. If a convention
here disagrees with the code, the code wins — the source of truth is
[harness/rig/transforms.py](../harness/rig/transforms.py) (`joint_transform`,
`forward_kinematics`, `joint_limit`) and [harness/rig/scene.py](../harness/rig/scene.py)
(`_check_frame_joints`, which enforces limits at load). For object pose, frames,
and scale, see [POSE.md](POSE.md).

You declare joints in `scene.py`; the harness applies **forward kinematics** per
frame on top of that frame's base pose.

---

## 1. The model

Where a part moves relative to the body **between frames** — a door opening, a
drawer sliding, a lid lifting — model it as a **joint with a per-frame state**, not
as a re-posed whole object.

This makes the deliverable a URDF-like **rig** rather than a frozen scan: a
canonical rest-pose mesh (one watertight part per component) plus articulation
parameters (each joint's `type`, `origin`, `axis`, `limit`, and each frame's DOF
value). The joint graph keeps parts kinematically attached; the mesh only needs
to depict geometry supported by the observations. The exported GLB is a visual
rig, not a validated collision model; collision geometry, swept clearance, and
self-collision checks are downstream concerns.

---

## 2. `JOINTS` schema

`JOINTS` is a **shared** list of joint *definitions* (no per-frame state here),
declared in the **canonical frame** (see [POSE.md](POSE.md) §1). Empty list = a
rigid object.

```python
JOINTS = [{
    "name":   "door_hinge",
    "type":   "revolute",              # "revolute" | "prismatic" | "fixed"
    "child":  ["door", "handle"],      # part name OR list = the rigid GROUP that rides it
    "origin": (x, y, z),               # joint origin, canonical frame
    "axis":   (x, y, z),               # unit joint axis, canonical frame
    "limit":  (lo, hi),                # valid range (see §4); rest 0 lies in it
    # "parent": "other_joint",         # OPTIONAL: name of the joint this hangs off (chain)
}]
```

| field | meaning |
|---|---|
| `name` | unique joint id; referenced by per-frame states |
| `type` | `revolute` (rotation), `prismatic` (slide), or `fixed` (rigid weld, no DOF) |
| `child` | the part name, or a **list** = the rigid group of parts that all ride this joint together |
| `origin` | joint origin `(x,y,z)` in the canonical frame — put it on the mechanism's axis (a hinge pin at the door's rim) |
| `axis` | unit joint axis `(x,y,z)` in the canonical frame — the rotation axis (revolute) or slide direction (prismatic) |
| `limit` | `(lo, hi)` valid range — see §4. Omit only for a genuinely continuous joint (e.g. a spinning wheel) |
| `parent` | optional: another joint's `name`, for kinematic chains (§5). Absent = attached to the object base |

**Units by type:** revolute state/limit is in **DEGREES**; prismatic is in
**canonical units** (the same units `build()` uses, longest dim ≈ 1). `fixed` has
no DOF. (Degrees are the agent-facing unit everywhere — scene.py, sweep/apply
specs, reports; the harness converts to radians only inside the FK matrix
kernel.)

---

## 3. Per-frame states

Each frame's DOF values live in `FRAMES` (**not** in `JOINTS`):

```python
FRAMES = {
    "000000.jpg": {"joints": {"door_hinge": 0.0}},    # rest = 0 = shut
    "000080.jpg": {"joints": {"door_hinge": -80.0}},  # 80° open
}
```

`joints` maps `joint_name -> value` (revolute degrees / prismatic canonical
units). For how a `joints` block is SERIALIZED alongside a pose — sibling of the
pose object, complete-or-absent, 6dp — see [state_json.md](state_json.md) §2.

### A joint state is ABSOLUTE, so it is never defaulted

**Every frame must declare a state for every articulated joint.** Omitting one — or
omitting `joints` entirely — is a **hard error at load** naming the frame and the joint
([scene.py](../harness/rig/scene.py), via [core/joints.py](../harness/core/joints.py)
`require_complete_states`). `fixed` joints are exempt: no DOF, identity at any state.

This is the opposite of how the pose DOFs work, and it is the rule to internalise:

| | measured from | omitted means |
|---|---|---|
| pose order (`roll`/`yaw`/`pitch`/`dpx`/`dpy`/`tz`) | an **increment** onto the frame's placement | `0` — a lossless no-op |
| joint state (`door_hinge: -80`) | **nothing**; it IS the state | *no such value exists* |

`0` is not a joint's identity but a *position* — "fully shut". Filling an unstated
joint with `0` therefore MOVES it, invisibly: a straightened door renders perfectly
well. So every layer refuses instead of guessing — the loader, the pool worker's
off-scene path, and the sweep engines (which hold an unswept joint at the frame's
declared state).

Carrying a state forward is likewise *your* job when handing a pose between tools: a
placement carried without its articulation is **the revert**, rendering the new pose at
the old joint states. So every report writes the **whole** state — each sweep candidate
and each object-verb frame block names every declared joint, held or swept — and
`pool.reseed` carries pose and joints from one address
([README](../harness/pool/README.md)).

---

## 4. Limits — enforced, not advisory

`limit = (lo, hi)` is the URDF `<limit lower= upper=>`. Rest (0) should lie inside
it. It exists so **impossible configurations cannot render or ship** (a door swung
360°, a drawer pulled past its travel):

- A per-frame `joints` state **outside** its joint's `(lo, hi)` is a **hard error
  at load** — the harness rejects the scene, naming the frame/joint/state
  ([scene.py](../harness/rig/scene.py) `_check_frame_joints`). Fix the state or widen
  the `limit`. The authored numbers, the render, and `pose.json` therefore always
  agree.
- A malformed `limit` (`lo > hi`, or not a 2-item pair) also fails loudly rather
  than silently disabling the clamp ([transforms.py](../harness/rig/transforms.py)
  `joint_limit`).
- A runtime clamp (`clamp_joint_state`) is a backstop for **programmatic** drivers
  (a future video optimizer) that bypass the authored-scene gate; for a normal
  authored render it is a no-op.
- `aggregate.py` reports each joint's **observed** state range across frames vs its
  declared `limit`, so you can tighten a too-wide range or widen one a state
  needed.

The `sweep` view confines its joint search to the declared `limit`.

---

## 5. Forward kinematics

The harness applies FK per frame on top of the base pose
([transforms.py](../harness/rig/transforms.py) `forward_kinematics`, `joint_transform`
— the source of truth):

- **revolute** → rotate `state` degrees about `axis` through `origin`:
  `Translation(origin) @ Rotation(state, axis) @ Translation(-origin)`.
  Right-handed: positive `state` turns counter-clockwise looking **down** `axis`
  (from `+axis` toward the origin).
- **prismatic** → translate `state` along `axis`.
- **fixed** → identity.

### The `axis` SIGN

Negating `axis` is exactly negating every `state`: a sign error does not break
the joint, it moves the group **the other way**. For a **revolute** joint this
hides from scored signals — rotation is periodic, so both extremes of a 0..180
hinge render identically under either sign (−180° ≡ +180°) and only the
intermediate states differ. (A **prismatic** sign error has no such hiding
place: only the zero state agrees.)

When the `mechanism` module is enabled (it is optional and off by default),
after authoring or changing `axis`/`origin`/`limit`/`child` the `mechanism`
view renders your joint **twice** — your declared axis and its negation, on blind
`A`/`B` labels — and asks which arc is the real mechanism. The pick is **graded**
against the declaration: if the declaration is mirrored relative to the arc you
identify, it BLOCKS and prints the sign to write here.

You answer this **once per mechanism**. The freshness hash covers the axis's
unsigned LINE, not its sign, because the two arms are that line's two signs and
flipping it re-renders the same pair of arcs — so writing the named sign makes
your existing pick correct and the gate passes with no re-render and no second
answer ([views/mechanism.md](../harness/views/mechanism.md)).

Each joint's canonical-frame local transform `L_j` accumulates down its parent
chain (`A_j = A_parent @ L_j`; cycles are detected), and the per-part world
transform is the base pose conjugating that accumulation:

```
W = base_pose · A_j · base_pose⁻¹
```

applied to every part in the joint's `child` group. Parts under no joint stay at
the base pose. `parent` builds kinematic chains (a handle joint hanging off a door
joint).

---

## 6. Attachment and geometry scope

The joint graph, not mesh intersection or modeled hardware, establishes
parent-child attachment. An articulated child and its parent do **not** need to
touch or overlap.

- Do not invent hidden hinge leaves, pins, axles, rails, or guides just to make
  the motion mechanically explanatory.
- Model support hardware when it is visible in the source or materially affects
  the object's silhouette, identity, or observed occlusion.
- Check observed states for clearly wrong part depth and gross
  interpenetration. Collision safety and unobserved swept clearance are out of
  scope.
- **Never boolean-UNION parts that articulate** relative to each other. Keep
  them as separate objects so forward kinematics can move them independently.

The turntable renders every observed state to check 3D shape and placement. See
[harness/shapes/README.md](../harness/shapes/README.md) for the build helpers and
boolean rules.

---

## 7. Worked example — a hinged door

```python
JOINTS = [{
    "name": "door_hinge", "type": "revolute",
    "child": ["door", "handle"],           # the door + its handle ride together
    "origin": (0.35, -0.2, 0.0),           # hinge at the door's rim, canonical frame
    "axis":   (0.0, 0.0, 1.0),             # swings about canonical up (+Z)
    "limit":  (-120.0, 0.0),               # degrees: 0 = shut, -120 = fully open
}]

FRAMES = {
    "000000.jpg": {"joints": {"door_hinge": 0.0}},     # reference: shut
    "000080.jpg": {"joints": {"door_hinge": -80.0}},   # 80° open
}
```

See [examples/scene_example.py](../examples/scene_example.py) for the full runnable
scene (the mug is rigid; the door hinge is shown there as a commented example).

---

## Source of truth

- FK + joint transforms + limits: [harness/rig/transforms.py](../harness/rig/transforms.py).
- Load-time limit enforcement + scene loading: [harness/rig/scene.py](../harness/rig/scene.py).
- Build helpers & boolean rules: [harness/shapes/README.md](../harness/shapes/README.md).
- The task that uses all of this: [AGENT_TASK.md](../AGENT_TASK.md).
