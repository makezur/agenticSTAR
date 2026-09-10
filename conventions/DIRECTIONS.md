# DIRECTIONS.md — what a positive sweep/apply DOF looks like on screen

The **single place** for the visual reading of each camera-frame order DOF's
sign. If a convention here disagrees with the code, the code wins — the source
of truth is [harness/rig/lie.py](../harness/rig/lie.py) (`roll_delta` /
`yaw_delta` / `pitch_delta` / `apply_order`, right-handed rotations about the
camera axes `Z_CAM`/`UP_CAM`/`RIGHT_CAM`). Each row below was verified
empirically: apply one DOF positive, watch the render. For the frames themselves see
[POSE.md](POSE.md); for joints see [JOINTS.md](JOINTS.md).

The camera-0 basis is OpenCV-style: **+X = image right, +Y = image DOWN
(camera-up is −Y), +Z = into the scene.** All rotations are right-handed about
those axes and orbit the object's bbox centre (the object spins/tilts in place;
framing is preserved).

## Camera-frame order DOFs (`sweep` / `apply` / `--sweep-start-shift`)

| DOF | + value looks like | − value looks like | directional preset tokens (+ / −) |
|---|---|---|---|
| `roll` | object spins **clockwise** on screen | counter-clockwise | `cw` / `ccw` |
| `yaw` | the object's **LEFT edge swings closer** to the camera; its right edge recedes (you see more of the left side) | right edge swings closer | `leftcloser` / `rightcloser` |
| `pitch` | the object's **TOP edge swings closer** (top tips toward the camera) | bottom edge swings closer (top tips away) | `topcloser` / `bottomcloser` |
| `dpx` | silhouette shifts image-**right** (fraction of frame width) | left | — |
| `dpy` | silhouette shifts image-**down** (fraction of frame height) | up | — |
| `tz` | object **recedes** (farther, slightly smaller) | comes closer | — |

Why `roll` + is clockwise: the rotation is right-handed about +Z-into-scene,
and image +Y points DOWN, so +X→+Y sweeps right→down — which reads clockwise
on screen.

Why the rotation tokens name which EDGE approaches: "turn left/right" and "tip
toward/away" are ambiguous (turning the object vs. its visible face; toward
what?), and the two readings are opposite. Which edge comes closer is
unambiguous: **+yaw = left edge closer** (right-handed about camera-up −Y),
**+pitch = top edge closer** (right-handed about camera-right +X).

## Directional angle presets — spend the whole budget where the pose is off

The `sweep`/`osweep` angle presets (`tiny ±5° / standard ±15° / large ±30° /
huge ±45°`) are **symmetric** bands by default. But when you can SEE which way
a pose is off (you almost always can, from the photo), append a **direction**
to the preset and the band becomes **one-sided**, spending the whole `2·half`
budget on the side you saw — same candidate count, none wasted on the wrong
side, and an underestimated defect still lands inside:

| axis | `axis:standard_<dir>` searches | which direction |
|---|---|---|
| `roll` | `cw` → `[0°, +30°]`, `ccw` → `[−30°, 0°]` | clockwise / counter-clockwise |
| `yaw` | `leftcloser` → `[0°, +30°]`, `rightcloser` → `[−30°, 0°]` | which edge comes closer |
| `pitch` | `topcloser` → `[0°, +30°]`, `bottomcloser` → `[−30°, 0°]` | which edge comes closer |

(Shown for `standard`; every band scales — `tiny_cw` = `[0°, +10°]`,
`large_cw` = `[0°, +60°]`, etc.) A **directional** preset is PREFERRED over the
symmetric one for every band except `tiny` (a genuine last-mile touch-up where
the sign no longer matters). Canonical `rx/ry/rz` (the object-centric views) have
**no**
directional tokens — a positive canonical rotation has no fixed on-screen
reading (it depends on the pose), so those bands stay symmetric.

## Object-canonical DOFs (`osweep` / `oapply` / `oapply_all`) — a DIFFERENT frame

`rx`/`ry`/`rz` form one rotation vector, right-handed about the object's OWN
canonical axes (+X right, +Y front, +Z up as built), right-multiplied onto a
frame's pose (`M' = M·R_extra`) — per frame in `osweep`/`oapply`, onto every
frame at once in `oapply_all`. `flip:x|y|z` are the discrete 180°
flips about them. What a positive `rz` looks like ON SCREEN therefore depends
on how the object is posed relative to the camera — there is no fixed visual
reading; the camera-frame table above does NOT transfer. Verify a canonical
rotation by rendering the flip/rotation panel and looking
([osweep.md](../harness/views/sweeps/osweep.md),
[oapply.md](../harness/views/sweeps/oapply.md),
[oapply_all.md](../harness/views/sweeps/oapply_all.md)).

Still, the sign IS fixed in the object's own frame: a positive rotation swings
the next axis cyclically toward the one after it — **`rx` turns +Y toward +Z,
`ry` turns +Z toward +X, `rz` turns +X toward +Y**. Positive reads **clockwise only
when you sight ALONG the positive axis** (axis pointing away, so screen +Y is
down — same reason `roll` + is clockwise above); from the axis TIP it is
counter-clockwise. Never write "clockwise" without saying which end you sight
from.

Apply the same `rx`/`ry`/`rz` to three different viewpoints at once and watch ONE
shared rotation land differently on each. That the three panels move differently
under the SAME rotation is the
"no fixed on-screen reading" claim, and it is why these axes get no directional
preset tokens. Measured: with the same `rz:+30`, gate IoU falls to 0.80 in one
view but 0.54 in another, and the apparent motion of a given landmark differs by
up to 169° between frames. That measurement is also why there is no whole-run
*searched* object-centric view: a shared rotation ranked by MEAN IoU can be
mediocre in every frame, so a shared rotation is committed imperatively
(`oapply_all`, a panel you look at) and per-frame fits are searched (`osweep`).

## Joints

A joint DOF in a sweep is the ABSOLUTE state (revolute: degrees; prismatic:
canonical units), clamped to the declared `limit`. Its visual direction is set
by the joint's declared `axis` in the canonical frame ([JOINTS.md](JOINTS.md))
— per-joint, not global; when in doubt render two states and look.
