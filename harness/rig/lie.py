"""lie.py — pure-numpy SO(3)/Sim(3) pose core + camera-frame increment operators.

WHY THIS MODULE EXISTS
----------------------
The object canonical->camera pose is a Sim(3): a uniform scale `s`, a rotation
`R`, and a translation `t`, acting on a canonical point as

    p_cam = s * R @ p_canon + t          (== the 4x4  M = T @ R @ S)

The rotation is a unit QUATERNION with proper SO(3)/Sim(3) exp/log (not Euler
angles, which suffer gimbal lock, order coupling, and have no clean way to
compose small increments), so an increment `delta_T` composes onto the current
pose `T` exactly and branch-free (T' = delta_T . T).

It also defines the sweep's new "order" vocabulary: intuitive, scale- and
resolution-invariant camera-frame increments (roll / yaw / pitch about the object
centre, an image-plane shift as a fraction of the frame, and a depth nudge as a
fraction of the object's depth) that build a left-multiplied delta_T — see
apply_order(). The order is RIGID: scale is global and not a sweepable DOF.

PURITY / PORTABILITY
--------------------
PURE NUMPY. No `bpy`, no `mathutils`, no `scipy`. This is deliberate: the module
must import BOTH inside Blender's bundled python (numpy 1.24, which drives the
render/sweep) AND in the `artscript` analysis env (numpy 2.x), so the same
math backs the harness and can be reused out-of-Blender. rig/transforms.py
converts between these numpy 4x4s and `mathutils.Matrix` at the Blender boundary.

CONVENTIONS (locked to the fixed camera-0 frame — see rig/camera.py, transforms.py)
-----------------------------------------------------------------------------------
  * Camera-0 world axes (OpenCV/SfM): optical axis  +Z = into the scene,
    camera-up = -Y (world +Y points DOWN in the image), camera-right = +X.
    These constants (Z_CAM / UP_CAM / RIGHT_CAM) MUST match camera.CAM0_UP
    (-Y) and the +X camera-right the sweep's yaw/pitch orbits about.
  * Quaternion layout: SCALAR-FIRST (w, x, y, z), unit-norm, an ACTIVE rotation
    of a point. This matches mathutils.Quaternion and three.js, so the layout
    is consistent across the whole stack.
  * A Pose acts as p_cam = s*R@p + t, i.e. pose_to_matrix(P) == T @ R @ S, the
    SAME composition as transforms.compose_pose.
"""

import sys
from dataclasses import dataclass
from numbers import Real

import numpy as np

# --------------------------------------------------------------------------- #
# camera-0 world axes (see module docstring; keep in lockstep with transforms.py)
# --------------------------------------------------------------------------- #
Z_CAM = np.array([0.0, 0.0, 1.0])      # optical axis, +Z into the scene (roll axis)
UP_CAM = np.array([0.0, -1.0, 0.0])    # camera-up = world -Y  (yaw axis)
RIGHT_CAM = np.array([1.0, 0.0, 0.0])  # camera-right = world +X (pitch axis)

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# shared scale contract
# --------------------------------------------------------------------------- #
def scalar_scale(value, label="scale"):
    """Return a numeric scalar scale or reject tuple/list/array scale.

    Scale is part of Sim(3), where it must commute with rotation. Axis-specific
    proportions belong in the canonical geometry, not in the object placement.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(
            f"{label} must be one numeric scalar; anisotropic scale sequences "
            "are forbidden because non-uniform scale does not commute with "
            "rotation. Encode axis proportions in build() geometry.")
    return float(value)


# --------------------------------------------------------------------------- #
# SO(3): quaternion algebra
# --------------------------------------------------------------------------- #
def quat_normalize(q):
    """Return `q` scaled to unit norm (guards against accumulated drift).

    A zero quaternion (degenerate) collapses to the identity (1,0,0,0)."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


# A serialized (six-decimal JSON) quaternion is non-unit only at the ~1e-6 level,
# and normalizing it away is silent and correct. A norm this far from 1 is NOT
# rounding — it is a malformed quaternion (unnormalized, scaled, or a rotation
# vector / euler triple pasted where a scalar-first quaternion belongs). The
# harness normalizes it anyway (so the render/analysis never crashes), but a
# quaternion that was authored wrong will read as the WRONG rotation once scaled
# to unit norm, so we shout rather than swallow it. 1e-2 leaves four orders of
# magnitude of headroom over JSON rounding.
QUAT_NORM_WARN_TOL = 1e-2


def check_quat_norm(q, where=""):
    """Warn LOUDLY (to stderr) if `q`'s norm is too far from 1 to be JSON rounding.

    Returns the norm. Non-fatal: callers still normalize (a slightly-off q is
    fine; the point is to surface a q that is off by more than serialization
    could ever explain — a sign the quaternion was authored/typed wrong, e.g. a
    rotation vector pasted into a quaternion slot). `where` names the source
    (a frame name / call site) so the warning is actionable."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        return float(np.linalg.norm(q))
    n = float(np.linalg.norm(q))
    if n > _EPS and abs(n - 1.0) > QUAT_NORM_WARN_TOL:
        loc = f" [{where}]" if where else ""
        print(f"WARNING: quaternion norm {n:.6g} is far from 1"
              f" (|norm-1| = {abs(n - 1.0):.3g} > {QUAT_NORM_WARN_TOL:g})"
              f"{loc}: too large for JSON rounding — the quaternion is likely "
              "MALFORMED (unnormalized, scaled, or a rotation-vector/euler triple "
              "in a scalar-first (w,x,y,z) quaternion slot). Normalizing it "
              f"anyway to {tuple(round(float(v), 4) for v in q / n)}, but if that "
              "is not the rotation you meant, fix the source quaternion.",
              file=sys.stderr, flush=True)
    return n


def quat_mul(q1, q2):
    """Hamilton product q1 * q2 (scalar-first).

    Composition convention: the product corresponds to matrix R1 @ R2, i.e.
    applying q2's rotation first, then q1's. So quat_to_matrix(quat_mul(a, b))
    == quat_to_matrix(a) @ quat_to_matrix(b)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q):
    """Conjugate (w, -x, -y, -z) — the inverse rotation for a UNIT quaternion."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_to_matrix(q):
    """Unit (or near-unit) quaternion -> 3x3 rotation matrix. Normalizes first."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R):
    """3x3 rotation matrix -> unit quaternion (w, x, y, z), canonicalized w >= 0.

    Shepperd's method: pick the branch keyed on the largest of the four
    {trace, R00, R11, R22} so the divisor is never near zero (numerically robust
    for any rotation, including 180-degree cases a naive trace formula botches)."""
    R = np.asarray(R, dtype=float)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0        # s = 4w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0   # s = 4x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0   # s = 4y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0   # s = 4z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0.0:            # canonicalize to the w>=0 hemisphere (shortest arc)
        q = -q
    return quat_normalize(q)


def so3_exp(omega):
    """Exponential map: rotation-vector (axis * angle, radians) -> unit quaternion.

    The rotation vector `omega` = theta * axis (|axis| = 1). Returns the unit
    quaternion (cos(theta/2), sin(theta/2)*axis). Uses a Taylor expansion of
    sin(theta/2)/theta as theta -> 0 so a tiny/zero rotation is exact and
    branch-safe."""
    omega = np.asarray(omega, dtype=float)
    theta = np.linalg.norm(omega)
    half = 0.5 * theta
    if theta < 1e-8:
        # sin(half)/theta = 0.5 - theta^2/48 + ...  -> 0.5 at theta=0.
        k = 0.5 - theta * theta / 48.0
        w = 1.0 - half * half / 2.0
    else:
        k = np.sin(half) / theta
        w = np.cos(half)
    return quat_normalize(np.array([w, k * omega[0], k * omega[1], k * omega[2]]))


def so3_log(q):
    """Logarithm map: unit quaternion -> rotation-vector (axis * angle, radians).

    theta = 2*atan2(|xyz|, w). We first flip q to the w>=0 hemisphere so theta
    lands in [0, pi] (the SHORTEST arc — resolves the quaternion double cover), so
    so3_log(so3_exp(w)) == w for |w| < pi. A Taylor branch handles the small-angle
    divisor. Returns a zero vector for the identity."""
    q = quat_normalize(q)
    if q[0] < 0.0:
        q = -q
    v = q[1:]
    vn = np.linalg.norm(v)
    w = q[0]
    if vn < 1e-8:
        # theta/sin(theta/2) -> 2 as theta -> 0; scale = angle / |v|.
        scale = 2.0 / (w if abs(w) > _EPS else 1.0)  # ~ 2 for w~1
        return scale * v
    theta = 2.0 * np.arctan2(vn, w)
    return (theta / vn) * v


def quat_angle(qa, qb):
    """Geodesic angle (rad, in [0, pi]) between two scalar-first quaternions.

    Double-cover safe (q and -q give 0). Inputs are normalized first, so any
    non-zero 4-vectors work."""
    qa = quat_normalize(np.asarray(qa, dtype=float))
    qb = quat_normalize(np.asarray(qb, dtype=float))
    dot = abs(float(np.dot(qa, qb)))
    return 2.0 * np.arccos(min(1.0, dot))


def rotvec_to_matrix(omega):
    """Rotation-vector -> 3x3 rotation matrix (== quat_to_matrix(so3_exp(omega)))."""
    return quat_to_matrix(so3_exp(omega))


def matrix_to_rotvec(R):
    """3x3 rotation matrix -> rotation-vector (== so3_log(matrix_to_quat(R)))."""
    return so3_log(matrix_to_quat(R))


# --------------------------------------------------------------------------- #
# Sim(3): uniform-scale + rotation + translation pose
# --------------------------------------------------------------------------- #
@dataclass
class Pose:
    """A Sim(3) placement: uniform scale `s`, unit quaternion `q` (w,x,y,z),
    translation `t` (3,). Acts as p_cam = s * R@p + t == pose_to_matrix @ [p;1].

    `s` is a uniform SCALAR — matching the harness's shared-SCALE model (a rigid
    object cannot 'breathe' across frames). Anisotropic scale is forbidden."""
    s: float
    q: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        self.s = scalar_scale(self.s, "Pose.s")
        self.q = quat_normalize(self.q)
        self.t = np.asarray(self.t, dtype=float).reshape(3)


def pose_identity():
    """The identity Sim(3): s=1, q=(1,0,0,0), t=0."""
    return Pose(1.0, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3))


def pose_to_matrix(P):
    """Pose -> 4x4 homogeneous matrix M = T @ R @ S (== transforms.compose_pose).

    The upper-left 3x3 is s*R (scale folded into the rotation block); the last
    column is t."""
    R = quat_to_matrix(P.q)
    M = np.eye(4)
    M[:3, :3] = P.s * R
    M[:3, 3] = P.t
    return M


def pose_from_matrix(M):
    """4x4 matrix (T @ R @ S, uniform scale) -> Pose.

    Recovers a uniform scale from the singular values of the 3x3 block, then
    ORTHONORMALIZES the rotation via SVD before extracting the quaternion — so a
    slightly non-orthonormal block (e.g. from a mathutils->numpy handoff, or fp
    drift) can't poison the quaternion. Rejects anisotropic scale."""
    M = np.asarray(M, dtype=float)
    A = M[:3, :3]
    singular_values = np.linalg.svd(A, compute_uv=False)
    s = float(np.mean(singular_values))
    if not np.allclose(singular_values, s, rtol=1e-5, atol=1e-8):
        raise ValueError(
            "pose matrix must contain one uniform scalar scale; anisotropic "
            "scale is forbidden because it does not commute with rotation")
    if s < _EPS:
        s = _EPS
    R = A / s
    # nearest orthonormal matrix (proper rotation) via SVD.
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0.0:            # guard against a reflection
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    return Pose(s, matrix_to_quat(R_ortho), M[:3, 3])


def pose_from_object_pose(object_pose, scale=None):
    """A pose.json `object_pose` dict -> lie.Pose, ALWAYS from `quaternion` +
    `translation` (+ scale) — the AUTHORITATIVE fields — and NEVER from a stored
    `matrix4x4`.

    `matrix4x4` (and `rotation_euler`) are derived caches: they can only ever be
    stale copies of the quaternion/translation/scale, so this — the one shared
    entry point every consumer uses — ignores them by construction. That makes a
    stale-matrix bug unrepresentable: there is no code path that reads a stored
    matrix back. Call `pose_to_matrix` on the result when a caller needs the 4x4.

    scale precedence: the object_pose's own `scale` (the per-frame echo of the
    shared SCALE) when present, else the passed `scale`, else 1.0.
    """
    p = dict(object_pose or {})
    q = p.get("quaternion", (1.0, 0.0, 0.0, 0.0))
    t = p.get("translation", (0.0, 0.0, 0.0))
    s = p.get("scale", scale)
    s = 1.0 if s is None else scalar_scale(s)
    # shout if the serialized quaternion is non-unit by more than JSON rounding
    # could explain (a malformed quaternion reads as the wrong rotation once
    # normalized) — Pose then normalizes it as usual.
    check_quat_norm(q, where=f"pose_from_object_pose q={tuple(q)}")
    return Pose(s, np.asarray(q, dtype=float), np.asarray(t, dtype=float))


def object_pose_matrix(object_pose, scale=None):
    """The 4x4 canonical->camera matrix for a pose.json `object_pose`, computed
    fresh from its quaternion/translation/scale via `pose_from_object_pose`.

    This REPLACES reading a stored `object_pose["matrix4x4"]`: the field is no
    longer serialized, so every consumer that needs the matrix derives it here
    (pure numpy — runs in Blender's bundled python and the analysis env alike).
    """
    return pose_to_matrix(pose_from_object_pose(object_pose, scale))


def pose_compose(A, B):
    """Compose two poses: (A . B) maps a point through B then A.

    pose_to_matrix(pose_compose(A, B)) == pose_to_matrix(A) @ pose_to_matrix(B).
    Derived: s = sA*sB, q = qA*qB, t = sA * (RA @ tB) + tA."""
    RA = quat_to_matrix(A.q)
    s = A.s * B.s
    q = quat_mul(A.q, B.q)
    t = A.s * (RA @ B.t) + A.t
    return Pose(s, q, t)


def pose_inverse(P):
    """The inverse pose: pose_compose(P, pose_inverse(P)) == identity."""
    q_inv = quat_conj(P.q)
    s_inv = 1.0 / (P.s if abs(P.s) > _EPS else _EPS)
    R_inv = quat_to_matrix(q_inv)
    t_inv = -s_inv * (R_inv @ P.t)
    return Pose(s_inv, q_inv, t_inv)


def pose_apply(P, p):
    """Apply a pose to a 3-point: s * R @ p + t."""
    p = np.asarray(p, dtype=float).reshape(3)
    return P.s * (quat_to_matrix(P.q) @ p) + P.t


def rescale_gauge(P, factor):
    """Apply the monocular depth/scale GAUGE to a pose: s -> factor*s, t -> factor*t.

    This is a uniform scaling by `factor` ABOUT THE CAMERA-0 CENTRE (the world
    origin, since camera 0 has the identity extrinsic). In the camera frame that
    is just p_cam -> factor * p_cam, and folding it into p_cam = s*R@p + t gives

        factor * p_cam = (factor*s) * R @ p + (factor*t)

    so the object's SIZE (s) and its DEPTH/position (t) scale together by the SAME
    `factor`, while rotation `q` and articulation (joint states) are untouched —
    they are scale-invariant. This is EXACTLY the transform a projective camera
    cannot see: pi(factor*p_cam) == pi(p_cam), so the silhouette is pixel-identical
    (the depth<->size ambiguity of a monocular, moving-object reconstruction).

    Because the gauge acts along each frame's camera ray about the SHARED camera
    centre, one global `factor` — recovered e.g. by depth alignment against a
    metric reference (see the `metrics` tool) — is applied to EVERY frame's pose
    identically here. Scaling `s` alone (object resizes, stays at the same depth)
    or `t` alone (object moves in depth, keeps its size) would BREAK the
    silhouette; both together keep it and make the pose metric.

    `factor` is a positive multiplicative scalar (1.0 is a no-op). Returns a NEW
    Pose; `P` is unchanged.
    """
    factor = scalar_scale(factor, "rescale_gauge factor")
    return Pose(factor * P.s, P.q, factor * P.t)


# --------------------------------------------------------------------------- #
# increment "order" operators — the sweep's camera-frame vocabulary
# --------------------------------------------------------------------------- #
# Each returns a left-multiplied increment delta_T such that a new pose is
# pose_compose(delta_T, T). Rotations orbit about the OBJECT CENTRE (a point c in
# the camera-0 world frame), so apparent framing is preserved: a rotation of
# angle theta about world axis `a` through `c` is the affine map x -> R(x-c)+c,
# i.e. Pose(1, exp(theta*a), c - R@c). Left-composing yaw(az) then pitch(el) is
# the increment form of a view-sphere re-orientation (M' = T . (Az . El . R) . S
# about the centre).
def rot_about_point(theta, axis, centre):
    """Increment: rotate `theta` rad about world `axis` through point `centre`.

    Fixes `centre` (pose_apply(result, centre) == centre)."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < _EPS:
        return pose_identity()
    q = so3_exp(theta * (axis / n))
    R = quat_to_matrix(q)
    centre = np.asarray(centre, dtype=float).reshape(3)
    return Pose(1.0, q, centre - R @ centre)


def roll_delta(theta, centre):
    """In-plane roll: rotate about the optical axis (+Z) through the centre.
    Positive theta rotates the image content (right-handed about +Z into scene)."""
    return rot_about_point(theta, Z_CAM, centre)


def yaw_delta(theta, centre):
    """Out-of-plane yaw: orbit about camera-up (-Y) through the centre."""
    return rot_about_point(theta, UP_CAM, centre)


def pitch_delta(theta, centre):
    """Out-of-plane pitch: orbit about camera-right (+X) through the centre."""
    return rot_about_point(theta, RIGHT_CAM, centre)


def scale_delta(sigma):
    """Multiplicative scale increment: s' = exp(sigma) * s (about the origin).

    sigma is a LOG-scale step, so it composes additively and is symmetric about 0
    (sigma and -sigma are reciprocal scalings)."""
    return Pose(np.exp(sigma), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3))


def pixel_translation_delta(dpx, dpy, depth_z, fx, fy):
    """Pixel-aligned translation increment: an image-plane shift of (dpx, dpy)
    PIXELS at object depth `depth_z`, given focal lengths (fx, fy).

    The camera-0 extrinsic is identity, so a camera-frame displacement adds
    directly in world coords. Inverting the pinhole projection the harness uses
    (u = fx*X/Z + cx,  v = fy*Y/Z + cy, image y-DOWN; see camera.debug_project —
    this is OpenCV, so world +Y projects DOWN, matching the camera-0 -Y-up frame):

        dt_x =  (dpx / fx) * depth_z     # +X is image-right
        dt_y =  (dpy / fy) * depth_z     # image y-down == world +Y (OpenCV)
        dt_z =  0

    So the projected object centre moves exactly dpx px right and dpy px down.

    This is the low-level primitive: `dpx`/`dpy` here are RAW pixels. The sweep's
    dpx/dpy DOFs are expressed as a FRACTION of the frame (width/height) and scaled
    to pixels (dpx*width, dpy*height) by apply_order — the image-plane analogue of
    tz's depth-fraction normalization, so every search DOF is resolution/scene
    invariant."""
    depth_z = float(depth_z)
    dt = np.array([(dpx / fx) * depth_z, (dpy / fy) * depth_z, 0.0])
    return Pose(1.0, np.array([1.0, 0.0, 0.0, 0.0]), dt)


def z_translation_delta(dtz):
    """Raw translation increment along the optical axis (+Z), in world units.

    This is the low-level primitive: `dtz` is a raw camera-frame +Z displacement.
    The sweep's `tz` DOF is expressed as a FRACTION of the object's depth and
    scaled to world units (dtz * depth_z) by apply_order — depth has no natural
    pixel analogue (a pixel shift is in-plane), so its DOF is normalized by depth
    rather than by focal length. This axis is what depth supervision targets."""
    return Pose(1.0, np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, float(dtz)]))


def apply_order(P, *, roll=0.0, yaw=0.0, pitch=0.0,
                dpx=0.0, dpy=0.0, dtz=0.0, centre, fx, fy, depth_z,
                width, height):
    """Apply the camera-frame "order" DOFs to pose `P` and return the new pose.

    PURELY RIGID: the increment has scale 1, so P's scale passes through
    untouched. There is no scale DOF here — SCALE is global, owned by the built
    object (see core/scene_contract.py), and no per-frame verb may search it. The
    order once carried a `sigma` log-scale step that pasted into the shared
    top-level SCALE; a sweep over N frames could produce N winning sigmas all
    writing the same global, last write wins. It was removed; `scale_delta` remains
    as the standalone Sim(3) primitive for callers that genuinely mean a scale
    increment.

    DOF units are all scene/scale/resolution-INVARIANT so one default box
    transfers across scenes: roll/yaw/pitch in radians, `dpx`/`dpy` a FRACTION OF
    THE FRAME (dt lands dpx*width px right, dpy*height px down — normalized by the
    frame's pixel size, so a step means the same visual shift at any resolution),
    and `dtz` a FRACTION OF THE OBJECT'S DEPTH (dt_z = dtz * depth_z world units
    along +Z — depth has no natural pixel unit, so it is normalized by depth). The
    two image-plane fractions and the depth fraction are the same idea on
    different axes.

    A left-multiplied increment  delta_T  (scale 1) composed onto P, written as the
    MATRIX PRODUCT (rightmost factor acts on the point first):

        delta_T = pixel_translate . z_translate . pitch . yaw . roll

    Rotations orbit about the object `centre` — roll applied first, then yaw, then
    pitch (the Az.El view-sphere order once left-multiplied) — so the object spins in
    place with its framing preserved; z_translate (the depth-fraction nudge, scaled to
    dtz*depth_z world units) then the pixel shift OUTERMOST (dpx*width /
    dpy*height pixels) act as final image-plane / depth nudges.

    The composite fixes `centre` under any ordering of the three rotations, so the
    order is not a correctness question — but it decides which axes the later
    rotations act about. The code is the product above.

    `centre` is the current posed object (bbox) centre in the camera-0 frame;
    `depth_z` is that centre's +Z (both the pivot depth AND the dtz depth-fraction
    normalizer); `fx, fy` are the frame's focal lengths and `width, height` its
    pixel dimensions, all at scoring resolution (fx/fy map the pixel shift to
    world; width/height turn the dpx/dpy frame-fractions into pixels). Any DOF left
    at 0 contributes an identity factor, so callers pass only the DOFs they sweep."""
    c = np.asarray(centre, dtype=float).reshape(3)
    delta = pose_identity()
    delta = pose_compose(roll_delta(roll, c), delta)
    delta = pose_compose(yaw_delta(yaw, c), delta)
    delta = pose_compose(pitch_delta(pitch, c), delta)
    delta = pose_compose(z_translation_delta(dtz * float(depth_z)), delta)
    delta = pose_compose(
        pixel_translation_delta(dpx * float(width), dpy * float(height),
                                depth_z, fx, fy), delta)
    return pose_compose(delta, P)                     # rigid: P's scale preserved


# --------------------------------------------------------------------------- #
# object-centric shared reorientation — the RIGHT-multiplied canonical rotation
# --------------------------------------------------------------------------- #
# apply_order (above) builds a CAMERA-frame delta and LEFT-multiplies it onto one
# frame's pose to fix THAT frame — good when a single frame's pose drifted. The
# operators here are the opposite: a rotation in the object's OWN CANONICAL frame,
# RIGHT-multiplied onto a pose (M' = M @ D with D = T(c)·R_extra·T(-c), i.e.
# p_cam = s*R@(R_extra@(p-c)+c)+t). Because R_extra acts before each frame's own
# rotation R, ONE shared R_extra re-orients the canonical geometry coherently
# across EVERY frame regardless of their differing camera views — the fix for an
# object built mis-oriented (e.g. flipped) so it reads wrong everywhere.
# Translation picks up only the pivot correction s*R@(I-R_extra)@c and (uniform)
# scale is untouched: the object spins about the canonical point `c`, staying put
# in each camera. `c` is the object's FK-AABB centre (core/centre_calculation.py)
# for the per-frame verbs; the default c=0 (the historical behavior) is exact only
# when the canonical geometry is truly centred at the origin — a CONVENTION, not a
# checked invariant (conventions/POSE.md), which is why callers that can measure
# the centre should pass it.
#
# Canonical object axes (see conventions/POSE.md: +Z up, centred at origin):
#   rx — canonical +X (right) : nod the object forward/back
#   ry — canonical +Y (front) : roll about the front-facing axis
#   rz — canonical +Z (up)    : turn the object about its up axis
# (rx, ry, rz) is a ROTATION VECTOR (axis * angle, so3_exp), not an Euler triple:
# the direction is the rotation axis, the norm the angle, so there is no
# composition order and no gimbal lock. A single-axis value reads exactly as
# "that many radians about that canonical axis". The names are axis-LITERAL on
# purpose: they once reused apply_order's CAMERA-frame yaw/pitch spellings, a
# collision that invited porting a camera-frame yaw into an object rotation (the
# two only coincide for an upright object).
RIGHT_CANON = np.array([1.0, 0.0, 0.0])  # canonical right (rx axis)
FWD_CANON = np.array([0.0, 1.0, 0.0])    # canonical front (ry axis)
UP_CANON = np.array([0.0, 0.0, 1.0])     # canonical up (rz axis)


def canonical_rotation(rx=0.0, ry=0.0, rz=0.0):
    """Shared object-frame rotation quaternion from the canonical ROTATION VECTOR
    (rx, ry, rz) — radians about canonical +X/+Y/+Z.

        canonical_rotation(rx, ry, rz) == so3_exp([rx, ry, rz])

    One rotation about the axis (rx,ry,rz)/|.| by angle |(rx,ry,rz)|: a
    single-axis call is exactly that many radians about that axis, and a
    multi-axis call is ONE rotation about the tilted axis — never a sequence, so
    there is no composition order to remember and no gimbal lock. The positional
    order matches the planner's CANON_DOFS tuple. All-zero is the identity.
    Returns a scalar-first unit quaternion suitable for right_reorient()."""
    return so3_exp(np.array([rx, ry, rz], dtype=float))


def right_reorient(P, q_extra, centre=None):
    """RIGHT-multiply a canonical rotation `q_extra` onto pose `P`, about `centre`.

    `centre` is the pivot in the CANONICAL frame (the object's FK-AABB centre,
    core/centre_calculation.py — the right update's counterpart of rot_about_point's
    camera-frame pivot). Returns pose_compose(P, Pose(1, q_extra, c - R_extra@c)):
    rotation q_P*q_extra with (uniform) scale PRESERVED and translation carrying
    only the pivot correction t = P.t + P.s*(R_P @ (I - R_extra) @ c), so the
    canonical point `centre` HOLDS STILL in the camera
    (pose_apply(result, centre) == pose_apply(P, centre)). pose_to_matrix of the
    result equals M_P @ T(c) @ R_extra @ T(-c): the object re-orients in its own
    canonical frame while `centre` holds its place in every camera.

    `centre=None` (and exactly 0) is the historical rotate-about-the-canonical-
    ORIGIN behavior — t and s untouched — which is only the object's centre when
    the canonical geometry is truly centred (a convention, not an invariant; see
    conventions/POSE.md). `q_extra` need not be pre-normalized."""
    q_extra = quat_normalize(q_extra)
    c = (np.zeros(3) if centre is None
         else np.asarray(centre, dtype=float).reshape(3))
    R = quat_to_matrix(q_extra)
    return pose_compose(P, Pose(1.0, q_extra, c - R @ c))
