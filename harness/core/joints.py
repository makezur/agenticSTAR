"""joints.py — shared joint-definition helpers (bpy-free).

The JOINTS schema is documented in conventions/JOINTS.md and (authoritatively
for the rig) in rig/transforms.py. The limit parser lives here in core/ so the
Blender side (rig.transforms, which needs a LOUD failure on a scene.py typo)
and the analysis env (pose_diff, aggregate, the temporal report — which must
tolerate whatever an old pose.json carries) share one implementation.

`require_complete_states` lives here for the same reason, and enforces the rule
that makes the two DOF grammars behave differently — see its docstring.
"""

import hashlib
import json
import math


KINEMATICS_SCHEMA_VERSION = 1
KINEMATICS_DP = 6


def _component(value):
    return round(float(value), KINEMATICS_DP) + 0.0


def _vector(value, normalize=False):
    if value is None:
        return None
    try:
        out = [_component(v) for v in value]
    except (TypeError, ValueError):
        return None
    if normalize:
        norm = math.sqrt(sum(v * v for v in out))
        if norm > 1e-9:
            out = [_component(v / norm) for v in out]
    return out


def canonical_joint_defs(joint_defs):
    """Stable semantic JOINTS representation for history reconciliation."""
    out = []
    for raw in joint_defs or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        child = raw.get("child")
        if isinstance(child, (list, tuple)):
            child = sorted(str(value) for value in child)
        elif child is not None:
            child = [str(child)]
        limit = joint_limit(raw, strict=False)
        out.append({
            "name": str(raw["name"]),
            "type": str(raw.get("type", "fixed")),
            "axis": _vector(raw.get("axis"), normalize=True),
            "origin": _vector(raw.get("origin")),
            "parent": (str(raw["parent"])
                       if raw.get("parent") is not None else None),
            "child": child,
            "limit": ([_component(v) for v in limit]
                      if limit is not None else None),
        })
    return sorted(out, key=lambda joint: joint["name"])


def kinematics_hash(joint_defs):
    payload = {
        "schema_version": KINEMATICS_SCHEMA_VERSION,
        "joint_defs": canonical_joint_defs(joint_defs),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def structural_joint(joint):
    """Canonical definition without its limit, which is reconcilable."""
    canonical = canonical_joint_defs([joint])
    if not canonical:
        return None
    out = dict(canonical[0])
    out.pop("limit", None)
    return out


def articulated_names(joint_defs):
    """Names of the joints that carry a DOF (everything but `fixed`).

    A `fixed` joint has no state to declare, so it is exempt from the
    completeness rule below; a state given for one is harmless (its transform is
    identity) and still permitted."""
    return [j.get("name") for j in (joint_defs or [])
            if j.get("name") and j.get("type", "fixed") != "fixed"]


def require_complete_states(joint_defs, states, where):
    """Refuse a joint-state dict that omits a joint the scene declares.

    THE ASYMMETRY THIS EXISTS FOR. The harness carries two DOF grammars and they
    measure from different origins:

      * a camera-frame POSE order (roll/yaw/pitch/dpx/dpy/tz) is an INCREMENT
        onto a base placement, so 0 IS the hold and an omitted DOF is a genuine,
        lossless no-op;
      * a JOINT state is ABSOLUTE, so 0 means "articulate this joint to zero" —
        a real move. There is no identity value, which means an omitted joint
        state cannot be resolved to any number. It can only be REFUSED.

    Defaulting a missing state to 0.0 therefore silently STRAIGHTENS the joint,
    and the render is a valid picture of the wrong configuration. This helper is
    the shared rule so no call site can reintroduce that default.

    Missing means "declared in JOINTS with a type that has a DOF, absent from
    `states`". Extra keys are NOT an error here: a pool order legitimately names
    joints in a scene whose defs the caller does not re-validate, and the scene
    gate (rig.scene._check_frame_joints) already ranges what is present.

    `where` names the seam in the message (a frame name, an order id, a view) —
    the error must say WHICH configuration was incomplete, not just that one
    was. Returns `states` unchanged so callers can wrap in place."""
    names = articulated_names(joint_defs)
    if not names:
        return states
    st = states or {}
    missing = [n for n in names if n not in st]
    if missing:
        raise ValueError(
            f"{where}: joint state(s) {', '.join(repr(m) for m in missing)} "
            f"missing from the joint states {sorted(st)!r}. Joint states are "
            "ABSOLUTE, not increments — there is no value that means 'leave "
            "this joint alone', so an omitted state cannot be defaulted (0.0 "
            "would articulate the joint to zero, i.e. straighten it, and "
            "render a plausible picture of the wrong configuration). Name "
            "every articulated joint explicitly.")
    return states


def joint_limit(joint, strict=False):
    """The joint's valid range as a (lo, hi) float tuple, or None if unbounded.

    None when `limit` is absent/None or the joint is 'fixed' (no DOF).

    strict=True (the rig / scene-load path): a malformed limit or lo > hi
    raises ValueError, so a typo in scene.py fails loudly rather than silently
    disabling the clamp.

    strict=False (the analysis / report-reading path): a malformed limit yields
    None and a reversed pair is normalized — a stale pose.json must not crash a
    reporter."""
    lim = joint.get("limit")
    if lim is None or joint.get("type", "fixed") == "fixed":
        return None
    try:
        lo, hi = (float(v) for v in lim)
    except (TypeError, ValueError):
        if strict:
            raise ValueError(
                f"joint {joint.get('name')!r}: limit must be a (lo, hi) pair, "
                f"got {lim!r}")
        return None
    if lo > hi:
        if strict:
            raise ValueError(
                f"joint {joint.get('name')!r}: limit lo ({lo}) > hi ({hi})")
        lo, hi = hi, lo
    return (lo, hi)
