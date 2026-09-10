"""fk.py — pure-numpy forward kinematics over the shared JOINTS definitions.

THE ONE IMPLEMENTATION of the joint arithmetic, on numpy 4x4s.
rig/transforms.py converts at the mathutils boundary and delegates — the same
split as rig/lie.py, for the same reason (see lie.py's PURITY note: importable
in Blender's python AND the `artscript` env).

The JOINTS schema is documented in conventions/JOINTS.md and, authoritatively
for the rig, at the schema comment in rig/transforms.py. Semantics:

  * revolute  — rotation `state` DEGREES about `axis` through `origin` (the
                agent-facing unit everywhere; deg->rad happens here, at the
                matrix seam). A degenerate axis falls back to +Z.
  * prismatic — translation `state` along `axis` (canonical units). A
                degenerate axis yields identity.
  * fixed     — identity at any state.
  * states are clamped to the joint's declared `limit` as a runtime backstop
    (authored scenes are already range-checked at load — rig.scene).
  * parent chains compose down the tree (A_j = A_parent @ L_j), with a cycle
    check; a missing state is 0.0 (arithmetic totality, NOT a licence to omit
    — see core.joints.require_complete_states, enforced at the pose seams).

Pure numpy + stdlib on core.joints and rig.lie — no bpy, no mathutils.
"""

import math

import numpy as np

from core import joints as joints_core
from rig import lie


def clamp_joint_state(joint, state):
    """Clamp a DOF `state` to the joint's `limit` (if any).

    A defensive RUNTIME backstop for PROGRAMMATIC drivers — the authored
    scene.py path is already hard-checked at load (rig.scene rejects an
    out-of-range authored state), so for a normal render this is a no-op; it
    only floors non-authored inputs, keeping the "impossible configs never
    render" guarantee for them too. STRICT limit parse: a malformed limit
    raises rather than silently disabling the clamp."""
    state = float(state)
    lim = joints_core.joint_limit(joint, strict=True)
    if lim is None:
        return state
    return min(max(state, lim[0]), lim[1])


def joint_transform(joint, state):
    """Local (canonical-frame) 4x4 a joint applies to its group at `state`.

    revolute  -> rotation `state` DEGREES about `axis` through `origin`.
    prismatic -> translation `state` along `axis`.
    fixed     -> identity.
    """
    jtype = joint.get("type", "fixed")
    origin = np.asarray(joint.get("origin", (0.0, 0.0, 0.0)), dtype=float)
    axis = np.asarray(joint.get("axis", (0.0, 0.0, 1.0)), dtype=float)
    state = clamp_joint_state(joint, state)
    if jtype == "revolute":
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            axis, norm = np.array([0.0, 0.0, 1.0]), 1.0
        M = np.eye(4)
        M[:3, :3] = lie.rotvec_to_matrix(axis / norm * math.radians(state))
        # rotate ABOUT origin: T(origin) @ R @ T(-origin)
        M[:3, 3] = origin - M[:3, :3] @ origin
        return M
    if jtype == "prismatic":
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return np.eye(4)
        M = np.eye(4)
        M[:3, 3] = axis / norm * state
        return M
    return np.eye(4)


def child_names(joint):
    """Normalize a joint's `child` (str or list) to a list of part names."""
    child = joint.get("child")
    if child is None:
        return []
    return [child] if isinstance(child, str) else list(child)


def accumulate_canonical(joints, states):
    """Canonical-frame accumulated transform A_j per joint, composing down the
    parent chain (A_j = A_parent @ L_j). Returns {joint_name: 4x4 ndarray}."""
    by_name = {j.get("name"): j for j in joints}
    local = {j.get("name"): joint_transform(j, states.get(j.get("name"), 0.0))
             for j in joints}
    acc = {}

    def resolve(name, seen):
        if name in acc:
            return acc[name]
        if name in seen:
            raise ValueError(f"joint parent cycle at {name!r}")
        seen.add(name)
        j = by_name[name]
        parent = j.get("parent")
        A = local[name]
        if parent is not None:
            if parent not in by_name:
                raise ValueError(f"joint {name!r} has unknown parent {parent!r}")
            A = resolve(parent, seen) @ A
        acc[name] = A
        return A

    for j in joints:
        resolve(j.get("name"), set())
    return acc


def forward_kinematics(joints, base_pose=None, states=None):
    """Per-PART additional world transform for the JOINTS at the given states.

    `joints`  — the shared joint DEFINITIONS (schema: rig/transforms.py).
    `base_pose` — the object's per-frame base placement 4x4 (ndarray), so joint
                transforms declared in the canonical frame act in the posed
                frame (conjugation base @ A @ base^-1). None -> identity, which
                collapses the conjugation and yields pure CANONICAL-frame
                transforms.
    `states`  — {joint_name: value} for THIS frame (missing -> 0.0 -> rest).

    THE MISSING -> 0.0 RULE IS ARITHMETIC, NOT A POLICY. It stays total so this
    function is defined for any partial dict a programmatic driver hands it. It
    is NOT a licence to omit states: a joint state is absolute, so 0.0 means
    "straighten this joint", and every path that turns states into pixels goes
    through rig.scene.pose_frame, which REFUSES an incomplete dict via
    core.joints.require_complete_states. The check lives there because that is
    where the joint DEFINITIONS are known; here we only have whatever `joints`
    was passed.

    Returns {part_name: extra_world_4x4} for every part in any joint's child
    group; parts under no joint are absent (they stay at the base pose). Empty
    `joints` -> {} (no articulation). Supports parent chains.
    """
    base = np.eye(4) if base_pose is None else np.asarray(base_pose, dtype=float)
    binv = np.linalg.inv(base)
    states = states or {}
    joints = [j for j in (joints or []) if j.get("name") is not None]
    if not joints:
        return {}
    acc = accumulate_canonical(joints, states)
    out = {}
    for j in joints:
        W = base @ acc[j.get("name")] @ binv
        for part in child_names(j):
            out[part] = W
    return out
