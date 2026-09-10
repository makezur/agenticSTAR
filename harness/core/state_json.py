"""state_json.py — shared serialization helpers for a frame's STATE in JSON.

The wire format is conventions/state_json.md: a pose is {quaternion (scalar-first
unit), translation, scale?} and nothing else, and `joints` rides BESIDE it as a
complete {name: state} map at the same precision. Both live here because both are
written by the same writers, at the same precision, into the same records — keeping
their rounding in one module is what stops the two halves of a frame's state from
drifting to different formats (which is how `joints` ended up nested inside the pose
in one dialect and missing from another).

The scale hunt (pass-local pose.json first, then the run's mesh/pose.json)
existed twice with drifting comments — shape_pass.read_scale and the depth
scorer's resolve_scale. The candidate ORDER is the caller's decision (they
anchor from different paths); the "first existing file with a non-null scale
wins" rule lives here once. Returns the RAW value — validation
(lie.scalar_scale) stays with the caller, since core/ sits below rig/.
"""

import json
import os

# Serialized precision for every pose component and joint state. 6dp is lossless
# at the pixel level for scene-unit translations, keeps reports diffable and paste
# blocks legible, and makes a round-trip through scene.py a no-op. It leaves a
# rounded unit quaternion non-unit at ~1e-6, which is renormalized once at the
# READ boundary (transforms._unit_quat / lie.check_quat_norm) — writers must not
# compensate. See conventions/state_json.md §5.
POSE_DP = 6

# Version of the FRAGMENT container (multiagent/windows.py `poses.json`, and the
# sweep family's `<stem>_poses.json`, which is merged by the same code). It lives
# here, below both, because the two writers sit in different environments — the
# sweep engines run inside Blender and cannot import multiagent.windows (it pulls
# in analysis/*). A literal in each writer is how the versions drift apart.
#
# 2 added the two fields a REFINER's fragment must now carry (a sweep-written
# fragment carries neither, and does not claim to — it has no `window_id`, so
# nobody refined it and its steps are the orchestrator's): `expected_motion`,
# written from the frame sheet BEFORE the first sweep, and `step_calls`, one
# temporal verdict per interior step, each stamped with a pose hash. See
# conventions/state_json.md §6 and multiagent/calls.py.
FRAGMENT_SCHEMA_VERSION = 2


def round_component(v):
    """One pose/joint scalar at the serialized precision."""
    return round(float(v), POSE_DP)


def round_pose(pose):
    """A pose dict rounded to the wire format: quaternion + translation, and
    `scale` only when the caller carries one (a per-frame PROPOSAL omits it, so
    it cannot smuggle a scale change through a merge — state_json.md §4)."""
    out = {"quaternion": [round_component(v) for v in pose["quaternion"]],
           "translation": [round_component(v) for v in pose["translation"]]}
    if pose.get("scale") is not None:
        out["scale"] = round_component(pose["scale"])
    return out


def paste_literal(v):
    """One scalar as a Python literal for a scene.py paste block — the SAME value
    the JSON report carries. `repr` of the rounded float is its shortest exact
    spelling, so pasting a tool's block reproduces the pose that tool scored.
    `%.6g` did not: 6 SIGNIFICANT digits silently truncated the 6 DECIMAL places
    the report wrote (12.345679 printed as 12.3457), so the paste and the JSON
    disagreed in the last places. See conventions/state_json.md §5."""
    return repr(round_component(v))


def round_joints(states):
    """Joint states at the serialized precision (revolute: degrees)."""
    return {n: round_component(v) for n, v in (states or {}).items()}


def first_scale(*pose_json_paths):
    """The `scale` field of the first existing pose.json that declares one,
    else None. Skips files whose scale is null rather than stopping."""
    for cand in pose_json_paths:
        if os.path.isfile(cand):
            with open(cand) as f:
                s = json.load(f).get("scale")
            if s is not None:
                return s
    return None
