"""dof_space.py — the unified DOF space for the sweep machinery (the "umbrella").

WHY THIS MODULE EXISTS
----------------------
A pose-only, a joint-only, and a coupled pose+joint sweep differ ONLY in how a
candidate becomes a posed scene; this module is that single seam. A `Space` is an
ordered list of named DOFs, each either:

  * a POSE-ORDER DOF — one of the reserved 6 camera-frame increments
    (roll/yaw/pitch/dpx/dpy/tz, see rig/lie.apply_order), applied onto a
    single FIXED base pose `P0` with a FIXED `OrderCtx` (pivot/depth/focal); or
  * a JOINT DOF — a named articulation joint whose value is an ABSOLUTE state fed
    to the forward-kinematics posing (rig/scene.pose_frame).

`realize(x)` turns a real parameter vector `x` into a concrete candidate: a
paste-ready placement (quaternion + translation + scale) plus a joint-states
dict. Pose-only, joint-only, and coupled sweeps
are then just Spaces with different DOF sets — one code path, no drift, and a grid
over pose AND joint components maps a coupled valley in one pass.

PURITY / PORTABILITY
--------------------
PURE (numpy + stdlib + the pure-numpy `rig.lie`). No `bpy`, no `mathutils`, no
`rig.transforms`. The Blender-dependent pieces — building `P0` from the base
placement, building the `OrderCtx` geometry (bbox centre / focal), and the actual
`pose_frame` render — live in `engine.py`, which hands the finished `P0`/
`OrderCtx`/limits into the Space constructor. So `realize()` is a pure mapping and
is unit-testable in the analysis env with a hand-built `lie.Pose` and `OrderCtx`.
"""

import math
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rig import lie


# The reserved camera-frame "order" DOFs. Any DOF name NOT in this set must be a
# declared joint (the space validates that when it is constructed). These names
# are therefore RESERVED — a scene may not declare a joint called `yaw` etc.
POSE_DOFS = ("roll", "yaw", "pitch", "dpx", "dpy", "tz")
POSE_DOF_SET = frozenset(POSE_DOFS)

# `sigma` is not a sweepable DOF: SCALE is global, owned by the built object
# (`core/scene_contract.py` declares it module-level; there is no per-frame scale
# key in FRAMES), so no per-frame verb may search it. The name stays RESERVED so a
# scene cannot declare a joint called `sigma` and a spec that names it fails LOUD
# with the reason instead of being read as a joint. See planner.removed_dof_error.
REMOVED_POSE_DOFS = ("sigma",)
REMOVED_POSE_DOF_SET = frozenset(REMOVED_POSE_DOFS)

# Kinds a DOF can have.
POSE = "pose"
JOINT = "joint"


# The geometry the pose-order DOFs need (moved here from runtime so the
# pure space can hold it without importing bpy; engine builds it, see
# _make_order_ctx). `centre` is the roll/yaw/pitch orbit PIVOT in the camera-0
# frame: the object's canonical-frame AABB centre AT THIS FRAME'S ARTICULATION,
# carried over by the frame's placement (c = s*R@c_canon + t) — a MATERIAL point of
# the object, so it follows the object rather than sliding as the base pose changes.
# See core/centre_calculation.py for why it is measured canonically and not from the
# posed bbox. `depth_z` is its +Z (pivot depth AND the tz depth-fraction
# normalizer), `fx,fy` the focal lengths and `width,height` the
# pixel dims — all at the SCORING resolution (so a dpx/dpy frame-fraction becomes
# pixels). See rig/lie.apply_order.
OrderCtx = namedtuple("OrderCtx", ["centre", "depth_z", "fx", "fy",
                                   "width", "height"])


@dataclass
class Space:
    """A unified DOF space over which the planner searches and the engine scores.

    Fields (all supplied by the engine, which owns the Blender-side construction):
      dof_names   — ordered DOF names; defines the layout of the vector `x` (the
                    planner sorts them for deterministic probe order).
      kinds       — {name: POSE|JOINT}.
      P0          — the fixed base pose (a lie.Pose) the pose-order DOFs increment.
      order_ctx   — the fixed OrderCtx (pivot/depth/focal) for apply_order.
      base_scale  — the shared uniform scalar SCALE, echoed by each candidate.
      base_states — {joint_name: held_state} for joints NOT swept (the frame's
                    declared states); swept joints override these in realize().
      joint_limits— {joint_name: (lo,hi)|None}; realize() clamps a swept joint's
                    value into its limit so the RECORDED state matches what the FK
                    posing (which also clamps) actually renders.
      joint_types — {joint_name: "revolute"|"prismatic"|...}; only consulted to
                    pick the type-appropriate DEFAULT span for an UNBOUNDED joint
                    (a missing entry falls back to revolute).
    """
    dof_names: List[str]
    kinds: Dict[str, str]
    P0: Any
    order_ctx: Optional[OrderCtx] = None
    base_scale: float = 1.0
    base_states: Dict[str, float] = field(default_factory=dict)
    joint_limits: Dict[str, Optional[Tuple[float, float]]] = field(default_factory=dict)
    joint_types: Dict[str, str] = field(default_factory=dict)

    # ---- the one place a vector becomes a posed scene ----------------------
    def realize(self, x) -> Dict[str, Any]:
        """Vector `x` (aligned to `dof_names`) -> a candidate dict.

        Returns:
          {
            "placement": {quaternion, translation, scale},  # paste-ready
            "order":     {dof: value for the 6 pose DOFs, 0 for un-swept},
            "joints":    {joint_name: state} (base_states + swept, clamped),
          }

        Pose-order DOFs compose onto the FIXED `P0` via lie.apply_order using the
        FIXED `order_ctx` — never re-based — so the pasted pose reproduces the
        scored render bit-for-bit and the pattern move's direction stays coherent.
        A space with no pose DOFs returns the base placement unchanged (P0). Joint
        DOFs override the held base states (clamped to their limit).
        """
        x = np.asarray(x, dtype=float)
        order = {d: 0.0 for d in POSE_DOFS}
        joints = dict(self.base_states)
        for i, name in enumerate(self.dof_names):
            v = float(x[i])
            if self.kinds[name] == POSE:
                order[name] = v
            else:
                lim = self.joint_limits.get(name)
                if lim is not None:
                    v = min(max(v, lim[0]), lim[1])
                joints[name] = v

        placement = self._placement_from_order(order)
        return {"placement": placement, "order": order, "joints": joints}

    def _placement_from_order(self, order) -> Dict[str, Any]:
        """Compose the pose-order increments onto P0 -> a paste-ready placement.

        `order`'s roll/yaw/pitch are agent-facing DEGREES; this is the single
        deg->rad seam for the camera grammar (rig.lie stays radians —
        dpx/dpy/tz are not angles and pass through untouched).

        The order is RIGID, so every candidate echoes the shared base SCALE
        unchanged — there is no scale DOF to search (see POSE_DOFS).

        Emits a scalar-first quaternion + translation + a scale echo that
        preserves the base SCALE's type. When no pose DOF is active this is P0
        itself, so a joint-only space still carries the base placement."""
        oc = self.order_ctx
        if oc is None:
            # joint-only space with no order geometry: the base pose is P0.
            P = self.P0
        else:
            P = lie.apply_order(
                self.P0, roll=math.radians(order["roll"]),
                yaw=math.radians(order["yaw"]),
                pitch=math.radians(order["pitch"]),
                dpx=order["dpx"], dpy=order["dpy"], dtz=order["tz"],
                centre=oc.centre, fx=oc.fx, fy=oc.fy, depth_z=oc.depth_z,
                width=oc.width, height=oc.height)
        return candidate_from_pose(P, self.base_scale)


def candidate_from_pose(P, base_scale) -> Dict[str, Any]:
    """A paste-ready placement dict from a lie.Pose `P`: a scalar-first
    (w,x,y,z) `quaternion` + `translation` and a scalar `scale` echo. These
    fields are authoritative and full-precision — the engine composes the
    scored 4x4 from them, and we cache no derived matrix, so what was scored and
    what the agent pastes into scene.py cannot drift apart."""
    lie.scalar_scale(base_scale, "base SCALE")
    q = P.q
    return {
        "quaternion": (float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        "translation": (float(P.t[0]), float(P.t[1]), float(P.t[2])),
        "scale": float(P.s),
    }


def removed_dof_error(flag: str, name: str):
    """The ValueError for a DOF REMOVED from the grammar, or None if `name` is fine.

    Lives here (not in planner) because `classify_dof` is the choke point every
    caller passes through — the planner's parsers, the engine's space resolution,
    and `--sweep-start-shift`. Raising from one place means a removed name cannot be
    quietly reinterpreted as a joint on some path nobody updated.

    Fails LOUD and teaches the reason. `sigma` is still named by archived orders and
    live agent prompts, and silently dropping it would mean the search the agent
    asked for did not happen while the report still looked successful.
    """
    if name not in REMOVED_POSE_DOF_SET:
        return None
    return ValueError(
        f"{flag}: {name!r} was REMOVED from the sweep grammar. SCALE is GLOBAL — "
        "it belongs to the built object, not to a frame — so no per-frame verb may "
        "search it (a sweep over N frames would produce N winning scales all "
        "pasting into the same top-level SCALE, last write wins). To resize the "
        "object, set the top-level SCALE in scene.py and re-run; to move a frame in "
        "depth instead — usually the DOF actually wanted, since size and depth trade "
        "off in a monocular view — sweep 'tz'.")


def classify_dof(name: str) -> str:
    """POSE for a reserved pose-order name, else JOINT. (Existence of the joint is
    validated by the planner against the scene's declared joints.)

    A REMOVED pose name (see REMOVED_POSE_DOFS) raises instead of classifying: it
    is neither a live pose DOF nor available as a joint name, and reading it as a
    joint would misdirect the error to "unknown joint"."""
    removed = removed_dof_error("sweep", name)
    if removed is not None:
        raise removed
    return POSE if name in POSE_DOF_SET else JOINT
