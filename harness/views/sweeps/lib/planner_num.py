"""planner_num.py — the numeric + angle-preset primitives BOTH sweep grammars share.

`planner_pose` (camera-frame) and `planner_canon` (object-frame) have disjoint DOF
names, disjoint spec syntax, and disjoint callers. What they genuinely share is small
and lives here, so neither has to import the other:

  * the named angle-preset SIZES (`tiny`/`standard`/`large`/`huge`) and the
    directional band tokens, plus `_preset_band` that turns one token into degrees;
  * `steps_for_quality`, the step count `--sweep-quality` implies;
  * `_linspace`, the sampler every lattice is built from.

The preset SIZES are deliberately shared: "large" must mean the same +/-30deg in both
frames, or an agent that learns one grammar mis-sizes the other. The direction TOKENS
are pose-only (a canonical axis has no on-screen "which edge comes
closer" reading, so the canonical parsers pass `{}`), but the parsing of
`size_direction` is one function either way.

PURITY: numpy + stdlib only. No `bpy`, no `dof_space`.
"""

import numpy as np


# ---- per-DOF natural step + default-box magnitudes ------------------------- #
# per-axis angle-preset half-magnitudes for roll/yaw/pitch (degrees). Let the agent
# size the initial search band by name instead of typing degrees (see
# --sweep-angle-preset): tiny ±5deg, standard ±15deg, large ±30deg, huge ±45deg.
# A full-swing search (unknown facing / plausible flip) is an explicit range,
# e.g. 'yaw:-180,180,9' — no named preset covers it.
ANGLE_PRESET_HALF = {"tiny": 5.0, "standard": 15.0, "large": 30.0, "huge": 45.0}
ANGLE_AXES = ("roll", "yaw", "pitch")
DEFAULT_ANGLE_PRESET = "standard"
DEFAULT_ANGLE_HALF = ANGLE_PRESET_HALF[DEFAULT_ANGLE_PRESET]


# per-axis DIRECTION tokens for one-sided preset bands. A bare preset
# (`standard`) is the SYMMETRIC ±half fallback; a directional preset
# (`standard_cw`) spends the WHOLE 2·half budget on ONE side — same candidate
# count, all in the direction the caller actually saw, so an underestimated
# defect still falls inside. Preferred over the symmetric band (except `tiny`,
# a genuine last-mile touch-up): a visual read can identify WHICH WAY the pose is
# off. The sign is the camera-frame convention from
# conventions/DIRECTIONS.md (verified empirically): +roll spins
# CLOCKWISE on screen, +yaw swings the object's LEFT edge CLOSER to the camera,
# +pitch swings its TOP edge CLOSER (tips the top toward the camera). The
# tokens name WHICH EDGE comes closer — unambiguous, unlike "turn left" or
# "tip toward".
ANGLE_DIRECTIONS = {
    "roll":  {"cw": +1.0, "ccw": -1.0},
    "yaw":   {"leftcloser": +1.0, "rightcloser": -1.0},
    "pitch": {"topcloser": +1.0, "bottomcloser": -1.0},
}


def _preset_band(flag, axis, token, directions):
    """One `preset` or `preset_direction` token -> a (lo, hi) degree band.

    A BARE preset (`standard`) is the SYMMETRIC band `(-half, +half)`. A
    DIRECTIONAL preset (`standard_cw`) is the ONE-SIDED band spending the WHOLE
    `2·half` budget toward that direction — `(0, +2·half)` or `(-2·half, 0)` per
    the sign in `directions` (from conventions/DIRECTIONS.md). `directions` is
    `{token: +1|-1}` for this axis (`{}` for a canonical axis with no visual
    direction). Raises ValueError (teaching the valid tokens) on a bad preset or
    direction so a typo fails loud."""
    base, _, direction = token.partition("_")
    if base not in ANGLE_PRESET_HALF:
        raise ValueError(
            f"{flag}: unknown preset {token!r} for {axis!r} "
            f"(known: {', '.join(ANGLE_PRESET_HALF)}"
            + (f", optionally suffixed _{'/_'.join(directions)}"
               if directions else "") + ")")
    half = ANGLE_PRESET_HALF[base]
    if not direction:
        return (-half, half)
    if direction not in directions:
        known = (", ".join(directions)
                 or "none — this axis has no directional band")
        raise ValueError(
            f"{flag}: unknown direction {direction!r} for {axis!r} in {token!r} "
            f"(known: {known})")
    span = 2.0 * half
    return (0.0, span) if directions[direction] > 0 else (-span, 0.0)


def steps_for_quality(quality):
    """Default per-DOF grid step count from --sweep-quality (coarse when low)."""
    return 3 if quality < 0.45 else (5 if quality < 0.75 else 7)


def _linspace(lo, hi, steps):
    """`steps` samples in [lo,hi]; steps<=1 -> the single midpoint (a hold)."""
    steps = int(steps)
    if steps <= 1:
        return [0.5 * (lo + hi)]
    return list(np.linspace(lo, hi, steps))
