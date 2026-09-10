"""planner_pose.py — candidate generation for the CAMERA-FRAME grammar
(`sweep`, `apply`).

Given a `Space` (`dof_space.py`) and the agent's range specs, this resolves continuous
bounds for DE or candidates for a full Cartesian grid (the explicit dense-map tool
for a tricky 2-DOF coupled valley).

It also owns range PARSING, unified across pose "order" increments
(roll/yaw/pitch/dpx/dpy/tz, left-multiplied onto ONE frame's pose) and absolute
JOINT states, with joint-limit clipping. It knows nothing about Blender: `grid()`
returns a list of vectors `x` aligned to `space.dof_names`; the engine turns vectors
into rendered, scored candidates via `space.realize` + `pose_frame`.

The OBJECT-frame grammar (rx/ry/rz, right-multiplied, shared across frames) is a
separate module, `planner_canon` — different DOF names, different spec syntax,
different engine, and no shared caller. `planner.py` is the facade that re-exports
both, so no import site had to change. Each grammar's parser does recognize the
OTHER's vocabulary in order to REDIRECT: `canon_cross_hint`, imported below, is what
appends "that is a canonical DOF, wrong verb" to this module's unknown-DOF errors.

PURITY
------
numpy + stdlib + the pure `dof_space` / `planner_num` / `planner_canon`. No `bpy`.
Unit-testable in the analysis env.
"""

import itertools
import json
import os

import numpy as np

from views.sweeps.lib import dof_space
from views.sweeps.lib.planner_canon import canon_cross_hint
from views.sweeps.lib.planner_num import (ANGLE_AXES, ANGLE_DIRECTIONS,
                                       DEFAULT_ANGLE_HALF, _linspace,
                                       _preset_band, steps_for_quality)


# ---- default-box magnitudes for this grammar ------------------------------- #

# Every empty-range rotation starts at the standard preset (a local ±15deg
# search; widen to `large` — or an explicit range for a full swing — when the
# orientation prior is weak). Translation and scale retain their historical
# local-search spans.
POSE_DEFAULT_HALF = {
    "roll": DEFAULT_ANGLE_HALF, "yaw": DEFAULT_ANGLE_HALF,
    "pitch": DEFAULT_ANGLE_HALF,
    "dpx": 0.08, "dpy": 0.08, "tz": 0.1,
}
# Coupled empty-range sweeps use the same default yaw span and a local depth span.
POSE_COUPLED_DEFAULT = {"yaw": DEFAULT_ANGLE_HALF, "tz": 0.1}

# joint default search spans when no explicit range is given: the joint's
# declared `limit` wins; else type-appropriate.
DEFAULT_REVOLUTE = (-180.0, 180.0)          # degrees (full swing either way)
DEFAULT_PRISMATIC = (-1.0, 1.0)             # canonical units (longest dim ~= 1)


# --------------------------------------------------------------------------- #
# range parsing
# --------------------------------------------------------------------------- #
# The removed-DOF error (the `sigma` teaching message) is owned by dof_space, beside
# REMOVED_POSE_DOFS and reachable from classify_dof; both grammars call through it
# rather than aliasing it, so there is exactly one owner. The planner facade
# re-exports it, which is how existing callers still reach it.


def default_ranges(space, quality, sweep_space, angle_presets=None):
    """The empty-`--sweep-ranges` default box for a space (see the CLI selection
    table). `sweep_space` is the resolved 'pose'|'joint'|'both'. `angle_presets` is
    an optional {axis: half_magnitude} (from `parse_angle_presets`) that overrides a
    named roll/yaw/pitch axis' default band; unnamed angles keep the default
    standard span and dpx/dpy/tz keep their local spans. Returns ranges over
    DOFs in `space.dof_names`."""
    n = steps_for_quality(quality)
    presets = angle_presets or {}
    out = {}
    pose_default = (POSE_COUPLED_DEFAULT if sweep_space == "both"
                    else POSE_DEFAULT_HALF)
    for d in space.dof_names:
        if space.kinds[d] == dof_space.POSE:
            if d in presets:
                lo, hi = presets[d]        # (lo, hi) band, possibly one-sided
                out[d] = (lo, hi, n)
            elif d in pose_default:
                h = pose_default[d]
                out[d] = (-h, h, n)
        else:
            lim = space.joint_limits.get(d)
            if lim is not None:
                out[d] = (lim[0], lim[1], n)
            elif space.joint_types.get(d) == "prismatic":
                out[d] = (*DEFAULT_PRISMATIC, n)  # unbounded slide: canonical units
            else:
                out[d] = (*DEFAULT_REVOLUTE, n)   # unbounded hinge: full swing
    return out


def parse_ranges(spec, space, quality, sweep_space, angle_presets=None):
    """'yaw:-15,15,7;door:0,80,5' -> {dof:(lo,hi,steps)} (degrees for angle DOFs).

    Every named DOF is validated against `space` (a pose-order name must be one of `POSE_DOFS`;
    any other name must be a declared joint), a joint span is CLIPPED to its
    limit (never sweep an impossible state), and an empty spec -> `default_ranges`
    (widened per `angle_presets`; see `parse_angle_presets`). Explicit numbers here
    win over any preset for that axis. Raises ValueError on an unknown DOF (with the
    valid names) so a typo fails loud.
    """
    if not spec or not spec.strip():
        return default_ranges(space, quality, sweep_space, angle_presets)
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, vals = part.split(":", 1)
        name = name.strip()
        if name not in space.kinds:
            removed = dof_space.removed_dof_error("--sweep-ranges", name)
            if removed is not None:
                raise removed
            raise ValueError(
                f"--sweep-ranges: unknown DOF {name!r} (known: "
                f"{', '.join(space.dof_names)})" + canon_cross_hint(name))
        lo, hi, steps = (float(x) for x in vals.split(","))
        if space.kinds[name] == dof_space.JOINT:
            lim = space.joint_limits.get(name)
            if lim is not None:
                lo, hi = _clip_span(name, lo, hi, lim)
        out[name] = (lo, hi, int(steps))
    return out


def parse_bounds(spec, space, quality, sweep_space, angle_presets=None):
    """DE bounds using ``dof:min,max`` (a legacy third step value is ignored).

    Empty specs reuse the same physically meaningful default box as grid search.
    Joint limits and DOF validation are identical to :func:`parse_ranges`.
    """
    if not spec or not spec.strip():
        return {name: (float(rng[0]), float(rng[1]))
                for name, rng in default_ranges(
                    space, quality, sweep_space, angle_presets).items()}
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, vals = part.split(":", 1)
        name = name.strip()
        if name not in space.kinds:
            removed = dof_space.removed_dof_error("--sweep-ranges", name)
            if removed is not None:
                raise removed
            raise ValueError(
                f"--sweep-ranges: unknown DOF {name!r} (known: "
                f"{', '.join(space.dof_names)})" + canon_cross_hint(name))
        numbers = [float(x) for x in vals.split(",")]
        if len(numbers) not in (2, 3):
            raise ValueError(
                f"--sweep-ranges: DE bound for {name!r} must be min,max "
                f"(got {vals!r})")
        lo, hi = numbers[:2]
        if hi < lo:
            raise ValueError(
                f"--sweep-ranges: lower bound exceeds upper bound for {name!r}")
        if space.kinds[name] == dof_space.JOINT:
            lim = space.joint_limits.get(name)
            if lim is not None:
                lo, hi = _clip_span(name, lo, hi, lim)
        out[name] = (lo, hi)
    return out


def parse_apply_pairs(spec):
    """'pitch:90;yaw:17;door_hinge:-45' -> an ORDERED {dof: value} dict (the
    imperative `apply` verb — set these DOFs, no search). Pure syntax: it splits and
    floats the ';'-separated `dof:value` pairs but does NOT classify or validate the
    names (the view does that, since it needs the names to build the Space first).
    Insertion order is preserved so the report echoes the DOFs as typed. Empty -> {}.
    """
    out = {}
    if not spec or not spec.strip():
        return out
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"--apply: expected 'dof:value', got {part!r}")
        name, val = part.split(":", 1)
        out[name.strip()] = float(val)
    return out


def apply_ranges_spec(spec):
    """'yaw:90;door:70' -> a one-point sweep range spec 'yaw:90,90,1;door:...'.

    The `apply` verb is a `sweep` of exactly one candidate: each applied `dof:value`
    becomes a degenerate `min,max,steps` range with min==max and steps=1, which
    `_linspace`/`grid` collapse to that single value. So the sole scored candidate is
    exactly the applied order — and DOF-name validation + joint-limit clipping then
    happen in `parse_ranges` / `_clip_span`, identical to a hand-written sweep (no
    parallel parser to drift). Empty -> "" (an empty sweep, which the engine skips).
    """
    pairs = parse_apply_pairs(spec)
    return ";".join(f"{d}:{v:.10g},{v:.10g},1" for d, v in pairs.items())


def parse_candidates(spec):
    """Parse --sweep-candidates as JSON text or a JSON file.

    The stable candidate schema is a list of objects carrying optional `pose` and
    `joints`, plus provenance (`candidate_id`, `hypothesis_id(s)`). Missing pose or
    joints inherit the current frame state in the Blender-side engine.
    """
    if not spec:
        return []
    if isinstance(spec, (list, tuple)):
        raw = list(spec)
    elif isinstance(spec, dict):
        raw = spec.get("candidates", [spec])
    else:
        text = str(spec)
        if os.path.isfile(text):
            with open(text) as f:
                parsed = json.load(f)
        else:
            parsed = json.loads(text)
        raw = parsed.get("candidates", []) if isinstance(parsed, dict) else parsed
    if not isinstance(raw, list):
        raise ValueError("--sweep-candidates must be a JSON list or {candidates:[...]}")
    out = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"--sweep-candidates entry {i} must be an object")
        pose = item.get("pose")
        joints = item.get("joints")
        if pose is not None and not isinstance(pose, dict):
            raise ValueError(f"candidate {i} pose must be an object")
        if joints is not None and not isinstance(joints, dict):
            raise ValueError(f"candidate {i} joints must be an object")
        rec = dict(item)
        rec["candidate_id"] = str(item.get("candidate_id") or f"candidate-{i:04d}")
        hs = item.get("hypothesis_ids", item.get("hypothesis_id", []))
        if isinstance(hs, str):
            hs = [hs]
        rec["hypothesis_ids"] = [str(v) for v in (hs or [])]
        out.append(rec)
    return out


def parse_angle_presets(spec):
    """'yaw:large;pitch:tiny;roll:standard_cw' -> {axis: (lo, hi)} in degrees.

    A per-axis size preset for the empty-`--sweep-ranges` default box: the agent
    names a rotation axis (roll/yaw/pitch) and a preset (tiny/standard/large/huge,
    see `ANGLE_PRESET_HALF`), OPTIONALLY suffixed with a DIRECTION so the band is
    one-sided (see `ANGLE_DIRECTIONS`, `_preset_band`): `roll:cw`/`roll:ccw`,
    `yaw:leftcloser`/`yaw:rightcloser`, `pitch:topcloser`/`pitch:bottomcloser`. A
    directional band is PREFERRED — it spends the whole budget where the pose is
    actually off. Empty/blank -> {}. Raises ValueError (with the valid names) on an
    unknown axis, preset, or direction so a typo fails loud, mirroring
    `parse_ranges`.
    """
    if not spec or not spec.strip():
        return {}
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"--sweep-angle-preset: expected 'axis:preset', got {part!r}")
        axis, preset = part.split(":", 1)
        axis, preset = axis.strip(), preset.strip()
        if axis not in ANGLE_AXES:
            raise ValueError(
                f"--sweep-angle-preset: unknown axis {axis!r} "
                f"(known: {', '.join(ANGLE_AXES)})")
        out[axis] = _preset_band("--sweep-angle-preset", axis, preset,
                                 ANGLE_DIRECTIONS.get(axis, {}))
    return out


def _clip_span(name, lo, hi, limit):
    """Clip [lo,hi] to a joint's (lo,hi) limit; collapse to the nearest endpoint if
    the window lies entirely outside (a degenerate 1-point search there)."""
    clo, chi = max(lo, limit[0]), min(hi, limit[1])
    if clo > chi:
        clo = chi = min(max(0.5 * (lo + hi), limit[0]), limit[1])
    if abs(clo - lo) > 1e-9 or abs(chi - hi) > 1e-9:
        print(f"[render_wrapper] sweep: joint {name!r} span [{lo:.4g}, {hi:.4g}] "
              f"clipped to its limit [{limit[0]:.4g}, {limit[1]:.4g}] -> "
              f"[{clo:.4g}, {chi:.4g}].")
    return clo, chi


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #
def grid(space, ranges, max_candidates=20000):
    """Cartesian product over the named DOFs -> a list of vectors `x` aligned to
    `space.dof_names` (un-ranged DOFs HELD at the value that changes nothing).

    The pose/joint split is just each DOF's kind. steps<=1 -> the midpoint.

    "Held" is KIND-DEPENDENT, because the two kinds measure from different origins
    (see dof_space): a pose order is a camera-frame INCREMENT onto P0, so 0 means "no
    change"; a joint value is an ABSOLUTE state that `realize` OVERRIDES the base
    state with, so 0 means "articulate this joint to zero" — a real move, not a
    hold. Held joints fill with their `base_states` value, so a DOF absent from
    `ranges` is a true no-op for either kind.

    There is no fallback BEHIND that base state: `build_space` is handed a fill for
    every joint it puts in the space, so a JOINT DOF missing from `base_states` is a
    bug in the space, not an input to guess at. It raises.
    """
    names = space.dof_names
    samples = {}
    for d in names:
        if d in ranges:
            samples[d] = _linspace(*ranges[d])
        elif space.kinds[d] == dof_space.JOINT:
            # absolute state: the hold is the frame's declared state, not 0.
            if d not in space.base_states:
                raise ValueError(
                    f"grid: joint {d!r} is in the space but has no base state, so "
                    "there is no value to HOLD it at while the other DOFs sweep. "
                    "Joint states are absolute: nothing means 'leave this alone', "
                    "so it cannot be defaulted (0.0 would straighten the joint in "
                    "every candidate).")
            samples[d] = [float(space.base_states[d])]
        else:
            samples[d] = [0.0]        # pose increment: 0 IS the hold
    combos = list(itertools.product(*(samples[d] for d in names))) or [()]
    if len(combos) > max_candidates:
        print(f"[render_wrapper] sweep: WARNING grid has {len(combos)} candidates "
              f"(> {max_candidates}); consider fewer steps / DOFs")
    return [np.array(c, dtype=float) for c in combos]


def refine_ranges(space, ranges, best_x, shrink):
    """Shrunk ranges re-centered on the best VECTOR for a coarse->fine grid pass.

    This stays in the FIXED vector space: it re-centers each swept DOF's window on that DOF's value in `best_x` and shrinks
    its half-span by `shrink` (0.5 = halve). Pose orders re-center on the best order
    value; joints on the best joint value (then clip to the joint limit). Same step
    count. Keeping the space fixed means every pass's candidates carry the exact
    composed matrix, so the pasted BEST always reproduces its scored render.
    """
    idx = {d: i for i, d in enumerate(space.dof_names)}
    out = {}
    for d, (lo, hi, steps) in ranges.items():
        c = float(best_x[idx[d]])
        half = shrink * (hi - lo) / 2.0
        nlo, nhi = c - half, c + half
        if space.kinds[d] == dof_space.JOINT:
            lim = space.joint_limits.get(d)
            if lim is not None:
                nlo, nhi = max(nlo, lim[0]), min(nhi, lim[1])
                if nlo > nhi:
                    nlo = nhi = min(max(c, lim[0]), lim[1])
        out[d] = (nlo, nhi, steps)
    return out


def grid_budget_estimate(space, ranges):
    """Rough total candidate count for seeding Progress.total: the grid product
    over the ranged DOFs (a DOF absent from `ranges` is held at one value)."""
    n = 1
    for d in space.dof_names:
        if d in ranges:
            n *= max(1, int(ranges[d][2]))
    return n
