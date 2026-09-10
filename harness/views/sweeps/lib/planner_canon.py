"""planner_canon.py — candidate generation for the OBJECT-FRAME (canonical)
grammar (`osweep`, `oapply`, `oapply_all`).

The camera-frame grammar in `planner_pose` searches increments left-multiplied onto
ONE frame's pose. This one searches a rotation in the object's own CANONICAL frame,
right-multiplied onto EVERY frame's pose (M' = M @ D; D pivots at the frame's
FK-AABB centre per-frame, at the canonical origin shared — shared_engine owns the
distinction, this module only generates the (rx,ry,rz)) — the correction for an
object that was BUILT mis-oriented, which no camera-frame yaw can express. Its three
DOFs (rx, ry, rz) form a ROTATION VECTOR — degrees about canonical +X (right) / +Y
(front) / +Z (up), axis = direction, angle = norm — so a multi-axis candidate is ONE
rotation about a tilted axis (no composition order, no gimbal lock). The engine
converts to radians at the single `rig.lie.canonical_rotation` seam (lie stays
radians).

The names are axis-LITERAL on purpose: sharing spellings with the camera-frame
POSE_DOFS would invite porting a camera-frame yaw into an object rotation (different
frames — they only coincide for an upright object). The two grammars have DISJOINT
names, and each parser recognizes the other's
vocabulary in order to redirect (see `_canon_unknown_dof_error`, and
`canon_cross_hint` for the other direction, which `planner_pose` imports).

This module holds three layers, from the most explicit spelling to the least:

  * `--osweep-ranges` -> `parse_canon_ranges` -> `canon_grid`: an rx/ry/rz lattice;
  * `--opreset` -> `parse_opreset` -> a `CanonPlan`: one axis, a named turn set or
    size band, sampled — or an `So3Plan` (`so3:N`): identity + N Haar-uniform
    random rotations from ONE baked seed, the re-localise grid for a lost pose;
  * `--oapply` -> `parse_canon_candidates`: a LABELLED candidate set (the symmetry
    flip panels), bypassing lattices entirely.

`resolve_canon_search` is the one place that decides which of the three a run used.

PURITY
------
numpy + stdlib + the pure `dof_space` / `planner_num`. No `bpy`. Unit-testable in the
analysis env.
"""

import itertools
import math

import numpy as np

from views.sweeps.lib import dof_space
from views.sweeps.lib.planner_num import (ANGLE_PRESET_HALF, DEFAULT_ANGLE_HALF,
                                       _linspace, _preset_band,
                                       steps_for_quality)

# ---- the canonical DOF vocabulary ------------------------------------------ #
# A ROTATION VECTOR, not Euler angles: see the module docstring for why the names
# are axis-literal and why the two grammars keep DISJOINT spellings.
CANON_DOFS = ("rx", "ry", "rz")
_CANON_SET = frozenset(CANON_DOFS)

# the retired canonical spellings -> their axis-literal replacements (kept ONLY to
# teach in error messages; the old names never parse).
_CANON_RENAMED = {"yaw": "rz", "pitch": "rx", "tilt": "ry"}

_CANON_AXES_BLURB = "rx (+X right), ry (+Y front), rz (+Z up)"


def _canon_unknown_dof_error(flag, name, extra_known=""):
    """The ValueError for a non-canonical DOF name in an osweep/oapply spec.

    Teaches instead of just rejecting: the retired yaw/pitch/tilt spellings map to
    their rx/ry/rz replacements, and a camera-frame sweep/apply DOF gets told it is
    the WRONG FRAME (the classic cross-verb transfer mistake), not just unknown."""
    if name in _CANON_RENAMED:
        also_camera = (
            f" If you meant the CAMERA-frame {name!r} of sweep/apply, that is a "
            f"DIFFERENT rotation (camera axes, ONE frame) and does not transfer "
            f"here." if name in dof_space.POSE_DOF_SET else "")
        return ValueError(
            f"{flag}: unknown DOF {name!r}. The canonical DOFs are axis-literal "
            f"{_CANON_AXES_BLURB}; the old canonical spelling {name!r} is now "
            f"{_CANON_RENAMED[name]!r}.{also_camera}")
    removed = dof_space.removed_dof_error(flag, name)
    if removed is not None:
        return removed
    if name in dof_space.POSE_DOF_SET:
        return ValueError(
            f"{flag}: {name!r} is a CAMERA-frame sweep/apply DOF (an increment "
            f"about the camera axes, left-multiplied onto ONE frame's pose). This "
            f"verb rotates the object in its own CANONICAL frame, shared across "
            f"EVERY frame — use {_CANON_AXES_BLURB}.")
    known = ", ".join(CANON_DOFS) + (f", {extra_known}" if extra_known else "")
    return ValueError(f"{flag}: unknown DOF {name!r} (known: {known})")


def canon_cross_hint(name):
    """A teaching suffix for CAMERA-frame grammar errors when the unknown name is a
    canonical rx/ry/rz DOF handed to the wrong verb. "" when it is not."""
    if name not in _CANON_SET:
        return ""
    return (f". NOTE {name!r} is a CANONICAL object-frame DOF (osweep/oapply: one "
            "shared rotation about the object's own axes, right-multiplied onto "
            "EVERY frame's pose); the sweep/apply grammar is CAMERA-frame — "
            "roll/yaw/pitch about the camera axes on ONE frame — plus declared "
            "joint names")


def default_canon_ranges(quality, angle_presets=None):
    """Empty-`--osweep-ranges` default box: the `standard` angle half-span (+/-15deg)
    on every canonical axis (overridable per axis by `angle_presets`), step count
    from quality. Mirrors default_ranges for the camera-frame space."""
    n = steps_for_quality(quality)
    presets = angle_presets or {}
    out = {}
    for d in CANON_DOFS:
        lo, hi = presets[d] if d in presets else (-DEFAULT_ANGLE_HALF,
                                                  DEFAULT_ANGLE_HALF)
        out[d] = (lo, hi, n)
    return out


def parse_canon_ranges(spec, quality, angle_presets=None):
    """'rz:-180,180,9;ry:0,0,1' -> {dof:(lo,hi,steps)} over rx/ry/rz (degrees).

    Empty spec -> default_canon_ranges (widened per `angle_presets`). Explicit
    numbers win over any preset. Raises ValueError on an unknown DOF (mirrors
    parse_ranges) so a typo fails loud — and TEACHES when the name is the retired
    yaw/pitch/tilt spelling or a camera-frame sweep DOF (the wrong verb)."""
    if not spec or not spec.strip():
        return default_canon_ranges(quality, angle_presets)
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, vals = part.split(":", 1)
        name = name.strip()
        if name not in _CANON_SET:
            raise _canon_unknown_dof_error("--osweep-ranges", name)
        lo, hi, steps = (float(x) for x in vals.split(","))
        out[name] = (lo, hi, int(steps))
    return out


def parse_canon_angle_presets(spec):
    """'rz:large;ry:tiny' -> {axis: (lo, hi)} over the canonical axes.

    The osweep twin of parse_angle_presets (which validates roll/yaw/pitch); this
    validates rx/ry/rz. Canonical axes have NO fixed visual direction (a positive
    `rz` reads differently depending on how the object is posed — see
    conventions/DIRECTIONS.md), so there are no directional suffixes here: every
    band is the SYMMETRIC `(-half, +half)`. Empty/blank -> {}. Raises ValueError
    on an unknown axis or preset."""
    if not spec or not spec.strip():
        return {}
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"--osweep-angle-preset: expected 'axis:preset', got {part!r}")
        axis, preset = (x.strip() for x in part.split(":", 1))
        if axis not in _CANON_SET:
            raise _canon_unknown_dof_error("--osweep-angle-preset", axis)
        out[axis] = _preset_band("--osweep-angle-preset", axis, preset, {})
    return out


def canon_grid(ranges):
    """Cartesian product over rx/ry/rz -> list of (rx, ry, rz) rotation-vector
    tuples. A DOF absent from `ranges` is HELD at 0.0 (identity on that axis).

    Accepts EITHER a `{dof: (lo, hi, steps)}` range dict (the `--osweep-ranges`
    lattice, sampled by `_linspace`) or a `CanonPlan` (the `--opreset` form, which
    carries its per-DOF sample LIST directly). Presets enumerate angle sets a
    triple cannot always express — `'z:0,90,180,270'` is four rotations, not a
    band — so the plan is the authority on what gets scored and this is the one
    place both forms turn into candidates."""
    if isinstance(ranges, So3Plan):
        # coupled triples, not a product: the plan already materialized them.
        return list(ranges.cands)
    if isinstance(ranges, CanonPlan):
        samples = {d: list(ranges.samples.get(d, [0.0])) for d in CANON_DOFS}
    else:
        samples = {d: (_linspace(*ranges[d]) if d in ranges else [0.0])
                   for d in CANON_DOFS}
    return [tuple(float(v) for v in c)
            for c in itertools.product(*(samples[d] for d in CANON_DOFS))]


def refine_canon_ranges(ranges, best, shrink):
    """Shrunk canonical ranges re-centered on the best (rx,ry,rz) for a
    coarse->fine pass (the osweep twin of refine_ranges; no joint limits to clip).

    Accepts a plain range dict or a `CanonPlan`. Every axis of a plain dict shrinks
    (it came from an explicit `lo,hi,steps` band). For a plan, only axes in
    `refinable` shrink; a DISCRETE turn-set axis is PINNED to the winner's own angle
    as a degenerate `(c, c, 1)` range. There is no basin between 90deg and 180deg to
    descend into, so shrinking a made-up band around a quarter-turn would score
    angles the agent never asked about — and simply dropping the axis would silently
    reset it to 0, throwing away the quarter-turn that won.

    An `So3Plan` becomes a PLAIN per-axis band dict centered on the winner's
    rotation vector, half-width `so3_cell_half_deg(n)` scaled by `shrink` (so the
    default 0.5 gives exactly the padded cell radius — the knob keeps meaning
    "smaller = tighter"). The winner's true pose lies inside its own Voronoi cell,
    and at that scale a rotation-vector box is a fine local chart. Later passes
    need no so3 branch at all: the returned dict is the fully-refinable plain form."""
    best_map = dict(zip(CANON_DOFS, best))
    if isinstance(ranges, So3Plan):
        h = so3_cell_half_deg(ranges.n) * (float(shrink) / 0.5)
        return {d: (best_map[d] - h, best_map[d] + h, ranges.refine_steps)
                for d in CANON_DOFS}
    if isinstance(ranges, CanonPlan):
        bands, refinable = ranges.bands, ranges.refinable
        pinned = {d: (best_map[d], best_map[d], 1)
                  for d in ranges.samples if d not in refinable}
    else:
        bands, refinable, pinned = (ranges or {}), set(ranges or {}), {}
    out = dict(pinned)
    for d, (lo, hi, steps) in bands.items():
        if d not in refinable:
            continue
        c = best_map[d]
        half = shrink * (hi - lo) / 2.0
        out[d] = (c - half, c + half, steps)
    return out


# ---- --opreset: one axis + a RANGE OF ANGLES, shared by all three oviews ----- #
# Axis tokens are x/y/z to match the existing `flip:x|y|z` sugar, with rx/ry/rz
# accepted as aliases.
#
# TURN-SETS are discrete angle enumerations; every one INCLUDES 0 so the no-op
# baseline always competes on the same table (the same reason 'flips' expands to
# identity|flip:x|flip:y|flip:z). SIZE BANDS reuse ANGLE_PRESET_HALF unchanged.
#
# There are deliberately NO directional suffixes here, unlike the camera-frame
# --sweep-angle-preset: a canonical axis has no fixed ON-SCREEN direction. The same
# shared rz:+30 moves a landmark by up to 169deg differently between two frames
# (measured; see conventions/DIRECTIONS.md), so a word like "clockwise" would be a
# lie in some frame of the very run it was ordered for. The positive SENSE is still
# fixed in the object's own frame (rz turns +X toward +Y), which is why signed
# numeric angles are fine — the ban is on screen-relative NAMES, not on signs.
_OPRESET_AXIS = {"x": "rx", "y": "ry", "z": "rz",
                 "rx": "rx", "ry": "ry", "rz": "rz"}

# {token: (samples, band_or_None)}. `band` is what refine may shrink; None marks a
# discrete set that refine must leave alone.
TURN_SETS = {
    "flip":     ((0.0, 180.0), None),
    "quarters": ((0.0, 90.0, 180.0, 270.0), None),
    "octants":  ((0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0), None),
    "full":     (None, (-180.0, 180.0)),        # a BAND: samples come from quality
}


# ---- so3:N — Haar-uniform re-localisation ----------------------------------- #
# The seed every bare `so3:N` grid is drawn from. A module constant, not a separate
# flag: the point of the preset is a REPRODUCIBLE grid an agent can name in two
# tokens ('so3:64'), and the same spec must mean the same rotations in every run.
# Sequential draws from one stream also make `so3:2N` a strict PREFIX extension of
# `so3:N`: raising granularity adds samples without reshuffling the ones already
# judged, so reports at different N stay comparable candidate-for-candidate (a
# legibility property — the engine re-renders everything either way).
#
# An UNHAPPY agent re-rolls IN the spec: 'so3:64@7' draws from seed 7 instead — a
# fresh independent Haar sample of the same size, for when every basin of the
# default draw judged wrong. That is not the silent-disagreement problem a seed
# flag would be, because the spec string itself names the seed it used.
#
# CHANGING THIS DEFAULT LATER IS SUPPORTED, HERE AND ONLY HERE: edit the constant.
# It is a one-way door per value — every bare `so3:N` grid ever scored was drawn
# from the default of its day, so a change re-rolls what the bare spec means and
# old reports stop being re-runnable by spec alone. That is why each so3 pass
# record stamps {n, seed} into the report: a report names ITS OWN generator and
# survives the constant moving under it. numpy's PCG64 stream is stable across
# versions and platforms, so a (seed, n) pair pins the grid exactly.
SO3_SEED = 0

# renders scale as (N+1) x frames x (1 + refine): fail loud, not slow. The visual
# budget guards IMAGES; this guards the scoring loop itself.
SO3_MAX = 4096


def so3_samples(n, seed=SO3_SEED):
    """n Haar-uniform random rotations -> [(rx, ry, rz), ...] rotation vectors in
    agent-facing DEGREES (norm <= 180, the convention `rig.lie.canonical_rotation`
    consumes).

    Method: 4D standard Gaussians normalized to unit quaternions — Haar-uniform by
    the rotation invariance of the isotropic Gaussian, which naive Euler sampling
    is NOT (it piles mass near the poles). The double cover is collapsed by
    flipping to w >= 0, so the rotation angle 2*atan2(|v|, w) lands in [0, 180].

    IID draws pay a modest coverage penalty vs an optimized point set (~1.5x on
    covering radius at n=64); `so3_cell_half_deg` carries that factor so the refine
    window still contains the true pose. Kept IID anyway because a low-discrepancy
    sequence is a contained swap behind this one function — decide before wide use,
    since after it byte-stability of the grid wins."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((int(n), 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0.0] *= -1.0                       # double cover: w >= 0
    w, v = q[:, 0], q[:, 1:]
    vn = np.linalg.norm(v, axis=1)
    theta = 2.0 * np.arctan2(vn, w)                # [0, pi]
    # unit axis, guarding the measure-zero identity draw (vn ~ 0 -> zero vector)
    axis = np.divide(v, vn[:, None], out=np.zeros_like(v), where=vn[:, None] > 0)
    rotvec = axis * np.degrees(theta)[:, None]
    return [tuple(float(x) for x in r) for r in rotvec]


def so3_cell_half_deg(n):
    """The refine half-window (degrees) around an so3:N winner: the covering-radius
    estimate of one sample's Voronoi cell, times a 1.5 IID safety factor.

    Volume argument: SO(3) has Haar volume pi^2 (in rotation-vector measure a ball
    of geodesic radius r has volume ~ 4/3*pi*r^3 for small r), so n cells of radius
    r cover it when 4/3*pi*r^3 * n ~ pi^2, i.e. r ~ (3*pi/(4n))^(1/3). IID samples
    leave gaps up to ~1.5x that radius (the log-factor penalty vs an optimized
    set), so the window is padded to still contain the true pose. n=64 -> ~29deg."""
    return 1.5 * math.degrees((3.0 * math.pi / (4.0 * float(n))) ** (1.0 / 3.0))


class So3Plan:
    """A resolved `so3:N` search: identity + n Haar-uniform rotations, plus the
    quality-derived step count refine will need (`refine_canon_ranges` has no
    quality argument, so the plan carries it from parse time).

    A sibling of `CanonPlan`, not a subclass: that plan is per-DOF sample lists
    Cartesian-producted, and a Haar set is a list of COUPLED triples no product can
    express. Identity is prepended on top of the n samples — the same rule as every
    turn set ("the no-op baseline always competes"): with tracking lost, "the pose
    was actually fine" must be a scoreable answer, and n keeps meaning n RANDOM
    samples. There is no lattice, so `canon_report_ranges` yields {} (panel binning
    falls back to score order — the honest answer, same as an uneven angle list)."""

    def __init__(self, n, refine_steps, seed=SO3_SEED):
        self.n = int(n)
        self.seed = int(seed)
        self.refine_steps = int(refine_steps)
        self.cands = [(0.0, 0.0, 0.0)] + so3_samples(self.n, self.seed)

    def __repr__(self):                                        # debugging only
        return f"So3Plan(n={self.n}, seed={self.seed})"


def _parse_so3_token(part, quality):
    """'so3:N' / 'so3:N@SEED' -> an So3Plan. Teaching errors for every wrong shape.

    The bare form draws from the baked SO3_SEED — no knob to remember, the same
    grid every time. '@SEED' is the RE-ROLL: when the draw's basins all judged
    wrong, a different seed is a fresh independent Haar sample of the same size,
    which is cheaper evidence than doubling N (whose first half you already
    scored). '@' rather than a comma because a comma already means "angle list"
    in this grammar. The seed rides into the report's pass record either way, so
    a re-rolled run names its own generator."""
    _, _, val = part.partition(":")
    tok, _, seed_tok = val.strip().partition("@")
    try:
        n = int(tok.strip())
    except ValueError:
        raise ValueError(
            f"--opreset: so3 takes a COUNT, e.g. 'so3:64' or 'so3:64@7' (got "
            f"{part!r}). The default seed is baked (one reproducible grid per N; "
            "see planner_canon.SO3_SEED); '@SEED' re-rolls the draw.") from None
    if n < 1:
        raise ValueError(f"--opreset: so3 needs a positive count (got {n})")
    if n > SO3_MAX:
        raise ValueError(
            f"--opreset: so3:{n} exceeds the {SO3_MAX}-candidate cap — renders "
            "scale as N x frames x passes, and a grid that size is almost always "
            "a mis-typed count. Coarse-to-fine (so3:64 with --sweep-refine) "
            "recovers a pose faster than brute density.")
    seed = SO3_SEED
    if seed_tok:
        try:
            seed = int(seed_tok.strip())
        except ValueError:
            raise ValueError(
                f"--opreset: so3 seed must be an integer, e.g. 'so3:{n}@7' "
                f"(got {part!r})") from None
        if seed < 0:
            raise ValueError(
                f"--opreset: so3 seed must be non-negative (got {seed})")
    return So3Plan(n, steps_for_quality(quality), seed)


class CanonPlan:
    """A resolved canonical search plan: per-DOF SAMPLE LISTS plus the bands (if
    any) those samples came from.

    Why a carrier instead of the `{dof: (lo, hi, steps)}` dict everything else
    uses: a triple can only express an EVENLY SPACED band, and `--opreset` must
    also express an arbitrary angle LIST ('z:0,90,180,270'). Rather than teach
    every consumer two shapes, the plan holds both — `samples` for
    `canon_grid`, `bands` for `refine_canon_ranges` and for the report's `ranges`
    block (which is also what `metrics.binnable_dofs` bins panels against).

    An evenly-spaced turn-set DOES get a synthetic band so panel binning keeps
    working: `quarters` is the lattice lo=0, step=90. It is recorded in
    `bands` but NOT in `refinable`, so panels bin while refine stays off — those
    are different questions about the same numbers.
    """

    def __init__(self, samples, bands, refinable):
        self.samples = {d: [float(v) for v in samples[d]] for d in samples}
        self.bands = dict(bands)
        # the subset of `bands` a coarse->fine pass may shrink.
        self.refinable = frozenset(refinable)

    def __repr__(self):                                        # debugging only
        return f"CanonPlan(samples={self.samples}, bands={self.bands})"

    @property
    def report_ranges(self):
        """The `ranges` block for the report / panel binning: the bands, which for a
        turn-set are the synthetic lattice its samples lie on."""
        return dict(self.bands)


def _even_band(samples):
    """(lo, hi, steps) for an evenly-spaced sample list, or None if it is uneven.

    A synthetic lattice for panel binning (see CanonPlan). Uneven lists get None
    rather than a fabricated step: `metrics.binnable_dofs` then reports "no
    lattice", which is the honest answer, and panels fall back to score order."""
    vals = [float(v) for v in samples]
    if len(vals) < 2:
        return None
    step = vals[1] - vals[0]
    if step <= 0:
        return None
    if any(abs((vals[i + 1] - vals[i]) - step) > 1e-9
           for i in range(len(vals) - 1)):
        return None
    return (vals[0], vals[-1], len(vals))


def _opreset_token(axis, token, quality):
    """One `--opreset` token -> (samples, band, refinable).

    `token` is a turn-set name, a size-band preset name, or an explicit
    comma-separated angle LIST. The numeric form is ALWAYS a list, never
    `lo,hi,steps`: a 3-value spec would otherwise be ambiguous with
    `--osweep-ranges`' triple, and deciding by counting values is exactly the kind
    of guess that produces a silently wrong grid. `lo,hi,steps` already has a flag;
    this one owns enumeration."""
    tok = token.strip()
    if not tok:
        raise ValueError(f"--opreset: empty preset for axis {axis!r}")
    low = tok.lower()
    if low in TURN_SETS:
        samples, band = TURN_SETS[low]
        if samples is None:                       # 'full': a band, quality-sampled
            n = steps_for_quality(quality)
            vals = _dedup_angles(_linspace(band[0], band[1], n))
            # keep the ORDERED band (that is the lattice panels bin against and the
            # window refine shrinks) even though a wrapped endpoint was dropped.
            return vals, (band[0], band[1], n), True
        vals = _dedup_angles(samples)
        return vals, _even_band(vals), False
    if low in ANGLE_PRESET_HALF:
        # size band: symmetric +/-half, sampled by quality. NO directional suffix
        # (see the comment above _OPRESET_AXIS), so _preset_band gets no directions.
        lo, hi = _preset_band("--opreset", axis, low, {})
        n = steps_for_quality(quality)
        return _dedup_angles(_linspace(lo, hi, n)), (lo, hi, n), True
    if any(ch.isdigit() for ch in tok):
        try:
            vals = [float(v) for v in tok.split(",") if v.strip()]
        except ValueError:
            raise ValueError(
                f"--opreset: bad angle list {tok!r} for axis {axis!r} (expected "
                "comma-separated DEGREES, e.g. 'z:0,90,180,270')") from None
        if not vals:
            raise ValueError(f"--opreset: empty angle list for axis {axis!r}")
        vals = _dedup_angles(vals)
        return vals, _even_band(vals), False
    raise ValueError(
        f"--opreset: unknown preset {tok!r} for axis {axis!r} (turn sets: "
        f"{', '.join(TURN_SETS)}; sizes: {', '.join(ANGLE_PRESET_HALF)}; or an "
        "explicit comma-separated angle list in DEGREES like '0,90,180,270'). "
        "There are no directional suffixes on a canonical axis — it has no fixed "
        "on-screen direction (see conventions/DIRECTIONS.md).")


def canon_searched_dofs(ranges):
    """The canonical DOFs a search actually varies, for logging.

    NOT the same as the report `ranges` keys: a preset axis with a single sample
    carries no band (nothing to bin or refine), yet it IS part of the rotation being
    scored, so it belongs in the "searching over ..." line."""
    if isinstance(ranges, So3Plan):
        return list(CANON_DOFS)       # a Haar sample genuinely turns all three
    if isinstance(ranges, CanonPlan):
        return [d for d in CANON_DOFS if d in ranges.samples]
    return [d for d in CANON_DOFS if d in (ranges or {})]


def canon_report_ranges(ranges):
    """The `{dof: (lo, hi, steps)}` view of either canonical range form.

    A `CanonPlan` yields its bands; a plain dict is already that shape. This is what
    the REPORT records and what `metrics.binnable_dofs` bins panels against, so
    callers that only need the lattice never have to know which form they were
    handed (`canon_grid`/`refine_canon_ranges` are the two that do). An `So3Plan`
    yields {} — random triples lie on NO lattice, and per `_even_band`'s
    philosophy "no lattice" is the honest answer (panels fall back to score
    order); the generator itself is recorded via the pass record's `so3` field."""
    if isinstance(ranges, So3Plan):
        return {}
    if isinstance(ranges, CanonPlan):
        return ranges.report_ranges
    return dict(ranges or {})


def _dedup_angles(vals):
    """Drop samples that name the SAME rotation, keeping the first and the order.

    Angles are compared modulo 360: `'z:full'` spans [-180, 180], whose endpoints
    are one rotation, and an explicit list may say `0,360`. Scoring both wastes a
    render per frame and, worse, would seat the same rotation in the ranked table
    twice — reading as two independent candidates that happen to tie."""
    out, seen = [], set()
    for v in vals:
        key = round(float(v) % 360.0, 6) % 360.0
        if key in seen:
            continue
        seen.add(key)
        out.append(float(v))
    return out


def parse_opreset(spec, quality):
    """'z:quarters' / 'z:0,90,180,270' / 'z:quarters;x:tiny' -> a CanonPlan;
    'so3:N' -> an So3Plan; empty -> None.

    The ONE preset flag all three object-centric views share. Grammar: ';'-separated
    `axis:preset`, where `axis` is x/y/z (rx/ry/rz accepted) and `preset` is a
    turn-set, a size band, or an explicit angle list in degrees. Several axes give
    the CARTESIAN PRODUCT of their sample lists.

    `so3:N` (or `so3:N@SEED` to re-roll the draw) is a PSEUDO-AXIS and must stand
    ALONE (like 'flip:AXIS' in --oapply): identity + N Haar-uniform rotations from
    the baked SO3_SEED (or the named one) — the re-localise grid when tracking is
    lost and no axis is suspected. A Cartesian product of Haar samples with an
    axis lattice would not be Haar-uniform and means nothing, so mixing raises
    rather than guessing.

    Empty/blank -> None, meaning "no preset given" so the caller can fall through to
    --osweep-ranges / --osweep-angle-preset. Raises ValueError on a bad axis (through
    the shared `_canon_unknown_dof_error`, so the camera-frame-DOF and retired
    yaw/pitch/tilt redirects come for free) or a bad token."""
    if not spec or not spec.strip():
        return None
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    so3_parts = [p for p in parts if p.lower() == "so3"
                 or p.lower().startswith("so3:")]
    if so3_parts:
        if len(parts) > 1:
            raise ValueError(
                f"--opreset: 'so3:N' must stand ALONE (got {spec!r}) — a "
                "Cartesian product of Haar-uniform samples with an axis lattice "
                "is not Haar-uniform. If you KNOW the lost axis, a per-axis "
                "preset like 'z:full' is the cheaper search.")
        if ":" not in so3_parts[0]:
            raise ValueError(
                "--opreset: so3 takes a COUNT, e.g. 'so3:64' (identity + 64 "
                "Haar-uniform rotations from a baked seed)")
        return _parse_so3_token(so3_parts[0], quality)
    samples, bands, refinable = {}, {}, set()
    for part in parts:
        if ":" not in part:
            raise ValueError(
                f"--opreset: expected 'axis:preset', got {part!r} (e.g. "
                "'z:quarters', 'z:0,90,180,270', 'x:large')")
        axis, token = (x.strip() for x in part.split(":", 1))
        dof = _OPRESET_AXIS.get(axis.lower())
        if dof is None:
            raise _canon_unknown_dof_error(
                "--opreset", axis, extra_known="or the x/y/z axis letters")
        vals, band, can_refine = _opreset_token(dof, token, quality)
        samples[dof] = vals
        if band is not None:
            bands[dof] = band
            if can_refine:
                refinable.add(dof)
    if not samples:
        return None
    return CanonPlan(samples, bands, refinable)


def resolve_canon_search(ranges_spec, preset_spec, angle_preset_spec, quality):
    """The three canonical-search spellings -> the ONE thing to grid (dict or CanonPlan).

    PRECEDENCE: explicit `--osweep-ranges` numbers > `--opreset` > the
    `--osweep-angle-preset` default box. Explicit numbers win because they are the
    least ambiguous statement of intent; the per-axis angle preset stays last since
    it only ever SIZED the empty-ranges default box, so a preset naming actual
    samples is strictly more specific.

    Lives here, not in the engine, so the ordering is one pure function every
    object-centric verb shares and a test can pin without rendering anything."""
    if (ranges_spec or "").strip():
        return parse_canon_ranges(ranges_spec, quality,
                                  parse_canon_angle_presets(angle_preset_spec))
    plan = parse_opreset(preset_spec, quality)
    if plan is not None:
        return plan
    return parse_canon_ranges("", quality,
                              parse_canon_angle_presets(angle_preset_spec))


def opreset_candidates(spec, quality):
    """`--opreset` -> a LABELLED candidate set [(label, (rx,ry,rz)), ...], the shape
    the imperative verbs (oapply / oapply_all) already consume.

    The search verb takes the CanonPlan and grids it; the imperative verbs want the
    named-set path instead — per-candidate renders, the frames x candidates matrix,
    a ranked-by-label manifest — so the same preset expands to labels here. Labels
    are filename-safe and sort in angle order (`rz+000`, `rz+090`, `rz+180`), and
    the all-zero rotation is named `identity` so a preset's no-op baseline reads the
    same as the flip panel's. Empty spec -> [].

    `so3:N` is REFUSED here, not expanded: the imperative verbs render NAMED
    rotations for an agent to LOOK at, and a random draw has no names — it is a
    search grid. On `oapply_all` it would be worse than pointless: ranking one
    shared rotation by mean IoU over a blind draw is exactly the whole-run
    searched mode this family deliberately does not have (see osweep.md)."""
    plan = parse_opreset(spec, quality)
    if plan is None:
        return []
    if isinstance(plan, So3Plan):
        raise ValueError(
            "--opreset: 'so3:N' is a SEARCH grid (random rotations have no names "
            "to panel and judge), so the imperative verbs do not take it — use "
            "the osweep view, which searches it per frame and refines each "
            "frame's winner.")
    out = []
    for c in canon_grid(plan):
        if all(abs(v) < 1e-9 for v in c):
            out.append(("identity", c))
            continue
        parts = [f"{d}{_angle_tag(v)}" for d, v in zip(CANON_DOFS, c)
                 if d in plan.samples and abs(v) > 1e-9]
        out.append(("_".join(parts) or "identity", c))
    return out


def imperative_candidates(oapply, opreset, quality):
    """The candidate SET the imperative canonical verbs try: `--oapply`'s named
    rotations, `--opreset`'s expanded angle set, or the union when both are given.

    `oapply` and `oapply_all` resolve their candidates identically — they differ only
    in whether each frame then picks its own winner — so the resolution lives here,
    beside the two functions it composes, rather than as the same six lines in both
    views. A preset expands to LABELLED candidates rather than a grid because that is
    the imperative path: every candidate rendered for every frame, a per-frame ranked
    table naming which one won. That is what a 'which quarter-turn is right?'
    question needs."""
    return merge_canon_candidates(parse_canon_candidates(oapply),
                                  opreset_candidates(opreset, quality))


def merge_canon_candidates(*lists):
    """Concatenate labelled candidate lists, dropping repeat ROTATIONS (first label
    wins) — the same rule `parse_canon_candidates` applies within one spec.

    Used when a verb is handed both `--oapply` and `--opreset`: they are two
    spellings of the same thing (a set of shared rotations to render and score), so
    the honest result is their union. Silently preferring one would mean a candidate
    the agent explicitly named never got scored, while the report still listed a
    winner."""
    out, seen = [], set()
    for lst in lists:
        for label, cand in (lst or []):
            key = tuple(round(float(v), 9) for v in cand)
            if key in seen:
                continue
            seen.add(key)
            out.append((label, tuple(float(v) for v in cand)))
    return out


def _angle_tag(v):
    """A signed, zero-padded, filename-safe angle tag: 90 -> '+090', -22.5 ->
    '-022p5'. Padded to 3 integer digits so labels sort in ANGLE order in a
    directory listing and on the report's ranked table, where a plain '%g' would
    put rz+180 before rz+90."""
    sign = "-" if v < 0 else "+"
    a = abs(float(v))
    whole = int(a)
    frac = a - whole
    tag = f"{sign}{whole:03d}"
    if frac > 1e-9:
        tag += "p" + f"{frac:.3f}".split(".")[1].rstrip("0")
    return tag


# ---- named candidate SETS for oapply (symmetry-flip panels) ----------------- #
# A near-symmetric object merged from independently-refined frame windows can land
# in different silhouette basins on the two sides of a seam; the standard check is
# "score the 180deg flip about each canonical axis and LOOK". flip:x/y/z name those
# flips (180deg about canonical +X/+Y/+Z, i.e. rx/ry/rz = 180); 'flips' is the
# default panel: the identity baseline + all three, competing on one table.
FLIP_AXIS_DOF = {"x": "rx", "y": "ry", "z": "rz"}
FLIPS_PANEL_SPEC = "identity|flip:x|flip:y|flip:z"


def _canon_candidate(part):
    """One '|'-separated candidate token -> (label, (rx, ry, rz)).

    Accepts 'identity', 'flip:x|y|z' (which must stand ALONE — a flip is a named
    180deg rotation, not a term to sum into a rotation vector; spell a custom axis
    explicitly), or a ';'-joined 'dof:value' list over rx/ry/rz (degrees). Labels
    are stable and filename-safe: 'identity', 'flip_x', or 'ry+12_rz+90'."""
    if part.lower() == "identity":
        return "identity", (0.0, 0.0, 0.0)
    toks = [t.strip() for t in part.split(";") if t.strip()]
    if any(t.lower().startswith("flip:") for t in toks) and len(toks) > 1:
        raise ValueError(
            f"--oapply: 'flip:AXIS' must stand alone in a candidate (got "
            f"{part!r}); spell a composed rotation explicitly in dof:value form")
    vals = {}
    for tok in toks:
        if ":" not in tok:
            raise ValueError(f"--oapply: expected 'dof:value', got {tok!r}")
        name, val = (x.strip() for x in tok.split(":", 1))
        if name.lower() == "flip":
            axis = val.lower()
            if axis not in FLIP_AXIS_DOF:
                raise ValueError(f"--oapply: unknown flip axis {val!r} "
                                 f"(known: {', '.join(sorted(FLIP_AXIS_DOF))})")
            return f"flip_{axis}", tuple(
                180.0 if d == FLIP_AXIS_DOF[axis] else 0.0 for d in CANON_DOFS)
        if name not in _CANON_SET:
            raise _canon_unknown_dof_error("--oapply", name,
                                           extra_known="flip:x|y|z, identity, "
                                                       "flips")
        try:
            vals[name] = float(val)
        except ValueError:
            raise ValueError(f"--oapply: bad value in {tok!r}") from None
    if not vals:
        raise ValueError(f"--oapply: empty candidate {part!r}")
    label = "_".join(f"{d}{vals[d]:+.3g}" for d in CANON_DOFS if d in vals)
    return label, tuple(vals.get(d, 0.0) for d in CANON_DOFS)


def parse_canon_candidates(spec):
    """--oapply spec -> ordered [(label, (rx, ry, rz)), ...].

    Grammar — '|' separates candidates, ';' separates dof:value terms inside one
    (the terms form ONE rotation vector in degrees, not a sequence):
      * 'rz:180'                      ONE custom candidate (the original grammar)
      * 'flip:z'                      a 180deg flip about a canonical axis
                                      (x -> rx, y -> ry, z -> rz)
      * 'identity'                    the no-flip baseline
      * 'flips'                       the DEFAULT SYMMETRY PANEL — expands to
                                      identity|flip:x|flip:y|flip:z; usable
                                      inline in a '|' list too
      * 'flip:z|rz:90;ry:12'          a custom set, freely mixed
    Duplicate rotations (after 'flips' expansion) are dropped keeping the FIRST
    label. Empty/blank -> []. Raises ValueError on an unknown DOF / flip axis
    (fails loud, mirroring parse_canon_ranges)."""
    out, seen = [], set()
    for part in (spec or "").split("|"):
        part = part.strip()
        if not part:
            continue
        expanded = (FLIPS_PANEL_SPEC.split("|") if part.lower() == "flips"
                    else [part])
        for p in expanded:
            label, cand = _canon_candidate(p)
            key = tuple(round(v, 9) for v in cand)
            if key in seen:
                continue
            seen.add(key)
            out.append((label, cand))
    return out
