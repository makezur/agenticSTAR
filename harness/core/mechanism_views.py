"""mechanism_views.py — the `mechanism` view's sampling, camera picking, and
filename grammar. Shared by the renderer (views/mechanism.py) and the sheet
builder (analysis/viz/mechanism_sheet.py); bpy-free so both envs can import it.

Why the view exists and how to read it: harness/views/mechanism.md. Short
version: flipping a revolute `axis` negates the state, so both endpoints of a
symmetric hinge render identically under either sign — only a swept arc shows
the direction.

    mechanism_<joint>_<arm>_r<R>_s<NN>_<state>.png

Every joint is swept TWICE — the declared `axis` and its negation — onto blind
`A`/`B` labels (views/mechanism.md). Arm leads row leads index leads state, so a
lexicographic sort is arm, then row, then sweep order (a bare signed state cannot
provide the latter: '-90' sorts before '-9').
"""

import math

# The two arms every joint is swept as. WHICH label is declared is not a
# convention: core.mechanism_calls derives it from the declaration hash.
ARMS = ("A", "B")


# Samples per joint per row. Tile size is page width / samples, and this page
# lives on tile legibility; 7 still leaves five interior states.
DEFAULT_SAMPLES = 7
MIN_SAMPLES = 2                       # an arc needs two ends to be an arc

# A joint with no `limit` is genuinely continuous (a wheel): a full turn IS its
# range.
CONTINUOUS_SPAN = (0.0, 360.0)


# --------------------------------------------------------------------------- #
# the camera picker
# --------------------------------------------------------------------------- #
# The sign of a revolute axis is invisible when the view direction lies in the
# swing plane (the circle projects to a line), so a viewpoint's legibility is
# asin(|view_dir . axis|): 0 = edge-on, 90 = straight down the axis. The picker
# CHOOSES among natural three-quarter views (the turntable's ring) rather than
# constructing one from the axis, which could land under/behind the object.
PALETTE_AZIMUTHS = (35.0, 80.0, 125.0, 170.0, 215.0, 260.0, 305.0, 350.0)
PALETTE_ELEVATIONS = (20.0, 45.0)

# The classic default; ties break toward it so the camera only moves when it is
# actually near-degenerate.
DEFAULT_VIEW = (35.0, 20.0)

# Rows (viewpoints) per joint. Two >=90-deg-apart azimuths cannot share a swing
# plane, and the second row resolves toward-vs-away ambiguity. Tweakable via
# --mechanism-rows.
DEFAULT_ROWS = 2
ROW_MIN_AZIMUTH_APART = 90.0

# Below this angle off the swing plane a row is flagged in its header (the arc
# is foreshortened; read the other row).
NEAR_EDGE_ON_DEG = 15.0

# At or past this angle a revolute view is simply good, so views tie and the
# tie-break keeps the default instead of chasing the largest angle.
GOOD_ENOUGH_DEG = 30.0


def view_dir(azimuth_deg, elevation_deg):
    """Unit vector camera -> target for an orbit viewpoint, in the canonical
    frame (rig.camera.orbit_offset's grammar: -Y up, az=0/el=0 looks down +Z).
    Restated rather than imported because rig/ is bpy-side."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return (-math.cos(el) * math.sin(az),
            math.sin(el),
            math.cos(el) * math.cos(az))


def swing_plane_angle_deg(azimuth_deg, elevation_deg, axis):
    """How far the viewpoint sits OFF the joint's swing plane, in degrees
    (0 = edge-on/degenerate, 90 = down the axis). One scalar serves prismatic
    joints too — their motion is a line along `axis`, so they prefer SMALL
    values (see _legibility)."""
    ax, ay, az_ = (float(axis[0]), float(axis[1]), float(axis[2]))
    norm = math.sqrt(ax * ax + ay * ay + az_ * az_)
    if norm < 1e-9:
        return 0.0                      # a zero axis has no swing plane at all
    vx, vy, vz = view_dir(azimuth_deg, elevation_deg)
    dot = abs(vx * ax + vy * ay + vz * az_) / norm
    return math.degrees(math.asin(min(1.0, dot)))


def _legibility(view, axis, jtype):
    """Higher is better, saturating at GOOD_ENOUGH_DEG (prismatic inverted)."""
    angle = swing_plane_angle_deg(view[0], view[1], axis)
    if jtype == "prismatic":
        angle = 90.0 - angle
    return min(angle, GOOD_ENOUGH_DEG)


def pick_views(axis, jtype="revolute", rows=DEFAULT_ROWS):
    """`rows` (azimuth, elevation) viewpoints for one joint, best first.

    Deterministic: best-scoring palette view first (ties toward DEFAULT_VIEW,
    then lower elevation, then azimuth), each further row the best view at
    least ROW_MIN_AZIMUTH_APART from all chosen — falling back to ignoring the
    spacing only if the palette is exhausted."""
    palette = [(az, el) for el in PALETTE_ELEVATIONS for az in PALETTE_AZIMUTHS]

    def rank(view):
        return (-round(_legibility(view, axis, jtype), 6),
                view != DEFAULT_VIEW, view[1], view[0])

    chosen = []
    for _ in range(max(1, int(rows))):
        def far_enough(view):
            return all(_az_apart(view[0], c[0]) >= ROW_MIN_AZIMUTH_APART
                       for c in chosen)
        remaining = [v for v in palette if v not in chosen]
        candidates = [v for v in remaining if far_enough(v)] or remaining
        if not candidates:
            break
        chosen.append(min(candidates, key=rank))
    return chosen


def _az_apart(a, b):
    """Circular azimuth separation in degrees, in [0, 180]."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def sample_states(limit, samples=DEFAULT_SAMPLES):
    """`samples` states spanning `limit` (a `(lo, hi)` tuple), endpoints
    included so the strip is anchored at rest and full travel. `limit=None`
    sweeps CONTINUOUS_SPAN."""
    samples = int(samples)
    if samples < MIN_SAMPLES:
        raise ValueError(
            f"mechanism sweep needs at least {MIN_SAMPLES} samples (an arc has "
            f"two ends), got {samples}")
    lo, hi = CONTINUOUS_SPAN if limit is None else (float(limit[0]),
                                                   float(limit[1]))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"mechanism sweep needs a finite limit, got {limit!r}")
    step = (hi - lo) / (samples - 1)
    # last sample set to `hi` EXACTLY so the arc provably reaches end of travel
    return [lo + step * i for i in range(samples - 1)] + [hi]


# --------------------------------------------------------------------------- #
# the filename grammar
# --------------------------------------------------------------------------- #
# mechanism_<joint>_<arm>_r<R>_s<NN>_<state>.png — the state is in the name so a
# tile can be quoted by degree; the sign is spelled `p`/`n` (not `-`) to stay
# shell- and glob-safe. The arm is the blind `A`/`B` label, NOT the axis sign:
# nothing a reader can see in a path tells them which arm is the declaration.
_PREFIX = "mechanism"


def state_field(state):
    """A state as its filename field: `p045` / `n120` / `p000` (whole
    degrees)."""
    value = float(state)
    sign = "n" if value < 0 else "p"
    return f"{sign}{abs(value):03.0f}"


def image_name(joint, arm, row, index, state, ext=".png"):
    """`mechanism_<joint>_A_r0_s03_p090.png` — the ONE place this is spelled."""
    return (f"{_PREFIX}_{joint}_{str(arm)}_r{int(row)}_s{int(index):02d}_"
            f"{state_field(state)}{ext}")


def image_names(joint, arm, row, states, ext=".png"):
    """One name per sample of one row, refused on (index) collisions rather
    than silently overwriting a render."""
    names = [image_name(joint, arm, row, i, s, ext=ext)
             for i, s in enumerate(states)]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(
            f"mechanism sweep produces duplicate filenames for joint "
            f"{joint!r}: {', '.join(dupes)}")
    return names


# What a consumer should glob for.
IMAGE_GLOB = f"{_PREFIX}_*_r*_s*_*.png"


def parse_image_name(name):
    """`(joint, arm, row, index, state)` from a mechanism filename, or None (so
    a caller can filter a directory listing). Parsed off the END so a joint name
    containing underscores survives.

    A pre-arm name (`mechanism_<joint>_r0_s00_p000.png`, written before the
    forced choice existed) parses with `arm=None` rather than as a joint whose
    name ends in `_A`: an old pass's tiles are still readable, and the sheet
    builder is what refuses to build a one-armed choice out of them."""
    stem, dot, _ext = str(name).rpartition(".")
    stem = stem if dot else str(name)
    head, sep, state_f = stem.rpartition("_")
    if not sep or len(state_f) < 2 or state_f[0] not in "pn":
        return None
    head, sep, index_f = head.rpartition("_s")
    if not sep:
        return None
    head, sep, row_f = head.rpartition("_r")
    if not sep or not head.startswith(_PREFIX + "_"):
        return None
    joint = head[len(_PREFIX) + 1:]
    if not joint:
        return None
    arm = None
    for candidate in ARMS:
        if joint.endswith("_" + candidate):
            joint, arm = joint[:-(len(candidate) + 1)], candidate
            break
    if not joint:
        return None
    try:
        row = int(row_f)
        index = int(index_f)
        state = float(state_f[1:]) * (-1.0 if state_f[0] == "n" else 1.0)
    except ValueError:
        return None
    return joint, arm, row, index, state


def negate_axis(axis):
    """The mirrored arm's axis: every component negated, `+ 0.0` folding -0.0
    onto 0.0 so a zero component does not read as a sign change in a header."""
    return tuple(-float(v) + 0.0 for v in axis)


def describe(joint, states, views=None, arms=ARMS):
    """A one-line log of the resolved sweep — the record of what was rendered.

    Names the ARM LABELS but never which axis each one got: this line lands in
    the render log the reader may well have open beside the page."""
    text = (f"{joint}: {len(states)} state(s) "
            f"{states[0]:+.0f} -> {states[-1]:+.0f}")
    if arms:
        text += f", arms {'/'.join(str(a) for a in arms)}"
    if views:
        text += ", rows " + ", ".join(f"az{az:.0f}/el{el:+.0f}"
                                      for az, el in views)
    return text
