"""turntable_views.py — WHICH directions the turntable looks from, and what the
resulting files are called.

Two modules need to agree on this and they live in different interpreters:
`views/turntable.py` places the camera (inside Blender, `bpy`) and
`analysis/viz/turntable_sheet.py` lays the renders out on a page (the analysis env,
`cv2`). So the view set AND the filename grammar live here — pure stdlib, no `bpy`
and no `cv2` — the same seam `core.sweep_family` and `core.visual_budget` use.
A sheet that parsed a format the renderer merely happened to write would be one
edit away from silently finding no tiles.

THE SET IS DETERMINISTIC, and that is the point. The turntable is a GATE read
every iteration, so its value comes from comparing this pass to the last one: "the
back caved in between iter03 and iter04" is only sayable if `az180_el+20` means
the same viewpoint both times. Randomly resampling the sphere each pass would
widen coverage across a run and destroy exactly that. `--turntable-jitter SEED`
exists for the deliberate "break the symmetry, look from somewhere new" pass; the
angles it lands on are still written into every filename, so a jittered pass is
self-describing rather than a set of images nobody can place.

WHY NOT FULL SO(3). A camera orbit has three DOF, but one of them — roll about the
view axis — adds no shape information at all (it rotates the picture, not the
object), so the space that matters is S²: a direction to look from, with a stable
up-vector. `rig.camera.aim_camera` supplies the up-vector (CAM0_UP), which is what
keeps every view right-side up and comparable.
"""

import random


# Elevations, in degrees, measured the way `rig.camera.orbit_offset` measures them:
# 0 is level with the object, positive lifts the camera toward CAM0_UP (looking
# DOWN at the object), negative drops it below (looking UP at the underside).
#
# ABSOLUTE, not relative to the ring's elevation. `--elevation` names the RING (it
# always did), and deriving the high/low pair from it would either push them past
# the pole for a raised ring or collapse them onto it for a lowered one — a knob
# whose safe range depends on another knob's value. These two numbers are the whole
# reason `sphere8` exists: the single ~20 degree ring the turntable had for its
# whole life cannot see a caved-in top or an unbuilt base, which is the "geometry
# wrong from a hidden side" failure the view claims to catch.
HIGH_EL = 60.0
LOW_EL = -30.0

# The high/low pairs sit 45 degrees off the ring's azimuths, so a raised view is
# never just a re-look down a bearing the ring already covered — it reads the
# corner between two ring views, where a seam between parts tends to hide.
_RING = (0.0, 90.0, 180.0, 270.0)
_HIGH = (45.0, 225.0)
_LOW = (135.0, 315.0)


def _ring4(ring_el):
    return [(az, ring_el) for az in _RING]


def _sphere8(ring_el):
    # RING FIRST: the four level views are the ones that read like a product shot,
    # so they are what the eye should land on; the raised and dropped views are the
    # follow-up that answers "and from above / underneath". The sheet re-groups
    # these by elevation band for the page, but the render ORDER is this one.
    return (_ring4(ring_el)
            + [(az, HIGH_EL) for az in _HIGH]
            + [(az, LOW_EL) for az in _LOW])


# name -> builder(ring_el). `ring4` is the set the turntable rendered for its whole
# life, kept as a named CHEAP option rather than deleted: a run with 11 articulation
# states pays 8 renders per state under the default, and "render half as many
# angles" is a reasonable thing to ask for explicitly.
VIEW_SETS = {
    "ring4": _ring4,
    "sphere8": _sphere8,
}

DEFAULT_VIEW_SET = "sphere8"

# How far a jittered view may wander from its nominal direction. The azimuth budget
# is half the ring's 90 degree spacing, so a jittered set still covers the object
# evenly instead of clumping two views together; the elevation budget is smaller
# because elevation is the axis with the hard ends (the poles).
JITTER_AZ_DEG = 22.5
JITTER_EL_DEG = 10.0

# Elevation is clamped inside the poles: at +-90 the view direction is parallel to
# CAM0_UP, `aim_camera`'s up-vector degenerates, and the image rolls arbitrarily —
# a picture with no stable orientation is not a comparable one.
MAX_EL = 80.0


def view_set(name=DEFAULT_VIEW_SET, ring_el=20.0, az_offset=0.0, jitter=None):
    """The (azimuth, elevation) directions to render, in render order.

    `ring_el` is the level ring's elevation (`--elevation`); `az_offset` rotates the
    WHOLE set about the up axis (`--azimuth`), which is what that flag has always
    claimed to do and never did. `jitter` is a seed (any hashable) or None: given
    one, each direction is perturbed by a deterministic amount drawn from that seed,
    so the same seed always yields the same set.
    """
    build = VIEW_SETS.get(str(name))
    if build is None:
        raise ValueError(
            f"unknown turntable view set {name!r} "
            f"(known: {', '.join(sorted(VIEW_SETS))})")
    views = [(az + float(az_offset), el) for az, el in build(float(ring_el))]
    if jitter is not None:
        rng = random.Random(jitter)
        views = [(az + rng.uniform(-JITTER_AZ_DEG, JITTER_AZ_DEG),
                  el + rng.uniform(-JITTER_EL_DEG, JITTER_EL_DEG))
                 for az, el in views]
    return [(az % 360.0, max(-MAX_EL, min(MAX_EL, el))) for az, el in views]


# --------------------------------------------------------------------------- #
# the filename grammar
# --------------------------------------------------------------------------- #
# turntable_<state>_az<AAA>_el<+EE>.png
#
# THE ANGLE IS IN THE NAME.
#
# The elevation carries an EXPLICIT SIGN (`el+20`, `el-30`) because the sign is the
# whole difference between looking down at the lid and up at the base, and a bare
# `el30` beside `el-30` is one character away from being read as the same view.
_PREFIX = "turntable"


def image_name(state, azimuth, elevation, ext=".png"):
    """`turntable_<state>_az135_el-30.png` — the ONE place this is spelled."""
    return (f"{_PREFIX}_{state}_az{azimuth % 360.0:03.0f}"
            f"_el{elevation:+03.0f}{ext}")


def image_names(state, views, ext=".png"):
    """One name per view, checked for collisions.

    A jittered set rounds its angles to whole degrees for the name, so two views
    COULD in principle round together — vanishingly unlikely at these budgets.
    Cheap to check, so it is checked."""
    names = [image_name(state, az, el, ext=ext) for az, el in views]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(
            f"turntable view set produces duplicate filenames for state "
            f"{state!r}: {', '.join(dupes)} — two views round to the same "
            "whole-degree direction")
    return names


# What a consumer should glob for. No index ceiling, unlike the `_0*` it replaces.
IMAGE_GLOB = f"{_PREFIX}_*_az*_el*.png"


def parse_image_name(name):
    """`(state, azimuth, elevation)` from a turntable filename, or None.

    Returns None rather than raising for anything that isn't one, so a caller can
    filter a directory listing with it. Parsed off the END (the az/el fields are
    fixed-shape) so a state label containing an underscore survives."""
    stem, dot, _ext = str(name).rpartition(".")
    stem = stem if dot else str(name)
    head, sep, el_field = stem.rpartition("_el")
    if not sep:
        return None
    head, sep, az_field = head.rpartition("_az")
    if not sep or not head.startswith(_PREFIX + "_"):
        return None
    state = head[len(_PREFIX) + 1:]
    if not state:
        return None
    try:
        azimuth, elevation = float(az_field), float(el_field)
    except ValueError:
        return None
    return state, azimuth, elevation


def describe(views):
    """A one-line log of the resolved set — the record of what a pass looked at."""
    return ", ".join(f"az{az:03.0f}/el{el:+03.0f}" for az, el in views)
