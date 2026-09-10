"""derivatives — velocity, acceleration, and their radial component.

analysis.pose_diff owns the two-pose residual and the per-frame LIFT it is built
from (`lift_placement`: `R_world = R_cam R_obj` and `offset_canon = R_cam tau`, the
camera->object vector in world axes and object units). This module is the layer a
pairwise residual cannot express: it differences a per-frame TRACK of those states
once for velocity and twice for acceleration.

The ladder, over frames i along a track (`u_i` = that frame's offset_canon):

    u_i                             position   object units, world axes
    v_i = u_i - u_{i-1}             velocity   at each STEP, indexed by its later
                                               frame — positive components mean the
                                               object moved that way
    a_i = v_{i+1} - v_i             accel      at each INTERIOR frame
        = u_{i+1} - 2 u_i + u_{i-1}

and the rotational counterpart, in the same body-frame convention pose_diff uses.
The second one is a RATIO, not a difference — rotations compose, they do not
subtract (`_angular_ratio` has the argument):

    Omega_i = W_{i-1}^T W_i               angular velocity increment
    omega_i = log(Omega_i)                reported as degrees (axis * angle)
    alpha_i = log(Omega_i^T Omega_{i+1})  angular acceleration

ALIGNMENT. `u_i` points from camera i's centre to the object, so its unit vector
`u_hat_i = u_i / |u_i|` is the RADIAL direction there, and each derivative is
projected onto it:

    radial   dot(u_hat_i, v_i)   signed: > 0 = moving AWAY from the camera

The across-view (tangential) part is NOT a separate field: it is
`sqrt(speed^2 - radial^2)`, so the row's own `speed_canon` / `accel_magnitude_canon`
already carries it. Read the two together — radial ~0 with a large speed is motion
across the view, radial ~0 with speed ~0 is no motion at all — or read the full
vector, which every row also carries.

READ alpha BESIDE the step's rotation_deg, never alone. `alpha` is a geodesic angle:
it PEAKS at a half-turn and folds past it, so a ~340 deg reversal reports 20 deg — the
same as a mild +10/+30 ramp. The per-step `rotation_deg` on the step record is what
separates them (170 each vs 10 and 30), which is why a reader needs both.
maths.md §3 has the full table. Translation's acceleration is a plain vector
difference in one basis and has no such fold.

None, never a fabricated zero, for anything not measured: a frame with no camera pose,
or a radial split about a `u` with no direction (object exactly at the camera centre).
Such rows are omitted from the attention order rather than ranked as 0 — "not
measured" is not "did not move".

NO DIVISION BY frame_gap; no thresholds anywhere in this package. It rides along so
unequal spacing is visible. maths.md §7 is why dividing by it would be unsound
rather than merely unhelpful, so "velocity" here means per-step delta, not
per-unit-time.

Pure numpy + stdlib; no file IO.
"""

import math

import numpy as np

from rig import lie


def _vec(value):
    """A track/step 3-vector as float ndarray, or None when absent."""
    return None if value is None else np.asarray(value, dtype=float)


def _radial(direction, vector):
    """The SIGNED component of `vector` along `direction`: `dot(d/|d|, v)`.

    Positive means along the direction, i.e. AWAY from the camera centre. The
    direction is unit-ized so the result is in the vector's own units and does not
    scale with how far away the object happens to sit.

    None when either input is absent, or when the direction has no length (an object
    exactly at the camera centre has no radial axis) — never a fabricated zero. The
    across-view component is not returned: it is `sqrt(speed^2 - radial^2)`, so the
    row's own magnitude already carries it."""
    if vector is None or direction is None:
        return None
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    return float(np.dot(direction / norm, vector))


def _angular_ratio(Om_in, Om_out):
    """The RATIO of two consecutive rotation increments, as a rotation vector in
    degrees: `log(Omega_in^T Omega_out)` — the rotation carrying the incoming
    increment onto the outgoing one. None when either is absent.

    Takes the increments as MATRICES, straight off the step records
    (`pose.rotation_increment`), because that is what they already are. Rebuilding
    them from the reported axis + angle would be `matrix -> axis+DEGREES -> radians
    -> matrix`: lossy in the last digits, and pointless when the matrix is in hand.

    NOT `omega_out - omega_in`. The two increments turn frames one step apart, so
    their log vectors are in DIFFERENT tangent spaces and subtracting them subtracts
    coordinate triples rather than taking the log of any rotation. Both forms agree
    on uniform motion (identity here) and on single-axis changes; they diverge once
    the axes differ (80 deg about Z then 80 about X: 108 deg ratio, 113 "difference").

    Being a rotation costs boundedness: the magnitude is a geodesic angle, so it
    peaks at 180 and folds past it, and consecutive ratios do not add up to
    anything. maths.md §3 is how to read that."""
    if Om_in is None or Om_out is None:
        return None
    return np.degrees(lie.matrix_to_rotvec(
        np.asarray(Om_in, dtype=float).T @ np.asarray(Om_out, dtype=float)))


def _orientation_matrix(row):
    """A track row's absolute world orientation as a 3x3, or None.

    The row stores `orientation_world_deg` (axis-angle in degrees, `log(W_i)`), so
    this converts back to radians before exponentiating — degrees are a reporting
    unit only and are never a Lie-algebra element."""
    deg = _vec(row.get("orientation_world_deg"))
    if deg is None:
        return None
    return lie.rotvec_to_matrix(np.radians(deg))


def velocities(track, steps):
    """Per-STEP velocity, indexed by the step's LATER frame.

    `track` — pose_diff.frame_track rows, for each frame's `offset_canon` (the
        radial direction is taken about it) and its absolute joint states.
    `steps` — the transition records for the same sequence
        (analysis.temporal.sequence).

    One row per step: {frame, from, frame_gap, velocity_canon (3),
    radial_velocity, joints}.

    THE FIRST DIFFERENCE IS NOT COMPUTED HERE, and the ROTATION HALF IS NOT REPORTED
    AT ALL. `v_i = u_i - u_{i-1}` and `omega_i = log(W_{i-1}^T W_i)` are *precisely*
    the step residual's `translation_canon` and rotation — a per-step first difference
    of placements IS the pairwise residual. So:

      * `velocity_canon` is READ off the step record, not re-derived;
      * `speed_canon` / `angular_speed_deg` are not emitted at all — the step already
        carries `translation_dist_canon` and `rotation_deg`;
      * no `angular_velocity_deg`. It was `rotation_axis * rotation_deg`, i.e. the
        step's own two fields multiplied together, and a 3-vector at that — a
        body-frame axis, so it is not even comparable between two steps. The step's
        scalar `rotation_deg` is the angular velocity; `rotation_increment` (a matrix)
        is what the acceleration composes.

    Two implementations of one quantity is exactly the kind of duplication that lets a
    viewer agree with itself while both it and the report are wrong.

    What this layer genuinely adds is the RADIAL projection and the joint states:
    `radial_velocity` is taken about `u_i`, the offset at the step's LATER frame, so
    it reads "was it moving away from where it is now".
    """
    by_frame = {r["frame"]: r for r in (track or [])}
    out = []
    for step in steps or []:
        cur = by_frame.get(step["to"]) or {}
        pose = step.get("pose") or {}
        v = _vec(pose.get("translation_canon"))

        jstep = step.get("joints") or {}
        jcur = cur.get("joints") or {}
        joints = {}
        for name in sorted(set(jstep) | set(jcur)):
            a, b = jstep.get(name) or {}, jcur.get(name) or {}
            joints[name] = {
                "delta": a.get("delta"),          # the step's own, not recomputed
                "state": b.get("state"),          # absolute, at the later frame
                # no percent-of-travel anywhere in the report: it is one division
                # from `delta` and `limit`, and one obvious name fitted two
                # different quantities — see pose_diff.joint_deltas.
                "type": b.get("type") or a.get("type"),
            }

        out.append({
            "frame": step["to"],
            "from": step["from"],
            "frame_gap": step.get("frame_gap"),
            "velocity_canon": None if v is None else [float(x) for x in v],
            "radial_velocity": _radial(_vec(cur.get("offset_canon")), v),
            # the increment MATRIX, carried through from the step so the angular
            # acceleration composes it directly instead of rebuilding it from
            # degrees. Numpy, not JSON; stripped before the report ships.
            "rotation_increment": pose.get("rotation_increment"),
            "joints": joints,
        })
    return out


def accelerations(track, vels):
    """Second derivative at each INTERIOR frame: translation as the difference
    `v_{i+1} - v_i`, rotation as the increments' ratio (`_angular_ratio`).

    `track` — the same rows `velocities` consumed (for each frame's `u`, which the
        radial split is taken about).
    `vels` — the output of `velocities` on that track.

    One row per interior frame: {frame, from, to, acceleration_canon (3),
    accel_magnitude_canon, radial_acceleration, angular_acceleration_deg (3),
    angular_accel_magnitude_deg, joints}.

    `radial_acceleration` is about `u_i` at the interior frame itself, so it reads
    "is it accelerating away from the camera". Read `angular_acceleration_deg`'s MAGNITUDE:
    its axis sits in frame i rotated by the incoming increment, so like the
    residual's `rotation_axis` it is not comparable across frames.

    Returns [] for fewer than two steps: an interior frame needs a step on each
    side. Per joint: `net` is the two deltas ADDED (where the joint ended up) and
    `change` the second difference; a joint whose two deltas cancel went out and
    came back.
    """
    rows = list(track or [])
    vels = list(vels or [])
    by_frame = {r["frame"]: r for r in rows}
    out = []
    for i in range(1, len(vels)):
        incoming, outgoing = vels[i - 1], vels[i]
        frame = incoming["frame"]              # the shared (interior) frame
        v_in = _vec(incoming.get("velocity_canon"))
        v_out = _vec(outgoing.get("velocity_canon"))
        a = None if v_in is None or v_out is None else v_out - v_in
        u = _vec((by_frame.get(frame) or {}).get("offset_canon"))
        radial = _radial(u, a)

        alpha = _angular_ratio(incoming.get("rotation_increment"),
                               outgoing.get("rotation_increment"))

        jin = incoming.get("joints") or {}
        jout = outgoing.get("joints") or {}
        joints = {}
        for name in sorted(set(jin) | set(jout)):
            da = (jin.get(name) or {}).get("delta")
            db = (jout.get(name) or {}).get("delta")
            both = da is not None and db is not None
            joints[name] = {
                "incoming_delta": da,
                "outgoing_delta": db,
                "net": (float(da) + float(db)) if both else None,
                "change": (float(db) - float(da)) if both else None,
                "type": ((jout.get(name) or {}).get("type")
                         or (jin.get(name) or {}).get("type")),
            }

        out.append({
            "frame": frame,
            "from": incoming["from"],
            "to": outgoing["frame"],
            "acceleration_canon": None if a is None else [float(x) for x in a],
            "accel_magnitude_canon": (None if a is None
                                      else float(np.linalg.norm(a))),
            "radial_acceleration": radial,
            "angular_acceleration_deg": (None if alpha is None
                                         else [float(x) for x in alpha]),
            "angular_accel_magnitude_deg": (None if alpha is None
                                            else float(np.linalg.norm(alpha))),
            "joints": joints,
        })
    return out


def attention_order(vels, accels):
    """Where to LOOK FIRST — the same rows, ordered by raw magnitude.

    Four independent biggest-first orderings, each a list of {frame, value}: the two
    RADIAL reads and the two acceleration magnitudes. Per-step magnitudes are NOT
    ordered here — they live on the step record and are
    read off the step record instead; this layer orders only what only it has.

    This is an ORDER, not a selection — every row with a measured value appears in
    its list, so nothing can be silently dropped; the reader decides how far down to
    read. Rows whose value is None (no camera pose, so nothing was measured) are
    omitted from that ordering rather than sorted as zero, since "not measured" is
    not "did not move".

    Sorting is by magnitude with the frame name as a deterministic tiebreak.
    """
    def by(rows, key):
        have = [r for r in (rows or []) if r.get(key) is not None]
        return [{"frame": r["frame"], "value": r[key]}
                for r in sorted(have, key=lambda r: (-abs(r[key]), r["frame"]))]

    return {
        "by_radial_velocity": by(vels, "radial_velocity"),
        "by_accel_magnitude_canon": by(accels, "accel_magnitude_canon"),
        "by_radial_acceleration": by(accels, "radial_acceleration"),
        "by_angular_accel_magnitude_deg": by(accels,
                                             "angular_accel_magnitude_deg"),
    }


def report(track, gaps=None):
    """The full threshold-free derivatives block for a per-frame track.

    {"track": the rows as given, "velocities", "accelerations",
    "attention_order"}. No flags, no verdict, nothing filtered."""
    vels = velocities(track, gaps)
    accels = accelerations(track, vels)
    return {
        "track": list(track or []),
        "velocities": vels,
        "accelerations": accels,
        "attention_order": attention_order(vels, accels),
    }


# --------------------------------------------------------------------------- #
# human-readable lines
# --------------------------------------------------------------------------- #
def _num(value, fmt=".4f", suffix=""):
    return "n/a" if value is None else f"{value:{fmt}}{suffix}"


def _speed(row):
    """|velocity_canon| for one velocity row, or None when unmeasured.

    Worked out for PRINTING only — the rows deliberately do not carry a magnitude
    (the step record's `translation_dist_canon` IS it; see `velocities`), but a
    printer sorting "biggest first" needs the number in hand."""
    v = _vec(row.get("velocity_canon"))
    return None if v is None else float(np.linalg.norm(v))


def velocity_lines(vels, limit=0):
    """Human lines for the per-step velocities, biggest speed first.

    The header carries the units and the sign convention because for a CLI reader it
    is the only documentation there is; it stays terse and states no verdict. The
    rotation half is NOT printed here — the step record's own `rotation_deg` is the
    angular speed (see `velocities`), and sequence's step lines already print it."""
    rows = sorted((vels or []), key=lambda r: (-abs(_speed(r) or 0.0), r["frame"]))
    if not rows:
        return ["velocities: none (needs at least 2 frames)"]
    shown = rows[:limit] if limit else rows
    lines = ["per-step velocity, in object units per step (an unequal frame_gap is "
             "reported, never divided out). Radial is signed along the "
             "camera->object direction (+ = moving away); compare it with the "
             "speed to see how much of the motion was across the view:"]
    for r in shown:
        gap = r.get("frame_gap")
        gap_note = f"  [gap {int(gap)}]" if gap and int(gap) != 1 else ""
        lines.append(
            f"  {r['from']} -> {r['frame']}: speed "
            f"{_num(_speed(r))} of object size "
            f"(radial {_num(r.get('radial_velocity'), '+.4f')}){gap_note}")
    if limit and len(rows) > limit:
        lines.append(f"  ... {len(rows) - limit} more step(s) in the JSON "
                     "(nothing was filtered — this is only the print cap)")
    return lines


def acceleration_lines(accels, limit=0):
    """Human lines for the interior-frame accelerations, biggest first.

    The header warns about the angular fold, because a CLI reader has nowhere else to
    learn that a small angular number can be a large reversal."""
    rows = sorted((accels or []),
                  key=lambda r: (-abs(r.get("accel_magnitude_canon") or 0.0),
                                 r["frame"]))
    if not rows:
        return ["accelerations: none (needs at least 3 frames)"]
    shown = rows[:limit] if limit else rows
    lines = ["step-to-step acceleration at each interior frame (translation as a "
             "second difference, rotation as the two increments' RATIO — a geodesic "
             "angle, so it wraps past 180; descriptive only):"]
    for r in shown:
        lines.append(
            f"  {r['frame']}: |accel| {_num(r.get('accel_magnitude_canon'))} "
            f"of object size (radial "
            f"{_num(r.get('radial_acceleration'), '+.4f')}), angular "
            f"{_num(r.get('angular_accel_magnitude_deg'), '.2f', ' deg')}")
        for name, j in (r.get("joints") or {}).items():
            if j.get("change") is None:
                continue
            unit = " deg" if j.get("type") == "revolute" else ""
            fmt = "+.2f" if j.get("type") == "revolute" else "+.4f"
            lines.append(
                f"      joint {name}: {float(j['incoming_delta']):{fmt}} in then "
                f"{float(j['outgoing_delta']):{fmt}} out{unit}, "
                f"net {float(j['net']):{fmt}}{unit}")
    if limit and len(rows) > limit:
        lines.append(f"  ... {len(rows) - limit} more interior frame(s) in the "
                     "JSON (nothing was filtered — this is only the print cap)")
    return lines


def attention_lines(order, limit=5):
    """Human lines for the attention order: the top `limit` of each ordering, with
    an explicit note that the rest is present, not discarded."""
    lines = ["look first — per-FRAME derivatives "
             "(biggest first; an ORDER, not a flag):"]
    specs = [
        ("by_radial_velocity", "radial velocity", "of object size", ".4f"),
        ("by_accel_magnitude_canon", "acceleration", "of object size", ".4f"),
        ("by_radial_acceleration", "radial acceleration", "of object size", ".4f"),
        ("by_angular_accel_magnitude_deg", "angular acceleration", "deg", ".2f"),
    ]
    for key, label, unit, fmt in specs:
        rows = (order or {}).get(key) or []
        if not rows:
            continue
        head = ", ".join(f"at {r['frame']} {r['value']:{fmt}} {unit}"
                         for r in rows[:limit])
        more = f"  (+{len(rows) - limit} more)" if len(rows) > limit else ""
        lines.append(f"  {label}: {head}{more}")
    return lines
