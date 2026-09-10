"""seams — which frames which refiner posed, and where two of them meet.

A run refined in WINDOWS split its frames among several refiner agents, each working
on its own chunk. A SEAM is a step whose two frames were posed by DIFFERENT agents:
neither saw the transition, so it is the one place two individually-plausible windows
can disagree with each other.

PURE OWNERSHIP BOOKKEEPING. Nothing here consults a magnitude, because a seam is
interesting for who posed it, not for how big it is — a 2 deg seam and a 178 deg seam
are equally unreviewed. That is the whole distinction between this module and the
measurement side (analysis.temporal.report), and it is why the two are separate: a
seam is a fact about the PROCESS, and a residual is a fact about the POSES.

Shared by multiagent.windows (merge + selfcheck), analysis.viz.seam_sheet (which
seams to draw) and analysis.temporal.report (which rows to mark), so all four agree
on what a seam is.

Pure stdlib; no numpy, no file IO.
"""


def step_key(step):
    """A step's stable identity: `(from, to)`.

    One place, so a step used as a dict key in one module matches the same step keyed
    in another."""
    return (step["from"], step["to"])


def crossing_steps(steps, owner):
    """The steps whose two frames have DIFFERENT owners — the seams.

    `owner` — {frame_name: owner_id}; a frame ABSENT from the map owns itself
    (`None`). So a step between a window frame and an unowned committed neighbour
    crosses, while two unowned frames beside each other do not.

    Consults no magnitude at all: this says "nobody owned both sides here", never
    "this looks wrong". Used with a plan-wide {frame: window_id} by
    multiagent.windows merge, and with {frame: True} for one window's own frames by
    multiagent.windows selfcheck (so any step touching a non-owned neighbour crosses).
    """
    return [s for s in (steps or [])
            if owner.get(s["from"]) != owner.get(s["to"])]


def step_lines(step):
    """One step's measured numbers as ONE compact line — the caption a seam band
    prints under its title (analysis.viz.seam_sheet's `note`).

    TERSE ON PURPOSE. A seam sheet is decided by LOOKING; the caption only has to say
    what the numbers were so the eye has something to check them against. Everything
    omitted is still in the report at full precision. Specifically NOT printed:
      * the frame names (the band title and each column header say them);
      * a unit legend (degrees for rotation and revolute joints, fractions of object
        size for translation — stated by the words "deg" / "of size");
      * pose_frame and frame_gap in the ordinary case, appended only when they are
        unusual, since a camera-frame step or an unequal gap changes how to read it.

    Magnitudes only: a single step has no derivative (that needs the per-frame
    track — analysis.temporal.derivatives). Descriptive; no verdict."""
    pose = (step or {}).get("pose") or {}
    parts = []
    if (step or {}).get("pose") is None:
        parts.append("no base pose (no camera pose) — joints only")
    else:
        parts.append(f"rot {pose.get('rotation_deg') or 0.0:.1f} deg")
        parts.append(f"trans {pose.get('translation_dist_canon') or 0.0:.3f} of size")
    for name, j in (step.get("joints") or {}).items():
        delta = j.get("delta")
        if delta is None:
            continue
        unit = " deg" if j.get("type") == "revolute" else ""
        parts.append(f"{name} {float(delta):+.1f}{unit}{_travel_note(j)}")
    frame = step.get("pose_frame", "oriented")
    if frame != "oriented":
        parts.append(f"{frame} frame")
    gap = step.get("frame_gap")
    if gap and int(gap) != 1:
        parts.append(f"gap {int(gap)}")
    return ["   ".join(parts)]


def _travel_note(joint):
    """" (-42% of range)" for a printed joint line, or "" without a limit.

    Computed for the PRINTED line only. The report deliberately carries `delta` and
    `limit` rather than their ratio (pose_diff.joint_deltas: one obvious name fitted
    two different quantities), but a person reading a caption benefits from the
    fraction being worked out."""
    lim = joint.get("limit")
    delta = joint.get("delta")
    if not lim or delta is None:
        return ""
    span = float(lim[1]) - float(lim[0])
    if abs(span) <= 1e-12:
        return ""
    return f" ({100.0 * float(delta) / span:+.0f}% of range)"


def rotation_order(steps):
    """The steps biggest-rotation-first — an ORDER, not a selection.

    Every step appears, so nothing can be silently dropped; the reader decides how far
    down to read. Frame name as a deterministic tiebreak. Used by a refiner's brief
    (multiagent.windows selfcheck) to say "if you check three steps, check these first" —
    which never implies the rest are fine.
    """
    rows = [(((s.get("pose") or {}).get("rotation_deg") or 0.0), s)
            for s in (steps or [])]
    return [{"from": s["from"], "to": s["to"], "value": v}
            for v, s in sorted(rows, key=lambda r: (-abs(r[0]), r[1]["to"]))]
