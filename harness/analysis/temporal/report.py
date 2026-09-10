#!/usr/bin/env python3
"""report — the CONSUMER-facing temporal report: one row per frame, scalars only.

analysis.temporal.sequence measures everything and reports it in the shape the maths
comes out in: step records, a placement track, velocity and acceleration rows, each
carrying vectors. That is the right shape for a producer and the wrong shape for a
reader. This module is the reader's view — a flat per-frame table where every value
is ONE NUMBER with a name that says what it is and a unit that says how to compare
it.

Why scalars only. A vector in this report is either world-axis components or a
body-frame axis. The first is three numbers a reader must reduce to a magnitude
before they mean anything; the second is worse, because a body-frame axis at frame i
and one at frame j live in DIFFERENT frames and cannot even be compared with each
other. Reducing them is not optional, so it happens once, here, rather than in every
consumer. The vectors remain in the sequence report for anything that genuinely
needs a direction.

THIS IS THE ENTRY POINT for every temporal read in the tree. `sequence` is the
producer beneath it and is not called directly by consumers: the report carries
both views of one computation — the per-frame `frames` table below, and the
per-step `steps` records it was built from, for a consumer whose subject is a
TRANSITION rather than a frame (which steps cross a window seam; what a temporal
verdict is stamped against). Reaching past this module for the steps meant
computing the same sequence twice, from two call sites that could drift.

One row per frame, in temporal order:

    {frame,
     velocity:     {rotation_deg, translation_canon, radial_canon},
     acceleration: {rotation_deg, translation_canon, radial_canon},
     joints:       {name: {velocity, acceleration, state}}}

Each joint's static facts — `type` and declared `limit` — are stated ONCE, in a
top-level `joints_meta`, not repeated on every row: a value that cannot change
frame-to-frame is not a per-frame value, and re-stating it N times is N-1 chances
for a reader to think it could.

VELOCITY is a per-STEP quantity indexed at the step's LATER frame; ACCELERATION is
per-INTERIOR-frame. So the first row has no velocity and the first and last have no
acceleration — reported as None, never as 0.0, because "no step ends here" is not
"nothing moved". Rows are in temporal order, so a consumer already knows which
frames a row sits between; `from`/`to` are not repeated on every row, and there is
no separate `order` list re-stating the frame column.

Units, once: `*_deg` are degrees; `*_canon` are object units (fractions of the
object's longest canonical dimension) per step, or per step squared for
acceleration; joint values are native (degrees revolute, canonical prismatic). No
value here is divided by `frame_gap` — see maths.md §7 — so a "velocity" is a
per-step delta, and the gap rides on the row.

THE TABLE IS THE REPORT; the JSON is the escape hatch. `table_lines` is what a reader
gets, and `--out` writes the same numbers nested at full precision for anything
programmatic. That is not a style preference — it is measured on a real 30-frame run:

    aligned table    ~1030 tokens
    JSON, compact    ~3250 tokens
    JSON, indented   ~4370 tokens   (3180 chars of it re-stating the same field
                                     names on every row)

4.6x is the cheap part. The real argument is that this data IS a matrix — frames x a
fixed set of columns, one type per column — so JSON's nesting adds structure the data
does not have, and a reader wanting "which frame is the outlier" has to hold 30
objects in mind to do what scanning one table column does at a glance. The JSON keeps
exact floats and needs no parsing, which is why it still exists.

READ-ONLY and derives NOTHING: every number here is already a scalar on the sequence
report — this module selects and renames, it does not compute. The magnitudes were
taken where the vectors are (analysis.pose_diff, analysis.temporal.derivatives), so a
consumer and the producer cannot disagree about one. The TABLE rounds for legibility
(2dp degrees, 4dp object units); the JSON keeps 6 decimals — past any physical
meaning here, but still cheap — so a consumer needing more precision than the table
prints has it.

VIEWS. The full temporal table is the default; `table_lines` also prints focused
views, all DISPLAY filters (the derivatives are always computed over the whole
sequence first, so a boundary row keeps its honest value): `seams_only` (the seam
steps + context), `frames` (a range spec over frame numbers), `window` (the rows one
refiner window owns), `columns` (only the named column groups), and `sort_by`
(biggest-|value| first instead of temporal). Elided rows are always counted, never
hidden. ONE BAN: `sort_by` with `seams_only` is refused — a seam is interesting for
WHO posed it, not how big it is, and a magnitude order over seam rows reads the
small ones as cleared when nothing cleared them. A sorted FULL table keeps the `>>`
seam marks, which is the composition that stays honest.

Runs from harness/:
  micromamba run -n artscript python -m analysis.temporal.report \
      --pose-json RUN/mesh/pose.json --out /tmp/temporal_report.json
"""

import argparse
import json
import os
import re

from analysis import frames as frames_lib
from analysis.lib import pose_read
from analysis.lib.io import write_json
from core import joints as joints_core
from analysis.temporal import sequence


# the CONSUMER report's own schema, independent of sequence's.
SCHEMA_VERSION = 6

UNITS = {
    "rotation_deg": "degrees (geodesic angle; acceleration folds past 180)",
    "translation_canon": "object units per step (of the object's longest dimension)",
    "radial_canon": "object units per step, signed: + = away from the camera",
    "joint": "native units per step (degrees revolute, canonical prismatic)",
    "frame_gap": "frames between this row and the previous one; NEVER divided by",
}


def window_owner(run_dir):
    """{frame_name: window_id} from the newest pose iteration's plan.json, or {}.

    A run refined in WINDOWS split its frames among several refiner agents, each
    working on its own chunk. This says which agent owned which frame, which is what
    makes a seam identifiable at all.

    Read straight off disk rather than through multiagent.windows, because analysis/
    must not import utils/ (the analysis env has no reason to carry the window
    tooling) and this needs only the plan's `windows[].frames` lists. Mirrors
    iteration counts: the highest numeric global iteration containing
    ``windows/plan.json``.
    """
    root = os.path.join(run_dir, "iterations")
    if not os.path.isdir(root):
        return {}
    candidates = []
    for name in sorted(os.listdir(root)):
        idir = os.path.join(root, name)
        if len(name) != 6 or not name.isdigit():
            continue
        metadata = {}
        try:
            with open(os.path.join(idir, "iteration.json")) as handle:
                metadata = json.load(handle)
        except (OSError, ValueError):
            pass
        if metadata.get("status") == "aborted":
            continue
        windows = os.path.join(idir, "windows")
        if os.path.isfile(os.path.join(windows, "plan.json")):
            candidates.append(windows)
    if not candidates:
        return {}
    try:
        with open(os.path.join(candidates[-1], "plan.json")) as f:
            plan = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(plan, dict):
        return {}
    return {frame: w.get("id")
            for w in plan.get("windows") or []
            for frame in w.get("frames") or []}


def _mark_seams(rows, owner):
    """Tag each row with the window that posed it, and whether the step ENDING there
    crossed into a different window.

    A SEAM is a step whose two frames were posed by DIFFERENT agents: neither of them
    saw the transition, so it is the one place two individually-plausible windows can
    disagree with each other. That makes a seam worth attention because of WHO posed
    it, not because of how big it is — a 2 deg seam and a 178 deg seam are equally
    unreviewed. Nothing here consults a magnitude.

    A frame absent from the plan owns itself (`None`), so a step between a window
    frame and an unowned committed neighbour crosses, while two unowned frames beside
    each other do not.
    """
    prev = None
    for row in rows:
        frame = row["frame"]
        row["window"] = owner.get(frame)
        row["is_seam"] = bool(prev is not None
                              and owner.get(prev) != owner.get(frame))
        prev = frame
    return rows


def _check_contiguous(order):
    """Warn LOUDLY when the requested frames are not consecutive in the capture.

    A report over a SUBWINDOW is a legitimate and common thing to want — a refiner
    reads its own window. But a report over frames with a HOLE in them is a trap: a
    "velocity" across a gap is the motion of several unseen steps summed into one, and
    an "acceleration" straddling it is that fiction differenced. Both are reported at
    face value beside honest numbers, and nothing in the row says which is which.

    So this does not fail — the frames may genuinely be all that exist — but it
    returns a warning the caller puts at the TOP of the report, naming the gaps. The
    per-row `frame_gap` shows the same thing, and is easy to skim past.

    Contiguity is judged by the MODE of the gaps, not by assuming 1: a run sampled
    every 10th frame has gap 10 everywhere and is perfectly contiguous.
    """
    nums = [frames_lib.frame_number(f) for f in order]
    if any(n is None for n in nums) or len(nums) < 3:
        return None
    gaps = [b - a for a, b in zip(nums[:-1], nums[1:])]
    stride = max(set(gaps), key=gaps.count)         # the run's own sampling stride
    holes = [(order[i], order[i + 1], gaps[i])
             for i in range(len(gaps)) if gaps[i] != stride]
    if not holes:
        return None
    detail = "; ".join(f"{a} -> {b} skips {g // stride - 1} frame(s)"
                       for a, b, g in holes[:4])
    more = f" (+{len(holes) - 4} more)" if len(holes) > 4 else ""
    return (f"!! NON-CONSECUTIVE FRAMES: {len(holes)} gap(s) in the requested order, "
            f"stride is {stride} elsewhere — {detail}{more}. A velocity across a gap "
            "is several unseen steps summed into ONE number, and an acceleration "
            "straddling it differences that fiction. Those rows are NOT comparable "
            "with the rest; check frame_gap on each.")


def _round6(value):
    """A measured scalar at 6 decimals, None passed through.

    The JSON's precision contract: 6 decimals is far past anything physically
    meaningful here (the table prints 2-4) while cutting the serialized floats
    roughly in half — a 17-significant-digit `15.030014470254867` is not extra
    information, it is fp noise dressed as precision."""
    return None if value is None else round(float(value), 6)


def build(pose_json, order=None, run_dir=None):
    """The per-frame scalar report for one run's pose.json.

    `run_dir` — optional. When given and the run refined in WINDOWS, each row is
        tagged with the `window` that posed its frame and `is_seam` when the step
        ending there crossed into a different window (see `_mark_seams`). Seams are
        the steps no single refiner saw both sides of, so they are where two
        individually-plausible windows can disagree.

    Returns {schema_version, pose_frame, units, joints_meta, frames, steps,
    totals, summary, warnings} where `frames` is the per-frame table (module
    doc), `steps` is the per-STEP transition records the table was built from
    (for a consumer reasoning about a TRANSITION rather than a frame — which
    steps cross a window seam, what a verdict is stamped against), `joints_meta`
    is each joint's static {type, limit} stated once, and `totals`/`summary` are
    sequence's own path summary (`totals` at this report's precision contract). `pose_frame` is "oriented" or "camera" and says what the numbers
    MEAN — with a camera pose the rotation is in world frame and the translation is
    world-oriented about each camera; without one both conflate the camera's own
    motion. `warnings` is a list of strings the table prints at the TOP; today the
    only one is the non-consecutive-frames warning (`_check_contiguous`).
    """
    seq = sequence.diff_sequence(pose_json, order=order)
    _require_joint_states(pose_json, seq["order"])
    steps = seq["steps"]
    deriv = seq["derivatives"]

    # index every layer by the frame it is reported AT, so one pass fills a row.
    vel = {v["frame"]: v for v in deriv["velocities"]}
    acc = {a["frame"]: a for a in deriv["accelerations"]}
    track = {t["frame"]: t for t in deriv["track"]}
    step = {s["to"]: s for s in steps}

    rows = []
    joints_meta = {}
    for frame in seq["order"]:
        v, a, s = vel.get(frame), acc.get(frame), step.get(frame)
        rows.append({
            "frame": frame,
            # the OTHER end of the step reported at this row. Every velocity here
            # is a per-step delta filed at the later frame, so a row without its
            # `from` names one endpoint of a two-frame measurement — and in a
            # SORTED table temporal adjacency is gone, so the reader cannot
            # recover it by looking up a line. None on the first frame, which
            # ends no step.
            "from": (s or {}).get("from"),
            "frame_gap": (v or {}).get("frame_gap"),
            # the step residual IS the first difference (pose_diff), so the velocity
            # magnitudes are read off it rather than recomputed.
            "velocity": {
                "rotation_deg":
                    _round6((s or {}).get("pose", {}).get("rotation_deg")),
                "translation_canon":
                    _round6((s or {}).get("pose", {}).get("translation_dist_canon")),
                "radial_canon": _round6((v or {}).get("radial_velocity")),
            },
            "acceleration": {
                "rotation_deg":
                    _round6((a or {}).get("angular_accel_magnitude_deg")),
                "translation_canon": _round6((a or {}).get("accel_magnitude_canon")),
                "radial_canon": _round6((a or {}).get("radial_acceleration")),
            },
            "joints": _joint_row(s, a, track.get(frame), joints_meta),
        })

    owner = window_owner(run_dir) if run_dir else {}
    if owner:
        _mark_seams(rows, owner)
    warnings = [w for w in [_check_contiguous(seq["order"])] if w]

    return {
        "schema_version": SCHEMA_VERSION,
        "pose_frame": seq["summary"]["pose_frame"],
        "units": UNITS,
        "joints_meta": joints_meta,
        "frames": rows,
        # The per-STEP transition records this table was built from, carried
        # rather than dropped. The rows answer "what happened at each frame";
        # a consumer deciding about a TRANSITION (which steps cross a window
        # seam, what to stamp a verdict against) needs the step, and its only
        # other route was to call sequence.diff_sequence a SECOND time on the
        # same document — same numbers, twice the work, two paths that could
        # disagree. One producer, one computation, both views.
        "steps": steps,
        # sequence's own path summary, at this report's precision contract
        "totals": {k: _round6(v) if isinstance(v, float) else v
                   for k, v in seq["summary"].items()},
        "summary": seq["summary"],
        "warnings": warnings,
    }


def _require_joint_states(pose_json, order):
    """Every ARTICULATED joint the scene declares must have a state on every frame.

    JOINTS.md §3 already states this rule ("Every frame must declare a state for every
    articulated joint... a hard error at load") and core.joints.require_complete_states
    already is it — so this delegates rather than re-implementing.

    Why the report needs its own check even though the loader has one: a pose.json is
    an artifact on disk. It can be hand-edited, produced by an older tool, or merged
    from fragments, so it reaches this module without passing the loader. The report is
    what a consumer reasons from, and a defaulted joint state renders a plausible
    picture of the wrong configuration — so it fails here too.

    `fixed` joints are exempt (no DOF, identity at any state) — which is exactly the
    kind of detail that argues for one shared implementation.
    """
    pj = pose_read.load_pose_json(pose_json)
    joint_defs = pj.get("joint_defs") or pj.get("JOINTS") or []
    for frame in order:
        states = pose_read.frame_entry(pj, frame)[1] or {}
        joints_core.require_complete_states(
            joint_defs, states, f"temporal report, frame {frame!r}")


def _joint_row(step, accel, track_row, joints_meta):
    """{name: {velocity, acceleration, state}} for one frame.

    `velocity` is the joint's per-step delta and `acceleration` the change in that
    delta across the frame — the same ladder as the base pose, in the joint's own
    native units. `state` is where the joint actually is, which is what makes the two
    deltas interpretable.

    The joint's `type` and declared `limit` cannot change frame-to-frame, so they go
    into the shared `joints_meta` (mutated here, once per joint) rather than onto
    every row — a reader forming a fraction of travel still has both operands
    (pose_diff.joint_deltas), stated once."""
    jstep = (step or {}).get("joints") or {}
    jaccel = (accel or {}).get("joints") or {}
    jtrack = (track_row or {}).get("joints") or {}
    out = {}
    for name in sorted(set(jstep) | set(jaccel) | set(jtrack)):
        st, ac, tr = jstep.get(name, {}), jaccel.get(name, {}), jtrack.get(name, {})
        out[name] = {
            "velocity": _round6(st.get("delta")),
            "acceleration": _round6(ac.get("change")),
            "state": _round6(tr.get("state")),
        }
        joints_meta.setdefault(name, {
            "type": tr.get("type") or st.get("type") or ac.get("type"),
            "limit": tr.get("limit") or st.get("limit"),
        })
    return out


# --------------------------------------------------------------------------- #
# human-readable table
# --------------------------------------------------------------------------- #
def _cell(value, width, fmt=".4f"):
    """One right-aligned cell of `width`, or a dash of the same width when the value
    was not measured. Padding the dash to the column keeps every row aligned — an
    un-padded placeholder shifts the whole rest of the line and makes a table that
    cannot be read down a column, which is the only reason to have a table."""
    if value is None:
        return "-".rjust(width)
    return f"{value:{fmt}}".rjust(width)


def _elide(rows, keep):
    """The rows at sorted indices `keep`, each carrying `elided` — how many rows sit
    between it and the previously kept one, INCLUDING a cut before the first kept
    row (easy to miss, usually the largest); the LAST kept row also carries
    `elided_after` for a cut running to the end of the table. Every focused view
    goes through this, so no view can hide rows without counting them: a view that
    did would read as "this is the whole table"."""
    out = []
    prev = -1
    for i in sorted(keep):
        row = dict(rows[i])
        row["elided"] = max(0, i - prev - 1)
        out.append(row)
        prev = i
    if out:
        out[-1]["elided_after"] = len(rows) - 1 - prev
    return out


def _seam_indices(rows, context):
    """Indices of the seam rows and `context` frames either side of each. The seam
    step spans rows i-1 -> i, so BOTH are the seam itself and the context extends
    outward from the pair. Empty when nothing is a seam."""
    keep = set()
    for i, row in enumerate(rows):
        if row.get("is_seam"):
            for j in range(i - 1 - context, i + 1 + context):
                if 0 <= j < len(rows):
                    keep.add(j)
    return keep


def _window_indices(rows, window, context):
    """Indices of the rows `window` posed, plus `context` either side — a refiner
    reading its own chunk, with the boundary steps visible. The context rows matter
    for the same reason seam context does: the step INTO the window's first frame is
    precisely a seam, and one row cannot show a step.

    Raises when the id matches nothing, naming what would have: a typo'd window
    silently showing an empty table would read as "this window has no rows"."""
    own = {i for i, r in enumerate(rows) if r.get("window") == window}
    if not own:
        have = sorted({r.get("window") for r in rows if r.get("window")})
        raise ValueError(
            f"window {window!r} owns no frame in this report; the plan has: "
            f"{', '.join(have) if have else 'no windows at all (no plan found)'}")
    return {j for i in own for j in range(i - context, i + context + 1)
            if 0 <= j < len(rows)}


def _frames_indices(rows, spec):
    """Indices matching a frame-range SPEC: comma-separated terms, each a single
    frame (`000120.jpg`, or a bare number `120`) or a range `A..B` over frame
    NUMBERS, inclusive, with either end open (`..000120`, `000180..`).

    Numbers, not list positions, so the spec means the same thing whatever subset
    of the capture the report covers. Raises on a term that matches nothing —
    a silently empty selection would read as "nothing happened there"."""
    nums = [frames_lib.frame_number(r["frame"]) for r in rows]
    byname = {r["frame"]: i for i, r in enumerate(rows)}

    def bound(text, which):
        if not text:
            return None
        n = frames_lib.frame_number(text)
        if n is None:
            raise ValueError(f"frames spec: {which} bound {text!r} has no "
                             "frame number in it")
        return n

    keep = set()
    for term in [t.strip() for t in spec.split(",") if t.strip()]:
        if ".." in term:
            lo_s, hi_s = term.split("..", 1)
            lo, hi = bound(lo_s, "low"), bound(hi_s, "high")
            hits = {i for i, n in enumerate(nums)
                    if n is not None
                    and (lo is None or n >= lo) and (hi is None or n <= hi)}
        elif term in byname:
            hits = {byname[term]}
        else:
            n = frames_lib.frame_number(term)
            hits = {i for i, m in enumerate(nums) if m is not None and m == n}
        if not hits:
            raise ValueError(
                f"frames spec: {term!r} matches no frame in this report "
                f"({rows[0]['frame']} .. {rows[-1]['frame']})")
        keep |= hits
    return keep


def seam_rows(report, context=1):
    """Just the SEAM rows and `context` frames either side of each — the focused read.

    A 30-frame table is 30 rows a reader has to scan; if the question is "did the
    windows agree at their boundaries", only a handful of rows can answer it. This
    keeps each seam plus its neighbours, because a seam step is only interpretable
    NEXT TO the steps around it: 40 deg at a seam means one thing when its neighbours
    are 5 deg and another when they are 45.

    Returns the rows in order with `elided` counts (`_elide`). Empty when there are
    no seams (no window plan, or one window covering everything). That is a real
    answer, not a failure: a run with no seams has no boundary anyone needs to
    review.
    """
    rows = report["frames"]
    keep = _seam_indices(rows, context)
    return _elide(rows, keep) if keep else []


# the sort spec's field names, per block — the table's own column words.
_SORT_FIELDS = {"rot": "rotation_deg", "trans": "translation_canon",
                "radial": "radial_canon"}


def _sort_value(row, spec):
    """The |value| a row sorts by under `spec`, or None when unmeasured.

    `spec` — `vel.<rot|trans|radial>`, `accel.<rot|trans|radial>`, or
    `<joint>.<vel|accel|state>` for a joint named in the report. Raises on anything
    else, naming what would have matched — a typo silently sorting by nothing would
    print a table that LOOKS sorted."""
    block, _, field = spec.partition(".")
    if block in ("vel", "accel") and field in _SORT_FIELDS:
        return row["velocity" if block == "vel" else "acceleration"][
            _SORT_FIELDS[field]]
    if block in row["joints"] and field in ("vel", "accel", "state"):
        key = {"vel": "velocity", "accel": "acceleration", "state": "state"}[field]
        return row["joints"][block][key]
    joints = sorted(row["joints"])
    raise ValueError(
        f"sort key {spec!r} is not a column: vel|accel . rot|trans|radial, or "
        f"a joint name ({', '.join(joints) or 'none in this report'}) . "
        "vel|accel|state")


def _select_rows(report, seams_only, context, frames, window):
    """The rows a focused view keeps, elision-counted, plus a scope note for the
    header. At most ONE selection at a time: each answers a different question
    (boundaries / a stretch of the capture / one refiner's chunk), and the meaning
    of two composed — union? intersection? — is exactly the ambiguity a reader
    should not have to guess at."""
    picked = [name for name, on in (("seams-only", seams_only),
                                    ("frames", frames is not None),
                                    ("window", window is not None)) if on]
    if len(picked) > 1:
        raise ValueError(f"pick ONE row selection, not {' + '.join(picked)}: each "
                         "answers a different question")
    rows = report["frames"]
    if seams_only:
        keep = _seam_indices(rows, context)
        return (_elide(rows, keep) if keep else []), f"seams +/-{context}"
    if frames is not None:
        return _elide(rows, _frames_indices(rows, frames)), f"frames {frames}"
    if window is not None:
        return (_elide(rows, _window_indices(rows, window, context)),
                f"{window} +/-{context}")
    return list(rows), None


def _parse_columns(spec, joints):
    """`columns` -> (show velocity?, show acceleration?, joint names to show).

    Tokens: `vel`, `accel`, `joints` (all of them), or a joint's name. The frame /
    gap / win columns and the seam marks always print — they are the row's identity
    and provenance, not data columns a reader opts into. Raises on an unknown token,
    naming the valid ones."""
    if spec is None:
        return True, True, joints
    show_vel = show_accel = False
    show_joints = []
    for token in [t.strip() for t in spec.split(",") if t.strip()]:
        if token == "vel":
            show_vel = True
        elif token == "accel":
            show_accel = True
        elif token == "joints":
            show_joints += [j for j in joints if j not in show_joints]
        elif token in joints:
            if token not in show_joints:
                show_joints.append(token)
        else:
            raise ValueError(
                f"columns: {token!r} is not a column group: vel, accel, joints, "
                f"or a joint name ({', '.join(joints) or 'none in this report'})")
    return show_vel, show_accel, show_joints


def table_lines(report, seams_only=False, context=1, frames=None, window=None,
                columns=None, sort_by=None):
    """The per-frame table as aligned text — the same numbers, for a terminal.

    Warnings go at the TOP, before any number, so a reader cannot skim past a caveat
    that changes how the whole table reads. A `>>` in the left margin marks a SEAM row
    (the step ending there crossed into another refiner's window) and the `win` column
    names the owner — both only appear when a window plan was supplied.

    FOCUSED VIEWS — all display filters over the same computed rows, so a kept row's
    numbers are identical to the full table's:
      * `seams_only` — the seam steps + `context` frames either side;
      * `frames`     — a range spec over frame NUMBERS (`"000100..000160"`, single
                       frames, comma-combinable, either end open);
      * `window`     — the rows one refiner window owns + `context` either side;
      * `columns`    — only the named column groups (`"vel,accel"`, a joint name);
      * `sort_by`    — rows biggest-|value| first by one column (`"vel.rot"`,
                       `"lid.accel"`) instead of temporal order; unmeasured rows
                       sink to the bottom rather than sorting as 0.
    One row selection at a time (`seams_only`/`frames`/`window`); `columns` and
    `sort_by` compose with any of them — EXCEPT `sort_by` with `seams_only`, which
    raises. A seam is interesting for WHO posed it, not how big it is: ordering seam
    rows by magnitude reads the small ones as cleared, and sorting destroys the
    adjacency that makes seam context mean anything. In a sorted table the seam
    marks still print; that is the honest composition.
    """
    if sort_by and seams_only:
        raise ValueError(
            "sort_by with seams_only is refused, on purpose: a seam matters for WHO "
            "posed it, not how big it is — a magnitude order over seam rows reads "
            "the small ones as cleared, and sorting breaks the neighbour adjacency "
            "the context rows exist for. Sort the full table (seam rows keep their "
            ">> marks there), or read the seams in temporal order.")
    rows, scope = _select_rows(report, seams_only, context, frames, window)
    all_joints = sorted({n for r in report["frames"] for n in r["joints"]})
    show_vel, show_accel, joints = _parse_columns(columns, all_joints)
    if sort_by:
        # validate against the first row (raises on a bad spec even when every
        # value is None); measured rows biggest first, unmeasured after them in
        # temporal order — never ranked as 0, "not measured" is not "did not move".
        if rows:
            _sort_value(rows[0], sort_by)
        rows = sorted(rows, key=lambda r: (
            (v := _sort_value(r, sort_by)) is None, -abs(v if v is not None else 0)))
        # adjacency gone; per-cut counts are meaningless (the header's "N of M
        # rows" already says how much a prior selection kept)
        rows = [dict(r, elided=0, elided_after=0) for r in rows]

    windowed = any(r.get("window") is not None for r in report["frames"])
    # the row identity is the STEP, so it prints both frames. `from` is provenance
    # of the same kind as the seam mark and the `win` column: not data a reader
    # opts into, but what the row's numbers were measured between. It matters most
    # under --sort-by, where the temporal neighbour is no longer the line above.
    head = (f"{'':>2}{'step (from->at)':>28} {'gap':>4}"
            + (f" {'win':>4}" if windowed else ""))
    band = f"{'':>2}{'':>28} {'':>4}" + (f" {'':>4}" if windowed else "") + " "
    if show_vel:
        head += f" | {'rot':>7} {'trans':>8} {'radial':>8}"
        band += f"|{'VELOCITY':^27}"
    if show_accel:
        head += f" | {'rot':>7} {'trans':>8} {'radial':>8}"
        band += f"|{'ACCELERATION':^27}"
    for n in joints:
        head += f" | {n[:12]:>12} {'vel':>8} {'accel':>8}"
    if joints:
        band += f"|{'JOINTS (state, velocity, accel)':^{33 * len(joints)}}"
    lines = []
    for warning in report.get("warnings") or []:
        lines += [warning, ""]
    if seams_only and not rows:
        return lines + ["temporal report: NO SEAMS — the run was not refined in "
                        "windows, or one window covers every frame, so no boundary "
                        "went unreviewed. (Drop --seams-only for the full table.)"]
    scope_note = (f", {len(rows)} of {len(report['frames'])} rows "
                  f"({scope})" if scope else "")
    sort_note = ([f"  SORTED by |{sort_by}| biggest first (unmeasured rows last) — "
                  "NOT temporal order; neighbours in this table are not neighbours "
                  "in time."] if sort_by else [])
    lines += [
        f"temporal report — {report['pose_frame']} frame, "
        f"{len(report['frames'])} frames{scope_note}",
        "  velocity is a per-STEP delta at the later frame; acceleration is per "
        "INTERIOR frame. '-' = not measured (no step ends here), never 0.",
        "  rot/radial in deg and object units; the angular acceleration is a "
        "geodesic angle and FOLDS past 180 — read it beside the rotation velocity.",
        *sort_note,
        "",
        band,
        head,
        "-" * len(head),
    ]
    for r in rows:
        if r.get("elided"):
            lines.append(f"{'':>2}{'...':>28}   {r['elided']} row(s) not shown")
        v, a = r["velocity"], r["acceleration"]
        mark = ">>" if r.get("is_seam") else "  "
        win = f" {(r.get('window') or '-'):>4}" if windowed else ""
        pair = (f"{r['from']}->{r['frame']}" if r.get("from")
                else r["frame"])          # first frame ends no step
        line = f"{mark}{pair:>28} {_cell(r['frame_gap'], 4, 'd')}{win}"
        if show_vel:
            line += (f" | {_cell(v['rotation_deg'], 7, '.2f')} "
                     f"{_cell(v['translation_canon'], 8, '.4f')} "
                     f"{_cell(v['radial_canon'], 8, '+.4f')}")
        if show_accel:
            line += (f" | {_cell(a['rotation_deg'], 7, '.2f')} "
                     f"{_cell(a['translation_canon'], 8, '.4f')} "
                     f"{_cell(a['radial_canon'], 8, '+.4f')}")
        for n in joints:
            j = r["joints"].get(n) or {}
            line += (f" | {_cell(j.get('state'), 12, '.1f')} "
                     f"{_cell(j.get('velocity'), 8, '+.1f')} "
                     f"{_cell(j.get('acceleration'), 8, '+.1f')}")
        lines.append(line)
        if r.get("elided_after"):
            lines.append(
                f"{'':>2}{'...':>28}   {r['elided_after']} row(s) not shown")
    t = report["totals"]
    seams = [r["frame"] for r in report["frames"] if r.get("is_seam")]
    lines += ["-" * len(head)]
    if windowed:
        lines.append(
            f"  >> {len(seams)} seam(s): {', '.join(seams) or 'none'} — the step "
            "ending there crossed windows, so no single refiner saw both sides. "
            "Interesting because of WHO posed it, not how big it is.")
    radial = t.get("max_radial_velocity")
    lines += [
        f"  path: {t['total_rotation_deg']:.1f} deg rotation, "
        f"{t['total_translation_canon']:.3f} translation (summed steps, whole "
        "sequence)",
        # every extremum names BOTH frames of its step: the quantity belongs to
        # the pair, and a mis-posed frame inflates the step on either side of it
        f"  biggest: {t['max_rotation_deg']:.1f} deg over "
        f"{sequence.step_label(t['max_rotation_at'])}, "
        f"{t['max_translation_canon']:.3f} translation over "
        f"{sequence.step_label(t['max_translation_at'])}",
        "  biggest radial: "
        + ("-" if radial is None else f"{radial:+.3f}")
        + f" over {sequence.step_label(t.get('max_radial_velocity_at'))} "
        "(+ = away)",
    ]
    return lines


def main():
    p = argparse.ArgumentParser(
        description="the consumer-facing temporal report: one row per frame, every "
                    "value a single scalar with its unit named. Reads only — every "
                    "number is a magnitude of something "
                    "analysis.temporal.sequence measured")
    p.add_argument("--pose-json", required=True, help="a scene's mesh/pose.json")
    p.add_argument("--run-dir", default="", help="the RUN_DIR, to mark WINDOW SEAMS: "
                   "steps whose two frames were posed by different refiner agents, so "
                   "nobody saw both sides. Defaults to the pose.json's own run "
                   "(../.. of mesh/pose.json); pass --no-seams to skip")
    p.add_argument("--no-seams", action="store_true",
                   help="do not read the window plan even if one exists")
    p.add_argument("--seams-only", action="store_true",
                   help="print only the SEAM rows and their neighbours — the focused "
                        "read when the question is whether the windows agreed at "
                        "their boundaries. Elided rows are counted, never hidden")
    p.add_argument("--frames", default=None,
                   help="print only these rows: a comma list of frames and/or "
                        "ranges over frame NUMBERS, inclusive, either end open "
                        "('000100..000160', '..000120', '000180..', '000040'). A "
                        "display filter — derivatives are still computed over the "
                        "whole sequence, so boundary rows keep honest values")
    p.add_argument("--window", default=None,
                   help="print only the rows this window id (e.g. 'w03') posed, "
                        "plus --context either side — a refiner reading its own "
                        "chunk. Same display-filter contract as --frames")
    p.add_argument("--context", type=int, default=1,
                   help="frames of context either side of each seam or window "
                        "(--seams-only / --window; default 1)")
    p.add_argument("--columns", default=None,
                   help="print only these column groups: comma list of 'vel', "
                        "'accel', 'joints', or a joint's name. Frame/gap/win and "
                        "the seam marks always print")
    p.add_argument("--sort-by", default=None,
                   help="order rows biggest-|value| first by ONE column — "
                        "'vel.rot', 'accel.trans', 'vel.radial', '<joint>.vel', "
                        "'<joint>.state' — instead of temporal order; unmeasured "
                        "rows sink to the bottom. Refused with --seams-only: a "
                        "seam matters for WHO posed it, not how big it is")
    p.add_argument("--order", default="", help="comma-separated frame order "
                   "(default: temporal order by frame number)")
    p.add_argument("--out", default="", help="write the JSON report here")
    args = p.parse_args()

    order = [s for s in args.order.split(",") if s] or None
    # default the run dir to the pose.json's own run, since RUN/mesh/pose.json is
    # where it always lives — a reader should not have to name it twice to see seams.
    run_dir = None if args.no_seams else (
        args.run_dir or os.path.dirname(os.path.dirname(
            os.path.abspath(args.pose_json))))
    report = build(args.pose_json, order=order, run_dir=run_dir)
    try:
        lines = table_lines(report, seams_only=args.seams_only,
                            context=max(0, args.context),
                            frames=args.frames, window=args.window,
                            columns=args.columns, sort_by=args.sort_by)
    except ValueError as exc:
        # a view-spec mistake (bad column, unknown window, the sort_by+seams_only
        # ban) is a usage error: say it the argparse way, not as a traceback.
        p.error(str(exc))
    print("\n".join(lines))
    if args.out:
        write_json(args.out, report)
        print(f"\n[temporal-report] wrote {args.out}")


if __name__ == "__main__":
    main()
