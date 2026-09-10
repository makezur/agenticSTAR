"""sequence — the PRODUCER: per-step residuals and a per-frame track.

For a frame sequence f0, f1, f2, ... this diffs each CONSECUTIVE pair
(f0->f1, f1->f2, ...) via analysis.pose_diff — the per-step residual, i.e. how
much the object moved between adjacent timesteps — and also lifts each frame's
placement into a per-frame TRACK, which analysis.temporal.derivatives differences
into velocity and acceleration. A smooth trajectory has small, steady steps; a
spike flags a jump / tracking glitch / re-grip.

Each step is one TRANSITION RECORD — the same shape
(analysis.temporal.report is the consumer-facing table over these):

    {"from", "to", "frame_gap",
     "pose_frame": "oriented" | "camera" | "unavailable",
     "pose": {rotation_deg, rotation_axis, translation_canon,
              translation_dist_canon, translation_dist (scene units),
              offset_canon_a, offset_canon_b}
             | null (iff pose_frame == "unavailable"),
     "joints": {name: {delta, state_a, state_b, type, limit}}}

`pose_frame` is "oriented" when both frames carry a camera pose — the rotation is
then in world frame and the translation is world-oriented about each camera, with
the camera's own TRANSLATION excluded (see pose_diff's module docstring: a
translating camera therefore does show up in the translation). It is "camera" when
either frame has no camera pose, and both halves then conflate the camera's motion.
A sequence step is never "unavailable" (that arm exists for candidates).

The rotation number is trustworthy (a real ~180 flip reads ~180, not ~0; robust
to the quaternion double cover — see the pose_diff ROTATION note); only vision
tells a genuine half-turn from a spurious flip, which is why this tool REPORTS
and never flags. There are no thresholds here: the report hands over raw
magnitudes and their derivatives, and the agent looking at the frames makes the
call.

Pure numpy + stdlib; file IO only in main(). Runs in the 'artscript' env, from
harness/:
  micromamba run -n artscript python -m analysis.temporal.sequence \
      --pose-json RUN/mesh/pose.json
"""

import argparse

import numpy as np

from analysis import frames as frames_lib
from analysis import pose_diff
from analysis.lib.io import write_json
from analysis.temporal import derivatives
from rig import lie

SCHEMA_VERSION = 5


def _travel_note(joint):
    """" (-42% of range)" for a printed joint line, or "" without a limit.

    Computed for the PRINTED line only: the report deliberately carries `delta` +
    `limit` and not their ratio (pose_diff.joint_deltas), because one obvious name
    fitted two different quantities. A person reading a terminal still benefits from
    the fraction being worked out."""
    lim = joint.get("limit")
    delta = joint.get("delta")
    if not lim or delta is None:
        return ""
    span = float(lim[1]) - float(lim[0])
    if abs(span) <= 1e-12:
        return ""
    return f" ({100.0 * float(delta) / span:+.0f}% of range)"


def _frame_gap(frame_a, frame_b):
    """`to` minus `from` as frame numbers, floored at 1; None when unnumbered."""
    na = frames_lib.frame_number(frame_a)
    nb = frames_lib.frame_number(frame_b)
    return max(1, nb - na) if na is not None and nb is not None else None


def _steps_from_track(track, scale, joint_defs, pose_frame):
    """Consecutive track rows -> transition records, with NO re-lifting.

    Each step's `pose` is pose_diff.placement_residual over the two rows' already
    lifted placements — the SAME implementation pose_diff.pair_residual calls, so a
    sequence step and a candidate transition are the same residual by construction
    rather than by two functions agreeing. `joints` comes from the rows' absolute
    states through the one pose_diff.joint_deltas.
    """
    steps = []
    for prev, cur in zip(track[:-1], track[1:]):
        pa, pb = prev["placement"], cur["placement"]
        states_a = {n: j["state"] for n, j in (prev["joints"] or {}).items()}
        states_b = {n: j["state"] for n, j in (cur["joints"] or {}).items()}
        steps.append({
            "from": prev["frame"],
            "to": cur["frame"],
            "frame_gap": _frame_gap(prev["frame"], cur["frame"]),
            "pose_frame": pose_frame,
            "pose": pose_diff.placement_residual(pa, pb, scale),
            "joints": pose_diff.joint_deltas(states_a, states_b, joint_defs),
        })
    return steps


def diff_sequence(pose_json, order=None):
    """Temporal residuals and derivatives along an ordered SEQUENCE of frames.

    `pose_json` — a loaded pose.json dict or a path to one.
    `order` — explicit frame-name order (a list). Defaults to the pose.json's
    frames sorted into TEMPORAL order by their numeric index (via
    analysis.frames.order_frames), so the per-step diffs follow the timeline
    even if the dict wasn't authored chronologically.

    Returns {"schema_version", "order", "steps", "summary", "derivatives"}:
      * steps — one transition record per consecutive pair (module doc);
      * summary — path length (sum of steps), the biggest rotation, translation and
        radial velocity each with the STEP it spans as a {"from", "to"} pair (a
        per-step extremum belongs to both its frames; naming the later one alone
        points past the defect as often as at it, because a mis-posed frame inflates
        the step on each side of it),
        n_steps, and pose_frame ("oriented" when every frame carried a camera pose,
        "camera" when none did). There is no "mixed": PARTIAL camera coverage RAISES,
        because residuals taken in the two frames of reference are not comparable, so
        a summary over both would be meaningless rather than merely caveated;
      * derivatives — analysis.temporal.derivatives.report over the per-frame
        placement track: velocity, acceleration, and each one's radial component
        about the camera->object direction.
      Descriptive; no flags, no thresholds, nothing filtered out. For the
      CONSUMER-facing view of all this — one row per frame, every value a scalar —
      see analysis.temporal.report.
    """
    pj = pose_diff.load_pose_json(pose_json)
    frames = pj.get("frames") or pj.get("FRAMES") or {}
    seq = list(order) if order is not None else frames_lib.order_frames(frames)

    # ALL frames or NONE may carry a camera pose. Checked up front, over the whole
    # sequence, so partial coverage names every offending frame at once instead of
    # surfacing as pose_diff's pair error at whichever step happens to straddle the
    # boundary first. There is no per-step fallback to reach for: mixing oriented
    # and camera-frame steps in one report is precisely what makes a summary
    # meaningless (pose_diff module docstring).
    without = [n for n in seq if pose_diff.frame_entry(pj, n)[2] is None]
    if without and len(without) != len(seq):
        raise ValueError(
            f"{len(without)} of {len(seq)} frames have no camera pose "
            f"({', '.join(without)}); a sequence needs a camera pose on EVERY "
            "frame (-> pose_frame 'oriented') or on none of them (-> 'camera'), "
            "because residuals taken in the two frames of reference are not "
            "comparable with each other")

    # ONE lift per frame. The track is the primitive: step residuals pair its
    # consecutive rows and the derivative ladder differences them, so no frame is
    # lifted twice and no quantity is computed by two code paths.
    scale = lie.scalar_scale(pj.get("scale", 1.0), "pose.json scale")
    joint_defs = pj.get("joint_defs") or pj.get("JOINTS") or []
    track = pose_diff.frame_track(pj, seq)
    pose_frame = "camera" if without else "oriented"
    steps = _steps_from_track(track, scale, joint_defs, pose_frame)

    rots = [s["pose"]["rotation_deg"] for s in steps]
    trans = [s["pose"]["translation_dist_canon"] for s in steps]
    step_frames = {s["pose_frame"] for s in steps}
    seq_frame = next(iter(step_frames)) if step_frames else (
        "camera" if without else "oriented")

    deriv = derivatives.report(track, steps)

    # `max_*_at` names the STEP, both endpoints — never one frame. A per-step
    # quantity belongs to a pair, and a mis-posed frame inflates the step on EACH
    # side of it, so naming only the later frame ("at 000310") points one frame
    # downstream of the defect as often as at it. The reader must be told which
    # two frames the extremum was measured between and left to judge which is
    # wrong; that is the whole reason this package states pairs and no verdicts.
    def _at(values):
        if not values:
            return None
        s = steps[int(np.argmax(values))]
        return {"from": s["from"], "to": s["to"]}

    # radial is promoted beside rotation and translation, not left in the table.
    # It is the WORST-constrained DOF under one camera (derivatives module: depth
    # is what a single view pins least), so it is the most likely to be wrong and
    # the least likely to be caught by eye — the opposite of a quantity to omit
    # from the digest. Signed, and ranked by |value| like the others, because
    # "+ = away" is the direction a reader needs, not just the magnitude.
    radials = [(v.get("radial_velocity"), v.get("frame"))
               for v in (deriv.get("velocities") or [])
               if v.get("radial_velocity") is not None]
    by_to = {s["to"]: s for s in steps}
    if radials:
        rv, rframe = max(radials, key=lambda p: abs(p[0]))
        rstep = by_to.get(rframe) or {}
        max_radial = float(rv)
        max_radial_at = ({"from": rstep["from"], "to": rstep["to"]}
                         if rstep else None)
    else:
        # never a fabricated zero: no measured radial is "not measured", not 0.0
        max_radial, max_radial_at = None, None

    summary = {
        "n_steps": len(steps),
        "pose_frame": seq_frame,
        "total_rotation_deg": float(sum(rots)),
        "total_translation_canon": float(sum(trans)),
        "max_rotation_deg": float(max(rots)) if rots else 0.0,
        "max_rotation_at": _at(rots),
        "max_translation_canon": float(max(trans)) if trans else 0.0,
        "max_translation_at": _at(trans),
        "max_radial_velocity": max_radial,
        "max_radial_velocity_at": max_radial_at,
    }
    # the raw numpy carriers have served their purpose: `placement` (W, u) let steps
    # be paired without re-lifting, and `rotation_increment` let the angular
    # acceleration compose Omega without a degrees round trip. Neither is JSON, so
    # both come off before the report ships.
    for row in track:
        row.pop("placement", None)
    for row in steps:
        (row.get("pose") or {}).pop("rotation_increment", None)
    for row in deriv.get("velocities") or []:
        row.pop("rotation_increment", None)
    return {"schema_version": SCHEMA_VERSION, "order": seq, "steps": steps,
            "summary": summary, "derivatives": deriv}


# --------------------------------------------------------------------------- #
# human-readable lines
# --------------------------------------------------------------------------- #
def step_lines(step):
    """Compact human lines for one transition record."""
    frame_notes = {
        "oriented": ("rotation world frame; translation world axes about each "
                     "camera (camera translation excluded)"),
        "camera": "camera frame (includes the camera's own motion — no camera pose)",
        "unavailable": "base pose unavailable (no camera pose)",
    }
    lines = [f"{step['from']} -> {step['to']}  "
             f"({frame_notes.get(step['pose_frame'], step['pose_frame'])})"]
    p = step.get("pose")
    if p is not None:
        lines += [
            f"  translation: {p['translation_dist_canon']:.4f} of object size "
            f"(canon delta {[round(v, 4) for v in p['translation_canon']]})",
            f"  rotation:    {p['rotation_deg']:.2f} deg  (~180 = a full flip)",
        ]
    if step.get("joints"):
        lines.append("  joints:")
        for name, j in step["joints"].items():
            # revolute deltas are DEGREES (say so; degree magnitudes don't need
            # 4 decimals) — prismatic stay canonical units at full precision.
            delta = (f"{j['delta']:+.1f} deg" if j.get("type") == "revolute"
                     else f"{j['delta']:+.4f}")
            note = _travel_note(j) or "   n/a (no limit)"
            lines.append(f"    {name:>16}: delta {delta} {note}")
    return lines


def step_label(at):
    """A `max_*_at` pair as "FROM->TO" — or "-" when nothing was measured.

    Both endpoints, always: the extremum is a property of the step, and which of
    the two frames is the bad one is the reader's call, not this function's."""
    if not at:
        return "-"
    return f"{at.get('from')}->{at.get('to')}"


def summary_lines(summary):
    """Compact human lines for a diff_sequence summary block."""
    radial = summary.get("max_radial_velocity")
    return [
        f"sequence: {summary['n_steps']} step(s)  "
        f"[{summary.get('pose_frame', 'camera')} frame]",
        f"  path length: {summary['total_rotation_deg']:.2f} deg rotation, "
        f"{summary['total_translation_canon']:.4f} translation (of object size)",
        f"  biggest step: {summary['max_rotation_deg']:.2f} deg over "
        f"{step_label(summary['max_rotation_at'])}, "
        f"{summary['max_translation_canon']:.4f} translation over "
        f"{step_label(summary['max_translation_at'])}",
        f"  biggest radial: "
        + ("-" if radial is None else f"{radial:+.4f}")
        + f" over {step_label(summary.get('max_radial_velocity_at'))} "
        "(+ = away from camera; depth is the least-constrained DOF)",
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="temporal residuals along a scene's frame sequence: diff "
                    "each consecutive pair (rotation as a geodesic angle in "
                    "degrees, translation as a fraction of object size, joints "
                    "as percent of declared travel), summarize the path, and "
                    "report per-step velocity/acceleration with their radial "
                    "component. Descriptive only — no thresholds and no "
                    "verdict; you decide against the frames whether the motion is "
                    "explained")
    p.add_argument("--pose-json", required=True, help="a scene's pose.json")
    p.add_argument("--order", default="", help="comma-separated frame order "
                   "(default: temporal order by frame number)")
    p.add_argument("--out", default="", help="write the JSON report here")
    p.add_argument("--top", type=int, default=0, help="print only this many "
                   "velocity / acceleration rows (0 = all; the JSON always has "
                   "every row)")
    args = p.parse_args()

    order = [s for s in args.order.split(",") if s] or None
    report = diff_sequence(args.pose_json, order=order)
    for step in report["steps"]:
        print("\n".join(step_lines(step)))
    print("\n".join(summary_lines(report["summary"])))
    kin = report["derivatives"]
    print("\n".join(derivatives.velocity_lines(kin["velocities"], limit=args.top)))
    print("\n".join(derivatives.acceleration_lines(kin["accelerations"],
                                                  limit=args.top)))
    print("\n".join(derivatives.attention_lines(kin["attention_order"])))
    if args.out:
        write_json(args.out, report)
        print(f"[temporal-sequence] wrote {args.out}")


if __name__ == "__main__":
    main()
