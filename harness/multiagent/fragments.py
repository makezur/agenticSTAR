"""fragments.py — what a refiner ships, and how it becomes the trajectory.

A FRAGMENT (`<wid>/poses.json`) is a refiner's whole deliverable: the poses it
settled, plus the two temporal fields that say it looked (`expected_motion`,
written before the first sweep, and a `step_calls` verdict on every interior
step). This is the one place a fragment is read, validated, and overlaid.

The two commands that consume one live here too: `selfcheck` (a refiner's
pre-commit read + call-sheet stamp) and `merge` (validate all fragments -> one
trajectory). They share `_overlay_pose`, which is why a refiner's pre-commit
table and the merge's are the same numbers by construction.

Neither picks out WHICH numbers matter: they print the whole table and name the
steps needing a verdict. Reading the columns is the agent's job — a tool that
summarized one number per step would be making the judgement being delegated.
See analysis/temporal/report.md.

Schema: conventions/state_json.md §6. Verdicts: calls.py.
"""

import json
import math
import os

from analysis import frames as frames_lib
from analysis import pose_diff
from analysis.lib.io import read_json, write_json
from analysis.temporal import report as temporal_report
from analysis.temporal import seams as seams_lib
from core import filehash, state_json
from multiagent import calls as calls_lib
from multiagent import scene_source
from multiagent.workspace import _active_workspace
from rig import lie


def log(msg):
    print(f"[windows] {msg}", flush=True)

# One constant for the fragment schema, shared with the sweep family's
# <stem>_poses.json writer (which merge() also consumes) — see core/state_json.py.
# One constant for the fragment schema, shared with the sweep family's
# <stem>_poses.json writer (which merge() also consumes) — see core/state_json.py.
SCHEMA_VERSION = state_json.FRAGMENT_SCHEMA_VERSION


# Tolerances for "did the fragment actually CHANGE this frame": an entry that
# just echoes the committed pose (a six-decimal serialization round-trip) is
# reported as kept, not updated. Joints serialize via %.6g (6 SIGNIFICANT
# digits), so at degree magnitudes (~100) the round-trip error reaches ~5e-4 —
# the joint tolerance must sit above that, not at the 1e-5 radian-era value.
ECHO_ROT_TOL_RAD = 1e-4


ECHO_TRANS_TOL = 1e-5


ECHO_JOINT_TOL = 1e-3


def _committed(pose_json, frame):
    op, joints, c2w = pose_diff.frame_entry(pose_json, frame)
    return {"pose": {"quaternion": op.get("quaternion"),
                     "translation": op.get("translation")},
            "joints": joints, "camera_c2w": c2w}


# --------------------------------------------------------------------------- #
# merge — validate fragments, overlay, temporal-check the result
# --------------------------------------------------------------------------- #
def _quat_angle(qa, qb):
    """lie.quat_angle with a missing/degenerate quaternion reading as maximally
    different (pi) — an absent pose must never look like an echo."""
    if not qa or not qb:
        return math.pi
    if not any(float(v) for v in qa) or not any(float(v) for v in qb):
        return math.pi
    return float(lie.quat_angle(qa, qb))


def _trans_dist(ta, tb):
    if not ta or not tb:
        return float("inf")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(ta, tb)))


def _echoes_committed(committed_entry, frag_entry):
    """True if a fragment entry just re-states the committed pose within
    serialization tolerance — reported as kept, not updated."""
    com_pose = committed_entry.get("pose") or {}
    frag_pose = frag_entry.get("pose") or {}
    if _quat_angle(com_pose.get("quaternion"),
                   frag_pose.get("quaternion")) > ECHO_ROT_TOL_RAD:
        return False
    if _trans_dist(com_pose.get("translation"),
                   frag_pose.get("translation")) > ECHO_TRANS_TOL:
        return False
    com_joints = committed_entry.get("joints") or {}
    frag_joints = frag_entry.get("joints")
    if frag_joints is not None:
        for name, value in frag_joints.items():
            if name not in com_joints:
                # A joint the fragment states and the committed entry does not is
                # NOT an echo. Joint states are absolute: an absent one is
                # unknown, not zero.
                return False
            if abs(float(value) - float(com_joints[name])) > ECHO_JOINT_TOL:
                return False
    return True


def load_fragment(run_dir, wid, wdir=None):
    wdir = wdir or _active_workspace(run_dir)
    path = os.path.join(wdir, wid, "poses.json")
    if not os.path.isfile(path):
        return None
    frag = read_json(path)
    if not isinstance(frag, dict):
        raise ValueError(f"{path} is not a JSON object")
    return frag


def _load_selfcheck_fragment(run_dir, wid, wdir=None):
    """Load final work when present, otherwise the worker's resumable draft."""
    wdir = wdir or _active_workspace(run_dir)
    frag = load_fragment(run_dir, wid, wdir=wdir)
    if frag is not None:
        return frag
    path = os.path.join(wdir, wid, "progress.json")
    if not os.path.isfile(path):
        return None
    frag = read_json(path)
    if not isinstance(frag, dict):
        raise ValueError(f"{path} is not a JSON object")
    return frag


def _fragment_report(wid, frag, owned):
    """Normalize a fragment's optional `report` — the refiner's handoff for things
    it SAW but could not fix (a flipped neighbor, a bad frame/mask).

    Advisory, so malformed input is COERCED, never rejected: a report must not be
    able to fail the merge. An out-of-window `frame` is kept and flagged rather
    than dropped ("my neighbor is flipped" legitimately names a frame it owns
    nothing of)."""
    raw = (frag or {}).get("report")
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    out = []
    for item in items:
        if isinstance(item, dict):
            frame = item.get("frame")
            note = item.get("note", item.get("notes", ""))
        else:
            frame, note = None, item
        obs = {"window": wid, "note": str(note)}
        if frame is not None:
            obs["frame"] = str(frame)
            if str(frame) not in owned:
                obs["out_of_window"] = True
        out.append(obs)
    return out


_MODEL_FEEDBACK_KINDS = {
    "missing_joint", "wrong_joint", "geometry", "scale", "input",
}


_MODEL_BLOCKING_KINDS = _MODEL_FEEDBACK_KINDS - {"input"}


_FEEDBACK_SEVERITIES = {"high", "med", "low"}


def _fragment_model_feedback(wid, frag, owned):
    """Validate and normalize a refiner's structured frozen-model findings."""
    raw = (frag or {}).get("model_feedback", [])
    if not isinstance(raw, list):
        raise ValueError("model_feedback must be a list")
    out = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"model_feedback[{i}] must be an object")
        kind = str(item.get("kind", "")).strip().lower()
        severity = str(item.get("severity", "")).strip().lower()
        observation = str(item.get("observation", "")).strip()
        if kind not in _MODEL_FEEDBACK_KINDS:
            raise ValueError(
                f"model_feedback[{i}].kind must be one of "
                f"{sorted(_MODEL_FEEDBACK_KINDS)}")
        if severity not in _FEEDBACK_SEVERITIES:
            raise ValueError(
                f"model_feedback[{i}].severity must be high, med, or low")
        if not observation:
            raise ValueError(f"model_feedback[{i}].observation is required")
        frames = item.get("frames", [])
        if isinstance(frames, str):
            frames = [frames]
        if not isinstance(frames, list):
            raise ValueError(f"model_feedback[{i}].frames must be a list")
        frames = [str(frame) for frame in frames]
        normalized = {
            "window": wid,
            "kind": kind,
            "severity": severity,
            "frames": frames,
            "part": str(item.get("part", "unknown")).strip() or "unknown",
            "observation": observation,
            "required_motion": str(item.get("required_motion", "")).strip(),
        }
        outside = [frame for frame in frames if frame not in owned]
        if outside:
            normalized["out_of_window_frames"] = outside
        out.append(normalized)
    return out


def _fragment_expected_motion(frag):
    """Validate `expected_motion` — the refiner's prediction of what the video
    shows, written from the frame sheet BEFORE its first sweep.

    Required and non-empty, because the field's whole value is its ORDER relative
    to the numbers: an expectation written after a residual table is a
    rationalization of it. Mandatory is the only enforcement available — the
    harness cannot see when it was typed.

    Shape is deliberately loose (`frames` is free text, not a parsed range): a
    strict grammar would only invite satisfying the parser."""
    raw = (frag or {}).get("expected_motion")
    if raw is None:
        raise ValueError(
            "expected_motion is required: state what you expect the video to "
            "show, from the frame sheet, BEFORE sweeping (see the brief)")
    if not isinstance(raw, list) or not raw:
        raise ValueError("expected_motion must be a non-empty list of "
                         '{"frames": "A..B", "expect": "<prose>"} items')
    out = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"expected_motion[{i}] must be an object")
        expect = str(item.get("expect", "")).strip()
        if not expect:
            raise ValueError(f"expected_motion[{i}].expect is required — say "
                             "what the object does over these frames")
        out.append({"frames": str(item.get("frames", "")).strip(),
                    "expect": expect})
    return out


def _interior_steps(window_frames):
    """The steps a window OWNS: the consecutive pairs among its OWN frames.

    A step from the window's first frame back to its committed prev-neighbor is
    NOT here — that one crosses ownership, so it is a seam and belongs to the
    orchestrator (analysis.temporal.seams). This is the whole ownership split, in
    one line: interior pairs to the refiner, crossings to whoever sees both
    windows."""
    return calls_lib.consecutive_steps(window_frames)


def _fragment_step_calls(wid, frag, window_frames, merged_frames, scale):
    """Validate one window's `step_calls` against the steps it owns.

    Freshness is checked against the MERGED frames — the poses about to be written
    — so a refiner that called a step and then re-posed it has a stale call, and
    this is where that surfaces while the fragment can still go back.

    Returns (calls, problems, routed_up). `flip` is refused on an interior step: a
    refiner can re-pose its own frame, so filing one just moves work to someone
    with less context. `unsure` goes to `routed_up` without failing the merge —
    blocking happens at one place (adjudicate)."""
    result = calls_lib.validate((frag or {}).get("step_calls"),
                                _interior_steps(window_frames),
                                merged_frames, scale, allow_flip=False)
    routed = [{**call, "window": wid} for call in result["blocking"]]
    return result["called"], result["problems"], routed


# The derived cache keys the committed object_pose no longer carries (see
# rig.transforms.frame_pose_to_dict); stripped defensively on every overlay in
# case an OLD pose.json still has them, so a consumer can never diff a stale
# matrix that describes the pre-refinement pose.
_DERIVED_CACHE_KEYS = ("matrix4x4", "rotation_euler", "euler_order")


def _frag_has_pose(frag_entry):
    """True if a fragment entry carries a complete quaternion+translation pose.

    Lets selfcheck run against an in-progress fragment: an entry without a full
    pose (a note-only stub, or a frame the refiner hasn't settled) is treated as
    'not updated' and keeps its committed pose rather than crashing the check."""
    p = (frag_entry or {}).get("pose") or {}
    return bool(p.get("quaternion")) and bool(p.get("translation"))


def _overlay_pose(committed_entry, frag_entry):
    """Overlay a fragment's fresh pose (+ any joints) onto a committed frame
    entry; returns a NEW entry. Shared by merge() and selfcheck(), which is what
    makes a refiner's pre-commit check see exactly what the merge will write.

    Drops the derived cache (_DERIVED_CACHE_KEYS): it described the OLD pose, and
    a consumer recomputing from it would diff the pre-refinement pose. Joints are
    OVERLAID, not replaced — replacing would reset an omitted joint to rest in FK;
    a refiner meaning rest states 0.0."""
    target = dict(committed_entry or {})
    merged_op = {k: v for k, v in (target.get("object_pose") or {}).items()
                 if k not in _DERIVED_CACHE_KEYS}
    merged_op["quaternion"] = frag_entry["pose"]["quaternion"]
    merged_op["translation"] = frag_entry["pose"]["translation"]
    target["object_pose"] = merged_op
    if frag_entry.get("joints") is not None:
        target["joints"] = {**(target.get("joints") or {}),
                            **frag_entry["joints"]}
    return target


# --------------------------------------------------------------------------- #
# selfcheck — a refiner's OWN pre-commit temporal read, and its call sheet
# --------------------------------------------------------------------------- #
def _window_review_frames(plan_doc, wid):
    """(window, [review frames]) for a window id: its own frames bracketed by its
    committed outside NEIGHBORS (prev + next when they exist), in temporal order.

    Including the neighbors is what lets a refiner see its SEAMS — a basin flip
    at a window boundary shows up as a big step into/out of a neighbor frame,
    exactly the seam the orchestrator would otherwise only catch post-merge."""
    w = next((x for x in plan_doc["windows"] if x["id"] == wid), None)
    if w is None:
        have = ", ".join(x["id"] for x in plan_doc["windows"])
        raise ValueError(f"no window {wid!r} in plan.json (have: {have})")
    nb = w.get("neighbors") or {}
    frames = set(w["frames"])
    if nb.get("prev"):
        frames.add(nb["prev"]["frame"])
    if nb.get("next"):
        frames.add(nb["next"]["frame"])
    return w, frames_lib.order_frames(frames)


def selfcheck(run_dir, wid, as_json=False):
    """Temporal self-check for ONE window's in-progress fragment, and the STAMP
    that turns its measured steps into an unfilled call sheet.

    Two jobs off one read: (1) print the table the orchestrator will read after
    the merge — the same `report` build, scoped to this window plus its committed
    neighbors, so a bad step surfaces while the refiner can still re-pose; and
    (2) stamp one `step_calls` record per interior step into the draft with the
    numbers and pose hash filled and the verdict blank. Re-running after a
    re-pose re-stamps, so a call whose frame moved comes back blank rather than
    stale.

    Overlays the final fragment (or the progress.json draft) through the shared
    `_overlay_pose`, so what is measured here is what the merge will write.
    `camera_c2w` rides along, so residuals are WORLD-frame given Pi3X poses."""
    run_dir = os.path.abspath(run_dir)
    wdir = _active_workspace(run_dir)
    plan_doc = read_json(os.path.join(wdir, "plan.json"))
    if not isinstance(plan_doc, dict):
        raise ValueError(f"no plan.json under {wdir} — run `plan` first")
    w, review = _window_review_frames(plan_doc, wid)
    pose_json = read_json(os.path.join(run_dir, "mesh", "pose.json"))
    committed = pose_json.get("frames") or {}

    frag = _load_selfcheck_fragment(run_dir, wid, wdir=wdir)
    owned = set(w["frames"])
    frag_frames = (frag or {}).get("frames") or {}
    overlaid = []
    review_frames = {}
    for name in review:
        entry = dict(committed.get(name) or {})
        fe = frag_frames.get(name)
        # only overlay a frame this window OWNS and whose fragment entry carries
        # a full pose — neighbors and un-settled stubs keep the committed pose.
        if name in owned and _frag_has_pose(fe):
            entry = _overlay_pose(entry, fe)
            overlaid.append(name)
        review_frames[name] = entry

    if len(review_frames) < 2:
        log(f"{wid}: only {len(review_frames)} frame(s) in range — nothing to "
            "diff (a window needs >= 2 frames, or a neighbor, to check motion)")
        return {"window_id": wid, "review_frames": list(review_frames),
                "overlaid_frames": overlaid, "steps": [],
                "seam_steps": [], "summary": {}, "step_calls": [],
                "uncalled": []}

    overlaid_doc = {**pose_json, "frames": review_frames}
    # ONE read, used for both jobs below: the table the refiner is shown and the
    # step records its call sheet is stamped from come out of the same
    # analysis.temporal.report build.
    table = temporal_report.build(overlaid_doc, order=list(review_frames),
                                  run_dir=run_dir)
    # a step is a SEAM for this window when it touches a frame the window does
    # not own (a neighbor) — the refiner must judge it against the image, and it
    # is NOT one of the steps this window owes a call on (a crossing step is the
    # orchestrator's; see _interior_steps).
    seams = seams_lib.crossing_steps(table["steps"], {f: True for f in owned})

    # The call sheet: stamp every INTERIOR step from what is overlaid right now,
    # then carry over any verdict the refiner has already written for a step
    # whose hash still matches. That is what makes re-running safe — work
    # already done survives, and a step whose frame moved comes back blank.
    scale = pose_json.get("scale", 1.0)
    interior = [(a, b) for a, b in _interior_steps(w["frames"])
                if a in review_frames and b in review_frames]
    by_key = {seams_lib.step_key(s): s for s in table["steps"]}
    stamped = calls_lib.template(
        [by_key.get((a, b)) or (a, b) for a, b in interior],
        review_frames, scale)
    prior = {calls_lib.step_key(c): c
             for c in ((frag or {}).get("step_calls") or [])
             if isinstance(c, dict)}
    for record in stamped:
        was = prior.get(calls_lib.step_key(record))
        if was and str(was.get("pose_hash") or "") == record["pose_hash"]:
            record["verdict"] = str(was.get("verdict") or "")
            record["evidence"] = str(was.get("evidence") or "")
    uncalled = [calls_lib.step_key(r) for r in stamped if not r["verdict"]]

    report = {
        "window_id": wid,
        "review_frames": list(review_frames),
        "overlaid_frames": overlaid,
        "pose_frame": table.get("pose_frame"),
        "summary": table["summary"],
        "steps": table["steps"],
        "seam_steps": seams,
        "step_calls": stamped,
        "uncalled": [list(k) for k in uncalled],
    }
    _stamp_draft_calls(wdir, wid, stamped)
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        for line in _selfcheck_lines(report, table):
            print(line)
    return report


def _stamp_draft_calls(wdir, wid, stamped):
    """Write the stamped call sheet back into the window's draft
    (`progress.json`), creating it if the refiner has not yet.

    The DRAFT, never the final `poses.json`: the fragment is the refiner's
    deliverable and this tool does not author deliverables. Writing the sheet
    where the refiner is already keeping state means the verdicts get written in
    the file the refiner is already editing, and the finished fragment is a copy
    of a draft that has them."""
    path = os.path.join(wdir, wid, "progress.json")
    draft = read_json(path) if os.path.isfile(path) else None
    if not isinstance(draft, dict):
        draft = {"window_id": wid, "frames": {}}
    draft["step_calls"] = stamped
    write_json(path, draft)


def _selfcheck_lines(report, table):
    """The WHOLE temporal table over this window, then the call sheet naming the
    steps that still need a verdict.

    NO COLUMN IS SUMMARIZED. The sheet lists steps and their state and stops: a
    reader shown one number per step reads the others as decoration."""
    wid = report["window_id"]
    n_over = len(report["overlaid_frames"])
    lines = [f"[windows] selfcheck {wid}: "
             f"{len(report['review_frames'])} frame(s) "
             f"({n_over} from your fragment, rest committed)"]
    try:
        # context 0: the neighbors are already in `review_frames`, so the window
        # view's own context would reach for frames this read does not carry.
        lines += temporal_report.table_lines(table, window=wid, context=0)
    except (ValueError, KeyError) as exc:
        # a window the report has no rows for (a legacy layout, a hand-made doc)
        # must not lose the call sheet below.
        lines.append(f"  (temporal table unavailable: {exc})")

    calls = report.get("step_calls") or []
    if not calls:
        lines.append("STEP CALLS: none — this window has no interior step.")
        return lines
    uncalled = {tuple(k) for k in report.get("uncalled") or []}
    lines += [
        "",
        f"STEP CALLS [{len(calls)} interior step(s), {len(uncalled)} uncalled] "
        f"— stamped into {wid}/progress.json with the pose hash filled in. "
        "Write a `verdict` + `evidence` for each, then copy them into your "
        "fragment:",
        "  READ EVERY COLUMN of the table above for the step you are calling — "
        "rotation, translation, radial (depth), and each joint, velocity AND "
        "acceleration. No column is the answer by itself: a step can be clean "
        "in rotation and still be the poses breathing in depth. report.md "
        "§ 'Reading the table' says what each column is telling you.",
        "  coherent — say what the pictures SHOW: the visible motion, or "
        "'static', or 'the basins tie here, kept the side continuous with the "
        "neighbors'. A big number is fine when the video shows the motion — a "
        "real throw of the lid is not a defect, so name it.",
        "  unsure   — you cannot explain what you see. This ROUTES UP to the "
        "orchestrator; it does not block your merge. Use it rather than "
        "guessing.",
        "  (`flip` is not yours to file on your OWN step: inside your window you "
        "can re-pose the frame, so fix it instead.)",
        "  When the pictures and the residual disagree, that is a pose to fix, "
        "not a number to explain.",
    ]
    for record in calls:
        key = (record["from"], record["to"])
        state = ("UNCALLED" if key in uncalled
                 else f"{record['verdict']}: {record['evidence'][:60]}")
        lines.append(f"  {record['from']} -> {record['to']}  [{state}]")
    return lines


# --------------------------------------------------------------------------- #
# merge — validate every fragment, fold them into ONE trajectory
# --------------------------------------------------------------------------- #
def merge(run_dir, allow_missing=False):
    """Validate every window's fragment and fold them into one trajectory.

    THE SINGLE-WRITER CONTRACT is enforced here: a refiner writes only its own
    `<wid>/poses.json`, this is the only thing that touches the trajectory, and
    `apply` is the only thing that rewrites `scene.py`. Every fragment carries the
    plan's `scene_sha1`, and a mismatch is fatal — poses fitted against different
    geometry are not poses of the same object.

    It also enforces the REFINERS' HALF of the temporal invariant: every window
    owes `expected_motion` and a `step_calls` verdict on every INTERIOR step.
    Missing, stale, out-of-vocabulary, or a `flip` a refiner should have fixed
    itself is a violation. A refiner's `unsure` does not block — it routes UP into
    the orchestrator's seam queue, so blocking happens at exactly one place."""
    run_dir = os.path.abspath(run_dir)
    wdir = _active_workspace(run_dir)
    plan_doc = read_json(os.path.join(wdir, "plan.json"))
    if not isinstance(plan_doc, dict):
        raise ValueError(f"no plan.json under {wdir} — run `plan` first")
    scene_path = os.path.join(run_dir, "scene.py")
    plan_ex_sha = plan_doc.get("scene_ex_frames_sha1")
    if plan_ex_sha:
        # ex-FRAMES: apply's own FRAMES rewrite is not a geometry change, so a
        # late fragment can still land after a partial merge+apply.
        current_sha = scene_source.ex_frames_sha1(scene_path)
        frozen_sha = plan_ex_sha
    else:
        current_sha = filehash.file_sha1(scene_path)
        frozen_sha = plan_doc["scene_sha1"]
    if current_sha != frozen_sha:
        raise ValueError(
            "scene.py changed since the plan was made "
            f"({frozen_sha} -> {current_sha}); the frozen-geometry "
            "contract is broken — re-plan (and re-run refiners) on the new scene")

    pose_json = read_json(os.path.join(run_dir, "mesh", "pose.json"))
    merged_frames = {}
    for name, entry in (pose_json.get("frames") or {}).items():
        merged_frames[name] = dict(entry)

    violations = []
    updated, kept = [], []
    missing_windows = []
    window_reports = []
    model_feedback = []
    expected_motion = []
    validated = {}
    for w in plan_doc["windows"]:
        frag = load_fragment(run_dir, w["id"], wdir=wdir)
        if frag is None:
            missing_windows.append(w["id"])
            kept.extend(w["frames"])
            continue
        if frag.get("scene_sha1") != plan_doc["scene_sha1"]:
            violations.append({"window": w["id"], "kind": "stale_scene",
                               "detail": f"fragment scene_sha1 "
                                         f"{frag.get('scene_sha1')!r}"})
            continue
        owned = set(w["frames"])
        window_reports.extend(_fragment_report(w["id"], frag, owned))
        try:
            model_feedback.extend(
                _fragment_model_feedback(w["id"], frag, owned))
        except ValueError as exc:
            violations.append({"window": w["id"],
                               "kind": "invalid_model_feedback",
                               "detail": str(exc)})
            continue
        # A `window_id` means a REFINER wrote this, so it owes the temporal half
        # of the contract. Without one it is a mechanical sweep product
        # (`<stem>_poses.json`, state_json.md §6): nobody inspected its steps, so
        # they route up instead of being silently unowned.
        refined = bool(frag.get("window_id"))
        if refined:
            try:
                expected_motion.extend(
                    {"window": w["id"], **item}
                    for item in _fragment_expected_motion(frag))
            except ValueError as exc:
                violations.append({"window": w["id"],
                                   "kind": "invalid_expected_motion",
                                   "detail": str(exc)})
                continue
        validated[w["id"]] = (w, frag, refined)
        for name, entry in (frag.get("frames") or {}).items():
            if name not in owned:
                violations.append({"window": w["id"], "kind": "out_of_window",
                                   "frame": name})
                continue
            if _echoes_committed(_committed(pose_json, name), entry):
                kept.append(name)
                continue
            merged_frames[name] = _overlay_pose(merged_frames.get(name), entry)
            updated.append(name)
        kept.extend(sorted(owned - set(frag.get("frames") or {})))

    # Calls are checked LAST, against `merged_frames` — the poses about to be
    # written, not what the refiner held while calling. Uncalled/stale/malformed
    # is a violation; `unsure` routes up instead (below).
    scale = pose_json.get("scale", 1.0)
    step_calls, routed_up = [], []
    for wid, (w, frag, refined) in validated.items():
        if not refined:
            # nobody refined these frames, so nobody has called their steps:
            # hand them to the orchestrator rather than leaving them unowned.
            routed_up.extend(
                {"window": wid, "from": a, "to": b, "verdict": "unsure",
                 "evidence": "no refiner posed this window (fragment carries no "
                             "window_id — a mechanical sweep product), so no "
                             "one has judged this step",
                 "pose_hash": ""}
                for a, b in _interior_steps(w["frames"]))
            continue
        called, problems, routed = _fragment_step_calls(
            wid, frag, w["frames"], merged_frames, scale)
        step_calls.extend({"window": wid, **call} for call in called)
        routed_up.extend(routed)
        violations.extend({"window": wid, "kind": f"step_call:{p['kind']}",
                           "frame": p.get("to"), "detail": p.get("detail", "")}
                          for p in problems)

    if missing_windows and not allow_missing:
        raise ValueError(
            f"window(s) without a poses.json fragment: "
            f"{', '.join(missing_windows)} (pass --allow-missing to merge "
            "anyway, keeping their committed poses)")
    if violations:
        write_json(os.path.join(wdir, "merge_report.json"),
                   {"ok": False, "violations": violations})
        report_path = os.path.join(wdir, "merge_report.json")
        raise ValueError(
            f"fragment violations (see {report_path}): " +
            "; ".join(f"{v['window']}:{v['kind']}:{v.get('frame', '')}"
                      for v in violations))

    merged_doc = {**pose_json, "frames": merged_frames}
    # ONE producer for every temporal number here: analysis.temporal.report. The
    # merge reads the same report an orchestrator does, so a seam's numbers match
    # the table's by construction, not by two call sites agreeing.
    seq = temporal_report.build(merged_doc,
                                order=frames_lib.order_frames(merged_frames))

    # A step across two windows is a SEAM: no refiner owed a call on it, so it is
    # the orchestrator's. EVERY crossing step is listed, by ownership alone with
    # no magnitude consulted: a 2 deg seam and a 178 deg seam are equally
    # unreviewed.
    owner = {f: w["id"] for w in plan_doc["windows"] for f in w["frames"]}
    seam_steps = seams_lib.crossing_steps(seq["steps"], owner)
    # `flip` never reaches here (a refiner may not file one on its own step) and
    # `coherent` needs nothing further, so the only calls with a life after the
    # merge are the `unsure` ones — inherited by adjudicate as seam-queue items.
    model_review_required = [
        item for item in model_feedback
        if item["severity"] == "high"
        and item["kind"] in _MODEL_BLOCKING_KINDS
    ]

    write_json(os.path.join(wdir, "merged_poses.json"), merged_doc)
    report = {
        "ok": True,
        "scene_sha1": plan_doc["scene_sha1"],
        "updated_frames": frames_lib.order_frames(set(updated)),
        "kept_frames": frames_lib.order_frames(set(kept)),
        "missing_windows": missing_windows,
        "summary": seq["summary"],
        "seam_steps": seam_steps,
        "expected_motion": expected_motion,
        "step_calls": step_calls,
        "routed_up": routed_up,
        "window_reports": window_reports,
        "model_feedback": model_feedback,
        "model_review_required": model_review_required,
    }
    write_json(os.path.join(wdir, "merge_report.json"), report)
    scene_source._write_paste(wdir, merged_frames, updated)
    log(f"merged: {len(set(updated))} frame(s) updated, "
        f"{len(set(kept))} kept; {len(step_calls)} interior step(s) called by "
        f"refiners, {len(seam_steps)} step(s) crossing window seams — "
        "adjudicate each one")
    if routed_up:
        log(f"{len(routed_up)} step(s) routed UP to you: a refiner could not "
            "judge them (`unsure`), or no refiner posed that window. They join "
            "the seam queue — adjudicate blocks until each is called")
    if window_reports:
        log(f"{len(window_reports)} refiner observation(s) handed up — see "
            "merge_report.json `window_reports` (things a refiner saw but could "
            "not fix: flipped neighbors, suspect geometry, bad frames/masks)")
    if model_review_required:
        log(f"{len(model_review_required)} high-severity frozen-model finding(s) "
            "require review; they remain visible during apply and verification")
    log(f"next: python -m multiagent.windows apply --run-dir {run_dir} "
        "(rewrites scene.py FRAMES mechanically; "
        f"{os.path.join(wdir, 'paste_frames.py')} is the review copy); then "
        "run composite_pass.sh and adjudicate every seam_step — draw it "
        "with analysis.viz.seam_sheet --pass-dir (so you see the committed-pose "
        "renders), look, and record the verdict")
    return report
