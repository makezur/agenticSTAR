#!/usr/bin/env python3
"""windows.py — the windows-round CLI, plus `plan` and `adjudicate`.

Parallel per-window pose refinement: cut the trajectory into contiguous
non-overlapping WINDOWS, one refiner subagent each against a frozen scene.py,
each writing a pose FRAGMENT that the merge folds back in. The design and the
round workflow are windows.md; the two contracts it all serves are (1) single
writer — only `merge` touches the trajectory, only `apply` writes scene.py; and
(2) every temporal step has exactly one owner who writes a verdict on it.

This module is the tool's front door: the CLI, `plan` (cutting windows), and
`adjudicate` (the steps that fall between them). The other three commands live
with the artifact they are about — `selfcheck` and `merge` in fragments.py,
`apply` in scene_source.py — and are re-exported here.

Schemas: conventions/state_json.md §6. Verdict vocabulary: calls.py. Freshness
hash: freshness.py. Every temporal number comes from analysis.temporal.report.

Pure analysis-env tool (no bpy). Run from harness/; `--help` on any subcommand.
"""

import argparse
import json
import math
import os
import re
import shutil
import sys

from analysis import frames as frames_lib
from analysis.lib.io import read_json, write_json
from analysis.temporal import report as temporal_report
from analysis.temporal import seams as seams_lib
from analysis.viz import seam_sheet
from bookkeeping import ledger as bookkeeping
from bookkeeping import pose_history
from core import filehash, modules, run_layout
from core import mechanism_calls as mechanism_calls_lib
from multiagent import calls as calls_lib
from multiagent.briefs import _write_brief
from multiagent.fragments import (   # the two fragment-consuming commands live
    SCHEMA_VERSION, _committed,      # with the fragment reader/validator they
    _interior_steps, load_fragment,  # share; re-exported here as the front door
    merge, selfcheck)                # noqa: F401
from multiagent import scene_source
from multiagent.scene_source import apply     # noqa: F401 (the FRAMES rewriter)
from multiagent.workspace import (          # noqa: F401 (PLAN_DERIVED_FILES,
    PLAN_DERIVED_FILES, WINDOWS_DIRNAME,     # WINDOWS_DIRNAME re-exported: this
    _active_workspace, _create_workspace,    # module is the tool's front door)
    _read_json_lenient)


def log(msg):
    print(f"[windows] {msg}", flush=True)


scene_sha1 = filehash.file_sha1


# --------------------------------------------------------------------------- #
# plan — split frames into non-overlapping windows
# --------------------------------------------------------------------------- #
def split_windows(frames, n_windows):
    """Contiguous NON-OVERLAPPING spans that partition `frames`; every frame
    belongs to exactly one span. n=27, K=5 -> sizes 5-6."""
    n = len(frames)
    if n < 2:
        raise ValueError(f"need >= 2 frames to window, have {n}")
    k = max(1, min(int(n_windows), n // 2))       # every span keeps >= 2 frames
    cuts = [round(i * n / k) for i in range(k + 1)]   # 0 .. n
    return [frames[cuts[i]:cuts[i + 1]] for i in range(k)]


def _parse_segment(spec, all_frames):
    """The --frames value -> the contiguous layout-frame SEGMENT it names.

    Takes a comma list, an inclusive 'a.jpg-c.jpg' range, or a list. Must be
    CONTIGUOUS — seams assume adjacency, so a gapped selection is refused rather
    than silently bridged — and >= 2 frames (one flipped frame is an `apply`-view
    fix, not a window round)."""
    index = {f: i for i, f in enumerate(all_frames)}
    if isinstance(spec, (list, tuple)):
        names = [str(s).strip() for s in spec if str(s).strip()]
    else:
        text = str(spec).strip()
        if "," in text:
            names = [s.strip() for s in text.split(",") if s.strip()]
        elif text in index:
            names = [text]
        else:
            lo, sep, hi = text.partition("-")
            lo, hi = lo.strip(), hi.strip()
            if not (sep and lo in index and hi in index):
                raise ValueError(
                    f"--frames {text!r}: not a layout frame, a comma list, or "
                    "an inclusive '<first>-<last>' range of layout frames")
            if index[lo] > index[hi]:
                raise ValueError(f"--frames range {text!r} runs backwards "
                                 f"({lo} is after {hi} in the timeline)")
            names = all_frames[index[lo]:index[hi] + 1]
    unknown = [n for n in names if n not in index]
    if unknown:
        raise ValueError("--frames names frames not in layout.json: "
                         + ", ".join(sorted(set(unknown))))
    idxs = sorted({index[n] for n in names})
    segment = all_frames[idxs[0]:idxs[-1] + 1]
    missing = sorted(set(segment) - set(names), key=index.get)
    if missing:
        raise ValueError(
            "--frames must name a CONTIGUOUS segment of the trajectory; "
            "missing in between: " + ", ".join(missing))
    if len(segment) < 2:
        raise ValueError(
            "--frames names a single frame — a window round needs >= 2; "
            "re-pose one frame with the `apply` view instead")
    return segment


def _neighbor(pose_json, frame):
    entry = _committed(pose_json, frame)
    return {"frame": frame, "pose": entry["pose"], "joints": entry["joints"]}


def _per_window_option(specs, wids, what):
    """Resolve a repeatable `TEXT` / `wNN=TEXT` option list into {wid: value}.

    Bare value = every window; `wNN=value` overrides one. An unknown window id
    raises rather than being silently dropped."""
    if specs is None:
        specs = []
    if isinstance(specs, (str, bytes)):
        specs = [specs]
    base, overrides = None, {}
    for spec in specs:
        text = str(spec)
        wid, sep, value = text.partition("=")
        if sep and re.fullmatch(r"w\d+", wid.strip()):
            overrides[wid.strip()] = value.strip()
        else:
            if base is not None:
                raise ValueError(
                    f"--{what} given twice without a window id "
                    f"({base!r} then {text!r}); use 'wNN=...' to differ "
                    "per window")
            base = text.strip()
    unknown = sorted(set(overrides) - set(wids))
    if unknown:
        raise ValueError(f"--{what} names window(s) not in this plan: "
                         + ", ".join(unknown) + f" (have: {', '.join(wids)})")
    return {wid: overrides.get(wid, base) for wid in wids}


def _gate_previous_round(run_dir):
    """Refuse to plan a new round while the newest one has an unadjudicated seam.

    Not tidiness: a new round re-poses frames, which stales the verdicts the open
    seam was waiting for, so the question becomes unanswerable rather than
    answered. Gates the NEWEST iteration only, and only once it has an ok merge."""
    wdir = _active_workspace(run_dir, required=False)
    if wdir is None:
        return
    merged = _read_json_lenient(os.path.join(wdir, "merge_report.json"))
    if not (isinstance(merged, dict) and merged.get("ok") is True):
        return
    plan_doc = _read_json_lenient(os.path.join(wdir, "plan.json"))
    if not isinstance(plan_doc, dict):
        return
    queue = _seam_queue(merged, plan_doc)
    if not queue:
        return
    calls_path = os.path.join(wdir, SEAM_CALLS_FILE)
    doc = _read_json_lenient(calls_path)
    merged_doc = _read_json_lenient(os.path.join(wdir, "merged_poses.json"))
    frames = (merged_doc or {}).get("frames") or {}
    result = calls_lib.validate((doc or {}).get("seam_calls"), queue.keys(),
                                frames, (merged_doc or {}).get("scale", 1.0),
                                allow_flip=True)
    if result["problems"] or result["blocking"]:
        n = len(result["problems"]) + len(result["blocking"])
        raise ValueError(
            f"iteration {plan_doc.get('iteration')} has {n} unadjudicated "
            f"seam step(s) — you cannot plan a new round over a step nobody has "
            f"judged (a new round re-poses frames, which stales the verdicts "
            f"the open seam is waiting for). Run:\n"
            f"  python -m multiagent.windows adjudicate --run-dir {run_dir} "
            f"--pass-dir <the verification pass>\n"
            f"call every step in {calls_path}, then plan again.")


def _gate_mechanism_pick(run_dir, pose_json):
    """Refuse to plan a round whose rig has an unanswered (or WRONG) mechanism.

    Windows refine per-frame states against a FROZEN scene.py, so a wrong
    joint declaration — a mirrored axis above all — invalidates every fragment
    the round would produce, and the refiners would chase it with pose sweeps
    that can never fix it. The A/B pick (analysis.mechanism_calls) is the check
    that catches it; verdicts are hash-stamped against the declaration, so a
    plan over an unchanged rig re-asks for nothing."""
    joint_defs = (pose_json.get("joint_defs") or pose_json.get("JOINTS") or [])
    result = mechanism_calls_lib.check_run(run_dir, joint_defs=joint_defs)
    if result.get("disabled"):
        log(modules.disabled_line("mechanism"))
        return
    if not result["ok"]:
        raise ValueError(
            "the committed rig has an unanswered or WRONG mechanism pick — "
            "refiners would sweep per-frame states against a joint declaration "
            "nobody has verified (a mirrored axis is exactly what a pose sweep "
            "can never fix):\n"
            + "\n".join(mechanism_calls_lib.problem_lines(result))
            + f"\nanswer every joint's A/B choice in {result['calls_path']} "
            "(python -m analysis.mechanism_calls ask/check --run-dir "
            f"{run_dir}), then plan again.")


def _warn_unmerged_fragments(run_dir, current_ex_sha):
    """Warn when the newest iteration holds validated-shape fragments for the
    CURRENT scene that were never merged — planning past them silently
    discards finished work (merge --allow-missing lands them; --seed-from
    carries them as hypotheses)."""
    wdir = _active_workspace(run_dir, required=False)
    if wdir is None:
        return
    plan_doc = _read_json_lenient(os.path.join(wdir, "plan.json"))
    if not (isinstance(plan_doc, dict)
            and plan_doc.get("scene_ex_frames_sha1") == current_ex_sha):
        return
    merged = _read_json_lenient(os.path.join(wdir, "merge_report.json"))
    if isinstance(merged, dict) and merged.get("ok") is True:
        # an ok merge consumed everything it saw — but a --allow-missing merge
        # is ok WITH windows outstanding, and a straggler landing after it is
        # exactly the finished-work-about-to-be-discarded this warning is for
        candidates = merged.get("missing_windows") or []
    else:
        candidates = [w["id"] for w in plan_doc.get("windows", [])]
    unmerged = [wid for wid in candidates
                if isinstance(_read_json_lenient(
                    os.path.join(wdir, wid, "poses.json")), dict)]
    if unmerged:
        log(f"WARNING: {wdir} has unmerged fragment(s) for the current "
            f"scene ({', '.join(unmerged)}); planning a new round discards "
            f"them — `merge --allow-missing` lands them, or pass "
            f"--seed-from {wdir} to carry them as hypotheses")


def plan(run_dir, n_windows=None, frames_per_window=None, known_issues=None,
         review_artifacts=None, frames=None, guidance=None, seed_from=None,
         seed_from_iteration=None):
    run_dir = os.path.abspath(run_dir)
    _gate_previous_round(run_dir)
    layout = run_layout.load_run_layout(run_dir)
    if layout is None:
        raise ValueError(f"missing or invalid layout.json in {run_dir}")
    scene = os.path.join(run_dir, "scene.py")
    pose_json = read_json(os.path.join(run_dir, "mesh", "pose.json"))
    if not isinstance(pose_json, dict) or not pose_json.get("frames"):
        raise ValueError("plan requires a committed mesh/pose.json (run a full "
                         "pass first — windows refine an EXISTING trajectory)")
    _gate_mechanism_pick(run_dir, pose_json)
    seeds = {}
    reconciliation = []
    seed_sources = list(seed_from or [])
    for selector in seed_from_iteration or []:
        seed_sources.append(pose_history.resolve_iteration(run_dir, selector))
    if seed_sources:
        seeds, reconciliation = pose_history.reconcile_sources(
            run_dir, pose_json, seed_sources)
    if isinstance(known_issues, str):
        known_issues = read_json(known_issues)
    if known_issues is not None and not isinstance(known_issues, dict):
        raise ValueError('known-issues must be a JSON object mapping '
                         '"<frame>.jpg" to a one-line orchestrator finding')
    review_artifacts = [os.path.abspath(path)
                        for path in (review_artifacts or [])]
    missing_reviews = [path for path in review_artifacts
                       if not os.path.isfile(path)]
    if missing_reviews:
        raise ValueError("review artifact does not exist or is not a file: "
                         + ", ".join(missing_reviews))
    review_names = [os.path.basename(path) for path in review_artifacts]
    if len(review_names) != len(set(review_names)):
        raise ValueError("review artifacts must have unique basenames: "
                         + ", ".join(review_names))
    all_frames = list(layout["frames"])
    segment = _parse_segment(frames, all_frames) if frames else all_frames
    unknown = set(known_issues or {}) - set(segment)
    if unknown:
        raise ValueError("known-issues for frames not in the planned "
                         "segment: " + ", ".join(sorted(unknown)))
    if frames_per_window:
        n_windows = max(1, math.ceil(len(segment) /
                                     max(1, int(frames_per_window))))
    if not n_windows:
        n_windows = max(1, math.ceil(len(segment) / 6))
    spans = split_windows(segment, n_windows)
    sha = scene_sha1(scene)
    ex_sha = scene_source.ex_frames_sha1(scene)
    _warn_unmerged_fragments(run_dir, ex_sha)

    # neighbor lookups index the FULL timeline, so a segment plan's boundary
    # windows see the committed frames just outside the segment as context.
    base = all_frames.index(segment[0])
    windows = []
    offset = 0
    for i, span in enumerate(spans):
        wid = f"w{i + 1:02d}"
        neighbors = {}
        if base + offset > 0:
            neighbors["prev"] = _neighbor(pose_json,
                                          all_frames[base + offset - 1])
        if base + offset + len(span) < len(all_frames):
            neighbors["next"] = _neighbor(
                pose_json, all_frames[base + offset + len(span)])
        w = {"id": wid, "frames": span, "neighbors": neighbors}
        owned_seeds = {f: seeds[f] for f in span if f in seeds}
        if owned_seeds:
            w["seeds"] = owned_seeds
        windows.append(w)
        offset += len(span)

    wids = [w["id"] for w in windows]
    guidances = _per_window_option(guidance, wids, "guidance")
    for w in windows:
        if guidances[w["id"]]:
            w["guidance"] = guidances[w["id"]]

    out = {
        "schema_version": SCHEMA_VERSION,
        "scene_sha1": sha,
        "scene_ex_frames_sha1": ex_sha,
        "ref_frame": layout["ref_frame"],
        "run_dir": run_dir,
        "spool": os.path.join(run_dir, "spool"),
        "windows": windows,
    }
    if reconciliation:
        out["seed_reconciliation"] = reconciliation
    if segment is not all_frames:
        out["segment"] = [segment[0], segment[-1]]
    iteration, wdir = _create_workspace(run_dir)
    out["iteration"] = iteration
    out["iteration_dir"] = wdir
    if known_issues is not None:
        issues_path = os.path.join(wdir, "known_issues.json")
        write_json(issues_path, known_issues)
        out["known_issues"] = os.path.relpath(issues_path, wdir)
    if review_artifacts:
        review_dir = os.path.join(wdir, "review")
        os.makedirs(review_dir)
        copied = []
        for source, name in zip(review_artifacts, review_names):
            destination = os.path.join(review_dir, name)
            shutil.copy2(source, destination)
            copied.append(os.path.relpath(destination, wdir))
        out["review_artifacts"] = copied
    write_json(os.path.join(wdir, "plan.json"), out)
    if reconciliation:
        write_json(os.path.join(wdir, "seed_reconciliation.json"), {
            "schema_version": 1,
            "sources": reconciliation,
        })
    for w in windows:
        os.makedirs(os.path.join(wdir, w["id"]), exist_ok=True)
        _write_brief(run_dir, layout, pose_json, out, w,
                     known_issues=known_issues)
    scope = (f"segment {segment[0]}..{segment[-1]} "
             f"({len(segment)} of {len(all_frames)} frames)"
             if segment is not all_frames else f"{len(segment)} frames")
    log(f"planned iteration {iteration:06d}: {len(windows)} window(s) over "
        f"{scope}; briefs under {wdir}/<wid>/brief.md")
    return out


# --------------------------------------------------------------------------- #
# adjudicate — the ORCHESTRATOR's verdicts on the steps no refiner owned
# --------------------------------------------------------------------------- #
SEAM_CALLS_FILE = "seam_calls.json"

# (sort spec, why it gets its own pass). One pass per column a pose defect can
# hide in: this was `accel.rot` alone, which made rotation the only defect the
# gate could see. The prose is printed, so it is the reader's explanation too.
FORCED_READS = (
    ("accel.rot",
     "A flip announces itself here: a spurious basin flip is a big rotation "
     "that REVERSES (large acceleration), a real turn is a big rotation that "
     "CONTINUES."),
    ("accel.trans",
     "One mis-posed frame between two good neighbors is a velocity "
     "out-and-back, which lands as a spike exactly on that frame — placement "
     "jitter no rotation column shows."),
    ("accel.radial",
     "The depth out-and-back, COMPUTED: r_out - r_in at each interior frame, "
     "signs inside the difference. One frame posed too near or too far makes "
     "two large opposite radial steps, so it scores about TWICE its depth "
     "error exactly AT that frame, while steady depth drift (same sign either "
     "side) cancels to near zero. Depth is what a single camera constrains "
     "WORST, so these are the likeliest errors and the hardest to catch by "
     "eye. No accel row exists for the first or last frame — a bad END frame "
     "shows only as one big radial step, so glance at vel.radial's extremes "
     "for the endpoints."),
)


def _seam_queue(report, plan_doc):
    """The steps the ORCHESTRATOR owes a call on: `(from, to)` -> why it is queued.

    Two sources — `seam_steps` (different owners posed the two frames, so no
    refiner saw the whole step) and `routed_up` (a refiner's `unsure`, plus every
    interior step of a window nobody refined). ONE queue, because the question is
    identical for both; only the reason differs, which is recorded, not branched
    on."""
    queue = {}
    for step in report.get("seam_steps") or []:
        queue[seams_lib.step_key(step)] = {"reason": "seam", "step": step}
    for call in report.get("routed_up") or []:
        key = (call.get("from"), call.get("to"))
        queue.setdefault(key, {
            "reason": "routed_up",
            "step": None,
            "from_window": call.get("window"),
            "refiner_said": call.get("evidence", ""),
        })
    return queue


SEAM_SHEETS_DIRNAME = "seam_sheets"

# the round's canonical sheet. Its own basename, so an ad-hoc look at one pair
# cannot overwrite the whole round's evidence with a one-band crop of it.
SEAM_SHEET_BASENAME = "round_seams"


def _seam_sheet_dir(wdir):
    """Where this round's sheets go: its iteration-owned ``windows/`` tree.

    The tool's default is one fixed `RUN_DIR/seam_sheets/seam_001.png`, so every
    round drew over the last and a run kept only its final sheet. A verdict is
    hash-stamped so re-posing stales it; the picture behind it has to be as
    durable, which means living with the round that made the call."""
    return os.path.join(wdir, SEAM_SHEETS_DIRNAME)


def _seam_sheet_line(run_dir, pass_dir, keys, out_dir):
    """The command that REDRAWS this queue — `adjudicate` already drew it.

    This is the one to VARY: drop all but one `--seam` for a single pair,
    `--context 1` for the local trend, `--seams-per-page 1` to go in close. It
    writes under `look/` so none of that overwrites the round's own sheet.

    Needs --pass-dir: the captured video is always smooth, so both sides of a flip
    look fine in the photos and only the COMMITTED renders show which basin a pose
    landed in. That is also why adjudication runs after apply."""
    seams = " ".join(f"--seam {a}:{b}" for a, b in keys)
    pass_part = f" --pass-dir {pass_dir}" if pass_dir else \
        "  # --pass-dir PASS_DIR  <-- ADD THIS: without the committed renders " \
        "you cannot tell a real half-turn from a flip"
    return (f"micromamba run -n artscript env PYTHONPATH=harness "
            f"python -m analysis.viz.seam_sheet --run-dir {run_dir} "
            f"--pose-json {os.path.join(run_dir, 'mesh', 'pose.json')} "
            f"{seams} --out-dir {os.path.join(out_dir, 'look')}{pass_part}")


def _draw_seam_sheet(run_dir, out_dir, pass_dir, keys):
    """DRAW this round's seam sheet as adjudication's own output.

    Same reason the tables are printed rather than recommended: a step that needs a
    second command sometimes does not happen — a real run wrote four `coherent`
    verdicts having never opened the sheet it generated. The manifest left here is
    what `evidence` can honestly cite.

    Failure is reported, never raised: a missing frames dir must not take out the
    gate that counts the calls."""
    try:
        _, manifest_path = seam_sheet.write_seam_sheets(
            [{"from": a, "to": b} for a, b in keys],
            run_dir=run_dir, pass_dir=pass_dir or "", out_dir=out_dir,
            basename=SEAM_SHEET_BASENAME,
            pose_json=os.path.join(run_dir, "mesh", "pose.json"))
        return manifest_path, ""
    except (ValueError, KeyError, OSError) as exc:
        return "", f"{type(exc).__name__}: {exc}"


def adjudicate(run_dir, pass_dir=None, check=False, as_json=False):
    """The orchestrator's side of the invariant: call every step no refiner owned.

    Prints the forced reads, emits/refreshes `seam_calls.json` (pre-stamped, with
    verdicts blank; a verdict survives a re-run if its hash still matches), and on
    `--check` turns the result into the gate's exit code. Sequence and semantics:
    windows.md §5 + "Round lifecycle".

    Two things the code alone would not tell you:

    * "Forced" means the tables are this command's OUTPUT, not an instruction
      hoping a second command gets run — you cannot pass the gate without having
      been shown them.
    * `flip` is allowed here though a refiner may not file one on its own interior
      step: at a seam the orchestrator is the only one who sees both windows. It
      passes `--check` (a decision, not an absence) while the re-pose it triggers
      stales it — which is what forces the re-look.
    """
    run_dir = os.path.abspath(run_dir)
    wdir = _active_workspace(run_dir)
    plan_doc = read_json(os.path.join(wdir, "plan.json"))
    if not isinstance(plan_doc, dict):
        raise ValueError(f"no plan.json under {wdir} — run `plan` first")
    report_path = os.path.join(wdir, "merge_report.json")
    report = read_json(report_path) if os.path.isfile(report_path) else None
    if not (isinstance(report, dict) and report.get("ok") is True):
        raise ValueError(
            f"no ok merge under {wdir} — adjudication is about the MERGED "
            "trajectory (and its seam verdicts need the committed renders, so "
            "run merge and apply first)")

    # the trajectory being judged is the merged one — the poses `apply` wrote.
    merged = read_json(os.path.join(wdir, "merged_poses.json"))
    frames = (merged or {}).get("frames") or {}
    scale = (merged or {}).get("scale", 1.0)
    queue = _seam_queue(report, plan_doc)

    lines = []
    if not check:
        lines += _forced_reads(merged, run_dir)

    calls_path = os.path.join(wdir, SEAM_CALLS_FILE)
    existing = read_json(calls_path) if os.path.isfile(calls_path) else None
    prior = {calls_lib.step_key(c): c
             for c in ((existing or {}).get("seam_calls") or [])
             if isinstance(c, dict)}

    # temporal order by the step's LATER frame, so the queue reads down the
    # timeline the way the tables above do.
    ordered = sorted(queue, key=lambda k: (frames_lib.frame_number(k[1]) or 0,
                                           k[0]))
    stamped = calls_lib.template(
        [queue[k]["step"] or k for k in ordered], frames, scale)
    for record, key in zip(stamped, ordered):
        record["reason"] = queue[key]["reason"]
        if queue[key].get("from_window"):
            record["from_window"] = queue[key]["from_window"]
        if queue[key].get("refiner_said"):
            record["refiner_said"] = queue[key]["refiner_said"]
        was = prior.get(key)
        if was and str(was.get("pose_hash") or "") == record["pose_hash"]:
            record["verdict"] = str(was.get("verdict") or "")
            record["evidence"] = str(was.get("evidence") or "")

    result = calls_lib.validate(stamped, ordered, frames, scale,
                                allow_flip=True)
    uncalled = [calls_lib.step_key(r) for r in stamped if not r["verdict"]]

    # A gate does not get to erase the record it gates. Bare `--check` (the
    # documented way to gate) knows no --pass-dir; only the run that looked writes
    # these three fields.
    sheets_dir = _seam_sheet_dir(wdir)
    draw_error = ""
    if check:
        keep = existing if isinstance(existing, dict) else {}
        provenance = {
            "pass_dir": str(keep.get("pass_dir") or ""),
            "seam_sheet": str(keep.get("seam_sheet") or ""),
            "seam_sheet_manifest": str(keep.get("seam_sheet_manifest") or ""),
        }
    else:
        manifest_path, draw_error = (
            _draw_seam_sheet(run_dir, sheets_dir, pass_dir, ordered)
            if ordered else ("", ""))
        provenance = {
            "pass_dir": pass_dir or "",
            "seam_sheet": _seam_sheet_line(run_dir, pass_dir, ordered,
                                           sheets_dir) if ordered else "",
            "seam_sheet_manifest": manifest_path,
        }

    doc = {
        "schema_version": SCHEMA_VERSION,
        "iteration": plan_doc.get("iteration"),
        "scene_sha1": plan_doc["scene_sha1"],
        **provenance,
        "seam_calls": stamped,
    }
    write_json(calls_path, doc)

    ok = not result["problems"] and not result["blocking"]
    doc["ok"] = ok
    if ordered:
        lines += _adjudicate_lines(doc, result, uncalled, calls_path, check,
                                   draw_error=draw_error)
    else:
        lines.append("[windows] no seams and nothing routed up — every step of "
                     "this round was interior to a window and called by its "
                     "refiner. Nothing to adjudicate.")
    if as_json:
        print(json.dumps(doc, indent=2))
    else:
        for line in lines:
            print(line)
    return doc


def _forced_reads(merged, run_dir):
    """The reads adjudication is not allowed to skip, as OUTPUT.

    ONE computed report, printed once per `FORCED_READS` entry plus the seams-only
    view — display orders over the same rows, so none hides a row without counting
    it. Several orders because a defect hides in whichever column nobody sorted by
    (see FORCED_READS). An order is not a verdict."""
    lines = []
    try:
        table = temporal_report.build(merged, run_dir=run_dir)
    except (ValueError, KeyError) as exc:
        return [f"[windows] temporal table unavailable: {exc}"]
    for sort_by, why in FORCED_READS:
        lines.append(f"=== FULL SEQUENCE, biggest |{sort_by}| first "
                     + "=" * max(3, 44 - len(sort_by)))
        lines.append(why)
        lines += temporal_report.table_lines(table, sort_by=sort_by)
        lines.append("")
    lines.append("=== SEAMS ONLY, in temporal order =========================="
                 "=================")
    lines += temporal_report.table_lines(table, seams_only=True)
    lines.append("")
    lines.append(f"Those are the SAME rows in {len(FORCED_READS) + 1} orders, "
                 "not that many findings. Nothing here is thresholded or "
                 "flagged: a small number is not a clearance and a large one is "
                 "not a defect. Read every column of a step you are about to "
                 "call — report.md § 'Reading the table' says what each column "
                 "is telling you.")
    lines.append("")
    return lines


def _adjudicate_lines(doc, result, uncalled, calls_path, check, draw_error=""):
    """Human lines for the seam queue: what to call, how to look, what is missing.

    Steps, NOT magnitudes: one number beside a step would answer the question the
    verdict is supposed to ask. They are in the tables above and on each record's
    `measured`."""
    stamped = doc["seam_calls"]
    lines = [f"[windows] {len(stamped)} step(s) are YOURS to call "
             f"({len(uncalled)} uncalled) — no refiner owned them:"]
    for record in stamped:
        why = ("crosses two windows' poses"
               if record["reason"] == "seam"
               else f"routed up from {record.get('from_window', '?')}: "
                    f"{record.get('refiner_said', '')[:70]}")
        state = ("UNCALLED" if not record["verdict"]
                 else f"{record['verdict']}: {record['evidence'][:50]}")
        lines.append(f"  {record['from']} -> {record['to']}  [{state}]")
        lines.append(f"      why yours: {why}")
    if draw_error:
        lines += ["",
                  f"SEAM SHEET NOT DRAWN ({draw_error}) — draw it before calling "
                  "anything; a verdict with no picture behind it is a guess:",
                  f"  {doc.get('seam_sheet', '')}"]
    elif doc.get("seam_sheet_manifest"):
        page = ""
        manifest = read_json(doc["seam_sheet_manifest"])
        for path in (manifest or {}).get("pages") or []:
            page = path
            break
        lines += ["",
                  "DRAWN for you — OPEN it. The source frames AND the committed "
                  "renders, one band per seam (both basins look smooth in the "
                  "video; only the renders show which side flipped):",
                  f"  {page or doc['seam_sheet_manifest']}",
                  f"  manifest (cite this in `evidence`): "
                  f"{doc['seam_sheet_manifest']}",
                  "  redraw to vary it (--context 1 for the local trend, "
                  "--seams-per-page 1 to go in close):",
                  f"    {doc['seam_sheet']}"]
    elif doc.get("seam_sheet"):
        lines += ["",
                  "LOOK at them together — the source frames AND the committed "
                  "renders (both basins look smooth in the video; only the "
                  "renders show which side flipped):",
                  f"  {doc['seam_sheet']}"]
    lines += [
        "",
        f"Write a verdict + evidence for each in {calls_path}:",
        "  coherent — name what the pictures SHOW. When the two basins tie by "
        "eye, keep the side continuous with the neighbors and SAY that is what "
        "you did.",
        "  flip     — an action item: cluster confirmed flips into a contiguous "
        "segment, check the shared un-flip with oapply_all on just that segment, "
        "commit, then re-plan the segment with --guidance naming the settled "
        "basin so refiners polish within it instead of re-searching.",
        "  unsure   — BLOCKS. There is no override, on purpose: an unsure you "
        "can flag past is not a blocker. Go and look again.",
    ]
    if result["problems"]:
        lines.append("")
        lines.append("NOT YET ADJUDICATED:")
        lines += calls_lib.problem_lines(result["problems"], what="seam")
    if result["blocking"]:
        lines.append(f"  {len(result['blocking'])} step(s) called `unsure` — "
                     "these block until you can say what you see")
    lines.append("")
    lines.append(f"[windows] adjudicate: {'PASS' if doc['ok'] else 'BLOCKED'}"
                 + ("" if check else "  (re-run with --check for the gate's "
                                     "exit code)"))
    return lines


# --------------------------------------------------------------------------- #
def status(run_dir):
    run_dir = os.path.abspath(run_dir)
    wdir = _active_workspace(run_dir)
    plan_doc = read_json(os.path.join(wdir, "plan.json"))
    if not isinstance(plan_doc, dict):
        raise ValueError(f"no plan.json under {wdir} — run `plan` first")
    out = {}
    for w in plan_doc["windows"]:
        frag = load_fragment(run_dir, w["id"], wdir=wdir)
        out[w["id"]] = {
            "frames": len(w["frames"]),
            "fragment": frag is not None,
            "fragment_frames": len(frag.get("frames") or {}) if frag else 0,
        }
    for wid, info in out.items():
        log(f"{wid}: {'DONE' if info['fragment'] else 'pending'} "
            f"({info['fragment_frames']}/{info['frames']} frames)")
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("plan", help="split frames into refinement windows")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--windows", type=int, default=None,
                    help="number of windows (default: ~6 frames per window)")
    sp.add_argument("--frames-per-window", type=int, default=None,
                    help="target frames per window (overrides --windows)")
    sp.add_argument("--known-issues", default=None,
                    help='JSON file: {"<frame>.jpg": "<one-line finding about '
                         'the committed pose>"}; each finding lands in the '
                         "owning window's brief and is snapshotted into the "
                         "iteration's windows/known_issues.json")
    sp.add_argument("--review-artifact", action="append", default=None,
                    metavar="FILE",
                    help="review input supporting this plan (for example a "
                         "pose review sheet); repeat as needed. Files are "
                         "copied into the iteration's windows/review/ directory "
                         "and recorded in plan.json")
    sp.add_argument("--frames", default=None,
                    help="plan only a CONTIGUOUS segment of the trajectory: a "
                         "comma list or an inclusive '<first>-<last>' range "
                         "of layout frames (default: all frames). Frames "
                         "outside the segment keep their committed poses and "
                         "bracket the plan as advisory neighbors")
    sp.add_argument("--guidance", action="append", default=None,
                    metavar="TEXT|wNN=TEXT",
                    help="free-text orchestrator intent rendered into the "
                         "brief's 'Orchestrator guidance' section, where it "
                         "OVERRIDES the general rules: emphasis, hypotheses, "
                         "context from previous rounds, and what kind of round "
                         "this is. A reconciliation round says so here (e.g. "
                         "'the basin is settled — polish within it, report a "
                         "disagreement, do not re-search facing'); there is no "
                         "flag for it, because what verifies a round is your "
                         "own read of the temporal report at adjudicate. A bare "
                         "TEXT applies to every window; repeat with wNN=TEXT to "
                         "differ per window")
    sp.add_argument("--seed-from", action="append", default=None,
                    metavar="ITER_DIR",
                    help="carry a previous iteration's fragment poses into "
                         "this plan as SEED HYPOTHESES "
                         "(iterations/NNNNNN/windows; "
                         "repeatable, later dirs win). Seeds land in the "
                         "owning window's brief as unverified starting "
                         "candidates — they carry no scores and no authority, "
                         "and never merge on their own. Joint states are "
                         "reconciled against the current JOINTS (unknown "
                         "joints dropped, values clamped to limits)")
    sp.add_argument("--seed-from-iteration", action="append", default=None,
                    metavar="previous|NNNNNN",
                    help="explicitly seed from a numbered history iteration "
                         "or the latest one; base poses carry automatically and "
                         "joint states are conservatively reconciled against "
                         "the current kinematics")
    sc = sub.add_parser("selfcheck",
                        help="a refiner's OWN temporal read of its window "
                             "(the whole table + its call sheet) before it "
                             "commits")
    sc.add_argument("--run-dir", required=True)
    sc.add_argument("--window", required=True,
                    help="window id to check (e.g. w03)")
    sc.add_argument("--json", action="store_true",
                    help="emit the full report as JSON instead of human lines")
    sm = sub.add_parser("merge", help="validate + merge window fragments")
    sm.add_argument("--run-dir", required=True)
    sm.add_argument("--allow-missing", action="store_true",
                    help="merge even if some windows have no fragment yet")
    sa = sub.add_parser("apply",
                        help="rewrite scene.py FRAMES from the merged trajectory")
    sa.add_argument("--run-dir", required=True)
    sj = sub.add_parser(
        "adjudicate",
        help="the ORCHESTRATOR's verdicts on the steps no refiner owned: prints "
             "the forced temporal reads, emits/refreshes seam_calls.json, and "
             "with --check exits non-zero until every one is called")
    sj.add_argument("--run-dir", required=True)
    sj.add_argument("--pass-dir", default=None,
                    help="the verification pass whose committed renders the "
                         "seam sheet should draw. Strongly recommended: both "
                         "basins look smooth in the source video, so only the "
                         "renders show which side of a seam flipped")
    sj.add_argument("--check", action="store_true",
                    help="THE GATE: exit non-zero unless every seam step is "
                         "called, fresh, and not `unsure` (no override exists)")
    sj.add_argument("--json", action="store_true",
                    help="emit seam_calls.json's content instead of human lines")
    ss = sub.add_parser("status", help="which windows have fragments")
    ss.add_argument("--run-dir", required=True)
    args = p.parse_args()

    if args.cmd == "plan":
        plan(args.run_dir, n_windows=args.windows,
             frames_per_window=args.frames_per_window,
             known_issues=args.known_issues,
             review_artifacts=args.review_artifact,
             frames=args.frames, guidance=args.guidance,
             seed_from=args.seed_from,
             seed_from_iteration=args.seed_from_iteration)
    elif args.cmd == "selfcheck":
        selfcheck(args.run_dir, args.window, as_json=args.json)
    elif args.cmd == "merge":
        merge(args.run_dir, allow_missing=args.allow_missing)
    elif args.cmd == "apply":
        apply(args.run_dir)
    elif args.cmd == "adjudicate":
        doc = adjudicate(args.run_dir, pass_dir=args.pass_dir,
                         check=args.check, as_json=args.json)
        # the exit code IS the gate — every other command returns 0
        if args.check and not doc.get("ok"):
            return 1
        if args.check:
            current = bookkeeping.active(args.run_dir, required=True)
            if current.get("kind") != "pose":
                raise ValueError(
                    f"adjudication belongs to a pose iteration, but active "
                    f"iteration {current['name']} is {current.get('kind')}")
            completed = bookkeeping.complete(
                args.run_dir, current["iteration"])
            log(f"completed global iteration {completed['name']} at "
                f"{completed['git_commit'][:12]}")
    else:
        status(args.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
