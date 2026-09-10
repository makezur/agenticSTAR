#!/usr/bin/env python3
"""mechanism_calls CLI — pose and GRADE the mechanism's blind A/B choice.

The mechanism page sweeps every joint twice — once about the declared `axis`,
once about its negation — onto blind `A`/`B` labels, and asks which arc is the
real mechanism. The harness knows the answer, so the pick is graded rather than
accepted: picking the mirror BLOCKS, with the sign to write spelled out.
`core/mechanism_calls.py` holds the schema, the declaration hash, the blind arm
assignment, and the grading; this is the front door.

    # after AUTHORING or CHANGING a joint, render + sheet it (shape_pass does
    # both), then ask the question:
    python -m analysis.mechanism_calls ask --run-dir RUN_DIR \\
        --pass-dir <the pass with mechanism_sheets/>

    # answer it in RUN_DIR/mechanism_calls.json (pick + evidence per joint):
    python -m analysis.mechanism_calls check --run-dir RUN_DIR

`ask` emits/refreshes `RUN_DIR/mechanism_calls.json`, pre-stamped from the pass's
sheet manifest with the pick blank; an answer survives a re-emit if its joint's
declaration hash still matches, so answering is owed only when a declaration
actually changed — NOT once per pass. `check` is the gate: exit non-zero until
every articulated joint's pick is fresh, evidenced from the INTERIOR states, and
correct. No override exists; `multiagent.windows plan` and the finalization lock
both run it (AGENT_TASK.md).

Neither subcommand ever prints which arm is the declared one — that is the
answer, and this tool is the examiner. It lands in the output only AFTER a wrong
pick, as the axis to write instead.

Pure analysis-env tool (no bpy). Run from harness/; `--help` on any subcommand.
"""

import argparse
import json
import os
import sys

from analysis.lib.io import read_json, write_json
from core import mechanism_calls as mc
from core import mechanism_views as mv
from core import modules


def log(msg):
    print(f"[mechanism-calls] {msg}", flush=True)


def _joint_defs(run_dir):
    """The COMMITTED declaration — mesh/pose.json's joint_defs. That is the rig
    a pick is evidence about; a pass-local copy could be a render of an edit
    that was never committed."""
    pose = read_json(os.path.join(run_dir, "mesh", "pose.json"))
    return ((pose or {}).get("joint_defs") or (pose or {}).get("JOINTS") or [])


def _manifest(pass_dir):
    path = os.path.join(os.path.abspath(pass_dir), "mechanism_sheets",
                        "mechanism_sheet_manifest.json")
    doc = read_json(path)
    if not isinstance(doc, dict) or not doc.get("joints"):
        raise ValueError(
            f"no mechanism sheet manifest under {pass_dir} (looked for "
            f"{path}) — render the mechanism view and build its sheets first "
            "(shape_pass.sh does both; or analysis.viz.mechanism_sheet on a "
            "pass holding mechanism renders)")
    return doc


def ask(run_dir, pass_dir, as_json=False):
    """Emit/refresh the pick template from `pass_dir`'s sheet manifest.

    Refuses a manifest whose declarations disagree with the committed ones — a
    pick recorded against a stale render would be born stale, and the tool
    saying so now beats `check` saying so after the reader argued a case.
    """
    run_dir = os.path.abspath(run_dir)
    if not modules.enabled(run_dir, "mechanism"):
        log(modules.disabled_line("mechanism"))
        log("no question to ask; nothing written")
        return None
    manifest = _manifest(pass_dir)
    joint_defs = _joint_defs(run_dir)
    by_name = {j.get("name"): j for j in joint_defs if j.get("name")}

    stale = []
    for entry in manifest.get("joints") or []:
        committed = by_name.get(entry.get("joint"))
        if committed is None:
            stale.append(f"{entry.get('joint')!r} is on the sheet but not in "
                         "the committed pose.json")
        elif (mc.declaration_hash(mc.rehydrate(entry.get("definition") or {},
                                               entry.get("axis_line")))
                != mc.declaration_hash(committed)):
            stale.append(f"{entry.get('joint')!r} was rendered from a "
                         "different declaration than the committed one")
    if stale:
        raise ValueError(
            "this pass's mechanism sheets do not show the committed rig:\n  "
            + "\n  ".join(stale)
            + f"\nrun a fresh pass (shape_pass.sh {run_dir} ...) and answer "
            "that one — a pick over a stale render would be born stale")

    prior = mc.read_calls(run_dir)
    doc = {
        "schema_version": mc.SCHEMA_VERSION,
        "pass_dir": os.path.abspath(pass_dir),
        "joints": mc.template(manifest, prior_doc=prior),
    }
    calls_path = os.path.join(run_dir, mc.CALLS_FILE)
    write_json(calls_path, doc)

    open_count = sum(1 for e in doc["joints"] if not e["pick"])
    log(f"{len(doc['joints'])} joint(s), {open_count} unanswered -> {calls_path}")
    for entry in doc["joints"]:
        arms = "/".join(entry["arms"])
        state = ("ANSWERED (still fresh)" if entry["pick"]
                 else f"pick one of {arms}")
        log(f"  {entry['joint']} [{entry['decl_hash']}]: {state}")
        if not entry["pick"] and entry.get("sheet"):
            log(f"    open {entry['sheet']}")
    if open_count:
        log("Which arm's arc is the real mechanism? The arms are mirror images "
            "and are IDENTICAL at both endpoints — read the INTERIOR tiles, "
            "where one swings the child into free space and the other drives it "
            "through the body.")
        log(f"  pick     — {'/'.join(mv.ARMS)}, or {mc.UNSURE} (blocks, no "
            "override)")
        log(f"  evidence — name >= {mc.MIN_INTERIOR_STATES} INTERIOR states and "
            "what the child does at each; cite landmark parts, not impressions")
        log(f"then: python -m analysis.mechanism_calls check --run-dir {run_dir}")
    if as_json:
        print(json.dumps(doc, indent=2))
    return doc


def check(run_dir, as_json=False):
    """THE GATE: every articulated joint's pick fresh, evidenced from the
    interior states, and CORRECT. Returns the result; main() turns `ok` into the
    exit code."""
    run_dir = os.path.abspath(run_dir)
    result = mc.check_run(run_dir, joint_defs=_joint_defs(run_dir))
    if as_json:
        print(json.dumps(result, indent=2))
        return result
    if result.get("disabled"):
        log(modules.disabled_line("mechanism") + ". PASS")
        return result
    if not result["joints"]:
        log("no articulated joints — a rigid object has no mechanism to "
            "answer for. PASS")
        return result
    if result["ok"]:
        log(f"all {len(result['joints'])} articulated joint(s): the arm picked "
            "off the sheet IS the declared one. PASS")
        return result
    log(f"BLOCKED — {len(result['problems'])} problem(s), "
        f"{len(result['blocking'])} wrong/unsure pick(s):")
    for line in mc.problem_lines(result):
        print(line)
    log(f"answer in {result['calls_path']} "
        "(python -m analysis.mechanism_calls ask --run-dir "
        f"{run_dir} --pass-dir <pass with mechanism_sheets/>)")
    return result


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser(
        "ask",
        help="emit/refresh RUN_DIR/mechanism_calls.json from a pass's mechanism "
             "sheets — one blind A/B question per joint, pick blank; fresh "
             "answers are carried forward")
    sm.add_argument("--run-dir", required=True)
    sm.add_argument("--pass-dir", required=True,
                    help="a pass holding mechanism_sheets/ rendered from the "
                         "COMMITTED declaration (shape_pass.sh builds them)")
    sm.add_argument("--json", action="store_true",
                    help="also emit the calls document as JSON")
    sc = sub.add_parser(
        "check",
        help="THE GATE: exit non-zero unless every articulated joint's pick is "
             "fresh, evidenced from the interior states, and the DECLARED arm "
             "(no override exists)")
    sc.add_argument("--run-dir", required=True)
    sc.add_argument("--json", action="store_true",
                    help="emit the result as JSON instead of human lines")
    args = p.parse_args()

    if args.cmd == "ask":
        ask(args.run_dir, args.pass_dir, as_json=args.json)
        return 0
    result = check(args.run_dir, as_json=args.json)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
