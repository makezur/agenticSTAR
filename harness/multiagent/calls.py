#!/usr/bin/env python3
"""calls.py — temporal verdicts as a checkable artifact: stamp them, check them.

THE INVARIANT this module serves:

    Every temporal step has exactly one owner. A step INTERIOR to a window is
    the refiner's; a step CROSSING ownership (a seam) is the orchestrator's.
    Every owner writes a verdict on their steps, and the verdict is stamped
    with a pose hash, so re-posing either side stales it mechanically.

Both halves of that invariant are here because both need the same three things
(the vocabulary, the "is every step called" bookkeeping, the freshness check),
and the halves must not drift: a refiner's `step_calls` over its interior steps
(multiagent.windows merge) and the orchestrator's `seam_calls` over the seams
(multiagent.windows adjudicate) are the SAME artifact at two ownership scopes.

THE VOCABULARY IS THREE WORDS, and the choice between them is the whole design:

  * `coherent` — with EVIDENCE, in prose, that names what the pictures show:
    the visible motion ("the lid swings clear of the rim"), or "static", or
    "the two basins tie here; kept the side continuous with the neighbors".
    The smoothness preference lives INSIDE this verdict, as a tie-break for
    when the eye reports a tie — never as a threshold, so a genuinely wild
    motion the video shows is `coherent` with the motion named, and the number
    being large is not itself an argument.
  * `flip` — an ACTION ITEM, not an explanation. It feeds segment diagnosis and
    un-flipping. A refiner may not ship `flip` on its own interior step: inside
    its own window it can re-pose the frame, so filing a flip it could fix is
    just deferring work to someone with less context.
  * `unsure` — BLOCKS. There is deliberately no escape flag. A refiner's
    `unsure` does not block the merge; it routes UP into the orchestrator's
    queue, so blocking happens at exactly one place (adjudicate) instead of at
    every layer.

NO THRESHOLDS LIVE HERE, and none can: nothing in this module reads a residual
magnitude. It checks that a step was called, that the call is well-formed, and
that it is still about the poses that are in the trajectory. Whether 47 degrees
is fine is a question for eyes, and the answer's only home is `evidence`.

Pure stdlib (freshness is too), so the process side can stamp and check without
importing the measurement stack.
"""

from multiagent import freshness

# The whole vocabulary. Anything else in a `verdict` field is a malformed call,
# not a fourth opinion — a private word ("probably_ok") would read as a decision
# while being invisible to every check here.
VERDICTS = ("coherent", "flip", "unsure")

# The verdicts that leave work behind, by owner scope:
#   * `unsure` always does — it is the blocking one;
#   * `flip` does for the orchestrator (it is the input to un-flipping) but is a
#     VIOLATION for a refiner on its own interior step (fix it, don't file it).
BLOCKING = ("unsure",)


def step_key(call):
    """A call's identity: `(from, to)` — the same key analysis.temporal.seams
    uses for a measured step, so a call and the step it is about match up by
    construction rather than by two modules agreeing on a format."""
    return (str(call.get("from")), str(call.get("to")))


def consecutive_steps(order):
    """The steps a frame `order` implies: every consecutive pair.

    This is what "every step is owned" is checked against, so it is deliberately
    dumb — the caller decides which frames are in scope (a window's own frames
    for `step_calls`, the seam list for `seam_calls`), and this only pairs them
    up."""
    order = list(order)
    return [(a, b) for a, b in zip(order[:-1], order[1:])]


def template(steps, frames, scale, note=None):
    """Pre-stamped, UNCALLED call records for `steps` — the artifact an owner
    fills in.

    `steps` — measured transition records (analysis.temporal.sequence) or bare
        `(from, to)` pairs.
    `frames` — {frame_name: pose.json-shaped entry} the hash is taken over.
    `scale` — the run's shared scale.
    `note` — optional one-line instruction stored on every record (e.g. the
        seam_sheet command that draws this step).

    Every record arrives with `pose_hash` and the measured numbers already
    filled and `verdict`/`evidence` EMPTY. That asymmetry is the point: the
    owner is asked for exactly the thing no tool can produce — a sentence about
    what the pictures show — and cannot be asked to retype a number, which is
    how a hand-copied hash ends up stamped to the wrong pose.
    """
    out = []
    for step in steps:
        if isinstance(step, (tuple, list)):
            frm, to, measured = str(step[0]), str(step[1]), None
        else:
            frm, to = step["from"], step["to"]
            pose = step.get("pose") or {}
            measured = {
                "rotation_deg": pose.get("rotation_deg"),
                "translation_canon": pose.get("translation_dist_canon"),
                "joints": {name: j.get("delta")
                           for name, j in (step.get("joints") or {}).items()},
                "frame_gap": step.get("frame_gap"),
            }
        record = {
            "from": frm,
            "to": to,
            "verdict": "",
            "evidence": "",
            "pose_hash": freshness.pose_hash(frames.get(frm), frames.get(to),
                                             scale),
        }
        if measured is not None:
            record["measured"] = measured
        if note:
            record["note"] = note
        out.append(record)
    return out


def _one_call(call, required, frames, scale, allow_flip):
    """Validate ONE call; returns a problem dict or None.

    Order matters: shape before freshness before verdict. A call missing its
    verdict should say so rather than first failing a hash check the owner
    cannot act on."""
    key = step_key(call)
    if key not in required:
        return {"kind": "step_not_owned", "from": key[0], "to": key[1],
                "detail": "this step is not one of the steps you own"}
    verdict = str(call.get("verdict") or "").strip().lower()
    if not verdict:
        return {"kind": "uncalled", "from": key[0], "to": key[1],
                "detail": "no verdict written"}
    if verdict not in VERDICTS:
        return {"kind": "bad_verdict", "from": key[0], "to": key[1],
                "detail": f"{verdict!r} is not one of {', '.join(VERDICTS)}"}
    if not str(call.get("evidence") or "").strip():
        # required for `unsure` too: an unsure with no words tells the person
        # inheriting it nothing about what to look at.
        return {"kind": "no_evidence", "from": key[0], "to": key[1],
                "detail": f"a {verdict!r} call must say what you saw"}
    if verdict == "flip" and not allow_flip:
        return {"kind": "flip_on_own_step", "from": key[0], "to": key[1],
                "detail": "a flip on a step you own is yours to FIX (re-pose "
                          "the frame), not to file"}
    fresh = freshness.pose_hash(frames.get(key[0]), frames.get(key[1]), scale)
    if str(call.get("pose_hash") or "") != fresh:
        return {"kind": "stale_call", "from": key[0], "to": key[1],
                "detail": f"stamped {call.get('pose_hash')!r}, poses now hash "
                          f"{fresh!r} — one of these frames moved since the "
                          "verdict was written, so it is uncalled again"}
    return None


def validate(calls, required, frames, scale, allow_flip=False):
    """Check a list of calls against the steps their owner must call.

    `calls` — the owner's call records.
    `required` — the `(from, to)` steps in this owner's scope.
    `frames` / `scale` — what the freshness hash is recomputed from.
    `allow_flip` — True for an ownership-crossing scope (a `flip` seam verdict
        is the orchestrator's action item); False inside one owner's own steps.

    Returns {"problems": [...], "blocking": [...], "called": [...]} —
    `problems` is every malformed / missing / stale call (the caller decides
    whether that fails a merge or an exit code), `blocking` is the well-formed
    calls whose verdict leaves work behind (`unsure`), and `called` is the
    normalized well-formed calls.

    Deciding nothing itself is deliberate: merge turns problems into violations
    and routes `blocking` upward, adjudicate turns the same two lists into an
    exit code. One rule set, two policies.
    """
    required = {(str(a), str(b)) for a, b in required}
    problems, blocking, called = [], [], []
    seen = {}
    for call in calls or []:
        if not isinstance(call, dict):
            problems.append({"kind": "malformed_call", "detail": str(call)})
            continue
        key = step_key(call)
        if key in seen:
            problems.append({"kind": "duplicate_call", "from": key[0],
                             "to": key[1],
                             "detail": "the same step is called twice; a step "
                                       "has exactly one owner and one verdict"})
            continue
        seen[key] = call
        problem = _one_call(call, required, frames, scale, allow_flip)
        if problem is not None:
            problems.append(problem)
            continue
        norm = {"from": key[0], "to": key[1],
                "verdict": str(call["verdict"]).strip().lower(),
                "evidence": str(call["evidence"]).strip(),
                "pose_hash": call["pose_hash"]}
        if call.get("measured") is not None:
            norm["measured"] = call["measured"]
        called.append(norm)
        if norm["verdict"] in BLOCKING:
            blocking.append(norm)
    for key in sorted(required - set(seen)):
        problems.append({"kind": "uncalled", "from": key[0], "to": key[1],
                         "detail": "no verdict on this step at all"})
    return {"problems": problems, "blocking": blocking, "called": called}


def problem_lines(problems, what="step"):
    """Human lines for a validate() problem list — one per problem, naming the
    step, because "3 violations" sends a reader back to the JSON to find out
    which."""
    lines = []
    for p in problems:
        where = (f"{p['from']} -> {p['to']}: " if p.get("from") else "")
        lines.append(f"  {what} {where}{p['kind']} — {p.get('detail', '')}")
    return lines
