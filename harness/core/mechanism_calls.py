"""mechanism_calls.py — the mechanism arc as a BLIND FORCED CHOICE.

Every joint is swept twice — the declared `axis` and its negation — onto blind
`A`/`B` labels, and the one question is which arm the real object articulates
like. The harness knows the answer, so the pick is GRADED. Why a forced choice
beats "does this look right", and how to read the page: views/mechanism.md.

  * `A` / `B` — the arm showing the real articulation, plus evidence naming
    >= MIN_INTERIOR_STATES interior states (the endpoints are identical under
    either sign, so an argument from them says nothing).
  * `unsure` — BLOCKS, no override; the fix is another viewpoint.

Freshness covers what would change the ARCS: `type`, `origin`, `limit`, `child`,
`parent`, and the axis's unsigned LINE — NOT its sign (`unsigned_axis` explains
the flip-flop that taught this). So a pass that touched no joint owes nothing,
and fixing a sign settles the existing pick instead of re-asking it.

Consumed by `analysis.mechanism_calls` (the CLI) and `multiagent.windows plan`.
Pure stdlib + core.joints, so both envs can stamp and check.
"""

import hashlib
import json
import math
import os

from core import joints as joints_core
from core import mechanism_views
from core import modules

SCHEMA_VERSION = 2

# The artifact's one home: RUN_DIR/mechanism_calls.json. Run-level, not
# pass-level, because its lifetime is the DECLARATION's, not a pass's — the
# freshness hash, not the pass counter, is what retires a pick.
CALLS_FILE = "mechanism_calls.json"

# A pick is an arm label or an admission. Anything else is malformed, not a
# third opinion.
UNSURE = "unsure"

# The states the pick must be justified against: a reader who names only the
# endpoints has named the two tiles that look identical under either sign, so
# the evidence must quote an INTERIOR state — the ones that carry the sign.
MIN_INTERIOR_STATES = 3

# Quantized at the wire precision so a declaration that round-trips through
# pose.json hashes the same on both sides (freshness.QUANT_DP's reasoning).
QUANT_DP = 6

# Short like freshness.HASH_LEN, same threat model: a verdict that quietly
# went stale, not an adversary forging one.
HASH_LEN = 12

# Bumped only if WHAT is hashed changes (which re-stales every pick in
# flight — the honest outcome, and the number is how a reader sees why).
PAYLOAD_VERSION = 1


def _component(value):
    """One scalar at wire precision; `+ 0.0` folds -0.0 onto 0.0 (json spells
    them differently, and a sign-of-zero round trip is not a moved axis)."""
    return round(float(value), QUANT_DP) + 0.0


def _unit_axis(axis):
    """The axis at wire precision, NORMALIZED to unit length. Re-spelling
    (0,0,1) as (0,0,2) is the same direction. A zero/malformed axis is kept
    verbatim: it is its own kind of wrong and two of them are the same wrong."""
    try:
        vec = [float(v) for v in axis]
    except (TypeError, ValueError):
        return None
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-9:
        return [_component(v) for v in vec]
    return [_component(v / norm) for v in vec]


def unsigned_axis(axis):
    """The axis's LINE — unit length, first non-zero component positive, so
    `(1,0,0)` and `(-1,0,0)` agree.

    What freshness hashes, because the two arms ARE the two signs of one line: a
    sign fix re-renders the same pair of arcs. Hashing the sign instead staled
    the pick the moment it was acted on AND re-labelled the arcs, so a correct
    answer came back wrong with the opposite fix — an infinite flip-flop,
    observed twice on one run.
    """
    vec = _unit_axis(axis)
    if vec is None:
        return None
    for v in vec:
        if abs(v) > 1e-9:
            return [_component(-x) for x in vec] if v < 0 else vec
    return vec                            # a zero axis has no direction to fix


def axis_is_canonical(joint_def):
    """Whether the DECLARED axis points along its canonical line. The one bit the
    page must not reveal — it is the answer."""
    declared = _unit_axis(joint_def.get("axis"))
    return declared is None or declared == unsigned_axis(joint_def.get("axis"))


def _child_group(child):
    """The child group as a SORTED name list — a group is a set, so reordering
    it is not a different mechanism."""
    if child is None:
        return None
    if isinstance(child, (list, tuple)):
        return sorted(str(c) for c in child)
    return [str(child)]


def declaration_hash(joint_def):
    """The freshness stamp: (type, axis LINE, origin, limit, child, parent).

    Two hashes agree exactly when the page would show the SAME TWO ARCS — hence
    the unsigned line (`unsigned_axis`). Deliberately NOT a scene.py content
    hash: gratuitous staleness trains re-stamping without re-looking
    (freshness.py's rule).
    """
    limit = joints_core.joint_limit(joint_def, strict=False)
    origin = joint_def.get("origin")
    try:
        origin = [_component(v) for v in origin] if origin is not None else None
    except (TypeError, ValueError):
        origin = None
    payload = {
        "v": PAYLOAD_VERSION,
        "type": str(joint_def.get("type", "fixed")),
        "axis": unsigned_axis(joint_def.get("axis")),
        "origin": origin,
        "limit": None if limit is None else [_component(v) for v in limit],
        "child": _child_group(joint_def.get("child")),
        "parent": (str(joint_def["parent"])
                   if joint_def.get("parent") is not None else None),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:HASH_LEN]


def canonical_arm(joint_def, arms=mechanism_views.ARMS):
    """Which arm label carries the axis's CANONICAL direction.

    Keyed to the sign-blind hash, so the label is glued to a PHYSICAL arc and
    survives a sign fix. Not a convention: the last nibble decides, so it varies
    joint to joint.
    """
    return arms[int(declaration_hash(joint_def)[-1], 16) % len(arms)]


def rehydrate(definition, axis_line):
    """A hashable joint definition from a BLINDED manifest entry.

    The sheet manifest publishes the definition without its signed `axis` plus
    the unsigned `axis_line` (mechanism_sheet._blind_definition). Since the hash
    is sign-blind, that pair carries everything freshness needs — this is the one
    place that re-assembles it, so no caller is tempted to reach for the sign."""
    out = dict(definition or {})
    if axis_line is not None:
        out["axis"] = list(axis_line)
    return out


def declared_arm(joint_def, arms=mechanism_views.ARMS):
    """WHICH arm shows the arc the DECLARATION produces: the canonical arm, or
    its mirror when the declared axis is the negation. Flipping a sign swaps this
    WITHOUT re-labelling the arcs, so a pick stays valid and becomes correct."""
    canon = canonical_arm(joint_def, arms=arms)
    return canon if axis_is_canonical(joint_def) else mirror_arm(canon, arms=arms)


def mirror_arm(arm, arms=mechanism_views.ARMS):
    """The other arm label."""
    return arms[(list(arms).index(str(arm)) + 1) % len(arms)]


def arm_axes(joint_def, arms=mechanism_views.ARMS):
    """`{arm_label: axis}` — the two signs of the axis LINE on their labels; the
    renderer's one source for what to sweep. Keyed to the CANONICAL direction, so
    fixing a sign re-renders a byte-identical pair."""
    canon = unsigned_axis(joint_def.get("axis")) or [0.0, 0.0, 1.0]
    canon = tuple(float(v) for v in canon)
    label = canonical_arm(joint_def, arms=arms)
    return {label: canon,
            mirror_arm(label, arms=arms): mechanism_views.negate_axis(canon)}


def template(manifest, prior_doc=None):
    """UNANSWERED pick records from a pass's sheet manifest — what the agent
    fills in. ONE entry per joint (the judged thing is which arc is the
    mechanism, a single answer), carrying the arms on offer, the swept states,
    and an empty `pick`/`evidence`. A prior pick is carried forward per
    (joint, hash); a changed declaration drops it rather than offering it for
    re-signing."""
    prior = {}
    for entry in ((prior_doc or {}).get("joints") or []):
        if isinstance(entry, dict) and entry.get("joint"):
            prior[(entry.get("joint"), entry.get("decl_hash"))] = entry

    out = []
    for entry in manifest.get("joints") or []:
        joint = entry.get("joint")
        definition = entry.get("definition") or {}
        if not definition:
            # No declaration means nothing to stamp the pick against, and a pick
            # stamped against nothing can never go stale — worse than none.
            raise ValueError(
                f"the sheet manifest carries no declaration for joint "
                f"{joint!r} (the pass had no readable pose.json), so there is "
                "nothing to stamp a pick against; render via shape_pass.sh "
                "or re-run the sheet builder on a pass that carries pose.json")
        definition = rehydrate(definition, entry.get("axis_line"))
        decl = declaration_hash(definition)
        arms = [str(a) for a in (entry.get("arms") or mechanism_views.ARMS)]
        states = sorted({_component(s.get("state", 0.0))
                         for arm in entry.get("arms_detail") or []
                         for row in arm.get("rows") or []
                         for s in row.get("states") or []})
        if not states:      # pre-arm manifest shape, or a sheet with no states
            states = sorted({_component(s.get("state", 0.0))
                             for row in entry.get("rows") or []
                             for s in row.get("states") or []})
        call = {
            "joint": joint,
            "decl_hash": decl,
            # The declaration MINUS the axis. Withholding it from the rendered
            # page while leaving it in the JSON beside that page would be
            # theatre: the axis is the answer, and `grep axis
            # mechanism_calls.json` is easier than opening an image. The line
            # (unsigned) is safe and is what freshness is about, so it travels
            # under its own key; the SIGN is what nothing exposes until the
            # pick is graded.
            "declaration": {k: definition.get(k)
                            for k in ("type", "origin", "limit", "child",
                                      "parent")
                            if definition.get(k) is not None},
            "axis_line": unsigned_axis(definition.get("axis")),
            "sheet": entry.get("page") or "",
            "arms": arms,
            "states": states,
            "pick": "",
            "evidence": "",
        }
        was = prior.get((joint, decl))
        if was:
            call["pick"] = str(was.get("pick") or "")
            call["evidence"] = str(was.get("evidence") or "")
        out.append(call)
    return out


def _norm_evidence(text):
    """Evidence normalized for comparison: case- and whitespace-folded."""
    return " ".join(str(text or "").split()).lower()


def _interior_states_named(evidence, states):
    """The INTERIOR swept states this evidence quotes, as degrees.

    The endpoints render identically under either sign, so an argument from them
    is no evidence at all — hence interior only. "+22"/"22 degrees"/"22.0" all
    count."""
    text = _norm_evidence(evidence)
    if not text or not states:
        return []
    lo, hi = min(states), max(states)
    found = []
    for state in states:
        if state in (lo, hi):
            continue                    # an endpoint proves nothing about sign
        for spelling in {f"{abs(state):g}", f"{abs(state):.0f}"}:
            if _mentions_number(text, spelling):
                found.append(state)
                break
    return found


def _mentions_number(text, number):
    """Whether `text` names `number` as a standalone figure — not as a digit
    inside a longer one (so "180" does not satisfy a claim about "18")."""
    start = 0
    while True:
        at = text.find(number, start)
        if at < 0:
            return False
        before = text[at - 1] if at else " "
        after_at = at + len(number)
        after = text[after_at] if after_at < len(text) else " "
        if not (before.isdigit() or before == ".") and \
           not (after.isdigit() or after == "."):
            return True
        start = at + 1


def validate(doc, joint_defs):
    """Grade a calls document against the CURRENT declaration.

    `problems` — missing/stale/malformed picks. `blocking` — well-formed and
    WRONG: a mirrored declaration (the sign to write is in the message) or an
    `unsure`. Decides nothing itself (calls.validate's contract): the CLI makes
    an exit code of it, `windows plan` a refusal.
    """
    entries = {}
    for entry in ((doc or {}).get("joints") or []):
        if isinstance(entry, dict) and entry.get("joint"):
            entries[str(entry["joint"])] = entry

    names = joints_core.articulated_names(joint_defs)
    by_name = {j.get("name"): j for j in (joint_defs or []) if j.get("name")}
    problems, blocking = [], []

    for name in names:
        entry = entries.get(name)
        if entry is None:
            problems.append({
                "kind": "unanswered", "joint": name,
                "detail": "no pick on this joint at all — render the mechanism "
                          "view and answer it (analysis.mechanism_calls ask)"})
            continue
        fresh = declaration_hash(by_name[name])
        if str(entry.get("decl_hash") or "") != fresh:
            problems.append({
                "kind": "stale_pick", "joint": name,
                "detail": f"picked against {entry.get('decl_hash')!r}, the "
                          f"committed declaration now hashes {fresh!r} — the "
                          "joint's type/axis/origin/limit/child changed since, "
                          "so the pick is about a rig that no longer exists "
                          "(and the arm labels may have moved with the hash); "
                          "re-render and re-answer"})
            continue
        arms = [str(a) for a in (entry.get("arms") or mechanism_views.ARMS)]
        states = [_component(s) for s in (entry.get("states") or [])]
        pick = str(entry.get("pick") or "").strip()
        evidence = " ".join(str(entry.get("evidence") or "").split())
        if not pick:
            problems.append({
                "kind": "unanswered", "joint": name,
                "detail": f"no pick written — say which of {'/'.join(arms)} "
                          "shows the real articulation"})
            continue
        if pick.lower() == UNSURE:
            blocking.append({
                "joint": name, "pick": UNSURE, "evidence": evidence,
                "detail": "unsure BLOCKS and has no override — the two arcs "
                          "are mirror images, so one of them puts the child "
                          "through the body at every interior state; add "
                          "--mechanism-rows for a viewpoint further off the "
                          "swing plane and look again"})
            continue
        if pick not in arms:
            problems.append({
                "kind": "bad_pick", "joint": name,
                "detail": f"{pick!r} is not one of {', '.join(arms)} "
                          f"or {UNSURE!r}"})
            continue
        if not evidence:
            problems.append({
                "kind": "no_evidence", "joint": name,
                "detail": f"a pick of {pick!r} must say what makes that arc the "
                          "real one and the other arc impossible — name the "
                          "interior states and what the child does there"})
            continue
        named = _interior_states_named(evidence, states)
        if states and len(named) < MIN_INTERIOR_STATES:
            problems.append({
                "kind": "endpoints_only", "joint": name,
                "detail": f"evidence names {len(named)} interior state(s); at "
                          f"least {MIN_INTERIOR_STATES} are needed. Both arms "
                          "render IDENTICALLY at the endpoints, so a pick "
                          "argued from them cites the only two tiles that "
                          "carry no information about the sign — quote the "
                          "interior angles and what the child does at each"})
            continue
        want = declared_arm(by_name[name])
        if pick != want:
            axis = by_name[name].get("axis")
            flipped = (mechanism_views.negate_axis(axis)
                       if axis is not None else None)
            blocking.append({
                "joint": name, "pick": pick, "evidence": evidence,
                "kind": "wrong_axis_sign",
                "detail": f"WRONG JOINT IMPLEMENTATION. Your pick ({pick}) "
                          f"stands. Set axis {flipped!r} in JOINTS (declared "
                          f"{tuple(axis)!r} swings the other way), then "
                          "re-render so mesh/pose.json carries it"})
    return {"problems": problems, "blocking": blocking, "joints": names}


def read_calls(run_dir):
    """RUN_DIR's calls document, or None. Stdlib on purpose — the process side
    (multiagent) reads it too, and a malformed file reads as absent rather
    than crashing the gate that counts the calls."""
    path = os.path.join(os.path.abspath(run_dir), CALLS_FILE)
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _scene_axis_signs(run_dir):
    """`{joint name: axis}` scraped from RUN_DIR/scene.py's JOINTS, or {}.

    Read as TEXT (scene.py imports bpy; this runs in the analysis env).
    Best-effort: a miss costs a slightly worse message and nothing else."""
    path = os.path.join(run_dir, "scene.py")
    try:
        with open(path) as f:
            source = f.read()
    except OSError:
        return {}
    try:
        import ast
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", "") == "JOINTS" for t in node.targets):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        for item in node.value.elts:
            if not isinstance(item, ast.Dict):
                continue
            fields = {}
            for k, v in zip(item.keys, item.values):
                key = getattr(k, "value", None)
                if key in ("name", "axis"):
                    try:
                        fields[key] = ast.literal_eval(v)
                    except (ValueError, SyntaxError):
                        pass
            if fields.get("name") and fields.get("axis") is not None:
                out[str(fields["name"])] = tuple(
                    float(x) for x in fields["axis"])
    return out


def check_run(run_dir, joint_defs=None):
    """The run-level check: RUN_DIR's calls file against its committed
    declaration (mesh/pose.json's `joint_defs` unless the caller already
    holds them). Returns validate()'s dict plus `ok` and `calls_path`.

    A rigid run (no articulated joints) is trivially ok — there is no
    mechanism to answer for, and the gate must not invent one.

    Re-words a wrong-sign block into a RE-RENDER instruction when `scene.py`
    already carries the fix: the gate grades `mesh/pose.json`, which only a render
    regenerates, so an agent that edits scene.py and re-checks would otherwise get
    the identical text back and read it as looping. Same verdict, different
    instruction.
    """
    run_dir = os.path.abspath(run_dir)
    # The module switch, checked HERE and nowhere else: every gate (the CLI,
    # windows plan, shape_pass's owed-ness) asks this one function, so a run
    # without the mechanism module owes nothing anywhere. RUN_DIR/modules.json
    # is read-only inside the sandbox — this is not an override the agent holds.
    if not modules.enabled(run_dir, "mechanism"):
        return {"problems": [], "blocking": [], "joints": [], "ok": True,
                "disabled": True,
                "calls_path": os.path.join(run_dir, CALLS_FILE)}
    if joint_defs is None:
        try:
            with open(os.path.join(run_dir, "mesh", "pose.json")) as f:
                pose = json.load(f)
        except (OSError, ValueError):
            pose = None
        joint_defs = ((pose or {}).get("joint_defs")
                      or (pose or {}).get("JOINTS") or [])
    result = validate(read_calls(run_dir), joint_defs)

    authored = _scene_axis_signs(run_dir)
    committed = {j.get("name"): j.get("axis") for j in (joint_defs or [])}
    for block in result["blocking"]:
        if block.get("kind") != "wrong_axis_sign":
            continue
        name = block.get("joint")
        src, pose_axis = authored.get(name), committed.get(name)
        if src is None or pose_axis is None:
            continue
        want = mechanism_views.negate_axis(pose_axis)
        if tuple(_component(v) for v in src) == tuple(_component(v)
                                                      for v in want):
            block["kind"] = "pose_json_behind_scene"
            block["detail"] = (
                f"scene.py ALREADY has axis {tuple(src)!r} — the fix is done, "
                "but mesh/pose.json still carries the old sign, and that is "
                "what this gate reads. RE-RENDER to regenerate it "
                "(harness/utils/shape_pass.sh RUN_DIR ...); do not edit the "
                "pick or the axis again")

    result["ok"] = not result["problems"] and not result["blocking"]
    result["calls_path"] = os.path.join(run_dir, CALLS_FILE)
    return result


def problem_lines(result):
    """One line per problem and per blocking pick, naming the joint. `detail`
    prints IN FULL — for a wrong sign that text IS the axis to write."""
    lines = []
    for p in result["problems"]:
        lines.append(f"  joint {p.get('joint', '?')}: {p['kind']} — "
                     f"{p.get('detail', '')}")
    for b in result["blocking"]:
        lines.append(f"  joint {b['joint']}: picked {b['pick']!r} — "
                     f"{b.get('detail', '')}")
    return lines
