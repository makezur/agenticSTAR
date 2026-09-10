"""briefs.py — render the one document a refiner subagent reads.

A brief is the ENTIRE interface to a refiner: it is spawned with a pointer to
`<wid>/brief.md` and nothing else, so everything it needs has to be in there —
paths, its own frames, the committed poses, the schema, the rules. That is why
the paths are absolute (a relative doc link broke every refiner's first action
once) and why reference material is INLINED rather than linked.

ONE template (`briefs/core.md`) carries the contract and the reference material.
Everything that varies per round is DATA rendered into it: per-frame
`known_issues` and free-text `guidance`.

There is deliberately no per-round-type template. This was three overlay files
(refine / polish / unflip) filling `${stance_*}` slots, on the theory that a
round type differs in RULES and the merge would validate against them — but that
validation never existed, and it should not: what verifies a round is the
orchestrator reading the temporal report over the merged sequence at
`adjudicate`, by eye. The overlays encoded one bit ("may a basin be searched")
plus three restatements of it, which is a sentence of `guidance`.

Prose is edited in core.md. Only values and flags live here.
"""

import json
import os
import re
import string

from multiagent.fragments import SCHEMA_VERSION, _committed, _interior_steps
from multiagent.workspace import WINDOWS_DIRNAME

# The authored brief source beside this module: core.md is the ONE template.
# Prose is edited THERE, only values/flags live in code.
_BRIEFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "briefs")


_BRIEF_TEMPLATE = os.path.join(_BRIEFS_DIR, "core.md")


def _read_text(path):
    """Read a UTF-8 text file whole (small docs pasted into the brief)."""
    with open(path, encoding="utf-8") as f:
        return f.read()


def _extract_section(md, heading, path=""):
    """The body of one `## <heading>` section of a Markdown doc, up to the next
    heading at the same level.

    For inlining a doc's ONE relevant chapter rather than the whole file: the
    refiner needs `report.md`'s "Reading the table" (what each column of the
    temporal table is telling you, which is the guide it has to have while
    writing a step verdict) and not its CLI reference, JSON schema, or error
    table. Raises if the heading is gone, so a doc edit that renames it fails the
    plan instead of silently shipping a brief with an empty section — the same
    strictness `_render_brief_template` has about placeholders."""
    lines = md.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if ln.strip().startswith("## ") and heading in ln), None)
    if start is None:
        raise ValueError(
            f"{path or 'document'}: no '## ...{heading}...' section to inline "
            "into the brief — it was renamed or removed; update the heading "
            "this brief asks for")
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start + 1:end]).strip("\n")


def _demote_headings(md, by):
    """Demote every ATX `#` heading in `md` by `by` levels (capped at 6) so a
    pasted document nests below the brief's own headings instead of competing
    with them. A `# Title` pasted under a `#### Section` at by=3 becomes
    `#### Title`. Non-heading lines (and fenced code) pass through unchanged."""
    out = []
    in_fence = False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        m = re.match(r"^(#{1,6})(\s)", line) if not in_fence else None
        if m:
            level = min(len(m.group(1)) + by, 6)
            out.append("#" * level + line[len(m.group(1)):])
        else:
            out.append(line)
    return "\n".join(out)


def _render_brief_template(text, values, flags):
    """Render the brief template: drop `<!--#`-prefixed template comments,
    resolve single-level `<!--IF flag--> ... <!--ELSE--> ... <!--ENDIF-->`
    blocks against `flags`, then substitute `${name}` placeholders from
    `values` — strict on purpose (a missing name raises KeyError), so the
    template and the planner cannot drift apart silently."""
    out = []
    keep = []                                  # truth of each open IF branch
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("<!--#"):
            continue
        m = re.fullmatch(r"<!--IF (\w+)-->", stripped)
        if m:
            keep.append(bool(flags[m.group(1)]))
            continue
        if stripped == "<!--ELSE-->":
            keep[-1] = not keep[-1]
            continue
        if stripped == "<!--ENDIF-->":
            keep.pop()
            continue
        if all(keep):
            out.append(line)
    if keep:
        raise ValueError(f"unclosed <!--IF--> in {_BRIEF_TEMPLATE}")
    return string.Template("\n".join(out)).substitute(values)


def _write_brief(run_dir, layout, pose_json, plan_doc, w, known_issues=None):
    """Render briefs/core.md into the self-contained per-window instruction file
    a refiner subagent gets pointed at. The brief is the SINGLE instruction
    source for a refiner: everything it needs is IN it (paths, committed poses,
    schema, rules, this round's guidance), so its context starts small and there
    is no second document to fetch. Doc pointers are ABSOLUTE paths."""
    harness = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(harness)
    directions_md = os.path.join(repo_root, "conventions", "DIRECTIONS.md")
    report_md = os.path.join(harness, "analysis", "temporal", "report.md")
    iteration_dir = plan_doc.get("iteration_dir") or \
        os.path.join(run_dir, WINDOWS_DIRNAME)
    wdir = os.path.join(iteration_dir, w["id"])
    run_name = os.path.basename(os.path.abspath(run_dir))
    run_match = re.fullmatch(r"(.+_v\d+)_c(\d+)", run_name)
    provenance = {
        "sequence": run_match.group(1) if run_match else run_name,
        "chunk": f"c{int(run_match.group(2)):02d}" if run_match else "",
        "iteration": int(plan_doc["iteration"]),
        "window": w["id"],
        "submitted_by": "window_agent",
    }
    issues = {f: (known_issues or {})[f] for f in w["frames"]
              if (known_issues or {}).get(f)}
    steps = _interior_steps(w["frames"])
    schema = {
        "schema_version": SCHEMA_VERSION,
        "window_id": w["id"],
        "scene_sha1": plan_doc["scene_sha1"],
        "expected_motion": [
            {"frames": f"{w['frames'][0]}..{w['frames'][-1]}",
             "expect": "<what the VIDEO shows the object doing here — written "
                       "from the frame sheet, BEFORE your first sweep>"}],
        "frames": {"<frame>.jpg": {
            "pose": {"quaternion": [1, 0, 0, 0], "translation": [0, 0, 2]},
            "joints": {"<joint>": 0.0},
            "confidence": "high|medium|low",
            "notes": "<what was rejected and why>"}},
        "step_calls": [
            {"from": steps[0][0] if steps else "<frame>.jpg",
             "to": steps[0][1] if steps else "<next frame>.jpg",
             "verdict": "coherent|unsure",
             "evidence": "<what the pictures SHOW>",
             "pose_hash": "<selfcheck stamps this — never type it yourself>"}],
        "report": [{"frame": "<frame>.jpg (optional)",
                    "note": "<something you SAW but cannot fix>"}],
        "model_feedback": [],
    }
    values = {
        "wid": w["id"],
        "provenance_json": json.dumps(provenance),
        "scene_sha1": plan_doc["scene_sha1"],
        "run_dir": run_dir,
        "wdir": wdir,
        "frag": os.path.join(wdir, "poses.json"),
        "progress": os.path.join(wdir, "progress.json"),
        "spool": plan_doc["spool"],
        "frames_csv": ", ".join(w["frames"]),
        "known_issue_lines": "\n".join(
            f"- {f}: {issues[f]}" for f in w["frames"] if f in issues),
        "seeds_json": json.dumps(w.get("seeds") or {}, indent=2),
        "neighbors_json": json.dumps(w["neighbors"], indent=2),
        "frames_dir": layout["frames_dir"],
        "masks_dir": layout["masks_dir"],
        "hand_masks_line": (f"- hand masks:    `{layout['hand_masks_dir']}`"
                            if layout.get("hand_masks_dir")
                            else "- hand masks:    (none)"),
        "pose_json_path": os.path.join(run_dir, "mesh", "pose.json"),
        "sweep_md": os.path.join(harness, "views", "sweeps", "sweep.md"),
        "osweep_md": os.path.join(harness, "views", "sweeps", "osweep.md"),
        "oapply_md": os.path.join(harness, "views", "sweeps", "oapply.md"),
        "oapply_all_md": os.path.join(harness, "views", "sweeps",
                                      "oapply_all.md"),
        "pool_md": os.path.join(harness, "pool", "README.md"),
        "directions_md": directions_md,
        "report_md": report_md,
        # inline the verified sign convention so the refiner never has to fetch
        # it: which way a +roll/+yaw/+pitch order (and each directional preset)
        # moves the object on screen is load-bearing for choosing what to order.
        # Headings are demoted so the pasted doc nests under the brief's ####.
        "directions_body": _demote_headings(_read_text(directions_md), 3),
        # and the temporal table's column semantics, for the same reason: the
        # refiner has to write a verdict on every interior step, and the ONE
        # thing it cannot do that from is a table whose columns it has to guess
        # at. Only the reading chapter — not report.md's CLI/schema/errors.
        "report_reading_body": _demote_headings(
            _extract_section(_read_text(report_md), "Reading the table",
                             path=report_md), 3),
        "committed_json": json.dumps(
            {f: _committed(pose_json, f)["pose"] for f in w["frames"]},
            indent=2),
        "schema_json": json.dumps(schema, indent=2),
        "guidance": w.get("guidance") or "",
        "n_steps": str(len(steps)),
        "frames_range": f"{w['frames'][0]}-{w['frames'][-1]}",
    }
    flags = {"known_issues": bool(issues),
             "neighbors": bool(w["neighbors"]),
             "guidance": bool(w.get("guidance")),
             "seeds": bool(w.get("seeds"))}
    with open(_BRIEF_TEMPLATE) as f:
        template = f.read()
    with open(os.path.join(wdir, "brief.md"), "w") as f:
        f.write(_render_brief_template(template, values, flags))
