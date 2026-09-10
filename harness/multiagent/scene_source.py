"""scene_source.py — read and rewrite `scene.py`'s FRAMES, mechanically.

`apply` (at the bottom of this file) is the only thing that writes the shared
`scene.py`, and it does so by regenerating the `FRAMES` literal from the merged
trajectory rather than by patching text. Everything above it exists to make that
safe without importing bpy: the emitters that format a frame entry (shared with
the review copy, so the two are identical by construction), and the `ast` checks
that refuse any file shape a line-splice rewrite could silently get wrong — a
duplicate assignment, a mutation of FRAMES elsewhere, a statement sharing the
assignment's lines.

The rewrite is then validated against the merged poses before it is allowed to
land: parse, literal-eval, compare every frame. A rewrite that does not reproduce
what the merge decided is a failure, not a warning.
"""

import ast
import math
import os
import shutil
import tempfile

from analysis import frames as frames_lib
from analysis.lib.io import read_json, write_json
from core import filehash
from multiagent.workspace import _active_workspace
from rig import lie

# How close a rewritten value must be to the merged one to count as the same
# pose. This is a SERIALIZATION tolerance — `_entry_lines` writes %.6g, so a
# round trip moves the last significant digit — and deliberately its own
# constant rather than fragments.py's echo tolerances: "did this rewrite
# reproduce the merge" and "did this fragment change anything" are different
# questions that happen to compare quaternions, and sharing a number would tie
# them together for no reason (it also made this module import fragments, which
# imports this one).
REWRITE_ROT_TOL_RAD = 1e-4


def log(msg):
    print(f"[windows] {msg}", flush=True)


def _quat_angle(qa, qb):
    """lie.quat_angle with a missing/degenerate quaternion reading as maximally
    different (pi) — an absent pose must never validate as a match."""
    if not qa or not qb:
        return math.pi
    if not any(float(v) for v in qa) or not any(float(v) for v in qb):
        return math.pi
    return float(lie.quat_angle(qa, qb))


def _entry_lines(name, entry, indent=""):
    """Source lines for one FRAMES entry (%.6g floats), from a pose.json-shaped
    merged frame entry. Shared by paste_frames.py and the apply rewrite so the
    two are guaranteed to format identically. Emits pose (always), joints
    (only if non-empty), moved (only if truthy) — matching the loader's
    defaults (rig.scene._normalize_frame)."""
    op = entry.get("object_pose") or {}
    if not op.get("quaternion") or not op.get("translation"):
        raise ValueError(f"frame {name!r} has no resolved quaternion+translation "
                         "in the merged trajectory — refusing to write an "
                         "empty pose into scene.py")
    q = ", ".join(f"{float(v):.6g}" for v in op["quaternion"])
    t = ", ".join(f"{float(v):.6g}" for v in op["translation"])
    lines = [f'{indent}"{name}": {{',
             f'{indent}    "pose": {{',
             f'{indent}        "quaternion": ({q}),',
             f'{indent}        "translation": ({t}),',
             f"{indent}    }},"]
    joints = entry.get("joints") or {}
    if joints:
        j = ", ".join(f'"{k}": {float(v):.6g}'
                      for k, v in sorted(joints.items()))
        lines.append(f'{indent}    "joints": {{{j}}},')
    if entry.get("moved"):
        lines.append(f'{indent}    "moved": True,')
    lines.append(f"{indent}}},")
    return lines


def _write_paste(wdir, merged_frames, updated):
    """Review copy of the UPDATED FRAMES entries (the same shape as the sweep's
    'BEST — paste into FRAMES' block). `apply` is the mechanical path that
    actually rewrites scene.py; refiners never touch it."""
    lines = ["# paste_frames.py — updated FRAMES entries from the window merge",
             "# (pose +, when present, joints/moved). Review copy: apply the",
             "# merge with `python -m multiagent.windows apply`, not by hand.",
             ""]
    for name in frames_lib.order_frames(set(updated)):
        lines += _entry_lines(name, merged_frames[name])
        lines.append("")
    with open(os.path.join(wdir, "paste_frames.py"), "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
# apply — mechanically rewrite scene.py's FRAMES from the merged trajectory
# --------------------------------------------------------------------------- #
def _frames_block(merged_frames):
    """The full `FRAMES = {...}` pure-literal assignment, frames in temporal
    order."""
    lines = ["FRAMES = {"]
    for name in frames_lib.order_frames(merged_frames):
        lines += _entry_lines(name, merged_frames[name], indent="    ")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _find_frames_assign(tree, path="scene.py"):
    """The single top-level `FRAMES = ...` assignment in a parsed scene.py.

    Rejects anything the line-splice rewrite could silently get wrong:
    multi-target assignments, other statements sharing the active assignment's
    lines, and any mutation of FRAMES outside the complete assignment."""
    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets
                     if isinstance(t, ast.Name)]
            if "FRAMES" in names:
                if len(node.targets) > 1:
                    raise ValueError(f"{path}: FRAMES in a multi-target "
                                     "assignment (line "
                                     f"{node.lineno}) — cannot rewrite safely")
                found.append(node)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "FRAMES":
                found.append(node)
    if not found:
        raise ValueError(f"{path}: no top-level `FRAMES = ...` assignment found")
    if len(found) > 1:
        lines = ", ".join(str(item.lineno) for item in found)
        raise ValueError(
            f"{path}: duplicate top-level FRAMES assignments at lines {lines}; "
            "keep exactly one assignment before applying merged poses")
    node = found[0]

    assignment_targets = {
        target
        for assignment in found
        for target in (
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else [assignment.target]
        )
    }
    for other in ast.walk(tree):
        if other in found or other in assignment_targets:
            continue
        if (isinstance(other, ast.Name) and other.id == "FRAMES"
                and isinstance(other.ctx, (ast.Store, ast.Del))):
            raise ValueError(
                f"{path}: FRAMES is also assigned/deleted at line "
                f"{other.lineno} — a mutation outside the rewritten "
                "assignment would silently re-shadow applied poses")
        if (isinstance(other, ast.Call)
                and isinstance(other.func, ast.Attribute)
                and isinstance(other.func.value, ast.Name)
                and other.func.value.id == "FRAMES"
                and other.func.attr in ("update", "setdefault", "pop",
                                        "popitem", "clear")):
            raise ValueError(
                f"{path}: FRAMES.{other.func.attr}(...) at line "
                f"{other.lineno} mutates FRAMES outside the rewritten "
                "assignment — cannot rewrite safely")
        if (isinstance(other, ast.Subscript)
                and isinstance(other.value, ast.Name)
                and other.value.id == "FRAMES"
                and isinstance(other.ctx, (ast.Store, ast.Del))):
            raise ValueError(
                f"{path}: FRAMES[...] is assigned/deleted at line "
                f"{other.lineno} — a mutation outside the rewritten "
                "assignment would silently re-shadow applied poses")

    for other in tree.body:
        if other is node:
            continue
        if other.lineno <= node.end_lineno and other.end_lineno >= node.lineno:
            raise ValueError(
                f"{path}: statement at line {other.lineno} shares source "
                "lines with the FRAMES assignment — cannot line-splice safely")
    return node


def ex_frames_sha1(scene_path):
    """sha1 of scene.py with the FRAMES assignment's lines removed.

    The merge/apply freshness gates compare THIS: `apply` rewrites FRAMES (and
    only FRAMES), so a whole-file sha would read the harness's own rewrite as
    a geometry change and kill any fragment landing after a partial apply.
    `_find_frames_assign` guarantees the assignment's lines are exclusively
    its own, so dropping them is a safe mechanical slice.

    A scene.py without a FRAMES assignment (or that does not parse) hashes
    whole — strictly narrower, never looser, than the whole-file gate."""
    with open(scene_path) as f:
        source = f.read()
    try:
        node = _find_frames_assign(ast.parse(source), path=scene_path)
    except (SyntaxError, ValueError):
        return filehash.bytes_sha1(source.encode("utf-8"))
    lines = source.splitlines(keepends=True)
    rest = lines[:node.lineno - 1] + lines[node.end_lineno:]
    return filehash.bytes_sha1("".join(rest).encode("utf-8"))


def _close(a, b):
    # %.6g keeps 6 SIGNIFICANT digits, so a coordinate >= 10 can round by more
    # than an absolute 1e-5 — compare with a relative term too.
    return math.isclose(float(a), float(b), rel_tol=1e-5, abs_tol=1e-5)


def _validate_rewrite(new_src, merged_frames, ref_frames, path="scene.py"):
    """Static (no-bpy) proof the rewritten source is what we meant to write:
    parses, FRAMES literal-evals, every frame matches the merged trajectory,
    reference frames are present."""
    tree = ast.parse(new_src)
    node = _find_frames_assign(tree, path=path)
    frames = ast.literal_eval(node.value)
    if set(frames) != set(merged_frames):
        raise ValueError(f"{path}: rewritten FRAMES keys do not match the "
                         "merged trajectory")
    for name, entry in frames.items():
        merged = merged_frames[name]
        op = merged.get("object_pose") or {}
        pose = entry.get("pose") or {}
        if _quat_angle(pose.get("quaternion"),
                       op.get("quaternion")) > REWRITE_ROT_TOL_RAD:
            raise ValueError(f"{path}: rewritten quaternion for {name!r} "
                             "does not match the merged trajectory")
        if not all(_close(a, b) for a, b in
                   zip(pose.get("translation") or (), op.get("translation") or [])):
            raise ValueError(f"{path}: rewritten translation for {name!r} "
                             "does not match the merged trajectory")
        written_joints = entry.get("joints") or {}
        for jname, val in (merged.get("joints") or {}).items():
            # A joint present in the merged trajectory but ABSENT from the
            # rewritten source is a failed rewrite, not a zero.
            if jname not in written_joints:
                raise ValueError(f"{path}: rewritten FRAMES for {name!r} omits "
                                 f"joint {jname!r}, which the merged trajectory "
                                 "sets — an absent joint state is not zero")
            if not _close(written_joints[jname], val):
                raise ValueError(f"{path}: rewritten joint {jname!r} for "
                                 f"{name!r} does not match the merged "
                                 "trajectory")
        if bool(entry.get("moved", False)) != bool(merged.get("moved", False)):
            raise ValueError(f"{path}: rewritten moved flag for {name!r} "
                             "does not match the merged trajectory")
    for ref in ref_frames:
        if ref and ref not in frames:
            raise ValueError(f"{path}: reference frame {ref!r} is not a key "
                             "of the rewritten FRAMES")


def rewrite_frames(scene_path, merged_frames, backup_path=None, ref_frames=()):
    """Atomically replace one scene's FRAMES with a pose-json-shaped mapping."""
    scene_path = os.path.abspath(scene_path)
    if not merged_frames:
        raise ValueError("cannot rewrite scene.py from an empty trajectory")
    with open(scene_path, "r") as handle:
        old_src = handle.read()
    tree = ast.parse(old_src, filename=scene_path)
    node = _find_frames_assign(tree, path=scene_path)
    if isinstance(node.value, ast.Dict) and all(
            isinstance(key, ast.Constant) for key in node.value.keys):
        old_keys = {key.value for key in node.value.keys}
        if old_keys != set(merged_frames):
            gone = sorted(old_keys - set(merged_frames))
            new = sorted(set(merged_frames) - old_keys)
            raise ValueError(
                "scene.py FRAMES and the replacement trajectory disagree "
                f"about the frame roster (only in scene.py: {gone or '-'}; "
                f"only in replacement: {new or '-'})")

    src_lines = old_src.splitlines(keepends=True)
    new_src = ("".join(src_lines[:node.lineno - 1])
               + _frames_block(merged_frames)
               + "".join(src_lines[node.end_lineno:]))
    scene_ref = next((
        item.value.value for item in tree.body
        if isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name)
                and target.id == "REFERENCE_FRAME"
                for target in item.targets)
        and isinstance(item.value, ast.Constant)
    ), None)
    _validate_rewrite(
        new_src, merged_frames,
        ref_frames=tuple(ref_frames) + (scene_ref,),
        path=scene_path)

    if backup_path:
        os.makedirs(os.path.dirname(os.path.abspath(backup_path)),
                    exist_ok=True)
        shutil.copyfile(scene_path, backup_path)
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(scene_path), prefix=".scene_frames_")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(new_src)
        os.replace(tmp, scene_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return {
        "scene_path": scene_path,
        "backup": backup_path,
        "new_scene_sha1": filehash.file_sha1(scene_path),
        "frames": len(merged_frames),
    }


# --------------------------------------------------------------------------- #
# apply — the command: consume a merge, rewrite FRAMES, never by hand
# --------------------------------------------------------------------------- #
def apply(run_dir):
    """Rewrite scene.py's FRAMES from the current iteration's merged poses.

    Consumes a completed merge; it does NOT re-run one. The merge report is the
    orchestrator's review point (model feedback, window reports, the seam list),
    and this gates on the frozen `scene_sha1`, so a second apply of the same
    merge fails loudly: a merge is consumed exactly once.

    Seam verdicts come AFTER this, not before — they need the COMMITTED renders,
    because both sides of a basin flip look smooth in the source video and only a
    render shows which side a pose landed on. `scene_before_apply.py` keeps the
    commit cheap to reverse in the meantime."""
    run_dir = os.path.abspath(run_dir)
    wdir = _active_workspace(run_dir)
    plan_doc = read_json(os.path.join(wdir, "plan.json"))
    if not isinstance(plan_doc, dict):
        raise ValueError(f"no plan.json under {wdir} — run `plan` first")
    merged_path = os.path.join(wdir, "merged_poses.json")
    report_path = os.path.join(wdir, "merge_report.json")
    if not (os.path.isfile(merged_path) and os.path.isfile(report_path)):
        raise ValueError(f"no completed merge under {wdir} — run `merge` first")
    report = read_json(report_path)
    if not (isinstance(report, dict) and report.get("ok") is True):
        raise ValueError(f"{report_path} is not an ok merge — fix the "
                         "violations and re-run `merge` first")
    model_review = report.get("model_review_required") or []
    if model_review:
        log(f"WARNING: applying with {len(model_review)} high-severity "
            "frozen-model finding(s); review `model_review_required` in "
            f"{report_path} during verification")

    scene_path = os.path.join(run_dir, "scene.py")
    plan_ex_sha = plan_doc.get("scene_ex_frames_sha1")
    merged_sha = filehash.file_sha1(merged_path)
    if plan_ex_sha:
        # ex-FRAMES: this apply's own FRAMES rewrites don't break the frozen-
        # geometry gate, so a re-merge (e.g. a late window after
        # --allow-missing) can be applied. Consumed-exactly-once therefore
        # moves to the merge fingerprint below.
        current_sha = ex_frames_sha1(scene_path)
        if current_sha != plan_ex_sha:
            raise ValueError(
                "scene.py geometry changed since the plan was made "
                f"({plan_ex_sha} -> {current_sha}); the frozen-geometry "
                "contract is broken — re-plan (and re-run refiners + merge) "
                "on the new scene")
        prior = read_json(os.path.join(wdir, "apply_report.json")) \
            if os.path.isfile(os.path.join(wdir, "apply_report.json")) else None
        if (isinstance(prior, dict) and prior.get("ok")
                and prior.get("merged_poses_sha1") == merged_sha):
            raise ValueError(
                f"this merge is already applied ({merged_sha[:8]}); a merge "
                "is consumed exactly once — re-run `merge` first if new "
                "fragments landed")
    else:
        current_sha = filehash.file_sha1(scene_path)
        if current_sha != plan_doc["scene_sha1"]:
            raise ValueError(
                "scene.py changed since the plan was made "
                f"({plan_doc['scene_sha1']} -> {current_sha}); the "
                "frozen-geometry contract is broken — if this apply already "
                "ran, the merge is consumed; otherwise re-plan (and re-run "
                "refiners + merge) on the new scene")

    merged_doc = read_json(merged_path)
    merged_frames = merged_doc.get("frames") or {}
    if not merged_frames:
        raise ValueError(f"{merged_path} has no frames")
    newest_frag = max((os.path.getmtime(os.path.join(wdir, w["id"],
                                                     "poses.json"))
                       for w in plan_doc["windows"]
                       if os.path.isfile(os.path.join(wdir, w["id"],
                                                      "poses.json"))),
                      default=0.0)
    if newest_frag > os.path.getmtime(merged_path):
        log("WARNING: window fragment(s) newer than merged_poses.json — "
            "re-run `merge` first if that edit was intentional")

    backup = os.path.join(wdir, "scene_before_apply.py")
    rewrite = rewrite_frames(
        scene_path, merged_frames, backup_path=backup,
        ref_frames=(plan_doc.get("ref_frame"),))
    new_sha = rewrite["new_scene_sha1"]
    updated = report.get("updated_frames") or []
    apply_report = {
        "ok": True,
        "plan_scene_sha1": plan_doc["scene_sha1"],
        "new_scene_sha1": new_sha,
        "merged_poses_sha1": merged_sha,
        "frames": len(merged_frames),
        "updated_frames": updated,
        "backup": backup,
        "model_review_required": model_review,
    }
    write_json(os.path.join(wdir, "apply_report.json"), apply_report)
    log(f"applied {len(merged_frames)} frame(s) ({len(updated)} updated) to "
        f"{scene_path}; sha1 {plan_doc['scene_sha1'][:8]} -> {new_sha[:8]} "
        f"(backup: {backup})")
    log("a running render pool will detect the scene edit and recycle its "
        "workers; next: harness/utils/composite_pass.sh RUN_DIR")
    return apply_report
