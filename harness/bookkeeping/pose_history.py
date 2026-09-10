"""Shared pose-history selection and kinematics reconciliation."""

import datetime
import json
import os

from bookkeeping import ledger
from core import joints as joints_lib


def _read_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def iteration_numbers(run_dir):
    root = os.path.join(os.path.abspath(run_dir), ledger.ITERATIONS_DIR)
    try:
        names = os.listdir(root)
    except OSError:
        return []
    return sorted(int(name) for name in names
                  if len(name) == 6 and name.isdigit()
                  and os.path.isdir(os.path.join(root, name)))


def resolve_iteration(run_dir, selector):
    """Resolve ``previous`` or a numeric selector to an iteration directory."""
    text = str(selector).strip().lower()
    numbers = iteration_numbers(run_dir)
    if text == "previous":
        if not numbers:
            raise ValueError(
                "--from-iteration previous: no iterations exist")
        number = numbers[-1]
    else:
        try:
            number = int(text)
        except ValueError as exc:
            raise ValueError(
                f"--from-iteration {selector!r}: expected 'previous' or an "
                "iteration number") from exc
    path = ledger.iteration_dir(run_dir, number)
    if not os.path.isdir(path):
        raise ValueError(
            f"--from-iteration {selector!r}: iteration "
            f"{ledger.iteration_name(number)} does not exist")
    return path


def _seed_source(path):
    """Return an iteration root, windows directory, and historical pose."""
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise ValueError(f"--seed-from: {path} is not a directory")
    if os.path.basename(path) == "windows":
        root, windows = os.path.dirname(path), path
    elif os.path.isdir(os.path.join(path, "windows")):
        root, windows = path, os.path.join(path, "windows")
    else:
        root, windows = path, path
    pose = _read_json(os.path.join(root, "pose.json"))
    if not isinstance(pose, dict):
        pose = _read_json(os.path.join(root, "pose_start.json"))
    return root, windows, pose if isinstance(pose, dict) else None


def _source_frame_map(windows, historical_pose):
    frames = {}
    if historical_pose is not None:
        for name, entry in (historical_pose.get("frames") or {}).items():
            pose = (entry or {}).get("object_pose")
            if pose and pose.get("quaternion") and pose.get("translation"):
                frames[name] = {
                    "pose": pose,
                    "joints": (entry or {}).get("joints") or {},
                }
    try:
        names = sorted(os.listdir(windows))
    except OSError:
        names = []
    for wid in names:
        for fname in ("poses.json", "progress.json"):
            fragment = _read_json(os.path.join(windows, wid, fname))
            if isinstance(fragment, dict) and fragment.get("frames"):
                break
        else:
            continue
        for name, entry in fragment["frames"].items():
            pose = (entry or {}).get("pose") or {}
            if pose.get("quaternion") and pose.get("translation"):
                frames[name] = {
                    "pose": pose,
                    "joints": (entry or {}).get("joints") or {},
                }
    return frames


def _joint_reconciliation(old_defs, current_defs):
    current = {
        joint["name"]: joint
        for joint in joints_lib.canonical_joint_defs(current_defs)
    }
    if old_defs is None:
        return current, {
            "compatible": [],
            "limit_changed": [],
            "added": [],
            "removed": [],
            "structurally_changed": [],
            "unknown": sorted(current),
        }
    old = {
        joint["name"]: joint
        for joint in joints_lib.canonical_joint_defs(old_defs)
    }
    common = set(old) & set(current)
    changed = sorted(
        name for name in common
        if joints_lib.structural_joint(old[name])
        != joints_lib.structural_joint(current[name]))
    limit_changed = sorted(
        name for name in common - set(changed)
        if old[name].get("limit") != current[name].get("limit"))
    compatible = sorted(common - set(changed) - set(limit_changed))
    return current, {
        "compatible": compatible,
        "limit_changed": limit_changed,
        "added": sorted(set(current) - set(old)),
        "removed": sorted(set(old) - set(current)),
        "structurally_changed": changed,
        "unknown": [],
    }


def _current_frame_joints(pose_json, frame):
    entry = (pose_json.get("frames") or {}).get(frame) or {}
    return entry.get("joints") or {}


def reconcile_sources(run_dir, current_pose, sources):
    """Load historical hypotheses and reconcile them with current kinematics."""
    del run_dir  # Kept in the API for provenance-oriented callers.
    current_defs = (current_pose.get("joint_defs")
                    if current_pose.get("joint_defs") is not None
                    else current_pose.get("JOINTS") or [])
    seeds = {}
    reports = []
    for source_dir in sources:
        root, windows, historical_pose = _seed_source(source_dir)
        label = os.path.basename(root)
        old_defs = None
        if historical_pose is not None:
            old_defs = (historical_pose.get("joint_defs")
                        if historical_pose.get("joint_defs") is not None
                        else historical_pose.get("JOINTS"))
        current_by_name, actions = _joint_reconciliation(
            old_defs, current_defs)
        unsafe = set(actions["added"] + actions["structurally_changed"]
                     + actions["unknown"])
        carry = set(actions["compatible"] + actions["limit_changed"])
        clamped = []
        missing_current = []
        frames = _source_frame_map(windows, historical_pose)
        for name, entry in frames.items():
            old_states = entry.get("joints") or {}
            current_states = _current_frame_joints(current_pose, name)
            states = {}
            for joint_name, joint in current_by_name.items():
                if joint.get("type") == "fixed":
                    continue
                if joint_name in carry and joint_name in old_states:
                    value = float(old_states[joint_name])
                    limit = joints_lib.joint_limit(joint)
                    reconciled = (min(max(value, limit[0]), limit[1])
                                  if limit else value)
                    states[joint_name] = reconciled
                    if reconciled != value:
                        clamped.append({
                            "frame": name,
                            "joint": joint_name,
                            "from": value,
                            "to": reconciled,
                        })
                elif joint_name in current_states:
                    states[joint_name] = float(current_states[joint_name])
                else:
                    missing_current.append({
                        "frame": name, "joint": joint_name})
            review = sorted(unsafe | {
                item["joint"] for item in missing_current
                if item["frame"] == name
            })
            seed = {
                "pose": {
                    "quaternion": entry["pose"]["quaternion"],
                    "translation": entry["pose"]["translation"],
                },
                "joints": states,
                "from": label,
            }
            if review:
                seed["reconciliation"] = {
                    "requires_review": True,
                    "joints": review,
                }
            seeds[name] = seed
        if not frames:
            raise ValueError(
                f"--seed-from: no usable fragment frames under {source_dir} "
                "(looked for pose.json and w*/poses.json or "
                "w*/progress.json)")
        reports.append({
            "source": root,
            "source_iteration": (
                int(label) if len(label) == 6 and label.isdigit() else None),
            "source_kinematics_hash": (
                joints_lib.kinematics_hash(old_defs)
                if old_defs is not None else None),
            "current_kinematics_hash":
                joints_lib.kinematics_hash(current_defs),
            "joint_actions": actions,
            "clamped_states": clamped,
            "missing_current_states": missing_current,
            "frames_seeded": len(frames),
            "requires_review": bool(unsafe or missing_current),
        })
    return seeds, reports


def _select_frames(spec, roster):
    if not spec:
        return list(roster)
    text = str(spec).strip()
    if "," in text:
        selected = [item.strip() for item in text.split(",") if item.strip()]
    elif text in roster:
        selected = [text]
    else:
        first, separator, last = text.partition("-")
        first, last = first.strip(), last.strip()
        if not separator or first not in roster or last not in roster:
            raise ValueError(
                f"--frames {text!r}: expected a frame, comma list, or "
                "inclusive '<first>-<last>' range")
        lo, hi = roster.index(first), roster.index(last)
        if lo > hi:
            raise ValueError(f"--frames range {text!r} runs backwards")
        selected = roster[lo:hi + 1]
    unknown = sorted(set(selected) - set(roster))
    if unknown:
        raise ValueError(
            "--frames names frames outside the current trajectory: "
            + ", ".join(unknown))
    return list(dict.fromkeys(selected))


def reseed_trajectory(run_dir, selector, frames=None):
    """Build an inspectable full-trajectory draft from one history iteration."""
    run_dir = os.path.abspath(run_dir)
    current_path = os.path.join(run_dir, "mesh", "pose.json")
    current_pose = _read_json(current_path)
    if not isinstance(current_pose, dict) or not current_pose.get("frames"):
        raise ValueError(
            f"{current_path} has no committed trajectory to reconcile")
    source = resolve_iteration(run_dir, selector)
    seeds, reports = reconcile_sources(run_dir, current_pose, [source])
    roster = list(current_pose["frames"])
    selected = _select_frames(frames, roster)
    selected_set = set(selected)
    reconciled_frames = {}
    reseeded = []
    for name, current in current_pose["frames"].items():
        seed = seeds.get(name) if name in selected_set else None
        if seed is None:
            reconciled_frames[name] = current
            continue
        reconciled_frames[name] = dict(current)
        reconciled_frames[name]["object_pose"] = seed["pose"]
        reconciled_frames[name]["joints"] = seed["joints"]
        if seed.get("reconciliation"):
            reconciled_frames[name]["reconciliation"] = seed["reconciliation"]
        reseeded.append(name)
    unavailable = [name for name in selected if name not in seeds]
    if not reseeded:
        raise ValueError(
            "the selected history iteration has no poses for the requested "
            "frame selection")
    return {
        "schema_version": 1,
        "created_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "run_dir": run_dir,
        "source": source,
        "source_iteration": reports[0]["source_iteration"],
        "selected_frames": selected,
        "reseeded_frames": reseeded,
        "unavailable_frames": unavailable,
        "frames": reconciled_frames,
        "reconciliation": reports[0],
        "requires_review": reports[0]["requires_review"],
    }
