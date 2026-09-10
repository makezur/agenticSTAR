"""Unified run iteration allocation, artifact ownership, and Git history.

The harness, not the agent, owns iteration numbers. A shape pass opens and
completes an iteration itself. A pose-window plan opens a pose iteration and
the verification shape pass later attaches to and completes it.
"""

import contextlib
import copy
import datetime
import fcntl
import json
import os
import shutil
import subprocess
import tempfile

from core import joints as joints_lib


ITERATIONS_DIR = "iterations"
STATE_DIR = ".harness"
STATE_FILE = "bookkeeping.json"
LOCK_FILE = "bookkeeping.lock"
SCHEMA_VERSION = 1


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


def _atomic_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def _locked(run_dir):
    state_dir = os.path.join(os.path.abspath(run_dir), STATE_DIR)
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, LOCK_FILE), "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def iteration_name(number):
    return f"{int(number):06d}"


def iteration_dir(run_dir, number):
    return os.path.join(os.path.abspath(run_dir), ITERATIONS_DIR,
                        iteration_name(number))


def artifact_dir(run_dir, name, requested=""):
    """Resolve an artifact directory under the active iteration when managed."""
    run_dir = os.path.abspath(run_dir)
    if not name or os.path.basename(name) != name or name in (".", ".."):
        raise ValueError(f"invalid artifact directory name {name!r}")
    requested = os.path.abspath(requested) if requested else ""
    state_path = os.path.join(run_dir, STATE_DIR, STATE_FILE)
    if not os.path.isfile(state_path):
        return requested or os.path.join(run_dir, name)

    current = active(run_dir, required=True)
    owner = os.path.realpath(current["path"])
    destination = requested or os.path.join(owner, name)
    resolved = os.path.realpath(destination)
    try:
        inside_owner = os.path.commonpath([owner, resolved]) == owner
    except ValueError:
        inside_owner = False
    if not inside_owner:
        raise ValueError(
            f"{name} for bookkeeping-managed run {run_dir} must be inside "
            f"active iteration {current['name']}: {owner}")
    return destination


def _metadata_path(path):
    return os.path.join(path, "iteration.json")


def _read_json(path, default=None):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def _state(run_dir):
    return _read_json(
        os.path.join(os.path.abspath(run_dir), STATE_DIR, STATE_FILE),
        {"schema_version": SCHEMA_VERSION, "active_iteration": None})


def _write_state(run_dir, state):
    _atomic_json(os.path.join(os.path.abspath(run_dir), STATE_DIR, STATE_FILE),
                 state)


def _compact_pose(path):
    """Return the trajectory state needed for history and recovery.

    Render-pass pose files also carry fixed camera tracking, intrinsics, part
    inventory, and descriptive conventions. Those belong to the pass that
    consumed them; pose history needs only the shared rig and editable state.
    """
    pose = _read_json(path)
    if not isinstance(pose, dict):
        return None
    defs = pose.get("joint_defs")
    if defs is None:
        defs = pose.get("JOINTS")
    canonical = joints_lib.canonical_joint_defs(defs)
    snapshot = {
        "joint_defs": canonical,
        "frames": {},
    }
    for key in ("scale", "reference_frame"):
        if pose.get(key) is not None:
            snapshot[key] = copy.deepcopy(pose[key])
    for name, raw in (pose.get("frames") or {}).items():
        if not isinstance(raw, dict):
            continue
        frame = {}
        for key in ("moved", "object_pose", "joints"):
            if key in raw:
                frame[key] = copy.deepcopy(raw[key])
        snapshot["frames"][str(name)] = frame
    return snapshot


def _snapshot_pose(source, destination):
    snapshot = _compact_pose(source)
    if snapshot is None:
        return None
    _atomic_json(destination, snapshot)
    return joints_lib.kinematics_hash(snapshot["joint_defs"])


def _iteration_numbers(run_dir):
    root = os.path.join(os.path.abspath(run_dir), ITERATIONS_DIR)
    try:
        names = os.listdir(root)
    except OSError:
        return []
    return sorted(int(name) for name in names
                  if len(name) == 6 and name.isdigit()
                  and os.path.isdir(os.path.join(root, name)))


def read_iteration(run_dir, number):
    path = iteration_dir(run_dir, number)
    doc = _read_json(_metadata_path(path))
    if not isinstance(doc, dict):
        raise ValueError(f"iteration {iteration_name(number)} has no readable "
                         "iteration.json")
    doc["path"] = path
    return doc


def active(run_dir, required=False):
    state = _state(run_dir)
    number = state.get("active_iteration")
    if number is None:
        if required:
            raise ValueError(f"no open iteration in {os.path.abspath(run_dir)}")
        return None
    doc = read_iteration(run_dir, number)
    if doc.get("status") != "open":
        state["active_iteration"] = None
        _write_state(run_dir, state)
        if required:
            raise ValueError(f"iteration {iteration_name(number)} is not open")
        return None
    return doc


def status(run_dir):
    """Inspectable recovery summary without requiring an active iteration."""
    run_dir = os.path.abspath(run_dir)
    numbers = _iteration_numbers(run_dir)
    current = active(run_dir, required=False)
    latest = read_iteration(run_dir, numbers[-1]) if numbers else None
    state = _state(run_dir)
    result = {
        "run_dir": run_dir,
        "active": current,
        "latest": latest,
        "iteration_count": len(numbers),
        "last_commit": state.get("last_commit"),
    }
    owner = current or latest
    if owner is not None:
        progress = _window_progress(run_dir, owner)
        if progress is not None:
            result["window_progress"] = progress
    return result


def history(run_dir):
    """Ordered machine-readable iteration ledger for resume and reseed."""
    run_dir = os.path.abspath(run_dir)
    entries = []
    for number in _iteration_numbers(run_dir):
        doc = read_iteration(run_dir, number)
        summary = {
            key: doc.get(key)
            for key in ("iteration", "name", "kind", "status", "label",
                        "created_at", "completed_at", "updated_at",
                        "start_kinematics_hash", "kinematics_hash", "passes")
            if doc.get(key) is not None
        }
        summary["path"] = doc["path"]
        composites = os.path.join(doc["path"], "composites")
        try:
            summary["composite_frames"] = len([
                name for name in os.listdir(composites)
                if name.endswith(".png")])
        except OSError:
            summary["composite_frames"] = 0
        progress = _window_progress(run_dir, doc)
        if progress is not None:
            summary["window_progress"] = progress
        entries.append(summary)
    return {"run_dir": run_dir, "iterations": entries}


def _window_progress(run_dir, doc):
    rel = doc.get("windows")
    if not rel:
        return None
    root = os.path.join(run_dir, rel)
    plan = _read_json(os.path.join(root, "plan.json"))
    if not isinstance(plan, dict):
        return {"path": rel, "windows": {}}
    out = {}
    for window in plan.get("windows", []):
        wid = window.get("id")
        if not wid:
            continue
        expected = len(window.get("frames") or [])
        fragment_path = os.path.join(root, wid, "poses.json")
        progress_path = os.path.join(root, wid, "progress.json")
        final = _read_json(fragment_path)
        partial = _read_json(progress_path)
        source = final if isinstance(final, dict) else partial
        authored = (source or {}).get("frames") or {}
        authored = authored if isinstance(authored, dict) else {}
        completed_frames = [
            frame for frame in (window.get("frames") or [])
            if frame in authored]
        completed_frames.extend(sorted(set(authored) - set(completed_frames)))
        count = len(authored)
        state = ("complete" if isinstance(final, dict)
                 else "partial" if isinstance(partial, dict) and count
                 else "pending")
        info = {
            "status": state,
            "frames": count,
            "expected_frames": expected,
            "completed_frames": completed_frames,
            "step_calls": len((source or {}).get("step_calls") or []),
            "model_feedback": len((source or {}).get("model_feedback") or []),
            "reports": len((source or {}).get("report") or []),
        }
        artifacts = [path for path in (progress_path, fragment_path)
                     if os.path.isfile(path)]
        if artifacts:
            recovery = fragment_path if isinstance(final, dict) else progress_path
            info["recovery_path"] = os.path.relpath(recovery, run_dir)
            latest = max(os.path.getmtime(path) for path in artifacts)
            info["last_activity_at"] = datetime.datetime.fromtimestamp(
                latest, datetime.timezone.utc).isoformat(timespec="seconds")
        out[wid] = info
    return {"path": rel, "windows": out}


def begin(run_dir, kind, label=None, attach_to=("shape", "pose", "custom")):
    """Open the next iteration, or attach to a compatible open iteration.

    `attach_to` lists open kinds this caller may join. Shape passes include
    ``pose`` so the verification pass closes the pose-window iteration.
    """
    run_dir = os.path.abspath(run_dir)
    kind = str(kind).strip().lower()
    if kind not in {"shape", "pose", "custom"}:
        raise ValueError(f"unknown iteration kind {kind!r}")
    with _locked(run_dir):
        current = active(run_dir, required=False)
        if current is not None:
            if current.get("kind") not in set(attach_to):
                raise ValueError(
                    f"iteration {current['name']} ({current.get('kind')}) is "
                    f"still open; cannot start {kind}")
            return current

        number = max(_iteration_numbers(run_dir), default=0) + 1
        path = iteration_dir(run_dir, number)
        os.makedirs(path, exist_ok=False)
        normalized_label = (str(label).strip() or None
                            if label is not None else None)
        doc = {
            "schema_version": SCHEMA_VERSION,
            "iteration": number,
            "name": iteration_name(number),
            "kind": kind,
            "status": "open",
            "label": normalized_label,
            "created_at": _now(),
            "updated_at": _now(),
            "passes": [],
        }
        start_hash = _snapshot_pose(
            os.path.join(run_dir, "mesh", "pose.json"),
            os.path.join(path, "pose_start.json"))
        if start_hash is not None:
            doc["start_kinematics_hash"] = start_hash
        _atomic_json(_metadata_path(path), doc)
        state = _state(run_dir)
        state["schema_version"] = SCHEMA_VERSION
        state["active_iteration"] = number
        _write_state(run_dir, state)
        doc["path"] = path
        return doc


def record_window_plan(run_dir, number, windows_dir):
    with _locked(run_dir):
        doc = read_iteration(run_dir, number)
        if doc.get("status") != "open":
            raise ValueError(f"iteration {doc['name']} is not open")
        doc["windows"] = os.path.relpath(windows_dir, run_dir)
        doc["updated_at"] = _now()
        path = doc.pop("path")
        _atomic_json(_metadata_path(path), doc)
        doc["path"] = path
        return doc


def _same_file(left, right):
    try:
        with open(left, "rb") as a, open(right, "rb") as b:
            return a.read() == b.read()
    except OSError:
        return False


def _git(run_dir, *args, check=True):
    result = subprocess.run(["git", "-C", run_dir, *args],
                            capture_output=True, text=True)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result


def ensure_git(run_dir):
    """Initialize the private per-run repository and its baseline commit."""
    run_dir = os.path.abspath(run_dir)
    if os.path.isdir(os.path.join(run_dir, ".git")):
        return
    _git(run_dir, "init", "-q")
    _git(run_dir, "config", "user.name", "Articulated Harness")
    _git(run_dir, "config", "user.email", "harness@localhost")
    ignore = os.path.join(run_dir, ".gitignore")
    with open(ignore, "w") as handle:
        handle.write("*\n")
    paths = [".gitignore"]
    for name in ("layout.json", "run_config.json", "depth_config.json",
                 "NOTES.md", "scene.py", "mesh/pose.json"):
        if os.path.isfile(os.path.join(run_dir, name)):
            paths.append(name)
    _git(run_dir, "add", "-f", "--", *paths)
    _git(run_dir, "commit", "-q", "-m", "initialize reconstruction run")


def _commit_iteration(run_dir, doc, path):
    ensure_git(run_dir)
    tracked = [
        os.path.relpath(_metadata_path(path), run_dir),
        os.path.relpath(os.path.join(path, "scene.py"), run_dir),
        os.path.relpath(os.path.join(path, "pose.json"), run_dir),
    ]
    pose_start = os.path.join(path, "pose_start.json")
    if os.path.isfile(pose_start):
        tracked.append(os.path.relpath(pose_start, run_dir))
    for name in ("scene.py", "mesh/pose.json", "NOTES.md"):
        if os.path.isfile(os.path.join(run_dir, name)):
            tracked.append(name)
    _git(run_dir, "add", "-f", "--", *tracked)
    label = f": {doc['label']}" if doc.get("label") else ""
    message = f"iteration {doc['name']} {doc['kind']} complete{label}"
    _git(run_dir, "commit", "-q", "--allow-empty", "-m", message)
    return _git(run_dir, "rev-parse", "HEAD").stdout.strip()


def _expected_composite_stems(run_dir):
    layout = _read_json(os.path.join(run_dir, "layout.json"), {}) or {}
    frames = layout.get("frames", [])
    if not isinstance(frames, list):
        return []
    return [os.path.splitext(str(frame))[0] for frame in frames]


def _stage_composites(run_dir, doc, iteration_path):
    """Build the promoted composite set without exposing it yet."""
    pass_dir = os.path.join(run_dir, doc["passes"][-1])
    composites = {}
    try:
        names = os.listdir(pass_dir)
    except OSError:
        names = []
    for name in names:
        if name.startswith("composite_") and name.endswith(".png"):
            stem = name[len("composite_"):-len(".png")]
            if stem:
                composites[stem] = os.path.join(pass_dir, name)

    if doc.get("kind") in {"shape", "pose"}:
        expected = _expected_composite_stems(run_dir)
        missing = [stem for stem in expected if stem not in composites]
        if not composites or missing:
            detail = f"; missing frames: {', '.join(missing)}" if missing else ""
            raise ValueError(
                f"iteration {doc['name']} verification pass has no complete "
                f"composite set{detail}")
    if not composites:
        return None, None

    final = os.path.join(iteration_path, "composites")
    if os.path.exists(final):
        raise ValueError(f"iteration {doc['name']} composites already exist")
    staging = tempfile.mkdtemp(prefix=".composites-", dir=iteration_path)
    try:
        for stem, source in sorted(composites.items()):
            destination = os.path.join(staging, f"{stem}.png")
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging, final


def record_pass(run_dir, number, pass_dir):
    """Attach one successful rendered pass and preserve its scene and pose."""
    run_dir = os.path.abspath(run_dir)
    pass_dir = os.path.abspath(pass_dir)
    with _locked(run_dir):
        doc = read_iteration(run_dir, number)
        if doc.get("status") != "open":
            raise ValueError(f"iteration {doc['name']} is not open")
        snapshot = os.path.join(pass_dir, "scene_snapshot.py")
        pose = os.path.join(pass_dir, "pose.json")
        scene = os.path.join(run_dir, "scene.py")
        if not os.path.isfile(snapshot) or not os.path.isfile(pose):
            raise ValueError(f"{pass_dir} lacks scene_snapshot.py or pose.json")
        if not _same_file(scene, snapshot):
            raise ValueError(
                "scene.py changed while the pass was rendering; the iteration "
                "remains open so the edited scene can be rendered explicitly")
        path = doc.pop("path")
        shutil.copy2(snapshot, os.path.join(path, "scene.py"))
        kinematics_hash = _snapshot_pose(
            pose, os.path.join(path, "pose.json"))
        if kinematics_hash is None:
            raise ValueError(f"{pose} is not a readable pose object")
        rel_pass = os.path.relpath(pass_dir, run_dir)
        if rel_pass not in doc["passes"]:
            doc["passes"].append(rel_pass)
        doc["kinematics_hash"] = kinematics_hash
        doc["updated_at"] = _now()
        _atomic_json(_metadata_path(path), doc)
        doc["path"] = path
        return doc


def complete(run_dir, number):
    """Close, commit, and publish an open iteration after its gates pass."""
    run_dir = os.path.abspath(run_dir)
    with _locked(run_dir):
        doc = read_iteration(run_dir, number)
        if doc.get("status") != "open":
            raise ValueError(f"iteration {doc['name']} is not open")
        path = doc.pop("path")
        if not (os.path.isfile(os.path.join(path, "scene.py"))
                and os.path.isfile(os.path.join(path, "pose.json"))
                and doc.get("passes")):
            raise ValueError(f"iteration {doc['name']} has no completed "
                             "verification pass")
        doc["status"] = "complete"
        doc["completed_at"] = _now()
        doc["updated_at"] = doc["completed_at"]
        staging = final = None
        try:
            staging, final = _stage_composites(run_dir, doc, path)
            _atomic_json(_metadata_path(path), doc)
            commit = _commit_iteration(run_dir, doc, path)
            if staging is not None:
                os.replace(staging, final)
                staging = None
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
        state = _state(run_dir)
        state["active_iteration"] = None
        state["last_completed_iteration"] = number
        state["last_commit"] = commit
        _write_state(run_dir, state)
        doc["path"] = path
        doc["git_commit"] = commit
        return doc


def complete_pass(run_dir, number, pass_dir):
    """Record a pass, then close and commit its iteration."""
    record_pass(run_dir, number, pass_dir)
    return complete(run_dir, number)


def abort(run_dir, number=None, reason=""):
    with _locked(run_dir):
        doc = active(run_dir, required=True) if number is None \
            else read_iteration(run_dir, number)
        path = doc.pop("path")
        doc["status"] = "aborted"
        doc["reason"] = str(reason).strip() or None
        doc["updated_at"] = _now()
        _atomic_json(_metadata_path(path), doc)
        state = _state(run_dir)
        if state.get("active_iteration") == doc["iteration"]:
            state["active_iteration"] = None
            _write_state(run_dir, state)
        doc["path"] = path
        return doc


def create_bundle(run_dir, out_path=None):
    """Export all per-run refs for collectors that omit the working tree."""
    run_dir = os.path.abspath(run_dir)
    ensure_git(run_dir)
    out_path = out_path or os.path.join(run_dir, "run-history.bundle")
    _git(run_dir, "bundle", "create", out_path, "--all")
    return out_path
