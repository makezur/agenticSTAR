"""Immutable render-pass directory allocation."""

import json
import os

from core import filehash

# The scene a pass rendered, copied into the pass dir. Lives here, bpy-free, so
# a reader can resolve it without importing render_wrapper (which imports bpy).
SCENE_SNAPSHOT = "scene_snapshot.py"

# The declared reason a pass exists (and what to keep if it is rejected).
# Written by render_wrapper at pass allocation; read back by shape_pass's
# lineage summary. Lives here, bpy-free, for the same reason SCENE_SNAPSHOT does.
INTENT_NAME = "intent.json"


def normalize_label(value):
    """Normalize optional metadata and reject the unsafe agent sentinel."""
    label = str(value or "").strip()
    if label.lower() == "final":
        raise ValueError(
            "--pass-label final is reserved and unsafe for agent workflows; "
            "omit --pass-label and use the printed numeric PASS_DIR.")
    return label


def allocate(root):
    """Atomically create and return the next zero-padded numeric pass directory."""
    os.makedirs(root, exist_ok=True)
    numbers = [
        int(name) for name in os.listdir(root)
        if name.isdigit() and os.path.isdir(os.path.join(root, name))
    ]
    candidate = max(numbers, default=0) + 1
    while True:
        name = f"{candidate:04d}"
        path = os.path.join(root, name)
        try:
            os.mkdir(path)
            return name, path
        except FileExistsError:
            candidate += 1


def snapshot_scene(pass_dir, scene_path, warn=print):
    """Snapshot the scene into the pass dir, and hash it, from ONE read.

    UNCONDITIONAL, every pass, no flag — same reason as the self-intersection
    report: it describes the state that was just rendered, and an agent cannot
    forget to ask for it.

    `scene_sha1` alone is a TRIPWIRE, not a record: it proves pass N's scene
    differs from pass N-1's, but a hash cannot say WHAT differs, and `runs/` is
    gitignored so there is no VCS to fall back on. Without a snapshot per pass
    the older states are simply gone — a run reconstructing "what did that pass
    change, and can I revert only part of it?" is left reading the CURRENT
    scene and guessing. With one, that question is a diff of two files.

    MUST BE CALLED BEFORE THE SCENE IS EXEC'D, and the snapshot and the hash
    come from the SAME `read()`. Both properties are load-bearing:

      * Before the exec, because renders take minutes. Read at the END of the
        pass (where the manifest is written) an edit landing mid-render is
        recorded as the scene that rendered, which it never was. The run lock
        serializes coordinators, not the agent's editor.
      * From one read, because the manifest documents "hashing the snapshot back
        MUST reproduce scene_sha1". Two separate reads make that check vacuous:
        the copy and the hash then agree with each other by construction,
        INCLUDING when both describe a file no render ever saw. One read is what
        gives the snapshot and any later re-hash independent provenance, so a
        disagreement means real corruption.

    Snapshotting before the exec also means a scene that FAILS to build still
    leaves the source that failed — the more useful artifact of the two.

    Returns {"scene_snapshot": basename or None, "scene_sha1": sha or None}; an
    unreadable scene degrades both (recorded in the manifest, not only on
    stdout) rather than failing a pass whose renders are fine.
    """
    try:
        with open(scene_path, "rb") as f:
            src = f.read()
    except OSError as exc:
        warn(f"WARNING: could not read scene {scene_path} to snapshot it: {exc}")
        return {"scene_snapshot": None, "scene_sha1": None}
    sha = filehash.bytes_sha1(src)
    try:
        with open(os.path.join(pass_dir, SCENE_SNAPSHOT), "wb") as f:
            f.write(src)
    except OSError as exc:
        # The hash survives a failed write: it describes bytes we DID read, so
        # the tripwire still works even when the record could not be kept.
        warn(f"WARNING: could not write scene snapshot into {pass_dir}: {exc}")
        return {"scene_snapshot": None, "scene_sha1": sha}
    return {"scene_snapshot": SCENE_SNAPSHOT, "scene_sha1": sha}


def write_intent(pass_dir, intent, rollback, warn=print):
    """Record WHY this pass exists, and what to keep if it is rejected.

    Written at pass allocation — before the render — because the declaration
    exists to survive an interruption, and a run that dies mid-render is
    exactly the interruption it must survive. Both strings come from the CLI,
    so they are fixed before any score exists to rationalize against.

    Absent flags write no file — this records a declaration, and inventing one
    would defeat the purpose.
    """
    intent = str(intent or "").strip()
    rollback = str(rollback or "").strip()
    if not intent and not rollback:
        return None
    path = os.path.join(pass_dir, INTENT_NAME)
    try:
        with open(path, "w") as f:
            json.dump({"pass": os.path.basename(pass_dir.rstrip(os.sep)),
                       "intent": intent or None,
                       "rollback": rollback or None}, f, indent=2)
    except OSError as exc:
        # Never fail a render over its own paper trail.
        warn(f"WARNING: could not write {INTENT_NAME}: {exc}")
        return None
    return path


def parent_of(root, name):
    """The pass `name` succeeds: the highest-numbered pass dir BELOW it, or None.

    Lineage is derived from the numbering rather than recorded at allocation
    time, so it is the same answer whoever asks and whenever they ask. `name`
    itself already exists on disk (allocate() created it), hence the strict <.

    Only zero-padded numeric dirs count — a legacy labelled dir is not a pass
    and cannot be a parent (same rule allocate() uses to pick the next number).
    """
    try:
        current = int(name)
    except (TypeError, ValueError):
        return None
    try:
        entries = os.listdir(root)
    except OSError:
        return None
    below = [
        int(entry) for entry in entries
        if entry.isdigit() and int(entry) < current
        and os.path.isdir(os.path.join(root, entry))
    ]
    return f"{max(below):04d}" if below else None
