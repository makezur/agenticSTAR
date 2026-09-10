#!/usr/bin/env python3
"""Process-lifecycle helpers shared by the GPU-pool coordinators (utils/shape_pass.py and
multiagent/pool_session.py): the per-run coordinator lock, a per-GPU host lock,
and the pool-process-group pidfile + orphan reaper.

Why this module exists
-----------------------
A coordinator (shape_pass / pool_session) spawns a resident render pool as a child.
If the coordinator is killed HARD (SIGKILL / a "window kill" of the foreground
tool call) its cleanup never runs, so the pool — double-forked through
`micromamba run` — survives and reparents to init. The next re-invocation of the
SAME command (by design: the Codex loop re-runs after a kill) then spawns a
SECOND pool on the same spool, and 2xG Blender workers duel over one GPU.

The primitives here close that hole:
  * acquire_run_lock  — one coordinator per run dir (advisory flock, auto-released
                        on death incl. SIGKILL).
  * acquire_gpu_lock  — one coordinator per physical GPU, so a second RUN on a
                        one-GPU box waits instead of thrashing the card.
  * write/read/clear_pool_pgid — record the pool's process-group id in the spool
                        so a later invocation can find and reap a leftover pool.
  * reap_pool         — kill any orphaned prior pool GROUP (killpg) before we reset
                        the spool and launch a fresh pool. Kills processes, never
                        the cached render/result FILES.
  * archive_spool     — the debuggable alternative to `rmtree(spool)`: RENAME the
                        previous spool aside to `spool-<timestamp>/` so its orders,
                        reports, and panels survive the next pass.
"""

import fcntl
import os
import shutil
import signal
import time


# --------------------------------------------------------------------------- #
# coordinator lock (per run dir)
# --------------------------------------------------------------------------- #
def acquire_run_lock(run_dir, name=".pool.lock"):
    """Hold an exclusive coordinator lock for this process and run directory.

    Advisory `flock` released automatically when the process dies (including
    SIGKILL), so a crashed coordinator never wedges the lock. ONE filename for
    every coordinator, on purpose: the contended resource is the single per-run
    GPU pool, which BOTH the shape pass (utils/shape_pass.py) and a windows
    round's pool (multiagent/pool_session.py) drive, so sharing the file
    serializes every same-run driver (pass x pass, round pool x round pool, and
    pass x round pool). The name states the RESOURCE."""
    path = os.path.join(run_dir, name)
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit(
            f"[run-lock] another coordinator is already running for {run_dir} "
            f"(lock: {path})")
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()) + "\n")
    handle.flush()
    return handle


# --------------------------------------------------------------------------- #
# GPU lock (per physical GPU id, host-wide)
# --------------------------------------------------------------------------- #
def acquire_gpu_lock(gpus, lock_dir="/tmp"):
    """Hold an exclusive host-wide lock per GPU id so a DIFFERENT run cannot drive
    the same physical GPU concurrently. `gpus` is the comma string passed to
    the manager's --gpus (e.g. "0" or "0,1"); an empty value means "GPU unspecified"
    -> no lock (matches the manager letting Blender see all devices).

    Keyed per id, so a real multi-GPU box still parallelizes across distinct GPUs.
    Returns the list of open lock handles (empty when `gpus` is empty); the caller
    keeps them open for the pool's lifetime and closes them to release."""
    ids = [g.strip() for g in str(gpus or "").split(",") if g.strip()]
    handles = []
    for gid in ids:
        path = os.path.join(lock_dir, f"artscript-gpu-{gid}.lock")
        handle = open(path, "a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            for h in handles:                       # release any already taken
                h.close()
            raise SystemExit(
                f"[run-lock] GPU {gid} is already in use by another run "
                f"(lock: {path})")
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()) + "\n")
        handle.flush()
        handles.append(handle)
    return handles


# --------------------------------------------------------------------------- #
# pool process-group pidfile (in the spool)
# --------------------------------------------------------------------------- #
def _pgid_path(spool):
    return os.path.join(spool, "pool.pgid")


def write_pool_pgid(spool, pgid):
    """Record the pool's process-group id so a later invocation can reap a leftover
    pool. Mirrors acquire_run_lock's `pid + "\\n"` format."""
    os.makedirs(spool, exist_ok=True)
    with open(_pgid_path(spool), "w") as f:
        f.write(str(int(pgid)) + "\n")


def read_pool_pgid(spool):
    """Return the recorded pool pgid, or None if the pidfile is missing/malformed.
    Never raises — a corrupt pidfile must not crash startup."""
    try:
        with open(_pgid_path(spool)) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def clear_pool_pgid(spool):
    """Remove the pidfile; ignore an already-absent file."""
    try:
        os.remove(_pgid_path(spool))
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------- #
# orphan reaper
# --------------------------------------------------------------------------- #
def group_alive(pgid):
    """True iff any member of process group `pgid` is alive. `killpg(pgid, 0)` is a
    permission/existence probe that signals nothing.

    Public because a recorded pgid is only meaningful together with this check:
    `read_pool_pgid` alone cannot distinguish a running pool from a pidfile a
    SIGKILLed one left behind, so anyone ASKING "is a pool up?" (e.g.
    multiagent.pool_session --check) needs both."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # A group we can't signal isn't ours to reap; treat as gone.
        return False


def reap_pool(spool, term_timeout=10.0, poll=0.1):
    """Kill any orphaned prior pool recorded in `spool/pool.pgid`, then clear it.

    The orphan is NOT our child (its coordinator died and init reparented it), so
    we probe/kill by process GROUP with signal 0 / SIGTERM / SIGKILL rather than
    waitpid. Called AFTER the run lock is held (so no other live coordinator is
    starting a pool) and BEFORE the spool reset + fresh launch, so a stale pool can
    never write results/ while we reset. Only processes are killed — the cached
    renders/ and results/ FILES are untouched, preserving the resume cache."""
    pgid = read_pool_pgid(spool)
    if pgid is None:
        return
    # Never signal our own group — a corrupt pidfile must not kill the coordinator
    # (or the Codex loop that spawned it).
    if pgid <= 1 or pgid == os.getpgrp():
        clear_pool_pgid(spool)
        return
    if not group_alive(pgid):
        clear_pool_pgid(spool)
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        clear_pool_pgid(spool)
        return
    deadline = time.monotonic() + term_timeout
    while time.monotonic() < deadline:
        if not group_alive(pgid):
            clear_pool_pgid(spool)
            return
        time.sleep(poll)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    clear_pool_pgid(spool)


# --------------------------------------------------------------------------- #
# spool reset — archive (debuggable) vs wipe
# --------------------------------------------------------------------------- #
ARCHIVE_PREFIX_DEFAULT = "spool"


def archive_spool(spool, keep=8, prefix=""):
    """Reset `spool` by RENAMING it aside instead of deleting it. Returns the
    archive path, or "" when there was no spool to move.

    A coordinator must start each pass from an empty spool: `pool_client.wait`
    returns as soon as `results/<id>.json` exists, and the shape pass reuses the SAME
    order ids every pass (id == frame stem), so a leftover result would be
    mistaken for this pass's render. `rmtree` satisfies that — and destroys the
    evidence. A rename satisfies it just as completely (the spool path is empty
    either way) while keeping every order, report, and panel readable.

    This matters more than "one tool’s own scratch": a run's `RUN_DIR/spool` is
    shared with the WINDOW AGENTS, whose sweep/oapply orders and candidate sheets
    are the only record of why a pose was chosen. A wipe takes those too.

    `os.rename` is atomic within a filesystem and does NOT follow into the tree,
    so it cannot race a straggler worker writing inside: the worker's open fds
    follow the inode to the new path and land in the archive, never in the fresh
    spool. Callers still `reap_pool` first — this is a fallback, not a licence to
    skip the kill.

    `keep` prunes the oldest archives to bound disk (0 or negative disables
    pruning). The timestamp is second-resolution, so a
    same-second second call disambiguates with a `-2`, `-3`, ... suffix rather
    than clobbering the earlier archive.
    """
    if not os.path.isdir(spool):
        return ""
    parent = os.path.dirname(os.path.abspath(spool)) or "."
    stem = prefix or os.path.basename(os.path.abspath(spool)) or ARCHIVE_PREFIX_DEFAULT
    base = os.path.join(parent, f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}")
    dest, n = base, 1
    while os.path.exists(dest):
        n += 1
        dest = f"{base}-{n}"
    os.rename(spool, dest)
    prune_spool_archives(parent, stem, keep)
    return dest


def _archive_dirs(parent, stem):
    """Existing `<stem>-<timestamp>` archive dirs under `parent`, oldest first.

    Ordered by mtime, NOT by name. Name order looks chronological for a fixed-width
    `%Y%m%d-%H%M%S` stamp, but pruning frees a name for reuse: once
    `<stem>-<ts>` is pruned, the next archive in that same second takes the bare
    `<stem>-<ts>` again (the `-2` suffix only avoids a LIVE collision), and that
    newest archive then sorts oldest and gets pruned first. mtime survives the
    rename — it is the time the spool's contents last changed, i.e. its pass — so
    it stays correct however names are recycled. The live `<stem>` dir never
    matches (no `-` suffix)."""
    try:
        names = os.listdir(parent)
    except OSError:
        return []
    pre = stem + "-"
    found = [os.path.join(parent, n) for n in names
             if n.startswith(pre) and os.path.isdir(os.path.join(parent, n))]

    def age(path):
        try:
            return (os.stat(path).st_mtime_ns, path)
        except OSError:                      # vanished under us — prune candidate
            return (0, path)
    return sorted(found, key=age)


def prune_spool_archives(parent, stem, keep):
    """Delete all but the newest `keep` archives. No-op when `keep` <= 0 (an
    explicit "never prune"), so an unbounded-retention caller can't lose data to a
    default."""
    if keep is None or keep <= 0:
        return []
    victims = _archive_dirs(parent, stem)[:-keep] if keep else []
    for path in victims:
        shutil.rmtree(path, ignore_errors=True)
    return victims
