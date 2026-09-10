"""session.py — the ONE owner of the render-pool process lifecycle.

Both pool drivers (the shape pass's per-pass throwaway pool and
multiagent.pool_session's long-lived windows-round pool) share the same
contract, previously duplicated in each:

  1. reap any ORPHANED pool over the spool (a coordinator killed hard leaves
     its manager's process group reparented to init, still claiming orders) — processes
     only, never the cached renders/results files;
  2. spawn `python -m pool.manager` under micromamba with
     start_new_session=True, so the whole micromamba -> manager -> Blender tree
     lives in ONE process group a single killpg can reach;
  3. record the pgid beside the spool (locks.write_pool_pgid) so a LATER
     invocation can reap this pool if we die before teardown runs;
  4. wait for order results (client.wait), failing hard on the first
     order that errors;
  5. tear the group down: optionally wait for a clean self-exit
     (--once-empty-exit pools drain and quit on their own), then SIGTERM the
     group, then SIGKILL as the backstop — and clear the recorded pgid.

The drivers keep what genuinely differs: WHAT they submit, resume semantics,
config resolution, and (multiagent.pool_session) the one-shot signal handlers +
GPU host locks around a long-lived pool.
"""

import os
import signal
import subprocess

from pool import client, locks

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(HARNESS)


def pool_cmd(scene, spool, workers, ref_frame, match_res,
             engine="BLENDER_EEVEE_NEXT", samples=1, tracking="", gpus="",
             once_empty_exit=False, run_dir=""):
    """The manager launch command both drivers build identically."""
    cmd = ["micromamba", "run", "-n", "artscript",
           "env", f"PYTHONPATH={HARNESS}",
           "python", "-m", "pool.manager",
           "--scene", scene, "--spool", spool,
           "--workers", str(workers),
           "--ref-frame", ref_frame,
           "--match-res", match_res,
           "--engine", engine, "--samples", str(samples)]
    if once_empty_exit:
        cmd.append("--once-empty-exit")
    if tracking:
        cmd += ["--tracking", tracking]
    if run_dir:
        cmd += ["--run-dir", run_dir]
    if gpus:
        cmd += ["--gpus", gpus]
    return cmd


def term_group(pgid, proc, timeout=30):
    """Tear down the whole pool process group: SIGTERM, wait for the group leader
    (our child), then SIGKILL if it won't go. Idempotent — safe to call from both
    a signal handler path and a finally. start_new_session at launch put the
    micromamba -> manager -> Blender tree under `pgid`, so one killpg reaches all
    of it (proc.terminate() would only hit the micromamba wrapper)."""
    if pgid is None:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        if proc is not None:
            proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


class PoolSession:
    """A spawned pool over one spool: pgid bookkeeping + teardown.

    Construct AFTER the spool is reaped/reset and the orders are decided; the
    session only owns the process tree. Use as:

        sess = PoolSession(cmd, spool)          # spawns, records pgid
        try:
            sess.wait_orders(ids, timeout)
        finally:
            sess.close(self_exit_timeout=30)    # or close() for immediate TERM
    """

    def __init__(self, cmd, spool, cwd=REPO):
        self.spool = spool
        self.proc = subprocess.Popen(cmd, cwd=cwd, start_new_session=True)
        self.pgid = os.getpgid(self.proc.pid)
        locks.write_pool_pgid(spool, self.pgid)

    def wait_orders(self, ids, timeout, error_cls=RuntimeError, label="sweep"):
        """Block for each order result; raise error_cls on the first failure.
        Returns {id: result}."""
        results = {}
        for rid in ids:
            res = client.wait(self.spool, rid, timeout=timeout)
            if not res.get("ok"):
                raise error_cls(
                    f"{label} {rid} failed: {res.get('error', 'unknown error')}")
            results[rid] = res
        return results

    def close(self, self_exit_timeout=None):
        """Tear the pool down and clear the recorded pgid (idempotent).

        self_exit_timeout: with --once-empty-exit the pool drains its queue and
        exits on its own — give it this many seconds to do so before the
        killpg backstop. None -> TERM the group immediately (a long-lived pool
        has no self-exit to wait for)."""
        if self.pgid is None:
            return
        if self_exit_timeout is not None:
            try:
                self.proc.wait(timeout=self_exit_timeout)
            except subprocess.TimeoutExpired:
                pass
        term_group(self.pgid, self.proc)
        locks.clear_pool_pgid(self.spool)
        self.pgid = None
