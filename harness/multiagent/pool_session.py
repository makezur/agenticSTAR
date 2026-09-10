#!/usr/bin/env python3
"""pool_session.py — start the shared render pool a windows round needs, SAFELY.

WHY THIS EXISTS. Window refiners all submit to ONE pool: N private pools of
resident EEVEE Blenders oversubscribe the GPU. So somebody has to start that one
pool and own it for the round, and a hand-launched `python -m pool.manager &` is
unprotected in ways that fail silently and expensively: a bare pool takes no
per-run lock (so a concurrent `shape_pass` pass
launches a SECOND pool over the same GPU), takes no per-GPU host lock (so
another RUN on the same machine does), and records no pgid (so when the
orchestrator's shell dies the Blender workers reparent to init and keep
claiming orders while the next round resets the spool underneath them).
`pool/locks.py` and `pool/session.py` provide each of those for the pass
coordinators (shape_pass / composite_pass); this makes the same protections
available to a windows round, as a command, so the safe path is the easy one.

WHAT IT DOES, in the order that matters:

  1. take the per-RUN coordinator lock (`pool/locks.acquire_run_lock`, the same
     lockfile the pass tool uses) — one GPU coordinator per run, released
     automatically even on SIGKILL;
  2. REAP any orphaned pool over this spool, only now that the lock is held: no
     other live coordinator can be starting a pool, so anything alive here is a
     dead coordinator's orphan. Processes only — never the cached renders, which
     is what lets a re-run reuse completed work;
  3. take the per-GPU HOST lock (`acquire_gpu_lock`) — serializes different runs
     on one physical GPU;
  4. spawn the manager through `pool/session.PoolSession`, which puts the whole
     micromamba -> manager -> Blender tree in one process group and records its
     pgid beside the spool, so a later invocation can reap it if we die;
  5. hold, printing the spool path refiners submit to, until interrupted — then
     tear the group down (killpg, not just the wrapper) and release both locks.

It stays in the FOREGROUND deliberately. A backgrounded pool is one whose owner
has already forgotten it; keeping it attached to a terminal (or a long-running
orchestrator step) means the process holding the locks is the process a human
can see, and Ctrl-C is a clean teardown rather than an orphaning event.

`--check` is the read-only question "is a pool already up for this run?", which
is what a refiner or a second orchestrator step should ask before assuming.

Runs from harness/:
    micromamba run -n artscript env PYTHONPATH=harness \\
        python -m multiagent.pool_session --run-dir RUN_DIR [--workers N]
    ... --run-dir RUN_DIR --check
"""

import argparse
import os
import signal
import sys

from core import run_layout
from pool import locks as pool_locks
from pool import manager as pool_manager
from pool import session as pool_session
from utils import _run_config


def log(msg):
    print(f"[pool-session] {msg}", flush=True)


def spool_path(run_dir):
    """The one spool a windows round shares. `plan` writes this same path into
    every brief (plan.json's `spool`), so the pool and the refiners agree on it
    without either being told twice."""
    return os.path.join(os.path.abspath(run_dir), "spool")


def _install_signals(pgid):
    """One-shot SIGTERM/SIGINT handlers that fast-TERM the pool group and then
    unwind through `serve`'s finally, which runs the full teardown (killpg +
    pgid clear + lock release).

    Without this an external SIGTERM skips the finally entirely and orphans the
    pool — the exact failure a hand-launched `pool.manager &` has. The handler
    does ONE async-safe syscall and then re-raises as SystemExit; everything
    else happens on the normal unwind path. Main-thread only (a signal.signal
    constraint), which `main()` satisfies."""
    old = {}

    def _handler(signum, frame):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        signal.signal(signum, old.get(signum, signal.SIG_DFL))    # one-shot
        raise SystemExit(128 + signum)            # -> finally does the rest

    for s in (signal.SIGTERM, signal.SIGINT):
        old[s] = signal.signal(s, _handler)
    return old


def _restore_signals(old):
    for s, handler in (old or {}).items():
        signal.signal(s, handler)


def check(run_dir):
    """Is a pool already up over this run's spool? Read-only.

    Answers the question a refiner (or a second orchestrator step) should ask
    before starting anything: `{"up": bool, "pgid": int|None, "spool": str}`.
    Reads the recorded pgid and tests whether that process group is actually
    alive, so a stale pidfile from a killed pool reads as `up: false` rather
    than as a pool nobody can find."""
    spool = spool_path(run_dir)
    pgid = pool_locks.read_pool_pgid(spool)
    up = bool(pgid and pool_locks.group_alive(pgid))
    return {"up": up, "pgid": pgid if up else None, "spool": spool}


def serve(run_dir, workers=None, gpus=None, engine="BLENDER_EEVEE_NEXT",
          samples=1, hold=True):
    """Start the round's shared pool under both locks and hold it.

    `workers` / `gpus` default to the run's `run_config.json` (the same
    config the pass tool reads, so one run has one answer), and `workers` is
    clamped to the box (`pool.manager.clamp_workers`) — 16 workers on a 4-core
    machine is the oversubscription this whole module exists to prevent.

    `hold=False` starts the pool, verifies it came up, and tears it straight
    back down: the smoke test, and what the tests use.
    """
    run_dir = os.path.abspath(run_dir)
    scene = os.path.join(run_dir, "scene.py")
    if not os.path.isfile(scene):
        raise ValueError(f"missing scene.py at {scene}")
    layout = run_layout.load_run_layout(run_dir)
    if layout is None:
        raise ValueError(f"missing or invalid layout.json in {run_dir}")
    spool = spool_path(run_dir)

    config = _run_config.load_config(run_dir)
    concurrency = _run_config.section(config, "concurrency")
    pool_cfg = _run_config.section(config, "pool")
    want = int(_run_config.pick(workers, concurrency, "workers", 3))
    clamped = pool_manager.clamp_workers(want, os.cpu_count())
    if clamped != want:
        log(f"clamping workers {want} -> {clamped} ({os.cpu_count()} cores; "
            "see pool.manager.clamp_workers)")
    gpu_value = gpus if gpus is not None else pool_cfg.get("gpus", "")
    gpu_spec = (",".join(str(v) for v in gpu_value)
                if isinstance(gpu_value, (list, tuple))
                else str(gpu_value or ""))

    # 1. per-RUN coordinator lock, before anything touches the spool or the GPU.
    lock = pool_locks.acquire_run_lock(run_dir)
    sess = None
    old_handlers = None
    gpu_locks = []
    try:
        # 2. only NOW is reaping safe: holding the run lock means no other live
        #    coordinator can be mid-launch, so any live pool here is an orphan.
        #    Kills processes via the recorded pgid; never touches cached renders.
        pool_locks.reap_pool(spool)
        # 3. per-GPU HOST lock: serialize different RUNS on one physical GPU.
        gpu_locks = pool_locks.acquire_gpu_lock(gpu_spec)
        # match_res is the REFERENCE FRAME'S IMAGE: renders come out at the
        # source resolution, so an IoU is computed against pixels of the same
        # size rather than a rescaled mask.
        cmd = pool_session.pool_cmd(
            scene, spool, clamped, layout["ref_frame"],
            os.path.join(layout["frames_dir"], layout["ref_frame"]),
            engine=engine, samples=samples,
            tracking=layout.get("tracking", ""), gpus=gpu_spec)
        log(f"starting {clamped} resident Blender worker(s) on "
            f"{gpu_spec or 'default GPU'}")
        # 4. PoolSession owns the process group + the recorded pgid.
        sess = pool_session.PoolSession(cmd, spool)
        old_handlers = _install_signals(sess.pgid)
        log(f"pool up (pgid {sess.pgid}); refiners submit to this spool:")
        log(f"  {spool}")
        log("locks held: per-run coordinator + per-GPU host. Leave this running "
            "for the whole round; Ctrl-C (or SIGTERM) tears the pool down "
            "cleanly and releases both.")
        if not hold:
            return {"ok": True, "pgid": sess.pgid, "spool": spool,
                    "workers": clamped}
        # 5. hold. The manager serves the spool; we exist to own the locks and
        #    the process group, so there is nothing to do but wait for it or for
        #    a signal.
        sess.proc.wait()
        log("the pool manager exited on its own")
        return {"ok": True, "pgid": None, "spool": spool, "workers": clamped}
    finally:
        # Restore handlers first, then tear the GROUP down and release the locks.
        # Runs on normal exit, on an exception, and on a graceful signal (whose
        # handler raises SystemExit into here). Only an external SIGKILL skips
        # it — which is what the recorded pgid + the next run's reap backstop.
        if old_handlers is not None:
            _restore_signals(old_handlers)
        if sess is not None:
            sess.close()
        for handle in gpu_locks:
            handle.close()
        lock.close()
        log("pool torn down; locks released")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--workers", type=int, default=None,
                   help="resident Blender workers (default: the run's "
                        "run_config.json, clamped to this host)")
    p.add_argument("--gpus", default=None,
                   help="GPU ids, comma-separated (default: the run's config)")
    p.add_argument("--engine", default="BLENDER_EEVEE_NEXT")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--check", action="store_true",
                   help="read-only: is a pool already up for this run? Exits 0 "
                        "when one is, 1 when not")
    p.add_argument("--no-hold", action="store_true",
                   help="start the pool, confirm it came up, tear it down again "
                        "(a smoke test, not a way to leave one running — an "
                        "unheld pool holds no locks)")
    args = p.parse_args()

    if args.check:
        state = check(args.run_dir)
        if state["up"]:
            log(f"a pool IS up for this run (pgid {state['pgid']}), serving "
                f"{state['spool']}")
            return 0
        log(f"no pool is up for this run; nothing is serving {state['spool']}")
        return 1
    serve(args.run_dir, workers=args.workers, gpus=args.gpus,
          engine=args.engine, samples=args.samples, hold=not args.no_hold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
