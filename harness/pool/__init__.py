"""pool — the resident Blender render pool: manager, worker, client, lifecycle.

Renders cost Blender startup + `build()` on every process. This package amortizes
that: G Blender workers stay RESIDENT (each has run `build()` + capture_canonical
once) and per-request renders are dispatched to them over pipes, so a scaled loop
pays the setup cost G times per scene edit instead of once per render. Layered as
a downward DAG, split by WHICH RUNTIME each piece lives in:

    session  →  manager  →  serve        (client / locks are shared leaves)
    (artscript)  (artscript)  (inside Blender)

  * `manager` — the pool manager (runs OUTSIDE Blender). Spawns/recycles the G
                workers, watches the file spool (orders → claimed → results),
                dispatches render requests, clamps the pool against cores and
                visible GPUs, pins workers to GPUs via bwrap. `python -m
                pool.manager`.
  * `serve`   — the resident worker loop, entered INSIDE Blender from
                `render_wrapper.py --serve`. Owns the request schema and the
                per-request sweep/apply/oapply/osweep override machinery, plus the
                SENTINEL stdout control protocol the manager parses. Kept
                `bpy`-free at module level so the manager and tests can import
                `SENTINEL` without Blender; `rig`/`views` load lazily.
  * `client`  — minimal file-spool client: drop an order, block on its result.
  * `session` — the ONE owner of the pool PROCESS lifecycle: reap orphans, spawn
                the manager under micromamba in its own process group, record the
                pgid, wait on orders, tear the whole group down with one killpg.
  * `locks`   — the locking primitives the lifecycle rests on: per-run coordinator
                flock, per-GPU host lock, pool-pgid pidfile + orphan reaper.
  * `ledger`  — the spool's append-only `ledger.jsonl`: submit/claim/result per
                order, worker deaths, pool start/recycle/stop. The manager's
                `log()` narration goes to stdout, which a session-spawned pool
                does not keep, so this is the only on-disk record of order
                TIMING, worker attribution, and retries. `python -m pool.ledger
                --spool DIR` reports it.

Callers (`utils/shape_pass.py`, `multiagent/pool_session.py`) drive the pool through
`session` + `client` and never touch `manager` directly. See pool/README.md for the
order schema, the worker protocol, and the clamping rules.

Invoked as a module from `harness/`:
    micromamba run -n artscript env PYTHONPATH=harness python -m pool.manager ...
"""
