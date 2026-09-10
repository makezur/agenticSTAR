# multiagent.pool_session — the round's shared render pool, under locks

## Why

A windows round has many refiner subagents and **one** render pool. That is not
a preference: N private pools of resident EEVEE Blenders oversubscribe the GPU,
so one pool = one GPU tenant.

Somebody has to start that pool and own it for the round. Do not hand-launch it:

```
python -m pool.manager --scene ... --spool ... --workers 3 &      # DON'T
```

A bare `pool.manager &` holds no locks, and every way that fails is silent:

| a bare `pool.manager &` | consequence |
|---|---|
| takes no per-run lock | a concurrent shape pass starts a **second** pool on the same GPU |
| takes no per-GPU host lock | another **run** on the same machine does the same |
| records no process group | when the launching shell dies, the Blender workers reparent to init and keep claiming orders — while the next round resets the spool underneath them |

`pool/locks.py` and `pool/session.py` provide each of those for the pass
coordinator. This module gives a windows round the same protections as a
command, so the safe path is also the easy one.

## Use

```
# start it, and LEAVE IT RUNNING for the whole round
micromamba run -n artscript env PYTHONPATH=harness \
    python -m multiagent.pool_session --run-dir RUN_DIR [--workers N] [--gpus 0]

# is one already up for this run?  (read-only; exit 0 = yes, 1 = no)
... python -m multiagent.pool_session --run-dir RUN_DIR --check
```

It serves `RUN_DIR/spool` — the same path `windows plan` writes into
`plan.json` and into every brief, so the pool and the refiners agree on it
without either being told twice.

`--workers` and `--gpus` default to the run’s `run_config.json` (the same
config the pass tool reads, so one run has one answer), and workers are clamped
to the box via `pool.manager.clamp_workers`.

## What it does, in the order that matters

1. **Per-run coordinator lock** (`pool.locks.acquire_run_lock`, the same
   `.pool.lock` the shape pass uses) — one GPU coordinator per run. Advisory
   `flock`, so it is released even on SIGKILL.
2. **Reap orphans** (`reap_pool`) — and only *now*, because holding the run lock
   means no other live coordinator can be mid-launch, so anything alive over
   this spool is a dead coordinator's orphan. Kills **processes** only; the
   cached `renders/` and `results/` files survive, which is what lets a re-run
   reuse completed work.
3. **Per-GPU host lock** (`acquire_gpu_lock`) — serializes different *runs* on
   one physical card. Keyed per GPU id, so a real multi-GPU box still
   parallelizes.
4. **Spawn through `pool.session.PoolSession`** — `start_new_session` puts the
   whole `micromamba -> manager -> Blender` tree in ONE process group, and the
   pgid is recorded beside the spool so a later invocation can reap it if this
   process dies.
5. **Hold**, then tear down: one-shot SIGTERM/SIGINT handlers fast-TERM the
   group and unwind through the `finally`, which does the full `killpg`, clears
   the pidfile, and releases both locks.

## It stays in the foreground on purpose

A backgrounded pool is one whose owner has already forgotten it. Attached, the
process holding the locks is a process you can see, and Ctrl-C is a clean
teardown instead of an orphaning event. `--no-hold` starts the pool, confirms it
came up, and stops — a smoke test, not a way to leave one running (an unheld
pool holds no locks).

## Refiners never start a pool

A refiner's brief tells it to submit orders to the spool and wait
(`pool.client --request`). If no pool is serving, the right response is to say
so, not to start one: a refiner that launches its own is exactly the
oversubscription this module prevents, and it would hold none of the locks.

## Files

- `harness/multiagent/pool_session.py` — this tool.
- `harness/pool/locks.py` — the run lock, the GPU lock, the pgid pidfile, and
  the orphan reaper (shared with `utils/shape_pass.py`).
- `harness/pool/session.py` — the process-group spawn/teardown both coordinators
  use.
- `harness/pool/README.md` § Order — the request schema refiners submit.
