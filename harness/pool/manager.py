#!/usr/bin/env python3
"""manager.py — the resident Blender worker-pool manager (the render pool).

Runs OUTSIDE Blender (in the artscript env). Keeps G Blender workers resident —
each has run build() + capture_canonical ONCE (see pool/serve.py) — and
dispatches per-request renders to them over pipes, so callers never pay the
Blender startup + build() cost on every render (the cost sweep.md calls out:
"each process pays Blender startup + build(), and EEVEE processes contend for one
GPU"). This is the "G knob" + "build dispatch to all workers" a scaled loop needs.

    micromamba run -n artscript env PYTHONPATH=harness python -m pool.manager \
        --scene RUN_DIR/scene.py --spool RUN_DIR/spool --workers 2 \
        [--tracking DIR] [--match-res IMG] [--engine ...] [--samples N] [--gpus 0,1]

Front door = a FILE SPOOL (not HTTP — deliberately, so we do not re-emulate a
per-request render.sh fork behind a server). Layout under --spool:
    orders/<id>.json    a client drops a request here (see pool/serve.py for schema;
                        omit "out" — the manager allocates it).
    claimed/<id>.json   the manager moves it here (atomic rename) once claimed.
    renders/<id>/       the manager-allocated output dir the worker renders into
                        (SOLE allocator -> no iterations/NNNNNN/renders/NNNN
                        check-then-create race).
    results/<id>.json   the manager writes the worker's result here ATOMICALLY
                        (temp + os.replace), so a polling client never reads a
                        half-written file.
    workers/<slot>/     each worker's own scratch --out (its harmless startup
                        pass dir lands here, off the shared root -> no race).
    ledger.jsonl        append-only lifecycle log: submit/claim/result per order
                        plus pool start/recycle/stop and worker deaths (see
                        pool/ledger.py). The only on-disk record of TIMING, the
                        worker slot, and retries — log() narrates those to stdout,
                        which a session-spawned pool has nowhere to keep.

Recycle ("build dispatch to all workers"): the manager watches --scene's sha1
(the same hash render_wrapper stamps into MANIFEST.json). On a change it DRAINS
in-flight requests, SIGTERMs all G workers, and relaunches G fresh ones (each
re-runs build()). Clean factory-startup bpy each time — identical to today's
proven cold path, just amortized across many requests between edits.

Scope: this is the pool + its file-spool API. `shape_pass.sh RUN_DIR --pool`
spawns one per pass (via pool/session.py) and drives it through pool/client.py,
the spool's order client. See pool/README.md.
"""

import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

from core import captures, filehash, run_layout
from pool import ledger, panels, reseed
from pool.serve import SENTINEL
from utils import _run_config

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RENDER_WRAPPER = os.path.join(REPO, "harness", "render_wrapper.py")
BLENDER = os.environ.get("BLENDER", "/home/ubuntu/blender/blender")


def _blender_system_resources():
    """The bundled <major>.<minor>/ resource dir adjacent to the Blender binary,
    or "" if not found. Blender normally discovers this by walking up from its
    own executable, but that walk fails in some sandboxes ("could not get a list
    of mounted file-systems"); passing BLENDER_SYSTEM_RESOURCES sidesteps it."""
    import glob
    base = os.path.dirname(BLENDER)
    for cand in sorted(glob.glob(os.path.join(base, "[0-9]*.[0-9]*"))):
        if os.path.isdir(os.path.join(cand, "scripts")) and \
           os.path.isdir(os.path.join(cand, "python")):
            return cand
    return ""


def log(msg):
    print(f"[pool] {msg}", flush=True)


# Blender/OIIO write this to stderr when a worker catches a signal mid-syscall
# during pool teardown (EINTR = errno 4). It is entirely benign — the worker is
# being shut down on purpose — but a bare "Interrupted system call" line reads
# like a crash and has repeatedly tricked the operator into "relaunching" a pool
# that shut down cleanly. Drop it so a normal teardown looks like one.
_BENIGN_STDERR_MARKERS = ("Interrupted system call", "error code 4")


def _is_benign_teardown_line(line):
    return all(m in line for m in _BENIGN_STDERR_MARKERS)


# Per-GPU worker cap: with the GPU-raster scoring loop (rig/raster.py) a sweep
# worker only touches the GPU briefly per candidate, and 16 resident workers per
# GPU is where throughput flattens with VRAM headroom left.
# Env-tunable (POOL_WORKERS_PER_GPU): retune whenever the per-candidate loop
# changes, don't hunt for this constant.
MAX_WORKERS_PER_GPU = int(os.environ.get("POOL_WORKERS_PER_GPU", "16"))


def visible_gpu_count():
    """Number of GPUs this process can reach, by counting /dev/nvidia[0-9]*
    device nodes (works both on the host and inside a device-isolated sandbox,
    where only the assigned subset is bound). 0 when none are visible."""
    import glob
    return len(glob.glob("/dev/nvidia[0-9]*"))


# --------------------------------------------------------------------------- #
# GPU pinning via device isolation (bwrap)
# --------------------------------------------------------------------------- #
# CUDA_VISIBLE_DEVICES pins CUDA (Cycles) but is IGNORED by EEVEE, whose EGL
# context enumerates GPUs itself and lands on GPU 0 regardless. The only lever
# that moves an EEVEE worker is device-node isolation: a mount namespace
# whose /dev holds ONLY that worker's /dev/nvidiaN. bwrap provides exactly
# that. Inside an already-sandboxed agent (codex-articulated) nested bwrap is
# impossible (the outer sandbox drops CAP ALL) — there the sandbox launcher is
# expected to have bound a GPU subset already, so the env-only fallback is
# correct: every visible GPU is ours.
_GPU_ISOLATION_SHARED_DEVICES = (
    "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools",
    "/dev/nvidia-modeset",
)


def bwrap_gpu_prefix(gpu):
    """The bwrap argv prefix that hides every GPU except /dev/nvidia<gpu>.

    `--bind / /` keeps the whole filesystem as-is (this is isolation, not a
    sandbox); `--dev /dev` swaps in a fresh minimal /dev; the dev-binds put
    back the one GPU node plus the shared control/UVM nodes. Returns None when
    the node for `gpu` doesn't exist (not a /dev/nvidiaN index)."""
    node = f"/dev/nvidia{gpu}"
    if not os.path.exists(node):
        return None
    prefix = ["bwrap", "--die-with-parent", "--bind", "/", "/", "--dev", "/dev",
              "--dev-bind", node, node]
    for dev in _GPU_ISOLATION_SHARED_DEVICES:
        if os.path.exists(dev):
            prefix += ["--dev-bind", dev, dev]
    if os.path.isdir("/dev/nvidia-caps"):
        prefix += ["--dev-bind", "/dev/nvidia-caps", "/dev/nvidia-caps"]
    if os.path.isdir("/dev/dri"):
        prefix += ["--dev-bind", "/dev/dri", "/dev/dri"]
    return prefix


def gpu_isolation_works():
    """True when a bwrap device-isolation prefix can actually run here.

    False inside an already-bwrapped sandbox (nested user namespaces are
    forbidden) or when bwrap is missing — callers then fall back to env-only
    pinning, which is fine exactly there (the sandbox launcher already bound
    only this run's GPU subset)."""
    if not shutil.which("bwrap"):
        return False
    probe = ["bwrap", "--bind", "/", "/", "--dev", "/dev", "true"]
    try:
        return subprocess.run(probe, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def clamp_workers(requested, cores, allow_oversubscribe=False, n_gpus=None):
    """Cap the pool at cores-1 AND (when the GPU count is known) at
    MAX_WORKERS_PER_GPU per visible GPU (min 1): too many EEVEE Blender
    processes contending for one GPU crash outright (GL context churn) rather
    than slowing down gracefully, so the per-GPU cap is the binding one."""
    if allow_oversubscribe:
        return requested
    capped = requested
    if cores:
        capped = min(capped, cores - 1)
    if n_gpus:
        capped = min(capped, MAX_WORKERS_PER_GPU * n_gpus)
    return max(1, capped)


# sha1 of scene.py — the recycle trigger. Whole-file digest (see core.filehash):
# any edit, incl. FRAMES/comments, may recycle on a pose-only edit too.
scene_sha1 = filehash.file_sha1_or_none


def resolve_hand_mask(req, run_dir):
    """Default-ON hand-occlusion waiver: fill sweep.hand_mask from the run
    layout when the order didn't decide for itself.

    On hand-held captures a sweep scored WITHOUT the hand mask silently gates on
    iou_raw — hand-occluded pixels count as misses and the numeric winner drifts
    toward covering the hand. Forgetting the key (or misplacing it at the
    request top level) is indistinguishable from opting out, so the manager
    resolves it: an order whose "sweep" object has NO "hand_mask" key gets
    <layout hand_masks_dir>/<frame stem>.png when that file exists (the shared
    per-frame join, `captures.mask_for`). An order that DOES carry the
    key — any value, including "" — has decided, and is left alone: "" is the
    explicit opt-out.

    Returns the filled path, or None when nothing was filled (no run layout,
    no hand_masks_dir, no per-frame file, no "sweep" object, or explicit key).
    """
    sweep = req.get("sweep")
    if not run_dir or not isinstance(sweep, dict) or "hand_mask" in sweep:
        return None
    frame = req.get("frame")
    if not frame:
        return None
    layout = run_layout.load_run_layout(run_dir)
    hand_dir = layout.get("hand_masks_dir") if layout else ""
    if not hand_dir:
        return None
    cand = captures.mask_for(hand_dir, str(frame))
    if not os.path.isfile(cand):
        return None
    sweep["hand_mask"] = cand
    return cand


def resolve_sweep_mask(req, run_dir):
    """Default-ON SCORING mask fill: sweep.mask from the run layout's masks_dir
    when a sweep/apply order didn't state it.

    `sweep.mask` is REQUIRED — without it `engine._prepare` prints "--sweep-mask ...
    is required; skipping" and returns before the first render, and because that is
    a clean return the worker would report ok:true with an EMPTY render dir. The
    mask path is a pure function of the frame, so it is derived here.

    Same contract as the other two fills: a key the order DOES carry — any value,
    including "" — has decided and is left alone. Unlike them, "" here does not mean
    "score without it" (there is no such mode); it means "I am handling this", and
    the engine's own guard still refuses an empty mask. Returns the filled path, or
    None."""
    sweep = req.get("sweep")
    if not run_dir or not isinstance(sweep, dict) or "mask" in sweep:
        return None
    # only the camera-frame verbs read sweep.mask; the object verbs take masks_dir
    # (resolve_object_masks). A `match`-only order has no scoring at all.
    from core.sweep_family import SWEEP_MASK_VIEWS
    views = [v.strip() for v in str(req.get("views") or "").split(",") if v.strip()]
    if not any(v in SWEEP_MASK_VIEWS for v in views):
        return None
    frame = req.get("frame")
    if not frame:
        return None
    layout = run_layout.load_run_layout(run_dir)
    masks_dir = layout.get("masks_dir") if layout else ""
    if not masks_dir:
        return None
    cand = captures.mask_for(masks_dir, str(frame))
    if not os.path.isfile(cand):
        return None
    sweep["mask"] = cand
    return cand


def resolve_object_masks(req, run_dir):
    """Default-ON masks fill for the MULTI-FRAME views: fill masks_dir /
    hand_masks_dir inside an order's "oapply"/"oapply_all"/"osweep" object from the
    run layout when the order didn't decide for itself.

    The one-shot render.sh path gets --masks-dir on the command line, but
    the manager's worker command doesn't carry it — without this fill every pooled
    object-centric order would have to restate paths the run layout already
    knows (and a forgotten masks_dir makes the view skip entirely). Same
    contract as resolve_hand_mask: a key the order DOES carry — any value,
    including "" — is left alone ("" is the explicit opt-out).

    Returns {key: filled_path} for what was filled (may be empty)."""
    filled = {}
    objs = [req.get(k) for k in ("oapply", "oapply_all", "osweep")]
    objs = [o for o in objs if isinstance(o, dict)]
    if not run_dir or not objs:
        return filled
    layout = run_layout.load_run_layout(run_dir)
    if layout is None:
        return filled
    for key in ("masks_dir", "hand_masks_dir"):
        mdir = layout.get(key) or ""   # already absolutized by load_run_layout
        if not mdir or not os.path.isdir(mdir):
            continue
        for obj in objs:
            if key not in obj:
                obj[key] = mdir
                filled[key] = mdir
    return filled


# --------------------------------------------------------------------------- #
# atomic result write (harness has NO atomic-write helper — this is net-new and
# is the guard for "a polling client never sees a partial results/<id>.json")
# --------------------------------------------------------------------------- #
def write_result_atomic(results_dir, rid, result):
    """Write results/<id>.json via a temp file + os.replace (atomic on the same
    filesystem). The '.' prefix keeps the temp out of a client's <id>.json glob."""
    os.makedirs(results_dir, exist_ok=True)
    final = os.path.join(results_dir, f"{rid}.json")
    tmp = os.path.join(results_dir, f".{rid}.json.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(result, indent=2))
    os.replace(tmp, final)
    return final


# --------------------------------------------------------------------------- #
# a single resident Blender worker
# --------------------------------------------------------------------------- #
class Worker:
    """One resident Blender --serve process. One request in flight at a time
    (renders are serial inside Blender), so a single slot thread owns it."""

    def __init__(self, idx, base_cmd, scratch_dir, gpu, verbose, isolate=False):
        self.idx = idx
        self.base_cmd = base_cmd
        self.scratch_dir = scratch_dir
        self.gpu = gpu
        self.verbose = verbose
        self.isolate = isolate
        self.p = None
        self.sha1 = None
        self._err_thread = None
        self._err_log = None

    def start(self):
        os.makedirs(self.scratch_dir, exist_ok=True)
        env = dict(os.environ)
        if not env.get("BLENDER_SYSTEM_RESOURCES"):
            res = _blender_system_resources()
            if res:
                env["BLENDER_SYSTEM_RESOURCES"] = res
        cmd = self.base_cmd + ["--out", self.scratch_dir]
        if self.gpu is not None:
            # CUDA_VISIBLE_DEVICES pins Cycles; EEVEE ignores it (its EGL context
            # picks GPU 0 regardless), so when the host allows it the worker is
            # additionally wrapped in a bwrap device namespace holding only its
            # /dev/nvidiaN — the one lever that actually moves an EEVEE worker.
            # Inside an already-bwrapped sandbox (no nesting) env-only is right:
            # the sandbox launcher bound this run's GPU subset already.
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
            if self.isolate:
                prefix = bwrap_gpu_prefix(self.gpu)
                if prefix:
                    cmd = prefix + cmd
        self.p = subprocess.Popen(
            cmd, cwd=REPO, env=env, text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=1)
        # Persist stderr to the slot's scratch dir — a worker that dies mid-render
        # (GPU/GL crash, signal) leaves its evidence HERE, not in the void.
        self._err_log = open(os.path.join(self.scratch_dir, "stderr.log"), "a")
        # Drain stderr in a daemon thread — Blender is verbose and a full stderr
        # pipe would deadlock the worker. Forward to console only when --verbose.
        self._err_thread = threading.Thread(
            target=self._drain_stderr, daemon=True)
        self._err_thread.start()

    def _drain_stderr(self):
        for line in self.p.stderr:
            if _is_benign_teardown_line(line):
                continue                        # benign EINTR at teardown; hide it
            if self._err_log is not None:
                try:
                    self._err_log.write(line)
                    self._err_log.flush()
                except (OSError, ValueError):
                    pass
            if self.verbose:
                sys.stderr.write(f"[w{self.idx}:err] {line}")

    def _read_control(self):
        """Read worker stdout until a SENTINEL control line; forward log lines.
        Returns the parsed control dict, or None if the worker's stdout hit EOF
        (the process died)."""
        while True:
            line = self.p.stdout.readline()
            if line == "":
                return None
            if line.startswith(SENTINEL):
                try:
                    return json.loads(line[len(SENTINEL):])
                except ValueError:
                    continue
            elif self.verbose:
                sys.stderr.write(f"[w{self.idx}] {line}")

    def wait_ready(self, timeout=180):
        """Block until the worker emits its 'ready' event (build() done)."""
        msg = self._read_control()
        if msg is None:
            raise RuntimeError(f"worker {self.idx} died before ready")
        if msg.get("event") != "ready":
            raise RuntimeError(f"worker {self.idx} first message was not ready: {msg}")
        self.sha1 = msg.get("scene_sha1")
        log(f"worker {self.idx} ready (pid {msg.get('pid')}, "
            f"gpu {self.gpu if self.gpu is not None else 'default'})")
        return True

    def render(self, req):
        """Send one request, block for its result. Never raises — a dead worker
        becomes an ok:false result so the order is reported, not swallowed."""
        try:
            self.p.stdin.write(json.dumps(req) + "\n")
            self.p.stdin.flush()
        except (BrokenPipeError, ValueError):
            return {"event": "result", "id": req.get("id"), "ok": False,
                    "error": "worker stdin closed (worker died)"}
        while True:
            msg = self._read_control()
            if msg is None:
                return {"event": "result", "id": req.get("id"), "ok": False,
                        "error": "worker died mid-render"}
            if msg.get("event") == "result":
                return msg
            # ignore any other control event (e.g. a stray ready)

    def alive(self):
        return self.p is not None and self.p.poll() is None

    def stop(self):
        """SIGTERM the worker and reap it (SIGKILL if it won't go)."""
        if self.p is None:
            return
        try:
            if self.p.poll() is None:
                # a shutdown request lets it exit its serve loop cleanly; TERM
                # is the backstop for a worker wedged inside a GPU call.
                try:
                    self.p.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    self.p.stdin.flush()
                except (BrokenPipeError, ValueError, OSError):
                    pass
                self.p.terminate()
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.p = None
        if self._err_log is not None:
            try:
                self._err_log.close()
            except OSError:
                pass
            self._err_log = None


# --------------------------------------------------------------------------- #
# the pool
# --------------------------------------------------------------------------- #
class Pool:
    def __init__(self, base_cmd, spool, n_workers, gpus, verbose, run_dir="",
                 isolate=False):
        self.base_cmd = base_cmd
        self.spool = spool
        self.n = n_workers
        self.gpus = gpus
        self.verbose = verbose
        # isolate: wrap each pinned worker in a bwrap GPU device namespace
        # (resolved once at startup by main(); False inside a sandbox).
        self.isolate = isolate
        # RUN_DIR (the --scene's dir): where layout.json lives, for the
        # default-ON sweep.hand_mask fill at claim time. Empty disables it.
        self.run_dir = run_dir
        config = _run_config.load_config(run_dir) if run_dir else {}
        self.candidate_sheet_max_dimension = \
            _run_config.candidate_sheet_max_dimension(config)
        self.side_by_side_max_dimension = \
            _run_config.side_by_side_max_dimension(config)
        self.workers = []
        self.q = queue.Queue()
        self.threads = []
        self.stop_evt = threading.Event()
        # recycle coordination: pause gates each slot BEFORE it picks up work;
        # busy counts slots currently mid-render (recycle waits for it to hit 0).
        self.pause_evt = threading.Event()   # set == running; clear == paused
        self.pause_evt.set()
        self.busy_lock = threading.Lock()
        self.busy = 0

    # ---- worker lifecycle ---- #
    def _spawn_worker(self, idx):
        gpu = None
        if self.gpus:
            gpu = self.gpus[idx % len(self.gpus)]
        scratch = os.path.join(self.spool, "workers", str(idx))
        w = Worker(idx, self.base_cmd, scratch, gpu, self.verbose,
                   isolate=self.isolate)
        w.start()
        w.wait_ready()
        return w

    def start(self):
        for d in ("orders", "claimed", "renders", "results", "workers"):
            os.makedirs(os.path.join(self.spool, d), exist_ok=True)
        log(f"launching {self.n} worker(s)...")
        self.workers = [self._spawn_worker(i) for i in range(self.n)]
        self.pool_sha1 = self.workers[0].sha1 if self.workers else None
        for i in range(self.n):
            t = threading.Thread(target=self._slot_loop, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)
        ledger.append(self.spool, "pool_ready", workers=self.n,
                      scene_sha1=self.pool_sha1, gpus=",".join(self.gpus),
                      pid=os.getpid(), pgid=os.getpgrp())
        log(f"pool ready ({self.n} workers, scene_sha1={self.pool_sha1})")

    def _slot_loop(self, idx):
        """One thread per slot: gate on pause, pull an order, render, write the
        result. Respawns its own worker if it died outside a recycle."""
        while not self.stop_evt.is_set():
            # gate: during a recycle, don't pick up new work.
            self.pause_evt.wait()
            if self.stop_evt.is_set():
                return
            try:
                order = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            rid, req = order
            with self.busy_lock:
                self.busy += 1
            try:
                res = self._render_with_retry(idx, rid, req)
                # Panels BEFORE the result is published: a client that sees
                # results/<id>.json already has its sheets, so an agent never
                # races a half-panelled dir (and never gets a bare render).
                if res.get("ok"):
                    self._build_panels(rid, req, res)
                write_result_atomic(os.path.join(self.spool, "results"), rid, res)
                ok = res.get("ok")
                # Ledger AFTER the result file: a client polls results/, so the
                # order is "done" the moment that lands — the ledger line is
                # debugging data trailing it, never a gate on delivery.
                ledger.append(self.spool, "result", id=rid, ok=bool(ok),
                              worker=idx, seconds=res.get("seconds"),
                              error=res.get("error"),
                              provenance=req.get("provenance"),
                              panels=(res.get("panels") or {}).get("error")
                              or len((res.get("panels") or {}).get("pages") or [])
                              or None)
                log(f"order {rid} -> {'ok' if ok else 'FAIL'} on worker {idx}"
                    + ("" if ok else f" ({res.get('error')})")
                    + (f"  {res.get('seconds')}s" if res.get("seconds") else ""))
            finally:
                with self.busy_lock:
                    self.busy -= 1
                self.q.task_done()

    def _build_panels(self, rid, req, res):
        """Panel a finished sweep-family order, recording the outcome in `res`.

        Never fails the order: the render already succeeded, so a panel problem is
        reported in result["panels"]["error"] rather than losing good renders."""
        if not panels.wants_panels(req):
            return
        out = res.get("out") or os.path.join(self.spool, "renders", rid)
        info = panels.build_for_order(
            req, out, run_dir=self.run_dir,
            max_dimension=self.candidate_sheet_max_dimension,
            side_by_side_max_dimension=self.side_by_side_max_dimension)
        res["panels"] = info
        if info.get("pages"):
            log(f"order {rid} -> {len(info['pages'])} candidate sheet page(s)")
        elif info.get("error"):
            log(f"order {rid} -> panels FAILED: {info['error']}")

    def _render_with_retry(self, idx, rid, req, retries=1):
        """Render an order, retrying it on a fresh worker when the WORKER died
        (a GPU/driver crash takes the whole Blender process). A request that
        FAILS while its worker survives is a deterministic error (bad ranges,
        unknown view, ...) and is not retried."""
        attempt = 0
        while True:
            w = self.workers[idx]
            if not w.alive():
                # A worker found dead BEFORE its render died on the previous
                # order (or at startup) — charge it here too, else the only trace
                # is a stdout line and a slot that mysteriously slowed down.
                # Only on the FIRST look: past attempt 0 this is the same death
                # the retry below already logged, and double-charging it would
                # inflate every retried order's death count.
                if attempt == 0:
                    ledger.append(self.spool, "worker_died", id=rid, worker=idx,
                                  attempt=attempt, retries=retries,
                                  before_render=True)
                log(f"worker {idx} not alive — respawning before order {rid}")
                try:
                    self.workers[idx] = self._spawn_worker(idx)
                except RuntimeError as e:
                    # the REPLACEMENT died before ready — the slot must still
                    # report the order (a silent slot-thread death strands the
                    # queue and hangs every waiting client).
                    return {"event": "result", "id": rid, "ok": False,
                            "error": f"respawn failed: {e}"}
                w = self.workers[idx]
            res = w.render(req)
            if res.get("ok") or w.alive() or attempt >= retries:
                return res
            attempt += 1
            # The dominant failure mode, and invisible in results/ once the retry
            # succeeds — the ledger is where "this order cost a worker" survives.
            ledger.append(self.spool, "worker_died", id=rid, worker=idx,
                          attempt=attempt, retries=retries)
            log(f"worker {idx} died on order {rid} — retry "
                f"{attempt}/{retries} on a fresh worker")

    # ---- recycle ---- #
    def recycle(self, new_sha1):
        """Kill+relaunch ALL workers so they pick up the new build() geometry.
        Drains in-flight first so no order is lost to a mid-render SIGTERM."""
        log(f"scene.py changed ({self.pool_sha1} -> {new_sha1}); recycling "
            f"{self.n} worker(s)...")
        self.pause_evt.clear()                      # stop slots taking new work
        while True:                                 # wait for in-flight to finish
            with self.busy_lock:
                if self.busy == 0:
                    break
            time.sleep(0.05)
        for w in self.workers:                      # SIGTERM all
            w.stop()
        self.workers = [self._spawn_worker(i) for i in range(self.n)]   # relaunch
        self.pool_sha1 = self.workers[0].sha1 if self.workers else new_sha1
        self.pause_evt.set()                        # resume (queued orders live on)
        ledger.append(self.spool, "recycle", scene_sha1=self.pool_sha1,
                      workers=self.n)
        log(f"recycle done; pool_sha1={self.pool_sha1}")

    # ---- spool intake ---- #
    def claim_new_orders(self):
        """Move each new orders/<id>.json to claimed/ (atomic rename) and enqueue
        it. The rename is the claim, so an order is never processed twice."""
        orders_dir = os.path.join(self.spool, "orders")
        try:
            entries = sorted(os.listdir(orders_dir))
        except FileNotFoundError:
            return
        for name in entries:
            if not name.endswith(".json") or name.startswith("."):
                continue
            src = os.path.join(orders_dir, name)
            try:
                with open(src) as f:
                    req = json.load(f)
            except (OSError, ValueError):
                continue                            # client still writing; next tick
            rid = str(req.get("id") or os.path.splitext(name)[0])
            req["id"] = rid
            claimed = os.path.join(self.spool, "claimed", f"{rid}.json")
            try:
                os.rename(src, claimed)             # THE claim (atomic)
            except OSError:
                continue
            req["out"] = os.path.join(self.spool, "renders", rid)
            # BEFORE the mask fills: a seed can introduce the `sweep` object those
            # fills key off, and a seed that cannot resolve should fail the order
            # rather than reach a worker with a reverted pose.
            try:
                seeded = reseed.resolve_order_seed(req)
            except reseed.ReseedError as exc:
                # The claim already happened (the rename is the claim), so the order
                # cannot go back to orders/ — it fails here with the reason, exactly
                # as a worker-side rejection would.
                msg = f"seed did not resolve: {exc}"
                write_result_atomic(os.path.join(self.spool, "results"), rid,
                                    {"id": rid, "ok": False, "error": msg})
                ledger.append(self.spool, "result", id=rid, ok=False, error=msg)
                log(f"order {rid}: {msg}")
                continue
            for key in sorted(seeded):
                log(f"order {rid}: {key} <- seed {req.get('seed_from')!r}")
            if seeded:
                # WHICH pose a hop descended from is the one thing that makes a
                # multi-hop fit auditable after the fact, and the claimed order is
                # deleted on recycle — so it goes in the ledger too.
                src_frame = reseed.order_seed_source_frame(req)
                if src_frame:
                    log(f"order {rid}: seed CROSSED FRAMES {src_frame} -> "
                        f"{req.get('frame')} (verbatim pose, camera motion NOT "
                        "compensated — the grid must cover it)")
                ledger.append(self.spool, "seed", id=rid,
                              seed_from=req.get("seed_from"),
                              filled=sorted(seeded),
                              cross_frame=src_frame,
                              joints=seeded.get("joints"))
            smask = resolve_sweep_mask(req, self.run_dir)
            if smask:
                log(f"order {rid}: sweep.mask <- {smask} (layout default)")
            filled = resolve_hand_mask(req, self.run_dir)
            if filled:
                log(f"order {rid}: sweep.hand_mask <- {filled} (layout default; "
                    'send "hand_mask": "" to opt out)')
            omasks = resolve_object_masks(req, self.run_dir)
            for key, path in sorted(omasks.items()):
                log(f"order {rid}: object-view {key} <- {path} (layout "
                    f'default; send "{key}": "" to opt out)')
            self.q.put((rid, req))
            ledger.append(self.spool, "claim", id=rid, frame=req.get("frame"),
                          views=req.get("views"),
                          provenance=req.get("provenance"),
                          queued=self.q.qsize())
            log(f"claimed order {rid}")

    def shutdown(self):
        log("shutting down...")
        # Written FIRST: the teardown SIGTERMs Blender and may itself be racing a
        # killpg, so an "I meant to stop" line recorded before the kills is what
        # distinguishes a clean drain from a pool that vanished mid-order.
        ledger.append(self.spool, "pool_stop", pending=self.q.unfinished_tasks)
        self.stop_evt.set()
        self.pause_evt.set()          # release any paused slot threads
        for w in self.workers:
            w.stop()


# --------------------------------------------------------------------------- #
def build_worker_cmd(args):
    """The blender --serve launch command shared by every worker (minus --out,
    which is per-slot). Forwards the render-config flags that fix the resident
    resolution/engine/geometry, so a pool render is byte-equivalent to a one-shot
    render.sh with the same flags."""
    # -t caps Blender's CPU thread pool. Untouched, EVERY worker spawns
    # ncores threads for no render gain (the per-candidate render is
    # GPU/driver-bound, not CPU-bound).
    cmd = [BLENDER, "--background", "--factory-startup",
           "-t", str(args.blender_threads),
           "--python", RENDER_WRAPPER, "--",
           "--serve", "--scene", args.scene]
    if args.tracking:
        cmd += ["--tracking", args.tracking]
    if args.match_res:
        cmd += ["--match-res", args.match_res]
    if args.ref_frame:
        cmd += ["--ref-frame", args.ref_frame]
    cmd += ["--engine", args.engine, "--samples", str(args.samples)]
    if args.device:
        cmd += ["--device", args.device]
    return cmd


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True, help="RUN_DIR/scene.py (built once "
                   "per worker; edits trigger a pool recycle)")
    p.add_argument("--spool", required=True, help="spool dir (orders/claimed/"
                   "renders/results/workers live under it)")
    p.add_argument("--workers", type=int, default=2,
                   help="G — number of resident Blender workers (default 2)")
    p.add_argument("--gpus", default="",
                   help="comma GPU ids to pin workers to, round-robin (default: "
                        "none set — Blender sees all and EEVEE piles onto GPU "
                        "0). Pinning = CUDA_VISIBLE_DEVICES (Cycles) + a bwrap "
                        "/dev/nvidiaN device namespace (EEVEE) when the host "
                        "allows it; inside an already-bwrapped sandbox the "
                        "namespace is skipped (the sandbox bound your subset).")
    p.add_argument("--tracking", "--pi3x", dest="tracking", default="",
                   help="capture tracking dir (per-frame cameras/K); "
                        "--pi3x is the old name, still accepted")
    p.add_argument("--match-res", default="", help="image whose pixel size sets "
                   "the resident render resolution (as render.sh --match-res)")
    p.add_argument("--ref-frame", default="", help="gauge frame (empty -> scene "
                   "REFERENCE_FRAME)")
    p.add_argument("--engine", default="BLENDER_EEVEE_NEXT",
                   choices=["CYCLES", "BLENDER_EEVEE_NEXT"])
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--device", default="", choices=["", "OPTIX", "CUDA", "CPU"],
                   help="Cycles device (ignored by EEVEE)")
    p.add_argument("--run-dir", default="",
                   help="RUN_DIR whose layout.json resolves the default-ON "
                        "sweep.hand_mask fill (default: --scene's directory). "
                        "Pass --no-hand-default to disable the fill entirely.")
    p.add_argument("--no-hand-default", action="store_true",
                   help="do NOT fill sweep.hand_mask from the run layout; only "
                        "orders that carry the key themselves get the hand "
                        "waiver (the pre-fill behavior)")
    p.add_argument("--verbose", action="store_true",
                   help="forward each worker's Blender stdout/stderr (debugging)")
    p.add_argument("--once-empty-exit", action="store_true",
                   help="exit after the orders queue drains once (for tests/batch "
                        "runs); default: run until SIGINT/SIGTERM")
    p.add_argument("--allow-oversubscribe", action="store_true",
                   help="skip the worker clamps (cores-1 and per-GPU; each "
                        "worker is a full EEVEE Blender — oversubscribing one "
                        "GPU crashes the pool via GL context churn)")
    p.add_argument("--blender-threads", type=int, default=8,
                   help="Blender -t per worker (default 8). 0 = ncores "
                        "(Blender's own default) — avoid: G workers x ncores "
                        "threads swamps the scheduler for no render gain.")
    args = p.parse_args()

    if not os.path.isfile(args.scene):
        raise SystemExit(f"[pool] no scene.py at {args.scene}")
    if args.workers < 1:
        raise SystemExit("[pool] --workers must be >= 1")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    # The clamp's GPU budget: the --gpus list when pinning, else however many
    # devices are visible (inside a device-isolated sandbox that is exactly the
    # subset the launcher bound; unpinned EEVEE workers all land on one GPU).
    n_gpus = len(gpus) if gpus else min(1, visible_gpu_count())
    capped = clamp_workers(args.workers, os.cpu_count(),
                           args.allow_oversubscribe, n_gpus=n_gpus)
    if capped != args.workers:
        log(f"clamping --workers {args.workers} -> {capped} "
            f"({os.cpu_count()} cores, {n_gpus or 'unknown'} GPU(s) x "
            f"{MAX_WORKERS_PER_GPU}/GPU; each worker is a full EEVEE Blender "
            "and oversubscription wipes the pool — pass --allow-oversubscribe "
            "to override)")
        args.workers = capped
    isolate = bool(gpus) and gpu_isolation_works()
    if gpus:
        log(f"GPU pinning: {','.join(gpus)} via "
            + ("bwrap device namespace + CUDA_VISIBLE_DEVICES" if isolate else
               "CUDA_VISIBLE_DEVICES only (no bwrap here — EEVEE pinning "
               "relies on the sandbox's own device isolation)"))
    run_dir = ""
    if not args.no_hand_default:
        run_dir = os.path.abspath(args.run_dir) if args.run_dir else \
            os.path.dirname(os.path.abspath(args.scene))

    pool = Pool(build_worker_cmd(args), os.path.abspath(args.spool),
                args.workers, gpus, args.verbose, run_dir=run_dir,
                isolate=isolate)

    stop = {"flag": False}

    def _sig(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    pool.start()
    try:
        poll = 0.25
        while not stop["flag"]:
            cur = scene_sha1(args.scene)
            if cur is not None and cur != pool.pool_sha1:
                pool.recycle(cur)
            pool.claim_new_orders()
            if args.once_empty_exit and pool.q.unfinished_tasks == 0:
                # drained at least one intake tick with nothing pending — done.
                break
            time.sleep(poll)
        # in --once-empty-exit, make sure the last batch's results are all written.
        if args.once_empty_exit:
            pool.q.join()
    finally:
        pool.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
