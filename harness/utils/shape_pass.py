#!/usr/bin/env python3
"""shape_pass.py — render and score the ONE committed state, on every frame.

Run in the 'artscript' micromamba env (use the harness/utils/shape_pass.sh wrapper):
  harness/utils/shape_pass.sh RUN_DIR --frames 000000.jpg,000040.jpg
  harness/utils/shape_pass.sh RUN_DIR                    # frames from layout.json
  harness/utils/shape_pass.sh RUN_DIR --pass-label shape-checkpoint \
      --critic-frames 000000.jpg

This is the main beat of the SHAPE loop: author `scene.py`, render, look, repeat.
It produces the full diagnostic set for the committed state on every frame.
Pose windows normally use the lighter `composite_pass.sh` after apply; run this
full pass there only when turntables, depth, mechanism views, critics, aggregate,
or self-intersection are deliberately needed.

The pass itself is a fixed sequence: render.sh (match,turntable,depth) -> then,
PER FRAME, a composite.py call threading the SAME pass dir through SIX path
args, plus a depth.py call -> then aggregate.py. This wraps that sequence so the
pass dir is CAPTURED (never hand-typed into six paths) and the per-frame paths
are built once from the frame stem.

Layout (which frames, and where the images/masks/tracking live) comes from
RUN_DIR/layout.json, written by run.sh from a CAPTURE dir (the canonical
materialized-sequence format — datasets/common/capture.py). There is NO
convention-based path guessing: the layout records explicit paths, so a missing
field is an error, not a guess. If a field is missing from the layout, pass it
explicitly (--tracking/--frames-dir/--masks-dir/--hand-masks-dir/--ref-frame/
--frames); CLI flags override the layout file.

What it runs, in order:
  1. render.sh RUN_DIR/scene.py ITERATION_DIR/renders --views match,turntable,depth
       --tracking ... --frames ... --ref-frame ... --match-res <ref image>
       [--engine/--samples] --export RUN_DIR/mesh/object.glb
     -> captures the "PASS <name> OUTPUT DIR: <dir>" line as PASS_DIR.
  2. per frame: composite.py (metrics_/overlap_/depth_residual_/composite_/
     side_by_side_) and depth.py (writes depth_<stem>.json — the file
     aggregate.py reads; the manual step-4 composite command never writes it).
  3. self_intersection.py on the pass's OWN pose.json + the GLB it just exported —
     UNCONDITIONAL, every pass, no flag to skip it (see run_self_intersection).
  4. critic.py for explicitly selected screening frames (soft; a non-zero exit is
     tolerated); a routine pass makes no VLM calls. Use --critic-frames for a
     coverage/suspect set or --critic-all for a deliberate checkpoint.
  5. unless --no-aggregate: aggregate.py --run-dir RUN_DIR --views-dir PASS_DIR.
  6. prints a per-frame summary read back from the JSON: the iou_raw / depth_mae_canon
     table, a SELF-INTERSECTION block (deepest part-pair overlap + the onsets), AND
     a VLM critic block (realism/identity + every discrepancy — all severities, any
     tag — with HIGH marked) so the critic's shape/pose/joint verdict is visible
     inline at decision time, not buried in critic_<frame>.json.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# analysis/ is a package now: tools run as `python -m analysis.<pkg>.<tool>` with
# cwd=HARNESS so `analysis` (and the shared `core`) resolve.
HARNESS = os.path.join(REPO, "harness")
DEPTH = "analysis.scorers.depth"
CRITIC = "analysis.scorers.critic"
AGGREGATE = "analysis.rollup.aggregate"
TURNTABLE_SHEET = "analysis.viz.turntable_sheet"
MECHANISM_SHEET = "analysis.viz.mechanism_sheet"
SELF_INTERSECTION = "analysis.self_intersection"

# shape_pass.sh runs this as a bare script (`python shape_pass.py`), so harness/ is
# not on sys.path by default. Add it so the sibling helpers import under the SAME
# module names every other caller uses (`from pool import ...`).
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)
from utils import _run_config as icfg  # noqa: E402
from utils import pass_pipeline  # noqa: E402
from core import mechanism_views  # noqa: E402
from core import modules  # noqa: E402
from core import turntable_views  # noqa: E402
from pool import client as pool_client  # noqa: E402
from pool import session as pool_session  # noqa: E402
from pool import locks as pool_locks  # noqa: E402
from pool.locks import acquire_run_lock  # noqa: E402,F401 (re-export)
from core import depth_config as dcfg  # noqa: E402
from bookkeeping import ledger as bookkeeping  # noqa: E402
from core import pass_dirs, state_json  # noqa: E402
from rig import lie  # noqa: E402


def log(msg):
    print(f"[shape-pass] {msg}", flush=True)


# acquire_run_lock lives in pool.locks (shared with the other GPU-pool
# coordinator, multiagent/pool_session.py); the re-export above keeps
# `shape_pass.acquire_run_lock` resolving for existing callers.


# --------------------------------------------------------------------------- #
# config: RUN_DIR/run_config.json (CLI flag > config json > default)
# --------------------------------------------------------------------------- #
def resolve_config(run_dir, args):
    """Fold RUN_DIR/run_config.json into `args` in place: resolve the
    concurrency knobs (N/C/G), the pool toggle/gpus, bg_mode, and the critic
    backend block. Config-settable flags parsed with a None default mean 'not
    passed on the CLI' -> fall through to the config value, then the built-in
    default. An absent/corrupt config leaves the built-in defaults, so older runs
    (no config file) behave exactly as before."""
    cfg = icfg.load_config(run_dir)
    conc = icfg.section(cfg, "concurrency")
    pool = icfg.section(cfg, "pool")
    visuals = icfg.section(cfg, "visuals")
    critic = icfg.section(cfg, "critic")

    args.ncpu = icfg.pick(args.ncpu, conc, "ncpu", min(4, os.cpu_count() or 1))
    args.critic_conc = icfg.pick(args.critic_conc, conc, "critic_conc", 8)
    args.workers = icfg.pick(args.workers, conc, "workers", 2)
    args.pool = bool(icfg.pick(args.pool, pool, "enable", False))
    args.archive_spool = bool(icfg.pick(args.archive_spool, pool,
                                        "archive_spool", True))
    args.keep_spools = icfg.pick(args.keep_spools, pool, "keep_spools", 8)
    gpus = icfg.pick(args.gpus, pool, "gpus", "")
    if isinstance(gpus, (list, tuple)):   # config may give [0, 1]; the pool wants "0,1"
        gpus = ",".join(str(g) for g in gpus)
    args.gpus = gpus
    args.bg_mode = icfg.pick(args.bg_mode, visuals, "bg_mode", "black")
    args.side_by_side_max_dimension = icfg.side_by_side_max_dimension(cfg)

    # observed-depth REPORT switch (depth_<stem>.json + residual panels). Precedence:
    # CLI --depth-report/--no-depth-report > RUN_DIR/depth_config.json "report" >
    # True. Lives in depth_config.json (NOT run_config.json) so the same file
    # that carries the depth backend/floor also carries its on/off — and so the
    # sweep's "cost" flag sits beside it. Camera seeds / measure_depth are untouched.
    if args.depth_report is None:
        args.depth_report = dcfg.depth_report_enabled(
            os.path.join(run_dir, "depth_config.json"))
    else:
        args.depth_report = bool(args.depth_report)

    # depth RENDER switch (the depth view -> depth_<stem>.npy). Resolved exactly
    # like "report" above and INDEPENDENT of it: our own depth render needs no GT,
    # so it feeds the object-units panel/sheet whether or not the scorer runs.
    # Turning it off is what makes a pass stop paying for depth entirely.
    if args.depth_render is None:
        args.depth_render = dcfg.depth_render_enabled(
            os.path.join(run_dir, "depth_config.json"))
    else:
        args.depth_render = bool(args.depth_render)

    # The critic backend block: keep values that stay None as None so
    # _critic_backend_flags omits them (critic.py falls back to its own default).
    args.critic = {
        k: icfg.pick(getattr(args, a), critic, k, None)
        for k, a in (("provider", "provider"), ("model", "model"),
                     ("api_key_env", "api_key_env"),
                     ("max_tokens", "max_tokens"),
                     ("pass_realism", "pass_realism"),
                     ("pass_identity", "pass_identity"),
                     ("max_turntable", "max_turntable"),
                     ("crops", "crops"), ("crop_pad", "crop_pad"))
    }


# --------------------------------------------------------------------------- #
# layout: manifest + CLI overrides (NO convention-based inference)
# --------------------------------------------------------------------------- #
def resolve_layout(run_dir, args):
    return pass_pipeline.resolve_layout(run_dir, args, tool_name="shape-pass")


def check_inputs(layout):
    return pass_pipeline.check_inputs(layout, tool_name="shape-pass")


# --------------------------------------------------------------------------- #
# render + pass-dir capture
# --------------------------------------------------------------------------- #
def _scan_render_log(path, tail_lines=50):
    return pass_pipeline._scan_render_log(path, tail_lines=tail_lines)


def resolve_existing_pass(iteration, requested):
    """Validate a score-only recovery pass owned by the open iteration."""
    if not requested:
        return ""
    pass_dir = os.path.realpath(os.path.abspath(requested))
    renders_root = os.path.realpath(os.path.join(iteration["path"], "renders"))
    try:
        owned = os.path.commonpath((renders_root, pass_dir)) == renders_root
    except ValueError:
        owned = False
    if not owned or pass_dir == renders_root:
        raise SystemExit(
            "[shape-pass] --existing-pass must name a pass below the active "
            f"iteration's renders directory: {renders_root}")
    missing = [
        name for name in ("scene_snapshot.py", "pose.json")
        if not os.path.isfile(os.path.join(pass_dir, name))
    ]
    if missing:
        raise SystemExit(
            f"[shape-pass] existing pass is incomplete ({', '.join(missing)} "
            f"missing): {pass_dir}")
    return pass_dir


def _render_sh(run_dir, layout, args, views, frames):
    """Invoke render.sh for the given comma view list + frames and return the
    PASS_DIR it printed. Aborts on failure. Both the cold one-shot (run_render) and
    the pool's gauge render (run_render_pool) go through here so they build the
    render command — engine/samples/tracking/ref-frame/match-res/export — identically."""
    extra = []
    if getattr(args, "turntable_views", None):
        extra += ["--turntable-views", args.turntable_views]
    if getattr(args, "turntable_jitter", None) is not None:
        extra += ["--turntable-jitter", str(args.turntable_jitter)]
    if getattr(args, "mechanism_rows", None):
        extra += ["--mechanism-rows", str(args.mechanism_rows)]
    if getattr(args, "mechanism_samples", None):
        extra += ["--mechanism-samples", str(args.mechanism_samples)]
    return pass_pipeline.render_pass(
        run_dir, layout, getattr(args, "iteration", None), views, frames,
        args.engine, args.samples, pass_label=args.pass_label,
        intent=getattr(args, "intent", ""),
        rollback=getattr(args, "rollback", ""), extra_args=extra, log=log,
        run=subprocess.run, tool_name="shape-pass")


def mechanism_owed(run_dir, args=None):
    """Whether this pass needs the mechanism A/B sweep.

    The sweep costs joints x 2 arms x rows x samples renders and feeds ONLY the
    pick gate, whose answers outlive the pass — so an unchanged rig would render
    arcs nobody is asked about. Owed = exactly `check_run`'s question. Errs toward
    RENDERING when it cannot tell: a wasted sweep costs GPU, a missing one makes
    the gate unanswerable. --mechanism / --no-mechanism override — but not past
    the run's module set: a run without the mechanism module never renders it.
    """
    if not modules.enabled(run_dir, "mechanism"):
        return False
    forced = getattr(args, "mechanism", None)
    if forced is not None:
        return bool(forced)
    # "no joints" must come from a pose.json that SAYS SO, not from a missing
    # one: before the first render there is no pose.json, and reading its absence
    # as "rigid" would skip the sweep on exactly the pass that owes the first pick.
    pose_path = os.path.join(run_dir, "mesh", "pose.json")
    try:
        with open(pose_path) as f:
            pose = json.load(f)
    except (OSError, ValueError):
        return True                       # cannot tell -> render (see docstring)
    joint_defs = (pose.get("joint_defs") or pose.get("JOINTS") or [])
    try:
        from core import mechanism_calls as mc
        result = mc.check_run(run_dir, joint_defs=joint_defs)
    except Exception:                     # never let a cost optimization break a pass
        return True
    if not result.get("joints"):
        return False                      # rigid: the view writes nothing anyway
    return not result.get("ok")


def _views_for(run_dir, args, extra=""):
    """The --views list for this pass, dropping `mechanism` when no pick is owed."""
    views = ["match", "turntable"] if not extra else list(extra.split(","))
    if mechanism_owed(run_dir, args):
        views.append("mechanism")
    elif not modules.enabled(run_dir, "mechanism"):
        log(modules.disabled_line("mechanism"))
    else:
        log("mechanism: every articulated joint's A/B pick is already fresh — "
            "skipping the sweep (--mechanism forces it)")
    # the depth view rides along unless the "render" switch is off: with observed depth it
    # feeds the depth scorer, and without (the monocular case — "report" off) it
    # still feeds the object-units DEPTH panel (composite --render-depth alone)
    # and the depth sheet (analysis.viz.depth_units), which need no GT. So
    # "report" does NOT gate it; only --no-depth-render / "render": false does.
    if getattr(args, "depth_render", True):
        views.append("depth")
    return ",".join(views)


def run_render(run_dir, layout, args):
    """The cold path (default): ONE render.sh renders match+turntable+depth for
    every frame and exports the GLB; return its PASS_DIR. The "report" flip
    switch (args.depth_report) gates only the observed-depth SCORING; the depth RENDER
    has its own switch (args.depth_render, default on), because the object-units
    panel and the depth sheet read depth_<stem>.npy with no GT at all.

    `mechanism` rides along only when a pick is OWED (see mechanism_owed)."""
    return _render_sh(run_dir, layout, args, _views_for(run_dir, args),
                      ",".join(layout["frames"]))


def run_render_pool(run_dir, layout, args):
    """The --pool path: split the render across a cheap cold GAUGE render and the
    resident worker pool, then reconcile so the returned PASS_DIR holds exactly
    what the cold path would have.

      1. GAUGE (one cold render.sh): the per-STATE turntable + pose.json + GLB.
         `pose.json` and the GLB are the reason this render exists at all: they are
         shared, once-per-pass artifacts a serve worker deliberately never writes
         (it writes only per-frame renders, into its own out dir). The turntable
         rides along because this process is already paying for itself — NOT because
         a worker couldn't render it. It could: `pool/serve.py` populates the state
         and canonical matrices the view needs, and `setup_lighting` ignores the
         bbox it is handed, so a resident worker's lighting is identical to this
         one's. Ordering the turntable per state from the pool is a live option if
         states x views ever makes this serial render the slow part.
      2. POOL: spawn a throwaway pool (--once-empty-exit) and submit one
         match+depth order per frame; G resident workers drain them in parallel.
      3. RECONCILE: copy each order's match_<stem>.png / depth_<stem>.npy into the
         gauge PASS_DIR, so _frame_paths / _score_cpu / _score_critic / read_scale
         see the single-pass-dir layout they already assume — no downstream change.
    """
    frames = layout["frames"]
    # 1. gauge render — turntable (per state) + pose.json + GLB; NO per-frame
    #    match/depth (the pool does those, so rendering them here would be wasted).
    # the mechanism sweep rides along with the turntable for the same reason: it
    # is per-JOINT, not per-frame, so a resident worker would have nothing to
    # parallelize over and this process is already paying for itself.
    gauge_views = "turntable,mechanism" if mechanism_owed(run_dir, args) \
        else "turntable"
    pass_dir = _render_sh(run_dir, layout, args, gauge_views, ",".join(frames))

    # 2. pool the per-frame match+depth renders.
    spool = os.path.join(run_dir, "spool")
    _run_pool(run_dir, layout, args, spool, frames)

    # 3. reconcile: copy the pool's per-frame renders into the gauge pass_dir.
    #    depth_<stem>.npy only exists when the "render" switch left the view on
    #    (the copy is guarded on existence anyway, so this is just intent).
    names = ("match_{s}.png", "depth_{s}.npy") if args.depth_render \
        else ("match_{s}.png",)
    for fr in frames:
        stem = os.path.splitext(fr)[0]
        src = os.path.join(spool, "renders", stem)
        for name in names:
            s = os.path.join(src, name.format(s=stem))
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(pass_dir, name.format(s=stem)))
    return pass_dir


def _run_pool(run_dir, layout, args, spool, frames):
    """Submit a match+depth order per frame, spawn a throwaway pool over `spool`,
    and block for every result. Aborts (SystemExit) if any order fails — same
    fail-hard contract as render.sh returning non-zero.

    Orders are submitted BEFORE the pool is spawned: with --once-empty-exit, it
    exits as soon as a claim tick finds the queue empty, so an order that arrives
    after its first tick could be missed. Dropping every order into spool/orders/
    up front guarantees they're all claimed on that first tick (submit only writes
    files; the resident build() the pool does at startup is far slower anyway)."""
    scene = os.path.join(run_dir, "scene.py")
    ref_img = os.path.join(layout["frames_dir"], layout["ref_frame"])

    # 0. reap any orphaned pool from a prior pass BEFORE wiping the spool — a
    #    coordinator killed hard leaves its pool group reparented to init, and it
    #    would keep claiming/rendering into the spool we're about to reuse. Reap
    #    kills processes (via the recorded pgid), not files. --once-empty-exit means
    #    such an orphan usually self-drains, but reaping makes it deterministic.
    pool_locks.reap_pool(spool)
    # then clear any spool from a previous pass. pool_client.wait returns as soon as
    #    results/<id>.json exists, so a stale result (or a renders/<id>/ dir this
    #    pass is concurrently rewriting) left over from an earlier --pool run would
    #    be picked up instantly instead of this pass's render.
    #
    #    HOW we clear it is the --archive-spool switch (default ON). Archiving
    #    RENAMES the old spool to spool-<timestamp>/ — the live path ends up just as
    #    empty as an rmtree leaves it, so the hermetic-pass guarantee is identical,
    #    but the previous pass's orders/reports/panels stay readable. That matters
    #    because RUN_DIR/spool is shared with the window agents: their sweep and
    #    oapply orders (and the candidate sheets that justify each accepted pose)
    #    live here, and an rmtree takes those too, not just this pass's match renders.
    if args.archive_spool:
        kept = pool_locks.archive_spool(spool, keep=args.keep_spools)
        if kept:
            log(f"pool: archived previous spool -> {kept}")
    else:
        shutil.rmtree(spool, ignore_errors=True)

    # 1. submit all orders first. id == frame stem, so the order's renders/<id>/
    #    dir carries the match_<stem>.png / depth_<stem>.npy names _frame_paths
    #    expects. The client never sets "out" (the manager allocates it). The depth
    #    view rides along unless the "render" switch is off (see _views_for — the
    #    object-units panel/sheet read it with no GT, so "report" does not gate it).
    order_views = "match,depth" if args.depth_render else "match"
    ids = [pool_client.submit(spool, {"id": os.path.splitext(fr)[0],
                                      "frame": fr, "views": order_views})
           for fr in frames]

    # 2. spawn the throwaway pool; --once-empty-exit drains the batch and exits.
    #    PoolSession owns the process-group bookkeeping (start_new_session +
    #    recorded pgid, so a later pass can reap us if we're killed mid-way).
    cmd = pool_session.pool_cmd(
        scene, spool, args.workers, layout["ref_frame"], ref_img,
        engine=args.engine, samples=args.samples,
        tracking=layout["tracking"], gpus=args.gpus, once_empty_exit=True)
    log(f"pool: {args.workers} worker(s) over {spool}")
    log("pool: " + " ".join(cmd))
    sess = pool_session.PoolSession(cmd, spool)
    try:
        # 3. block for each result (workers drain the pre-filled queue in parallel).
        sess.wait_orders(ids, args.pool_timeout, error_cls=SystemExit,
                         label="[shape-pass] pool render")
    finally:
        # --once-empty-exit lets the pool exit on its own once the queue drains; wait
        # for that clean shutdown, then killpg as the backstop.
        sess.close(self_exit_timeout=30)


def read_scale(run_dir, pass_dir):
    """The shared object SCALE for depth's canonical-units error. Prefer the
    authoritative pass-local pose snapshot; retain the latest-copy fallback for
    legacy renders."""
    s = state_json.first_scale(os.path.join(pass_dir, "pose.json"),
                              os.path.join(run_dir, "mesh", "pose.json"))
    return None if s is None else lie.scalar_scale(s, "pose.json scale")


# --------------------------------------------------------------------------- #
# per-frame scoring
# --------------------------------------------------------------------------- #
def _frame_paths(pass_dir, layout, frame, depth_report=True):
    """The full bundle of PASS_DIR paths + inputs for one frame, derived ONCE from
    the frame stem so the pass number can't drift between them. Pure — no side
    effects — so _score_cpu and _score_critic can each call it and agree byte-for-
    byte on every path they touch.

    `depth_report` (depth_config.json "report" flip switch, default on) gates
    `has_depth` — the observed-depth SCORING (depth_<stem>.json + the residual
    panel). `has_render_depth` is independent of it: our own depth render needs
    no GT, so the object-units DEPTH panel is drawn from it even when scoring
    is off (the monocular case)."""
    paths = pass_pipeline.composite_paths(pass_dir, layout, frame)
    render_depth = os.path.join(pass_dir, f"depth_{paths['stem']}.npy")
    paths.update({
        "render_depth": render_depth,
        "depth_residual": os.path.join(
            pass_dir, f"depth_residual_{paths['stem']}.png"),
        "depth_json": os.path.join(pass_dir, f"depth_{paths['stem']}.json"),
        "critic_out": os.path.join(pass_dir, f"critic_{paths['stem']}.json"),
        "has_depth": (depth_report and bool(layout["tracking"])
                      and os.path.isfile(render_depth)),
        "has_render_depth": os.path.isfile(render_depth),
    })
    return paths


def _score_cpu(run_dir, pass_dir, layout, frame, args, scale,
               screen_critic=False):
    """The CPU, HARD-FAIL half of scoring one frame: composite.py (metrics +
    overlap) then depth.py (depth_<stem>.json). A failure here raises
    SystemExit — inside a worker thread that becomes the future's exception, which
    the dispatcher re-raises to abort the whole pass (a broken composite/depth is
    not something to score around). Returns the result dict; the `critic` path is
    filled deterministically so _score_critic just has to write to it."""
    p = _frame_paths(pass_dir, layout, frame, args.depth_report)
    stem = p["stem"]

    # --- composite: metrics + overlap (+ depth panels when available)
    if p["has_depth"]:
        comp = pass_pipeline.composite_command(
            p, bg_mode=args.bg_mode, tracking=layout["tracking"],
            render_depth=p["render_depth"],
            depth_residual_out=p["depth_residual"],
            side_by_side_max_dimension=getattr(
                args, "side_by_side_max_dimension", 0))
    elif p["has_render_depth"]:
        # no GT to score against — composite draws the object-units DEPTH
        # panel from our render alone (depth / the pose.json scale).
        comp = pass_pipeline.composite_command(
            p, bg_mode=args.bg_mode, render_depth=p["render_depth"],
            side_by_side_max_dimension=getattr(
                args, "side_by_side_max_dimension", 0))
    else:
        comp = pass_pipeline.composite_command(
            p, bg_mode=args.bg_mode,
            side_by_side_max_dimension=getattr(
                args, "side_by_side_max_dimension", 0))
    _run_analysis(comp, f"composite {stem}")

    # --- depth: writes depth_<stem>.json (the file aggregate.py reads)
    if p["has_depth"]:
        # composite.py already wrote the standalone depth-residual PNG
        # (--depth-residual-out); depth.py's job here is depth_<stem>.json.
        dep = ["python", "-m", DEPTH,
               "--tracking", layout["tracking"], "--render-depth", p["render_depth"],
               "--mask", p["mask"], "--source", p["img"], "--frame-name", frame,
               "--out", p["depth_json"]]
        if p["hand"]:
            dep += ["--hand-mask", p["hand"]]
        if scale is not None:
            dep += ["--scale", str(scale)]
        _run_analysis(dep, f"depth {stem}")
    else:
        log(f"{stem}: no observed/rendered depth — skipping depth score")

    return {"frame": frame, "metrics": p["metrics_out"],
            "depth": p["depth_json"] if p["has_depth"] else "",
            "critic": p["critic_out"] if screen_critic else ""}



# The critic backend settings (provider/model/thresholds) live in the resolved
# critic config (CLI flag > run_config.json > critic.py's own default). Each is
# appended ONLY when set — an unset value is omitted so critic.py falls back to its
# own default (provider auto / per-provider model / max-tokens 6000 / ...).
def _critic_backend_flags(critic_cfg):
    """CLI args to append to a `python -m analysis.scorers.critic` command from the
    resolved critic config dict (values may be None = 'use critic.py's default')."""
    flags = []
    for key, flag in (("provider", "--provider"), ("model", "--model"),
                      ("api_key_env", "--api-key-env"),
                      ("max_tokens", "--max-tokens"),
                      ("pass_realism", "--pass-realism"),
                      ("pass_identity", "--pass-identity"),
                      ("max_turntable", "--max-turntable"),
                      ("crop_pad", "--crop-pad")):
        v = critic_cfg.get(key)
        if v is not None and v != "":
            flags += [flag, str(v)]
    if critic_cfg.get("crops"):          # store_true on critic.py — presence only
        flags.append("--crops")
    return flags


def _score_critic(run_dir, pass_dir, layout, frame, args):
    """The NETWORK, SOFT half of scoring one frame: critic.py. Runs after the
    frame's composite (it reads overlap_<stem>.png). Soft — a non-zero exit is
    tolerated; skips cleanly (exit 0, writes nothing) with no VLM creds. Writes to
    the same critic_<stem>.json _score_cpu already put in the result dict."""
    p = _frame_paths(pass_dir, layout, frame, args.depth_report)
    stem = p["stem"]
    crit = ["python", "-m", CRITIC,
            "--source", p["img"], "--render", p["match"],
            "--overlap", p["overlap_out"],
            "--turntable", os.path.join(pass_dir, turntable_views.IMAGE_GLOB),
            "--bg-mode", args.bg_mode, "--mask", p["mask"],
            "--frame", stem, "--pose", os.path.join(pass_dir, "pose.json"),
            "--out", p["critic_out"]]
    if p["hand"]:
        crit += ["--hand-mask", p["hand"]]
    # numeric grounding is OFF by default (it can confuse the visual judge);
    # opt in with --critic-ground to feed this frame's metrics/depth.
    if args.critic_ground:
        if os.path.isfile(p["metrics_out"]):
            crit += ["--metrics", p["metrics_out"]]
        if p["has_depth"] and os.path.isfile(p["depth_json"]):
            crit += ["--depth", p["depth_json"]]
    crit += _critic_backend_flags(getattr(args, "critic", {}) or {})
    _run_analysis(crit, f"critic {stem}", soft=True)
    _stamp_critic_output(p["critic_out"], pass_dir)


def _stamp_critic_output(path, pass_dir):
    """Attach render-pass freshness metadata to a pass-driven critic result."""
    critic = _load_json(path)
    if critic is None:
        return
    manifest = _load_json(os.path.join(pass_dir, "MANIFEST.json")) or {}
    critic["screening"] = {
        "pass": os.path.basename(pass_dir.rstrip(os.sep)),
        "scene_sha1": manifest.get("scene_sha1"),
    }
    with open(path, "w") as f:
        json.dump(critic, f, indent=2)
        f.write("\n")


def select_critic_frames(args, frames):
    """Resolve the explicit VLM screening policy for this pass.

    The routine default is empty. ``--critic-frames`` screens a known coverage or
    suspect set; ``--critic-all`` is reserved for deliberate checkpoints.
    ``--no-critic`` remains as a compatibility spelling for the default.
    """
    requested = [f.strip() for f in
                 str(getattr(args, "critic_frames", "") or "").split(",")
                 if f.strip()]
    all_frames = bool(getattr(args, "critic_all", False))
    disabled = bool(getattr(args, "no_critic", False))
    if disabled and (all_frames or requested):
        raise SystemExit("[shape-pass] --no-critic cannot be combined with "
                         "--critic-frames/--critic-all")
    if all_frames and requested:
        raise SystemExit("[shape-pass] choose either --critic-frames or --critic-all")
    unknown = [f for f in requested if f not in frames]
    if unknown:
        raise SystemExit("[shape-pass] critic frame(s) not in the rendered layout: "
                         + ", ".join(unknown))
    return set(frames if all_frames else requested)


def score_frames(run_dir, pass_dir, layout, args, scale):
    """Score every frame numerically, then VLM-screen only selected frames."""
    frames = layout["frames"]
    critic_frames = getattr(args, "selected_critic_frames", None)
    if critic_frames is None:
        critic_frames = select_critic_frames(args, frames)
    results = {}
    with ThreadPoolExecutor(max_workers=args.ncpu) as cpu, \
         ThreadPoolExecutor(max_workers=args.critic_conc) as net:
        cpu_futs = {
            cpu.submit(_score_cpu, run_dir, pass_dir, layout, fr, args, scale,
                       fr in critic_frames): fr
            for fr in frames
        }
        critic_futs = []
        try:
            for fut in as_completed(cpu_futs):
                fr = cpu_futs[fut]
                results[fr] = fut.result()   # re-raises composite/depth SystemExit
                if results[fr].get("critic"):
                    critic_futs.append(net.submit(_score_critic, run_dir, pass_dir,
                                                  layout, fr, args))
        except BaseException:
            # a CPU worker hard-failed — cancel everything still queued and abort.
            cpu.shutdown(wait=False, cancel_futures=True)
            net.shutdown(wait=False, cancel_futures=True)
            raise
        # drain the critic futures (soft — _score_critic does not raise on a critic
        # failure; result() surfaces only an unexpected dispatcher-side bug).
        for fut in as_completed(critic_futs):
            fut.result()
    return [results[fr] for fr in frames]   # RE-SORT to layout order


def build_turntable_sheets(pass_dir, run_dir=""):
    """Lay this pass's turntable renders out as ONE PAGE PER ARTICULATION STATE,
    each page opened by the state's SOURCE photo (resolved via the run's
    layout.json) as a reference tile.

    SOFT, deliberately. The sheet is an index onto renders that already succeeded —
    the individual PNGs are on disk and the gate is readable without it — so a cv2
    problem must not fail a pass whose geometry rendered fine. The same reasoning
    `pool/panels.py` applies to candidate sheets.

    Skips quietly when the pass has no turntable renders: shape_pass.sh's own passes
    always render it, but `--views match` exists and a missing-view error here would be a
    lie about what went wrong.
    """
    import glob as _glob
    if not _glob.glob(os.path.join(pass_dir, turntable_views.IMAGE_GLOB)):
        log("turntable sheets: no turntable renders in this pass — skipping")
        return
    cmd = ["python", "-m", TURNTABLE_SHEET, "--pass-dir", pass_dir]
    if run_dir:
        cmd += ["--run-dir", run_dir]
    _run_analysis(cmd, "turntable sheets", soft=True)


def build_mechanism_sheets(pass_dir, columns=None):
    """Lay this pass's mechanism renders out as ONE STRIP PER ROW per joint —
    the joint swept across its declared limit, in sweep order, no source tile.

    SOFT and skip-quiet for the same reasons as the turntable sheets above: the
    renders already succeeded, and a rigid object legitimately has no sweep (the
    view says so and writes nothing, so "no renders" here is not an error).
    """
    import glob as _glob
    if not _glob.glob(os.path.join(pass_dir, mechanism_views.IMAGE_GLOB)):
        log("mechanism sheets: no mechanism renders in this pass "
            "(a rigid object has no joint to sweep) — skipping")
        return
    cmd = ["python", "-m", MECHANISM_SHEET, "--pass-dir", pass_dir]
    if columns:
        cmd += ["--columns", str(columns)]
    _run_analysis(cmd, "mechanism sheets", soft=True)


def self_intersection_paths(pass_dir):
    """Where this pass's self-intersection report lands (txt read + JSON)."""
    return (os.path.join(pass_dir, "self_intersection.txt"),
            os.path.join(pass_dir, "self_intersection.json"))


def run_self_intersection(run_dir, pass_dir):
    """Part-pair overlap volumes for the state this pass just rendered —
    MANDATORY, every pass, no flag to skip it.

    Both inputs already exist by this point (the pass's OWN pose.json — so the
    states measured are the states rendered — and the GLB this render exported),
    it needs no renders and no network, and costs ~1s, so there is nothing to
    trade off.

    MANDATORY MEANS THE REPORT, NOT A VERDICT: the tool still thresholds nothing
    (analysis/self_intersection.md). A non-zero exit means the report could not be
    PRODUCED (unreadable GLB, incomplete joint states) — a defect in the state
    this pass claims to have rendered — so it aborts the pass like any other hard
    scorer.

    Covers every frame the scene declares, not just a --frames subset: an overlap
    in an unrendered frame is still wrong in the committed state.
    """
    out_table, out_json = self_intersection_paths(pass_dir)
    _run_analysis(["python", "-m", SELF_INTERSECTION,
                   "--pose-json", os.path.join(pass_dir, "pose.json"),
                   "--glb", os.path.join(run_dir, "mesh", "object.glb"),
                   "--out-table", out_table, "--out", out_json],
                  "self-intersection")
    return out_json


def _self_intersection_lines(report):
    """The pass summary's self-intersection block: the deepest part-pair overlap,
    the ONSETS (the step INTO that frame drove the parts together — the row to read
    in the temporal report), and any part excluded as non-watertight.

    A few lines only; the full table sits beside it in the pass dir. Printed inline
    so an overlap is SEEN at decision time — same reason as the critic's verdict."""
    if report is None:
        return ["[shape-pass]   (no report — see the self-intersection output above)"]
    frames = report.get("frames") or []
    part_size = {n: p.get("size") for n, p in (report.get("parts") or {}).items()}
    floor = report.get("mark_floor") or 0.0

    worst = None
    for row in frames:
        for o in row.get("overlaps") or []:
            if worst is None or o["frac_smaller"] > worst[1]["frac_smaller"]:
                worst = (row["frame"], o)

    out = []
    skipped = report.get("skipped_parts") or []
    if skipped:
        out.append("[shape-pass]   ⚠ NOT watertight, excluded from every pair "
                   "(overlaps UNKNOWN, not zero): "
                   + ", ".join(f"{s['name']} ({s.get('why', '?')})"
                               for s in skipped))
    if worst is None:
        out.append(f"[shape-pass]   no articulating-pair overlap in any of the "
                   f"{len(frames)} frame(s)")
        return out

    frame, w = worst
    a, b = w["pair"]

    def side(name):
        s = part_size.get(name)
        return f"{s:.2f}" if isinstance(s, (int, float)) else "?"
    mark = "⚠" if w["frac_smaller"] >= floor else "·"
    out.append(f"[shape-pass]   {mark} deepest: {a}({side(a)}) ∩ {b}({side(b)}) at "
               f"{frame} — ovl {w['size']:.3f} (equivalent-cube side), "
               f"{w['frac_smaller']:.2%} of the smaller part")
    onsets = report.get("onsets") or {}
    if onsets:
        shown = list(onsets.items())[:4]
        out += [f"[shape-pass]     onset {pair}: "
                + ", ".join(os.path.splitext(f)[0] for f in fs)
                for pair, fs in shown]
        if len(onsets) > len(shown):
            out.append(f"[shape-pass]     (+{len(onsets) - len(shown)} more pair(s) "
                       "in the table)")
    return out


def _run_analysis(cmd, label, soft=False):
    # cmd is ["python", "-m", "analysis.<pkg>.<tool>", ...]; the module target at
    # index 2 is already the readable label, so log the command verbatim.
    log(label + ": " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=HARNESS, capture_output=True, text=True)
    if r.stdout:
        sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        if soft:
            log(f"{label}: exit {r.returncode} (soft — continuing)")
        else:
            raise SystemExit(f"[shape-pass] {label} FAILED (exit {r.returncode}).")


# --------------------------------------------------------------------------- #
# aggregate + summary
# --------------------------------------------------------------------------- #
def run_aggregate(run_dir, pass_dir):
    _run_analysis(["python", "-m", AGGREGATE, "--run-dir", run_dir,
                   "--views-dir", pass_dir], "aggregate")


# intent.json is written by render_wrapper at pass allocation (pre-render, so
# it survives a mid-render death); this module only forwards the flags
# (_render_sh) and reads the file back (_parent_intent_lines).
INTENT_NAME = pass_dirs.INTENT_NAME


def _read_num(path, key):
    try:
        with open(path) as f:
            return json.load(f).get(key)
    except (OSError, ValueError):
        return None


def _load_json(path):
    """Whole JSON dict, or None if the file is absent/unreadable/not a dict."""
    if not path:
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _critic_lines(frame, critic_path):
    """Lines summarizing this frame's VLM critic verdict for the pass summary.

    The critic (critic.py) writes realism/identity scores, a one-line summary, and
    a ranked `discrepancies` list — each tagged shape/pose/joint with a severity.
    We surface it inline so the agent SEES it when deciding the next move, instead
    of having to open critic_<frame>.json by hand. We print EVERY discrepancy
    (high/med/low, any tag) in the critic's own biggest-first order, so no signal is
    hidden; HIGH-severity fixes (ANY tag) are marked ⚠ because those are the
    reconcile-before-finalize items (aggregate.py's FINALIZE REVIEW REQUIRED).
    Degrades gracefully: a missing file / an error status / no creds prints a single
    skipped line and never crashes."""
    c = _load_json(critic_path)
    if c is None or c.get("status") == "error":
        return [f"[shape-pass]   {frame}: critic (skipped — no verdict this pass)"]

    def fnum(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else "-"
    head = (f"[shape-pass]   {frame}: realism={fnum(c.get('realism'))} "
            f"identity={fnum(c.get('identity'))} — {c.get('summary') or ''}".rstrip())
    out = [head]

    ds = [d for d in (c.get("discrepancies") or []) if isinstance(d, dict)]
    if not ds:
        out.append("[shape-pass]     (no fixes — critic clean)")
        return out
    for d in ds:  # biggest-first order the critic returned; show all severities
        sev = str(d.get("severity", "")).strip().lower() or "?"
        tag = str(d.get("tag", "?")).strip().lower() or "?"
        part = str(d.get("part", "?")).strip() or "?"
        adj = (d.get("adjustment") or d.get("observation") or "").strip()
        # HIGH fixes are the gating ones — mark them so they stand out from the
        # advisory med/low notes.
        marker = "⚠ HIGH" if sev == "high" else f"· {sev}"
        out.append(f"[shape-pass]     {marker} {tag}/{part}: {adj}")
    return out


def _resolve_parent_snapshot(root, parent):
    """(parent_name, snapshot_path) for the nearest ancestor that KEPT a
    snapshot, starting at `parent` and walking down.

    The manifest's `parent` is frozen at write time, but pass dirs get pruned by
    hand — so the recorded parent may no longer exist while an older pass with a
    perfectly good snapshot does. Re-deriving on a miss routes around the hole
    instead of silently dropping the diff, which is the whole value of computing
    lineage from the numbering rather than recording it.

    Returns (parent, None) when the chain runs out, so a caller can still name
    the recorded parent while explaining why there is no diff.
    """
    candidate = parent
    while candidate:
        path = os.path.join(root, candidate, pass_dirs.SCENE_SNAPSHOT)
        if os.path.isfile(path):
            return candidate, path
        candidate = pass_dirs.parent_of(root, candidate)
    return parent, None


def _lineage_lines(pass_dir):
    """Where this pass sits in the chain, and how to read what it changed.

    Printed because a scene_sha1 nobody resolves is a hash nobody uses: naming
    the parent's snapshot beside it turns "the scene changed" into a diff the
    agent can actually run.

    The parent's `intent`/`rollback` are echoed here too. They are written so a
    resumed run reads the reason instead of inferring it — which only works if
    something on the other side of the interruption SHOWS them, and this summary
    is the one thing every pass prints.
    """
    manifest = _load_json(os.path.join(pass_dir, "MANIFEST.json")) or {}
    parent = manifest.get("parent")
    scene_sha = str(manifest.get("scene_sha1") or "-")[:12]
    if not parent:
        return [f"[shape-pass] lineage: first pass (no parent), "
                f"scene {scene_sha}"]
    root = os.path.dirname(os.path.abspath(pass_dir.rstrip(os.sep)))
    lines = [f"[shape-pass] lineage: parent {parent}, scene {scene_sha}"]
    resolved, prev = _resolve_parent_snapshot(root, parent)
    if prev and manifest.get("scene_snapshot"):
        if resolved != parent:
            lines.append(f"[shape-pass]   (pass {parent} is gone; diffing "
                         f"against the nearest kept snapshot, {resolved})")
        lines.append(f"[shape-pass]   what this pass changed: diff {prev} "
                     f"{os.path.join(pass_dir, manifest['scene_snapshot'])}")
    elif not manifest.get("scene_snapshot"):
        lines.append("[shape-pass]   no diff: THIS pass kept no scene snapshot")
    else:
        lines.append(f"[shape-pass]   no diff: no ancestor of {parent} kept a "
                     f"scene snapshot (passes predating snapshots?)")
    lines.extend(_parent_intent_lines(root, parent))
    return lines


def _parent_intent_lines(root, parent):
    """The parent's declared intent / rollback scope, if it made one.

    This is the read side of `--rollback`: "revert the geometry, KEEP the 000250
    hinge" is written for whoever resumes, and without this it was written for
    nobody.
    """
    doc = _load_json(os.path.join(root, parent, INTENT_NAME)) or {}
    lines = []
    if doc.get("intent"):
        lines.append(f"[shape-pass]   parent {parent} intended: "
                     f"{doc['intent']}")
    if doc.get("rollback"):
        lines.append(f"[shape-pass]   parent {parent} rollback scope: "
                     f"{doc['rollback']}")
    return lines


def print_summary(pass_dir, scored, self_intersection=None):
    print("\n[shape-pass] " + "=" * 60)
    print(f"[shape-pass] PASS_DIR: {pass_dir}")
    for line in _lineage_lines(pass_dir):
        print(line)
    print(f"[shape-pass] {'frame':<14}{'iou_raw':>10}{'iou_visible':>13}"
          f"{'depth_canon':>13}")
    for s in scored:
        iou = _read_num(s["metrics"], "iou_raw")
        iouv = _read_num(s["metrics"], "iou_visible")
        dcanon = _read_num(s["depth"], "depth_mae_canon") if s["depth"] else None

        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else "-"
        print(f"[shape-pass] {s['frame']:<14}{fmt(iou):>10}{fmt(iouv):>13}"
              f"{fmt(dcanon):>13}")

    # Self-intersection — every pass, so an impossible configuration is seen HERE
    # rather than in a report nobody opened. A warning, never a gate.
    if self_intersection:
        print("[shape-pass]")
        print("[shape-pass] Self-intersection (part-pair overlaps, all declared "
              "frames):")
        for line in _self_intersection_lines(_load_json(self_intersection)):
            print(line)
        table, _ = self_intersection_paths(pass_dir)
        print(f"[shape-pass]   -> full per-frame table: {table} (read the ONSETS "
              "beside the temporal report — the step INTO an onset frame is what "
              "drove the parts together). A WARNING, not a gate: a modelled "
              "contact is fine once you can name it.")

    # VLM critic — an explicit checkpoint/escalation screen, surfaced inline when
    # requested so its shape/pose/joint finding can route the next round.
    if any(s.get("critic") for s in scored):
        print("[shape-pass]")
        print("[shape-pass] VLM screening (selected frames only):")
        for s in scored:
            if s.get("critic"):
                for line in _critic_lines(s["frame"], s["critic"]):
                    print(line)
        print("[shape-pass]   -> HIGH fixes (ANY tag) must be reconciled before finalize. "
              "A pose `sweep` CANNOT fix a `shape` flag — fix it in build() first; "
              "sweeping against wrong geometry just entrenches it.")

    # the strip is `composite.side_by_side_panel` (see the --side-by-side-out arg
    # built above), so this names ITS columns in ITS order: SOURCE | RENDER |
    # MASKED SOURCE (`match` survives only as a column ALIAS).
    print("[shape-pass] Read the individual images in PASS_DIR each iteration: "
          "side_by_side_<f>_preview.png when present (then side_by_side_<f>.png "
          "at native image size when needed; "
          "source | render | masked source), "
          "overlap_<f>.png, and depth_residual_<f>.png.")
    # The turntable is named separately because what you READ for it is the SHEET,
    # not the tiles: one page per articulation state, every orbit view on it,
    # labelled with the direction it was shot from. The tiles are still there for
    # when a suspect region needs full resolution.
    print("[shape-pass] For 3D coherence read ONE page per articulation state: "
          "turntable_sheets/turntable_sheet_<state>.png (open the individual "
          f"{turntable_views.IMAGE_GLOB} tiles when a region needs full "
          "resolution).")
    # The mechanism hints belong to the mechanism module: a run without it has
    # no sheets to read and no pick to grade, and telling it otherwise would send
    # the agent hunting for files this run never renders.
    if not modules.find_modules(pass_dir)["mechanism"]:
        return
    print("[shape-pass] After AUTHORING or CHANGING a joint (axis/origin/limit/"
          "child), read mechanism_sheets/mechanism_sheet_<joint>_A.png AND "
          "_<joint>_B.png (the stacked _<joint>.png is the same tiles for "
          "same-state comparison). They are TWO arcs of that joint: one swept "
          "about the declared `axis`, one about its MIRROR. Decide WHICH ARM is "
          "the real mechanism. The arms render identically at both endpoints, so "
          "the answer is in the INTERIOR tiles — in one arm the child swings "
          "into free space, in the other it drives through the body it should be "
          "opening away from. Name landmark parts (which face is toward camera, "
          "what occludes what), not just 'looks right'.")
    print("[shape-pass] The pick is GRADED (it gates windows plan and finalize; "
          "an answer survives until the declaration changes): "
          "python -m analysis.mechanism_calls ask --run-dir RUN_DIR "
          f"--pass-dir {pass_dir} , write the arm + evidence in "
          "RUN_DIR/mechanism_calls.json, then "
          "python -m analysis.mechanism_calls check --run-dir RUN_DIR . "
          "A MIRRORED DECLARATION blocks and prints the one axis sign to write; "
          "writing it settles the SAME pick (no re-render, no re-answer).")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="the run directory (runs/<name>)")
    p.add_argument("--frames", default="",
                   help="comma list of frames to render+score; default: the "
                        "frames in layout.json")
    p.add_argument("--existing-pass", default="",
                   help="score and complete an existing render pass below the "
                        "currently open iteration without rerendering")
    p.add_argument("--no-critic", action="store_true",
                   help="make no VLM calls (compatibility flag; this is the default)")
    p.add_argument("--critic-frames", default="",
                   help="comma list of rendered frames to VLM-screen; routine "
                        "iterations screen none")
    p.add_argument("--critic", "--critic-all", dest="critic_all",
                   action="store_true",
                   help="VLM-screen every rendered frame at a deliberate checkpoint")
    p.add_argument("--critic-ground", action="store_true",
                   help="feed the critic numeric grounding (metrics/depth) for each "
                        "frame; OFF by default (grounding can confuse the visual "
                        "judge — it stays an independent image-only critic unless "
                        "you opt in)")
    p.add_argument("--bg-mode", default=None, choices=["alpha", "black"],
                   help="backdrop the composites and the critic use for match/turntable/masked-source "
                        "panels: 'black' (default, uniform — dark objects stay visible) "
                        "or 'alpha' (keep the renders' transparent film)")
    p.add_argument("--engine", default="BLENDER_EEVEE_NEXT",
                   choices=["CYCLES", "BLENDER_EEVEE_NEXT"],
                   help="render engine (default EEVEE; CYCLES for a final pass)")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--turntable-views", default=None,
                   choices=sorted(turntable_views.VIEW_SETS),
                   help="turntable view set per articulation state: 'sphere8' "
                        "(default — a level ring plus a raised and a dropped pair, "
                        "so the top and underside are seen) or 'ring4' (the level "
                        "ring only, half the renders per state). Cost is "
                        "states x views. See views/turntable.md")
    p.add_argument("--turntable-jitter", type=int, default=None,
                   help="SEED — perturb the turntable directions for a deliberate "
                        "look-from-somewhere-new pass. Unset (default) keeps every "
                        "pass on the SAME directions, which is what makes one "
                        "iteration's views comparable with the last's")
    p.add_argument("--mechanism-rows", type=int, default=None,
                   help="viewpoints per joint for the mechanism sweep (default 2, "
                        "picked >=90 deg apart in azimuth). 1 = quick look; 3-4 "
                        "when a joint stays ambiguous. Cost is joints x rows x "
                        "samples renders")
    p.add_argument("--mechanism-samples", type=int, default=None,
                   help="states per joint per row for the mechanism sweep "
                        "(default 9, endpoints included)")
    p.add_argument("--mechanism-columns", type=int, default=None,
                   help="tiles per line on the mechanism sheet before a row's "
                        "strip wraps (default 9 — a default sweep stays one "
                        "line). Match to --mechanism-samples if you raise it")
    p.add_argument("--pass-label", default="",
                   help="store optional descriptive metadata in the pass manifest; "
                        "directories remain numeric and 'final' is reserved")
    p.add_argument("--intent", default="",
                   help="what this pass CHANGES and why, in your own words, "
                        "recorded to <pass>/intent.json. Stated before the render, "
                        "so a resumed run reads the reason instead of inferring it "
                        "from the scene")
    p.add_argument("--rollback", default="",
                   help="what to KEEP if this pass is rejected (e.g. 'revert the "
                        "cover extents, keep the 000250 hinge'). A pass that "
                        "changes two independent things otherwise reads as one "
                        "atomic experiment after a resume, and rejecting the bad "
                        "half discards the good half with it")
    p.add_argument("--no-aggregate", action="store_true",
                   help="skip the final aggregate.py roll-up")
    p.add_argument("--depth-report", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="score + panel observed depth this pass (depth_<stem>.json, "
                        "the depth-residual images, and the depth_mae_canon "
                        "summary column). --no-depth-report skips depth SCORING "
                        "only; every downstream depth reporter then goes quiet on "
                        "its own, but the depth view is still rendered (see "
                        "--no-depth-render) because the object-units DEPTH panel "
                        "and depth sheet need no GT. Camera seeds / measure_depth "
                        "are UNAFFECTED. Config: depth_config.json \"report\" "
                        "(default on)")
    p.add_argument("--depth-render", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="render the depth view this pass (depth_<stem>.npy), which "
                        "feeds the depth scorer AND the GT-free object-units DEPTH "
                        "panel / depth sheet. --no-depth-render drops the view, "
                        "saving one render per frame per pass and leaving every "
                        "depth visual with nothing to draw — the switch to use when "
                        "a run should not pay for depth at all. INDEPENDENT of "
                        "--depth-report. Config: depth_config.json \"render\" "
                        "(default on)")
    p.add_argument("--mechanism", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="render the mechanism A/B sweep (both arms of every "
                        "articulated joint). DEFAULT: only when a pick is OWED — "
                        "i.e. some joint's A/B answer is missing or stale. The "
                        "sweep costs joints x 2 arms x rows x samples renders and "
                        "feeds nothing but the pick gate, whose answers survive "
                        "until a declaration changes, so an unchanged rig would "
                        "render arcs nobody is asked about. --mechanism forces it "
                        "(a fresh look at an unchanged rig); --no-mechanism skips "
                        "it even when owed, which leaves the gate unanswerable")
    # --- concurrency (G/N/C) — CLI flag > run_config.json > default ----------
    # These default to None so an omitted flag defers to run_config.json's
    # `concurrency` block (else the built-in default). See utils/_run_config.py.
    p.add_argument("--ncpu", type=int, default=None,
                   help="N — CPU scoring workers (concurrent composite+depth "
                        "subprocesses; default min(4, cpu_count); 1 = serial, "
                        "reproduces the old behavior). Config: concurrency.ncpu")
    p.add_argument("--critic-conc", type=int, default=None,
                   help="C — VLM critic workers (concurrent critic.py subprocesses; "
                        "network-bound; default 8; 1 = serial). Config: "
                        "concurrency.critic_conc")
    # --- resident-pool rendering (opt-in; cold render.sh is the default) ---------
    p.add_argument("--pool", action=argparse.BooleanOptionalAction, default=None,
                   help="render per-frame match/depth through the resident pool "
                        "worker pool (build Blender once, render many) instead of a "
                        "cold render.sh. A cheap gauge render still produces the "
                        "turntable + pose.json + GLB. Config: pool.enable "
                        "(default off — unchanged cold path)")
    p.add_argument("--workers", type=int, default=None,
                   help="G — resident Blender workers when --pool (default 2). "
                        "Config: concurrency.workers")
    p.add_argument("--gpus", default=None,
                   help="comma GPU ids the pool pins workers to, round-robin "
                        "(default: none — Blender sees all). Config: pool.gpus")
    p.add_argument("--pool-timeout", type=float, default=600.0,
                   help="per-order wait timeout (s) for a pool render (default 600)")
    p.add_argument("--archive-spool", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="reset the spool by RENAMING it to spool-<timestamp>/ "
                        "instead of deleting it, so a previous pass's orders, "
                        "sweep reports and candidate sheets stay debuggable "
                        "(default ON). --no-archive-spool restores the rmtree. "
                        "Config: pool.archive_spool")
    p.add_argument("--keep-spools", type=int, default=None,
                   help="how many spool archives to retain, oldest pruned first "
                        "(default 8; 0 = keep every archive forever). Config: "
                        "pool.keep_spools")
    # --- VLM critic backend — CLI flag > run_config.json > critic.py default -
    # Each stays None when unset so critic.py's own default applies. Config: the
    # `critic` block. See analysis/scorers/critic/critic.md for meaning.
    p.add_argument("--provider", default=None,
                   choices=["auto", "anthropic", "openai"],
                   help="VLM backend for the critic (default auto: whichever API key "
                        "is available). Config: critic.provider")
    p.add_argument("--model", default=None,
                   help="critic model id override (default per provider). Config: "
                        "critic.model")
    p.add_argument("--api-key-env", default=None,
                   help="env var holding the API key for the critic. Config: "
                        "critic.api_key_env")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="critic max output tokens (default 6000). Config: "
                        "critic.max_tokens")
    p.add_argument("--pass-realism", type=float, default=None,
                   help="critic realism soft-gate (default 0.7). Config: "
                        "critic.pass_realism")
    p.add_argument("--pass-identity", type=float, default=None,
                   help="critic identity soft-gate (default 0.7). Config: "
                        "critic.pass_identity")
    p.add_argument("--max-turntable", type=int, default=None,
                   help="cap on turntable images sent to the critic (default 16). "
                        "Spent across articulation states, so a capped screen drops "
                        "angles rather than hiding a configuration. "
                        "Config: critic.max_turntable")
    p.add_argument("--crops", action=argparse.BooleanOptionalAction, default=None,
                   help="send zoomed detail crops to the critic (off by default). "
                        "Config: critic.crops")
    p.add_argument("--crop-pad", type=float, default=None,
                   help="detail-crop padding fraction (default 0.12). Config: "
                        "critic.crop_pad")
    # layout overrides (default: from RUN_DIR/layout.json)
    p.add_argument("--tracking", default="",
                   help="capture tracking dir override (cameras.npz + "
                        "keyframes.json); default = <layout capture>/tracking")
    p.add_argument("--frames-dir", default="")
    p.add_argument("--masks-dir", default="")
    p.add_argument("--hand-masks-dir", default="")
    p.add_argument("--ref-frame", default="")
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"[shape-pass] no such run dir: {run_dir}")
    _run_lock_handle = acquire_run_lock(run_dir)

    resolve_config(run_dir, args)

    layout = resolve_layout(run_dir, args)
    check_inputs(layout)
    args.selected_critic_frames = select_critic_frames(args, layout["frames"])
    log(f"frames: {', '.join(layout['frames'])}  (ref {layout['ref_frame']})")
    if args.selected_critic_frames:
        selected = [f for f in layout["frames"] if f in args.selected_critic_frames]
        log("VLM screening: " + ", ".join(selected))
    else:
        log("VLM screening: none (numeric/visual iteration)")
    if not args.depth_report:
        log("depth reporting: OFF (depth_config.json report=false / "
            "--no-depth-report) — no depth score/residual panels; the depth view "
            "is still rendered (--no-depth-render turns that off too)")
    if not args.depth_render:
        log("depth render: OFF (depth_config.json render=false / "
            "--no-depth-render) — no depth_<stem>.npy, so no object-units DEPTH "
            "panel and no depth sheet either")

    if args.existing_pass:
        args.iteration = bookkeeping.active(run_dir, required=True)
        if args.iteration.get("kind") not in {"shape", "pose"}:
            raise ValueError(
                f"--existing-pass cannot resume {args.iteration.get('kind')} "
                f"iteration {args.iteration['name']}")
    else:
        args.iteration = bookkeeping.begin(
            run_dir, "shape", label=args.pass_label,
            attach_to=("shape", "pose"))
    log(f"iteration {args.iteration['name']} "
        f"({args.iteration['kind']}): {args.iteration['path']}")

    pass_dir = resolve_existing_pass(args.iteration, args.existing_pass)
    if pass_dir:
        log(f"resuming existing render pass without rerendering: {pass_dir}")
    else:
        render_fn = run_render_pool if args.pool else run_render
        pass_dir = render_fn(run_dir, layout, args)
    # intent.json was written by render_wrapper at pass allocation (pre-render,
    # so it survives a mid-render death); here it is only confirmed.
    if os.path.isfile(os.path.join(pass_dir, INTENT_NAME)):
        log(f"intent recorded: {os.path.join(pass_dir, INTENT_NAME)}")
    build_turntable_sheets(pass_dir, run_dir=run_dir)
    build_mechanism_sheets(pass_dir,
                           columns=getattr(args, "mechanism_columns", None))
    scale = read_scale(run_dir, pass_dir)

    scored = score_frames(run_dir, pass_dir, layout, args, scale)

    # MANDATORY, no flag: the state was just rendered, so say whether it is
    # physically possible. Cheap (no renders, no network) and hard-failing only
    # when the report cannot be produced at all — see run_self_intersection.
    self_intersection = run_self_intersection(run_dir, pass_dir)

    if not args.no_aggregate:
        run_aggregate(run_dir, pass_dir)

    print_summary(pass_dir, scored, self_intersection=self_intersection)
    recorded = bookkeeping.record_pass(
        run_dir, args.iteration["iteration"], pass_dir)
    if recorded["kind"] == "pose":
        log(f"recorded verification pass for pose iteration "
            f"{recorded['name']}; adjudicate --check will close it")
    else:
        completed = bookkeeping.complete(run_dir, recorded["iteration"])
        log(f"completed iteration {completed['name']} at "
            f"{completed['git_commit'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
