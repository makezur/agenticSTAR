"""_run_config.py — resolve a run's per-run tool settings from a config json.

`RUN_DIR/run_config.json` carries the concurrency knobs (G/N/C), render-pool
settings, visual presentation settings, and the VLM-critic backend settings, so
none of them need re-typing on every invocation. Both GPU-pool drivers read it —
`utils/shape_pass.py` and `multiagent/pool_session.py` — which is why the FILE is named for the run rather
than for either tool: the `pool` block is not the shape pass's property. It
mirrors the `depth_config.json` precedent (a per-run json a tool reads instead of
pure CLI flags), with the same precedence rule:

    explicit CLI flag  >  config json value  >  built-in default

Unlike `core.depth_config.resolve_conf_thr` (which walks UP the tree because the depth
scorers only get a render/out path), both readers always know RUN_DIR, so the
config is read directly — no walk-up needed.

Schema (all fields optional; an absent/corrupt file -> {} -> pure CLI defaults, so
older runs are unchanged):

    {
      "concurrency": { "workers": 3, "ncpu": 4, "critic_conc": 8 },
      "visuals": { "bg_mode": "black" },
      "critic": { "provider": "auto", "model": "", "api_key_env": "",
                  "max_tokens": 6000, "pass_realism": 0.7, "pass_identity": 0.7,
                  "max_turntable": 16, "crops": false, "crop_pad": 0.12 },
      "pool": { "enable": false, "gpus": [0],
                "archive_spool": true, "keep_spools": 8,
                "candidate_sheet_max_dimension": 0,
                "side_by_side_max_dimension": 0 }
    }
"""
import json
import os

CONFIG_NAME = "run_config.json"
# The legacy config name, still read when no run_config.json is present so a
# run already on disk keeps its settings instead of falling back to defaults.
LEGACY_CONFIG_NAME = "iterate_config.json"


def config_path(run_dir):
    """The config file this run actually has: the current name if it exists, else
    the legacy one if THAT exists, else the current name (for a writer)."""
    current = os.path.join(run_dir, CONFIG_NAME)
    if os.path.isfile(current):
        return current
    legacy = os.path.join(run_dir, LEGACY_CONFIG_NAME)
    return legacy if os.path.isfile(legacy) else current


def load_config(run_dir):
    """The parsed run config, or {} if absent/unreadable/not a dict (so a missing
    config silently falls back to CLI defaults). Reads `run_config.json`, or the
    legacy `iterate_config.json` when that is what the run has."""
    try:
        with open(config_path(run_dir)) as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, ValueError):
        return {}


def section(cfg, name):
    """A named top-level block ('concurrency' | 'visuals' | 'critic' | 'pool') as a dict —
    {} when absent or the wrong type."""
    s = cfg.get(name)
    return s if isinstance(s, dict) else {}


def pick(cli_value, cfg_section, key, default):
    """Resolve one setting with precedence CLI > config json > default.

    `cli_value` is the argparse value; config-settable flags use a `None` default
    so `None` means 'not passed on the CLI' (fall through to config/default). A
    config value of `None` is likewise treated as absent."""
    if cli_value is not None:
        return cli_value
    v = cfg_section.get(key)
    if v is not None:
        return v
    return default


def _pool_max_dimension(cfg, key):
    """One validated pool-panel preview target (0 means no preview)."""
    value = section(cfg, "pool").get(key, 0)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"run_config.json pool.{key} must be an integer >= 0, got {value!r}")
    return value


def candidate_sheet_max_dimension(cfg):
    """The pool's candidate-sheet preview longest-edge target (0 means none).

    This is deliberately strict rather than coercing strings/floats: a misspelled
    or malformed visual budget must not quietly make a run's visual evidence much
    larger or smaller than its recorded configuration says.
    """
    return _pool_max_dimension(cfg, "candidate_sheet_max_dimension")


def side_by_side_max_dimension(cfg):
    """The recurring review-strip preview longest-edge target (0 means none)."""
    return _pool_max_dimension(cfg, "side_by_side_max_dimension")


# --------------------------------------------------------------------------- #
# scaffold — the starter config run.sh writes into a fresh RUN_DIR
# --------------------------------------------------------------------------- #
# The scaffold is a deliberate full-size profile, not the built-in defaults
# above: pool on, 16 workers/ncpu (the per-GPU cap,
# pool.manager.MAX_WORKERS_PER_GPU), alpha-backed visual panels, one critic
# worker. The built-in defaults stay conservative so a bare run with NO config
# file doesn't try to spawn a 16-worker pool. The profile lives HERE, beside the schema and the
# defaults, so the two cannot drift apart.
def scaffold_config(workers=16, ncpu=16, gpus=(0,),
                    candidate_sheet_max_dimension=0,
                    side_by_side_max_dimension=0, keep_spools=8,
                    critic_provider="auto"):
    return {
        "concurrency": {"workers": int(workers), "ncpu": int(ncpu),
                        "critic_conc": 1},
        "visuals": {"bg_mode": "alpha"},
        "critic": {"provider": critic_provider, "pass_realism": 0.7,
                   "pass_identity": 0.7, "max_turntable": 16},
        "pool": {"enable": True, "gpus": [int(g) for g in gpus],
                 "archive_spool": True,
                 "keep_spools": int(keep_spools),
                 "candidate_sheet_max_dimension": int(
                     candidate_sheet_max_dimension),
                 "side_by_side_max_dimension": int(
                     side_by_side_max_dimension)},
    }


def _main(argv=None):
    """CLI for run.sh: write RUN_DIR/run_config.json unless it exists.

    A --no-timestamp re-run must never clobber a hand-edited config, so an
    existing file is kept (unlike layout.json/depth_config.json, which are
    derived from run.sh's args and always rewritten)."""
    import argparse
    p = argparse.ArgumentParser(
        description="scaffold RUN_DIR/run_config.json (kept if it exists)")
    p.add_argument("run_dir")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--ncpu", type=int, default=16)
    p.add_argument("--gpus", default="0",
                   help="comma-separated GPU ids for pool.gpus (default '0')")
    p.add_argument(
        "--candidate-sheet-max-dimension", type=int, default=0,
        help="target candidate-sheet preview longest edge; canonical pages stay "
             "native and previews never drop below 0.5x (0 = no preview)")
    p.add_argument(
        "--side-by-side-max-dimension", type=int, default=0,
        help="target side-by-side preview longest edge; canonical strips stay "
             "native and previews never drop below 0.5x (0 = no preview)")
    p.add_argument(
        "--keep-spools", type=int, default=8,
        help="number of spool archives to retain (default 8; 0 = never prune)")
    p.add_argument(
        "--critic-provider", default="auto", choices=["auto", "anthropic", "openai"],
        help="VLM backend for the critic block (default auto: whichever API key "
             "the sandbox has)")
    a = p.parse_args(argv)
    out = os.path.join(a.run_dir, CONFIG_NAME)
    if os.path.isfile(out):
        print(f"run config: kept existing {out}")
        return 0
    if a.workers < 1 or a.ncpu < 1:
        p.error("--workers/--ncpu must be positive")
    if a.candidate_sheet_max_dimension < 0:
        p.error("--candidate-sheet-max-dimension must be >= 0")
    if a.side_by_side_max_dimension < 0:
        p.error("--side-by-side-max-dimension must be >= 0")
    if a.keep_spools < 0:
        p.error("--keep-spools must be >= 0")
    try:
        gpus = [int(g) for g in a.gpus.replace(" ", "").split(",") if g != ""]
    except ValueError:
        p.error(f"--gpus must be comma-separated GPU ids, got {a.gpus!r}")
    cfg = scaffold_config(
        a.workers, a.ncpu, gpus or (0,),
        candidate_sheet_max_dimension=a.candidate_sheet_max_dimension,
        side_by_side_max_dimension=a.side_by_side_max_dimension,
        keep_spools=a.keep_spools, critic_provider=a.critic_provider)
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"run config: wrote {out} (workers={a.workers}, ncpu={a.ncpu}, "
          f"pool.gpus={cfg['pool']['gpus']}, "
          f"pool.keep_spools={cfg['pool']['keep_spools']}, "
          f"critic.provider={cfg['critic']['provider']}; edit to pin "
          "non-default G/N/C, visual or critic settings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
