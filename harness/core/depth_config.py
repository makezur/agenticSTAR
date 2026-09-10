"""depth_config.py — resolve the per-run depth confidence FLOOR (conf_thr) and
the depth flip switches (cost / report).

`run.sh` writes `RUN_DIR/depth_config.json` = {"backend": ..., "conf_thr": ...,
"weight": ..., "cost": ..., "report": ...} at scaffold time (the backend comes
from the capture manifest's `depth` kind; its floor/weight from
harness/depth_config.json, plus the depth toggles). The depth scorers
auto-resolve that per-run file by walking UP from the render/out path, so the
floor follows the capture with no per-tool flag. An explicit --conf-thr always
wins; a missing file falls back to the default. `weight`
is the backend's default sweep depth-supervision weight, resolved the same way
(an explicit --sweep-depth-weight / order depth_weight wins).

Lives in bpy-free core/ so Blender's python (views/sweeps) and the analysis env
(scorers, shape_pass) share ONE walk-up.

THREE optional booleans toggle the depth pipeline (all default TRUE, so an absent
field / missing file behaves exactly as before). They are INDEPENDENT, and none of
them touches the per-frame tracking camera seeds or measure_depth — those read the
pointmap directly and always work:

    "cost":   depth SUPERVISION in the sweep ranking (combined = IoU - weight *
              depth_canon). false -> the sweep forces depth_weight 0 (pure IoU).
    "report": the depth SCORER / panels (depth_<stem>.json, residual
              images). false -> the shape pass skips depth
              scoring, and every downstream reporter that reads those files goes
              quiet on its own. The depth RENDER is NOT affected — see below.
    "render": our rendered depth artifacts: the depth VIEW (depth_<stem>.npy)
              and same-render depth for panelled sweep/apply candidates.
              Independent of "report" because neither needs GT: with it, a
              monocular run still gets object-units DEPTH panels and sheets
              (analysis.viz.depth_units). false -> those renders are omitted.
              This is the knob to reach for when a run does not want to pay for
              depth at all; "report": false alone does not stop the render.
"""
import json
import os

DEFAULT_CONF_THR = 0.1
DEFAULT_DEPTH_WEIGHT = 0.1


def _find_config(start_path):
    """The nearest depth_config.json dict walking UP from `start_path` (a render
    depth / out path), or None if none is found / unreadable. One walk-up, shared
    by conf_thr and the cost/report flags so they resolve identically."""
    d = os.path.dirname(os.path.abspath(start_path))
    prev = None
    while d and d != prev:
        cand = os.path.join(d, "depth_config.json")
        if os.path.isfile(cand):
            try:
                with open(cand) as f:
                    cfg = json.load(f)
                if isinstance(cfg, dict):
                    return cfg
            except (ValueError, OSError):
                pass
        prev, d = d, os.path.dirname(d)
    return None


def resolve_conf_thr(start_path, override=None, default=DEFAULT_CONF_THR):
    """Confidence floor for depth scoring.

    `override` (an explicit --conf-thr) wins when not None. Otherwise walk up from
    `start_path` (a render depth / out path) looking for `depth_config.json` and
    return its `conf_thr`. Fall back to `default` when no config is found."""
    if override is not None:
        return float(override)
    cfg = _find_config(start_path)
    if cfg is not None:
        ct = cfg.get("conf_thr")
        if ct is not None:
            return float(ct)
    return float(default)


def resolve_depth_weight(start_path, override=None,
                         default=DEFAULT_DEPTH_WEIGHT):
    """Sweep depth-supervision weight for the run that owns `start_path`.

    `override` (an explicit --sweep-depth-weight / order depth_weight) wins when
    not None. Otherwise walk up looking for `depth_config.json` and return its
    `weight` (the backend's default, copied from harness/depth_config.json at
    scaffold — a noisy estimator can supervise more gently than exact GT depth
    without every order restating it). Fall back to `default` when neither
    exists (old runs keep the historic 0.1)."""
    if override is not None:
        return float(override)
    cfg = _find_config(start_path)
    if cfg is not None:
        w = cfg.get("weight")
        if w is not None:
            return float(w)
    return float(default)


def _flag(start_path, key, default=True):
    """A boolean toggle from the nearest depth_config.json (default when the file /
    key is absent). A None value is treated as absent -> default."""
    cfg = _find_config(start_path)
    if cfg is None:
        return default
    v = cfg.get(key)
    return default if v is None else bool(v)


def depth_cost_enabled(start_path, default=True):
    """Whether observed-depth SUPERVISION (the sweep's depth penalty) is on for the
    run that owns `start_path`. False only when depth_config.json says
    "cost": false."""
    return _flag(start_path, "cost", default)


def depth_report_enabled(start_path, default=True):
    """Whether the observed-depth SCORER / panels are on for the run that owns
    `start_path`. False only when depth_config.json says "report": false."""
    return _flag(start_path, "report", default)


def depth_render_enabled(start_path, default=True):
    """Whether our depth artifacts are rendered for the owning run.

    False only when depth_config.json says "render": false. Independent of
    "report": they feed GT-free object-units visuals as well as the scorer, so
    scoring being off is not a reason to stop rendering. An absent key means
    TRUE."""
    return _flag(start_path, "render", default)
