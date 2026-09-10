"""views — one module per render view; VIEWS maps a --views name to its entry.

Each view module exposes `render_view(ctx)` taking a rig.render.Context. The
dispatcher (render_wrapper.py) looks up requested view names in VIEWS and calls
them in order. To add a view: drop in a module with render_view(ctx), register
it here, and add a <name>.md doc next to it.

All FIVE candidate-report verbs — `sweep`/`apply` and `osweep`/`oapply`/`oapply_all`
— live in the `views.sweeps` subpackage together with the machinery they share, so a
verb sits beside the engine it drives. This module registers them; it does not house
them. Everything left here is a plain one-frame renderer.

`VIEWS` is built LAZILY (PEP 562 `__getattr__`) so importing the `views` package
itself is `bpy`-free: the view modules — which `import bpy` — load only when
`VIEWS` is first accessed (inside Blender, from render_wrapper.py). This lets the
sweeps' pure, `bpy`-free submodules (`dof_space`, `metrics`, `planner`) be
imported and unit-tested in the analysis env without pulling in Blender.
"""


# The five verbs that write a judgeable candidate report live in ONE list, shared
# with pool/panels.py's sheet decision, pool/serve.py's request whitelist, and
# analysis/viz/sweep_sides.py's report lookup.
# Re-exported here (not redefined) because a view NAME is this package's vocabulary:
# every member is required to be really registered in VIEWS.
from core.sweep_family import SWEEP_VIEWS      # noqa: F401  (re-export)

# A CALLING CONVENTION, not a fitting one: these verbs are invoked ONCE with the
# resolved FrameSpec list (ctx.frames) and write ONE report, instead of being called
# per frame like every other view. The list may hold a single frame — `--frames` /
# an order's `"frames"` key scopes it, and `_prepare` only bails on ZERO
# (`shared_engine.py`: "no frames resolved").
#
# Only `oapply_all` NEEDS several: its ranking key is the mean across frames.
# `osweep`/`oapply` are PER_FRAME (N independent argmaxes — see `config.py`'s `mode`)
# and take the list purely so N frames share one Blender process, one candidate grid,
# and one report.
#
# Kept here (bpy-free) as the single source shared by render_wrapper's one-shot
# dispatch and pool/serve.py's resident-worker dispatch.
MULTI_FRAME_VIEWS = ("osweep", "oapply", "oapply_all")

# Views that were RENAMED or removed, with the message that teaches the replacement.
# A stale prompt or saved order naming one must fail loudly and usefully: silently
# treating it as unknown would print "known: match, turntable, ..." and leave the
# agent to guess which of two verbs it now wants.
RETIRED_VIEWS = {
    "osweep_all": (
        "there is no 'osweep_all' — a SHARED rotation searched by MEAN IoU can be "
        "mediocre in every frame (the same rz:+30 scores 0.80 in one view and 0.54 "
        "in another). Use 'oapply_all' with --oapply 'flips' or --opreset 'z:full' "
        "for ONE shared rotation chosen from a panel you can look at, or 'osweep' "
        "for an independent search per frame."),
}


def _build_views():
    from views import crop, depth, ids, match, mechanism, turntable, visibility
    from views.sweeps import apply, oapply, oapply_all, osweep, sweep
    return {
        "match": match.render_view,
        "turntable": turntable.render_view,
        "mechanism": mechanism.render_view,
        "ids": ids.render_view,
        "crop": crop.render_view,
        "visibility": visibility.render_view,
        "sweep": sweep.render_view,
        "apply": apply.render_view,
        "osweep": osweep.render_view,
        "oapply": oapply.render_view,
        "oapply_all": oapply_all.render_view,
        "depth": depth.render_view,
    }


def unknown_view_error(name, known):
    """The message for a view name that isn't registered — teaching the replacement
    when the name is a RETIRED one, else listing what is known."""
    retired = RETIRED_VIEWS.get(str(name))
    if retired:
        return f"unknown view {name!r}: {retired}"
    return f"unknown view {name!r} (known: {', '.join(known)})"


def __getattr__(name):
    """Lazily construct VIEWS on first access (keeps package import bpy-free)."""
    if name == "VIEWS":
        views = _build_views()
        globals()["VIEWS"] = views     # cache so later access skips __getattr__
        return views
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
