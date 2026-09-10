"""panels.py — the pool panels every sweep-family order: a sheet AND per-candidate
strips.

INVARIANT: if we render images, we render PANELS. A bare match PNG makes an agent
open two viewers to judge one candidate, and the sweep-family renders exist purely
to be looked at, so a bare render in the spool is wasted capacity.

Both media come from `analysis.viz.rows` — the same builder behind the per-frame
composite strip — with four columns,
SOURCE | RENDER | SILHOUETTE(overlap) | DEPTH, built HERE, in the manager, the
moment the order's renders land. DEPTH is our candidate render in object units;
it does not load observed/Pi3X depth.

  * `candidate_sheet_NNN.png` — several candidates per page. Answers "WHICH of
    these", and is what the manifest indexes for auditable row-to-candidate IDs.
  * `side_by_side_<image>.png` — one candidate across the full width, at
    STRIP_HEIGHT. Answers "is THIS one right", where facing, hinge opening and
    forward-vs-mirrored branding survive at a readable size — the things the brief
    warns "vanish in dense sheets".

Panelling both media here, in the manager, means:

  * an agent never has to remember a post-hoc tool, and never sees a bare render;
  * `sweep_sides` / `candidate_sheet` stay useful unchanged for re-panelling with
    different columns, a different height, or rolling several orders onto one page.

WHY THE MANAGER CAN DO THIS. `pool.manager` runs in the `artscript` env with no
`bpy` — it only ever talks to Blender over pipes. So it can import cv2/numpy and
the analysis viz stack directly, which a Blender-side worker cannot. Panels are
built in the slot thread between the render and the atomic result write, so a
client that sees `results/<id>.json` already has its panels. Cost is ~50 ms of cv2
against a multi-second render, and slots run in parallel.

OPT-OUT: `"raw": true` on the order. That is the agent's explicit "I want the bare
renders" escape, per order, and the only way to get one.
"""

import json
import os
import traceback

from core import visual_budget
from core.sweep_family import SWEEP_VIEWS

# The sweep FAMILY: orders whose whole purpose is candidates an agent judges by
# eye. A shape pass's per-frame match/depth views are excluded — those are
# RECONCILEd into the pass dir where composite.py already writes
# side_by_side_<stem>.png, so panelling them here would duplicate a panel per
# frame per pass.
PANEL_VIEWS = SWEEP_VIEWS


def wants_panels(req):
    """Should this order get panels? The sweep family, unless it asked for raw."""
    if _truthy(_setting(req, "raw")):
        return False
    if _visuals(req) == "none":
        return False
    return bool(panel_views(req))


def panel_views(req):
    """The order's requested views that belong to the sweep family."""
    views = str(req.get("views") or "match").replace(" ", "")
    return [v for v in views.split(",") if v in PANEL_VIEWS]


def _truthy(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _setting(req, key):
    """One panel setting, top level or inside a per-view settings object.

    `visuals` / `waive_visual_budget` are accepted BOTH at the top level (where the
    manager reads them) and inside any PANEL_VIEWS settings object — "sweep",
    "apply", "oapply", "oapply_all", "osweep" (where they
    reach the engine, which decides how many candidates are rendered at all). An
    agent that states it in one place must not get the other place's default."""
    if key in req:
        return req[key]
    for view in PANEL_VIEWS:
        obj = req.get(view)
        if isinstance(obj, dict) and key in obj:
            return obj[key]
    return None


def _visuals(req):
    """The order's visuals level: auto (default) | all | none.

    Normalized by `core.visual_budget`, the one owner of the level vocabulary, so
    the manager and the engines cannot disagree about what an order asked for."""
    return visual_budget.visuals_level(_setting(req, "visuals"))


# How many per-candidate STRIPS one order may produce. The sheet paginates, so it
# scales to a waived `visuals: "all"` order of 100 candidates; a directory of 100
# strips does not — past a dozen the medium is wrong and the sheet is the thing to
# read. Truncation is reported (`strips_available` beside the written list), never
# silent: "12 of 100" is a number an agent can act on, an unexplained 12 is not.
MAX_STRIPS = 12

# The strip's own geometry. Taller than a sheet row (`height=360`) because a strip
# spends the whole width on ONE candidate, which is the reason to open one at all:
# facing and branding that vanish in a dense sheet read here.
STRIP_HEIGHT = 512


def build_for_order(req, out_dir, run_dir="", rows_per_page=3, height=360,
                    bg_mode="black", max_strips=MAX_STRIPS, max_dimension=0,
                    side_by_side_max_dimension=0):
    """Write candidate sheets AND per-candidate strips for one order's render dir.

    Returns {"pages": [...], "manifest": path, "strips": [...]} on success,
    {"skipped": reason} when the order does not want panels or has nothing
    panellable, or {"error": msg} if sheet building failed. NEVER raises: a panel is
    a convenience on top of a render that already succeeded, so a cv2/mask problem
    must not turn a good render into a failed order. The error rides along in the
    result instead.

    TWO MEDIA, ONE DECISION. The paginated sheet answers "which of these candidates"
    — several rows on a page, comparable at a glance. A strip answers "is THIS
    candidate right" — one candidate across the full width, where facing, hinge
    opening and whether branding reads forward survive at a readable size. Agents
    were building the second kind by hand, per order, with a documented `sweep_sides`
    invocation in every window brief, because the pool only ever wrote the first.
    Both come from `analysis.viz.rows.build_row` with the same columns, so they are
    the same panel at two densities rather than two conventions.

    A strip failure does NOT discard the sheets: they are independent evidence and
    the sheets are the ones the manifest indexes. It lands in `strips_error`.
    """
    if not wants_panels(req):
        if _truthy(_setting(req, "raw")):
            return {"skipped": "raw"}
        return {"skipped": "visuals=none" if _visuals(req) == "none"
                else "not-panelled"}
    try:
        # imported lazily: keeps `pool.manager --help` and the unit tests free of
        # a hard cv2 dependency, and keeps import cost off the hot claim path.
        from analysis.viz import candidate_sheet, sweep_sides
        from analysis.viz import rows as rows_lib
        from core.visual_budget import VisualBudgetError
    except ImportError as exc:                       # pragma: no cover - env issue
        return {"error": f"panel deps unavailable: {exc}"}

    try:
        columns = rows_lib.DEFAULT_COLUMNS + ("depth",)
        pages, manifest = candidate_sheet.write_sheets(
            out_dir, run_dir=run_dir, rows_per_page=rows_per_page,
            columns=columns, height=height, bg_mode=bg_mode,
            max_dimension=max_dimension,
            waive_budget=_truthy(_setting(req, "waive_visual_budget")))
        try:
            with open(manifest) as handle:
                sheet_manifest = json.load(handle)
        except (OSError, ValueError):
            sheet_manifest = {}
        originals = sheet_manifest.get("original_pages") or pages
        info = {"pages": [os.path.basename(p) for p in pages],
                "original_pages": [os.path.basename(p) for p in originals],
                "manifest": os.path.basename(manifest),
                "candidate_sheet_max_dimension": max_dimension,
                "side_by_side_max_dimension": side_by_side_max_dimension}
    except VisualBudgetError as exc:
        # A budget refusal is the agent's problem to see, not a crash.
        return {"error": str(exc)}
    except ValueError as exc:
        # no rendered candidates / no mask: normal for an order that scored only.
        return {"skipped": str(exc)}
    except Exception as exc:                         # pragma: no cover - defensive
        return {"error": f"{exc.__class__.__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3)}

    # ...then the strips, from the SAME columns the sheet just drew — including the
    # silhouette overlap, which the hand-run strips never had (STRIP_COLUMNS is the
    # older three-panel default) and which is the column that settles a silhouette
    # trap. `candidate_sheet` is the owner of the sheet's own hand dilation, so the
    # value is read from it rather than restated.
    try:
        strips = sweep_sides.write_sides(
            out_dir, run_dir=run_dir, height=STRIP_HEIGHT, bg_mode=bg_mode,
            columns=columns, limit=max(0, int(max_strips)),
            # one strip per CANDIDATE: the winner is rendered twice (*_best.png and
            # the rank-0 panel) and judging it twice is exactly the redundancy the
            # sheet already drops.
            drop_duplicate_winner=True,
            hand_dilate=candidate_sheet.HAND_DILATE,
            max_dimension=side_by_side_max_dimension,
            return_variants=True)
        if strips and isinstance(strips[0], dict):
            info["strips"] = [
                os.path.basename(item["preview"]) for item in strips]
            info["original_strips"] = [
                os.path.basename(item["original"]) for item in strips]
            info["strip_raster"] = [{
                **item,
                "preview": os.path.basename(item["preview"]),
                "original": os.path.basename(item["original"]),
                **({"manifest": os.path.basename(item["manifest"])}
                   if item.get("manifest") else {}),
            } for item in strips]
        else:
            # Compatibility with older/fake panel backends.
            info["strips"] = [os.path.basename(s) for s in strips]
            info["original_strips"] = list(info["strips"])
        available = len(candidate_sheet.rendered_candidates(
            sweep_sides.find_report(out_dir)[0]))
        if available > len(strips):
            # the cap is a stated fact, not a shortfall to be inferred from a count.
            info["strips_available"] = available
    except Exception as exc:
        # The sheets are already written and are the primary evidence; a strip
        # problem must not retract them.
        info["strips_error"] = f"{exc.__class__.__name__}: {exc}"
    return info
