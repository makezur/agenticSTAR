"""metrics.py — scoring, landscape stats, and the sweep report writer.

"Calculate the metrics / stats" lives in its own module, distinct from the
dispatcher that renders candidates. This is that module: the
combined-score `score_of`, the per-DOF landscape/plateau statistics, the 2-D score
FIELD for a dense two-DOF grid, the depth verdict, and the single unified
legend/manifest writer that serves every space (pose-only, joint-only, coupled) by
emitting a pose paste-block and/or a joints paste-block according to which DOF KINDS
were swept.

PURITY
------
Pure numpy + stdlib + json. No `bpy`. Candidate records are plain dicts (built by
`engine`), so all of this is unit-testable in the analysis env. Rotation-format
helpers accept a placement dict carrying a scalar-first `quaternion` (the form
`dof_space.candidate_from_pose` writes); a legacy euler placement is converted via a
tiny local quaternion-from-matrix (no `mathutils`).
"""

import json
import math
import os

import numpy as np

from core import state_json, visual_budget
from rig import lie
from views.sweeps.lib import dof_space


# --------------------------------------------------------------------------- #
# combined score
# --------------------------------------------------------------------------- #
def score_of(rec, gate_field, depth_weight=0.0):
    """Ranking score: the gate IoU minus a depth penalty when supervision is on.

        combined = gate_iou - depth_weight * min(depth_canon, 1.0)

    IoU is depth-blind: a nearer+smaller (or farther+bigger) pose projects the
    same silhouette, so a family of depth-wrong poses ties on IoU. The penalty
    breaks the tie with the candidate's RAW depth error in canonical units
    (fraction of the object's longest dimension). depth_weight <= 0 (or no --tracking)
    -> the plain gate IoU. A candidate with no scorable depth
    pixels while supervision is on (depth_canon None) takes the FULL penalty. A
    None visible-IoU ranks worst of all. A behind-camera candidate scores its
    iou_raw 0 (never a real hit)."""
    if gate_field == "iou_visible":
        v = rec.get("iou_visible")
        iou = v if v is not None else -1.0
    else:
        iou = rec.get("iou_raw", 0.0)
    if depth_weight <= 0.0:
        return iou
    canon = rec.get("depth_canon")
    penalty = 1.0 if canon is None else min(float(canon), 1.0)
    return iou - depth_weight * penalty


# --------------------------------------------------------------------------- #
# pivot serialization — shared by BOTH report families
# --------------------------------------------------------------------------- #
def pivot_out(pivot):
    """A pivot record (core.centre_calculation.pivot_record) rounded to the wire
    precision, or None.

    Rounded with `state_json.round_component` — the SAME 6dp as the poses and joint
    states it sits beside (conventions/state_json.md §5). A pivot printed to a
    different number of places than the poses it explains is the exact drift that
    module exists to prevent. Lives here (bpy-free) because both the camera-frame
    report (engine) and the canonical per-frame report (shared_report) write it."""
    if not pivot:
        return None
    return {
        "centre": [state_json.round_component(v) for v in pivot["centre"]],
        "depth_z": state_json.round_component(pivot["depth_z"]),
        "canonical_centre": [state_json.round_component(v)
                             for v in pivot["canonical_centre"]],
        "joint_states": state_json.round_joints(pivot["joint_states"]),
        "source": pivot["source"],
    }


# --------------------------------------------------------------------------- #
# search PASSES — the lattice each pass actually scored
# --------------------------------------------------------------------------- #
# A sweep is (1 + refine) passes over DIFFERENT lattices. `passes` records one
# entry per pass, carrying the lattice, the count, and (for a refine pass) the
# center it descended from and the shrink that produced it. Together with each
# ranked entry's `pass` index, every candidate on disk is placeable in the search
# that produced it.
def pass_record(index, ranges, n, kind="coarse", center=None, shrink=None,
                slice_of=None, so3=None):
    """One search pass -> its manifest entry.

    `ranges` is the lattice THIS pass scored ([lo, hi, steps] per DOF), not the
    order's original band — that is the whole point. `n` is what was actually
    scored, so a pass cut short by the timeout is visible rather than inferred.
    `center`/`shrink` are set for a refine pass (which candidate it re-centered on,
    and by how much the band shrank). `slice_of` is [index, total] for a sharded
    coarse pass, which is why that pass has fewer candidates than its lattice.
    `so3` is {n, seed} for a kind="so3" coarse pass: its candidates lie on NO
    lattice (`ranges` is honestly empty), so the record names the GENERATOR
    instead — with it the exact grid is re-drawable even if the baked default
    seed (planner_canon.SO3_SEED) later changes."""
    out = {"pass": int(index), "kind": str(kind), "n": int(n),
           "ranges": {k: list(v) for k, v in (ranges or {}).items()}}
    if center is not None:
        out["center"] = center
    if shrink is not None:
        out["shrink"] = round(float(shrink), 6)
    if slice_of is not None:
        out["slice"] = [int(v) for v in slice_of]
    if so3 is not None:
        out["so3"] = {"n": int(so3["n"]), "seed": int(so3["seed"])}
    return out


def pass_center(space, x):
    """The DOF vector a refine pass re-centered on, as {dof: value}.

    Named by DOF rather than positional: the vector's order is an internal detail of
    `Space`, but the report is read by agents and viewers."""
    return {d: round(float(v), 6) for d, v in zip(space.dof_names, x)}


def passes_summary(passes):
    """The one-line `.txt` account of the search: how many passes, over what.

    Prints each pass's candidate count and the SPAN of its band per DOF, so a reader
    sees the windows nesting without having to open the json."""
    if not passes or len(passes) < 2:
        return ""
    parts = []
    lattice_only = all(p.get("kind") in ("coarse", "refine") for p in passes)
    for p in passes:
        spans = []
        for d, rng in sorted((p.get("ranges") or {}).items()):
            try:
                lo, hi = float(rng[0]), float(rng[1])
            except (TypeError, ValueError, IndexError):
                continue
            steps = None
            if len(rng) >= 3:
                try:
                    steps = int(rng[2])
                except (TypeError, ValueError):
                    pass
            spans.append(f"{d} {lo:+.3g}..{hi:+.3g}"
                         + (f"/{steps}" if steps is not None else ""))
        parts.append(f"pass {p['pass']} ({p.get('kind', '?')}, {p.get('n', 0)} cand"
                     + (f": {'; '.join(spans)}" if spans else "") + ")")
    if lattice_only:
        return ("SEARCH PASSES — each pass scored its OWN lattice; a winner from a "
                "later pass sits off the coarse grid BY DESIGN. "
                + " -> ".join(parts)
                + ". Full bands + centers in the json's `passes`; each ranked "
                  "entry's `pass` says which lattice it came from.")
    return ("SEARCH STAGES — " + " -> ".join(parts)
            + ". Bounds and optimizer metadata are in the json; each ranked "
              "entry's `pass` identifies its stage.")


# --------------------------------------------------------------------------- #
# panel selection — the best candidate per ORDERED grid cell
# --------------------------------------------------------------------------- #
def bin_step(rng):
    """The ordered grid's cell width for one DOF, or None if there is no lattice.

    `rng` is the report's [lo, hi, steps] triple. None means the DOF cannot be
    binned — either it carries no range at all, or it is PINNED (lo == hi, or a
    single step), in which case it holds the same value for every candidate and so
    cannot distinguish them anyway."""
    try:
        lo, hi, steps = float(rng[0]), float(rng[1]), int(rng[2])
    except (TypeError, ValueError, IndexError):
        return None
    if steps < 2 or hi == lo:
        return None
    return (hi - lo) / (steps - 1)


def binnable_dofs(dof_kinds, ranges):
    """The swept DOFs that carry a usable lattice: [(name, kind, lo, step), ...].

    A DOF with no range, or a PINNED one (`bin_step` None), is skipped — it holds
    the same value for every candidate and cannot distinguish them. An EMPTY result
    means the order has no agent-supplied scale at all, which is the honest signal
    that binning does not apply (see `bin_topk`)."""
    ranges = ranges or {}
    out = []
    for name, kind in dof_kinds.items():
        step = bin_step(ranges.get(name))
        if step is not None:
            out.append((name, kind, float(ranges[name][0]), step))
    return out


def cell_key(rec, binnable):
    """The record's cell in the ordered grid, or None if no DOF is measurable.

    `round((v - lo) / step)` per binnable DOF. NOT clamped to the ordered band:
    refine re-centers, so a refined value can leave the band and must keep its
    out-of-range index — clamping would silently merge an out-of-band winner into
    the edge cell. A record whose every DOF is unmeasurable (a behind-camera pose,
    missing joints) has no cell at all and returns None, so it can never occupy one.
    """
    key = []
    measurable = False
    for name, kind, lo, step in binnable:
        v = value_of(rec, name, kind)
        if v is None:
            key.append(None)
            continue
        measurable = True
        key.append(int(math.floor((v - lo) / step + 0.5)))
    return tuple(key) if measurable else None


def count_cells(ranked, dof_kinds, ranges):
    """How many DISTINCT ordered cells the whole scored set occupies, or None when
    the order has no lattice. Independent of any panel quota — this is what tells an
    agent whether a short sheet means a degenerate grid or a small `dump_topk`."""
    binnable = binnable_dofs(dof_kinds, ranges)
    if not binnable:
        return None
    return len({k for k in (cell_key(r, binnable) for r in ranked)
                if k is not None})


def bin_topk(ranked, dof_kinds, ranges, count):
    """Keep the BEST candidate in each distinct cell of the grid the agent ordered.

    Walk candidates in plain score order and accept one only if its cell is still
    empty, where the cell is `round((v - lo) / step)` per swept DOF from the
    report's own `ranges`. Stop at `count`. Nothing is ever accepted out of score
    order, so the k-th panel is simply the best candidate in the k-th distinct
    ordered cell — junk can never be injected to fill a quota.

    The scale comes from the grid the agent ASKED for, so no tolerance constant is
    needed, and it is stable under --sweep-refine (which changes the explored span
    but never the ordered one). Whether the triple came from an explicit
    --sweep-ranges or from the empty-spec preset defaults makes no difference: by
    here it is just a lattice. Refined values landing between coarse cells snap to
    the nearest (see `cell_key` for why out-of-band indices are kept as-is).

    Bins on every DOF that HAS a usable step and ignores the rest, so a sweep that
    pinned one DOF still bins on the others. With NO binnable DOF (an explicit
    candidate set, or a report whose `ranges` is empty) there is no agent-supplied
    scale, so ordinary score order is preserved rather than inventing one.

    A candidate with NO measurable DOF (behind-camera pose, missing joints) occupies
    no cell and is only appended once every real cell is exhausted, as a last-resort
    filler — it can never take a slot from a real cell.

    HONEST SHORTFALL: if fewer than `count` cells are occupied, returns FEWER --
    possibly ONE. That is the truth about the sweep (every candidate sits in one
    ordered cell, e.g. after refine converged) and is better evidence than filler
    panels from elsewhere in the band. Records `_rank` for report diagnostics.
    """
    wanted = max(1, int(count))
    if not ranked:
        return []
    for rank, rec in enumerate(ranked):
        rec["_rank"] = rank

    binnable = binnable_dofs(dof_kinds, ranges)
    if not binnable:
        return list(ranked[:wanted])

    selected, seen, cellless = [], set(), []
    for rec in ranked:
        key = cell_key(rec, binnable)
        if key is None:
            cellless.append(rec)
            continue
        if key in seen:
            continue
        seen.add(key)
        selected.append(rec)
        if len(selected) == wanted:
            return selected
    # Every occupied cell is represented and we are still short. Only now may a
    # cell-less candidate fill a slot (score-ordered), because it is better than
    # nothing but is never evidence about a cell.
    selected.extend(cellless[:wanted - len(selected)])
    return selected


# How the panelled candidates were chosen — reported verbatim as
# meta.panels.selection, so it must never claim a mechanism that did not run.
BY_CELL = "best-per-ordered-cell"
BY_SCORE = "score-order-no-lattice"
EVERY = "every-scored-candidate"
BY_RESTART = "best-per-de-restart"


def panel_plan(ranked, dof_kinds, ranges, dump_topk, visuals="auto",
               waived=False, what="sweep", set_limit=0, labelled=False,
               multiplier=1, defer_budget=False):
    """Decide WHICH candidates become images.

    Returns (retained, dump_n, selection, cells_occupied).

    Three levels. The invariant behind them: if we render images at all, we render
    PANELS, and an order never lands as a bare render an agent has to pair against a
    source by hand (pool/panels.py builds the sheet; pool/README.md § panels).

      auto  honour `dump_topk`, best-per-ordered-cell (bin_topk). dump_topk 0 still
            panels ONE candidate -- the winner -- because an order must never land
            as a bare render an agent has to go pair up by hand;
      all   EVERY scored candidate, in score order with duplicates kept (binning
            would collapse each cell to its best, the opposite of "all"). Checked
            against the visual budget: "all" is a property of the GRID, not of the
            agent's intent, so a 400-cell sweep is refused unless `waived`;
      none  no candidate images at all -- the scores-only escape. <stem>_best.png is
            still the view's own output and is written either way.

    `selection` states which mechanism actually ran: BY_CELL when at least one swept
    DOF carried a lattice, BY_SCORE when none did (an explicit --sweep-candidates
    set, where there is no ordered grid to bin against), EVERY for visuals=all.
    `cells_occupied` is the distinct-cell count over ALL of `ranked` -- NOT the
    number retained -- or None when there is no lattice.

    `set_limit`/`labelled` serve the canon views (osweep/oapply): a small NAMED
    candidate set panels every member whatever `dump_topk` says, because flip twins
    tie on IoU and the whole point is that the agent looks. Grid mode there bins
    like any other sweep. `multiplier` is how many IMAGES one retained candidate
    becomes — 1 for the single-frame sweep family, len(frames) for the canon views,
    where the panel count multiplies across frames.

    `defer_budget` skips the visuals=all budget check so the CALLER can make it.
    PER-FRAME osweep/oapply plans once per frame (the frame is the unit, so
    `multiplier=1`), and 50 candidates x 10 frames would sail through as ten
    independent 50s while actually meaning 500 images. The per-frame caller sums the
    retained counts and makes ONE check for the whole order — one honest message
    about its real size. Nobody else should pass this: an unchecked "all" is exactly
    the runaway the budget exists to catch.

    Pure so the decision is testable without Blender; the engine only renders it.
    """
    visuals = visual_budget.visuals_level(visuals)
    cells = count_cells(ranked, dof_kinds, ranges) if ranked else None
    mult = max(1, int(multiplier or 1))
    if visuals == "all":
        if not defer_budget:
            visual_budget.check_budget(
                len(ranked) * mult, waived=bool(waived),
                what=f"{what} visuals=all",
                detail=(f"{len(ranked)} candidates scored" if mult == 1 else
                        f"{len(ranked)} candidates x {mult} frames"))
        dump_n = len(ranked)
        for rank, rec in enumerate(ranked):
            rec["_rank"] = rank
        return list(ranked), dump_n, EVERY, cells
    if visuals == "none":
        return [], 0, (BY_CELL if cells is not None else BY_SCORE), cells
    # A named set small enough to show whole: every member is a panel.
    if labelled and set_limit and len(ranked) <= int(set_limit):
        for rank, rec in enumerate(ranked):
            rec["_rank"] = rank
        return list(ranked), len(ranked), EVERY, cells
    dump_n = max(0, int(dump_topk or 0))
    # invariant: the winner is always panelled, so an order never lands as a bare
    # render an agent has to go pair up by hand.
    retained = bin_topk(ranked, dof_kinds, ranges, dump_n or 1)
    selection = BY_CELL if cells is not None else BY_SCORE
    return retained, dump_n, selection, cells


# The near-optimal PLATEAU of a swept DOF is the band of values scoring within
# PLATEAU_TOL of the peak marginal score — candidates this close are practically
# TIED to the silhouette. Its WIDTH (in the DOF's native units) is how poorly the
# optimum is localized; a wide plateau means a whole band ties, so the "best" is
# nearly arbitrary (the sunscreen-lid trap). PLATEAU_TOL only LABELS the width — it
# is not a pass/fail gate; the agent judges the landscape against the photo.
PLATEAU_TOL = 0.01


# --------------------------------------------------------------------------- #
# per-DOF value extractors (unified — no separate pose/joint functions)
# --------------------------------------------------------------------------- #
def value_of(rec, name, kind):
    """The candidate's value for DOF `name` of the given kind.

    POSE  -> rec["order"][name] (the camera-frame increment the grid applied);
             None for a behind-camera candidate (iou 0, not a real score — it
             would fake a swing at the pose extremes) or a missing order map.
    JOINT -> rec["joints"][name] (the absolute state).
    """
    if kind == dof_space.POSE:
        if rec.get("behind_camera"):
            return None
        order = rec.get("order")
        if not order or name not in order:
            return None
        try:
            return float(order[name])
        except (TypeError, ValueError):
            return None
    j = rec.get("joints") or {}
    return None if name not in j else float(j[name])


def dof_landscape_stats(scored, dof_kinds, gate_field, depth_weight=0.0):
    """Per swept DOF: the RAW SHAPE of the combined-score landscape across the
    explored values — descriptive stats, NOT an ok/low verdict.

    `dof_kinds` is {name: POSE|JOINT}. For each DOF we keep, per explored value,
    its BEST combined score across all OTHER swept DOFs (the marginal profile:
    "at the best of everything else, how good is this value?"). A small
    dyn_range/score_std (metric barely moves) or a large plateau_width (a wide band
    ties near the top) both mean the silhouette does not pin the DOF.

    Per DOF returns span / n_values / best_value / dyn_range / score_std /
    plateau_width / at_boundary / profile — see the field comments below. The field
    names and their meanings are FROZEN: they are part of sweep.json's contract and
    existing readers (analysis/viz, the pool's sheets) depend on them.
    """
    out = {}
    for name, kind in dof_kinds.items():
        by_val = {}
        for rec in scored:
            v = value_of(rec, name, kind)
            if v is None:
                continue
            v = round(float(v), 6)
            s = score_of(rec, gate_field, depth_weight)
            if v not in by_val or s > by_val[v]:
                by_val[v] = s
        info = {"span": None, "n_values": len(by_val), "best_value": None,
                "dyn_range": 0.0, "score_std": 0.0, "plateau_width": None,
                "at_boundary": None, "profile": []}
        if len(by_val) < 2:
            if by_val:
                k = next(iter(by_val))
                info["best_value"] = round(k, 6)
                info["profile"] = [[round(k, 6), round(by_val[k], 6)]]
            out[name] = info
            continue
        vals = sorted(by_val)
        lo_e, hi_e = vals[0], vals[-1]
        best_val = max(by_val, key=lambda v: by_val[v])
        best_score = by_val[best_val]
        dyn_range = best_score - min(by_val.values())
        score_std = float(np.std(list(by_val.values())))
        near = [v for v in vals if by_val[v] >= best_score - PLATEAU_TOL]
        plateau_width = (max(near) - min(near)) if near else 0.0
        at_boundary = (best_val <= lo_e + 1e-6) or (best_val >= hi_e - 1e-6)
        info.update({
            "span": [round(lo_e, 6), round(hi_e, 6)],
            "best_value": round(best_val, 6),
            "dyn_range": round(dyn_range, 4),
            "score_std": round(score_std, 4),
            "plateau_width": round(plateau_width, 6),
            "at_boundary": at_boundary,
            "profile": [[round(v, 6), round(by_val[v], 6)] for v in vals],
        })
        out[name] = info
    return out


# --------------------------------------------------------------------------- #
# 2-D score field — the dense two-DOF grid map
# --------------------------------------------------------------------------- #
def score_field(scored, dof_a, dof_b, kinds, gate_field, depth_weight=0.0):
    """The dense combined-score field over one or two grid-swept DOFs, or None.

    `dof_b` None gives the ONE-DOF field (the marginal) in the same shape as the
    pair, so every reader has one code path:
      {"dofs":[dof_a] | [dof_a,dof_b], "a_values":[...], "b_values":[...],
       "scores":[[combined | null, ...], ...]}    rows indexed by a, cols by b
    A one-DOF field has a single column, and `b_values` is [None].

    Each cell keeps the BEST combined score of every candidate landing on it: with
    other DOFs also swept, many candidates share a cell, and the max is what makes
    the projection meaningful ("at the best of everything else, how good is this?").
    Cells never scored are null. None if an axis has fewer than 2 distinct values —
    a degenerate field says nothing and would draw as a single stripe.

    A per-axis marginal alone cannot show COUPLING: it collapses a diagonal valley
    to two flat axes, which is exactly why the pair field exists. Both come from
    here so they cannot drift apart.
    """
    ka = kinds[dof_a]
    kb = kinds[dof_b] if dof_b else None
    cell = {}
    a_set, b_set = set(), set()
    for rec in scored:
        va = value_of(rec, dof_a, ka)
        if va is None:
            continue
        if dof_b:
            vb = value_of(rec, dof_b, kb)
            if vb is None:
                continue
            vb = round(vb, 6)
        else:
            vb = None
        va = round(va, 6)
        a_set.add(va)
        b_set.add(vb)
        s = score_of(rec, gate_field, depth_weight)
        key = (va, vb)
        if key not in cell or s > cell[key]:
            cell[key] = s
    a_values = sorted(a_set)
    b_values = sorted(b_set, key=lambda v: (v is not None, v))
    if len(a_values) < 2 or (dof_b and len(b_values) < 2):
        return None
    scores = [[(round(cell[(a, b)], 6) if (a, b) in cell else None)
               for b in b_values] for a in a_values]
    return {"dofs": ([dof_a, dof_b] if dof_b else [dof_a]),
            "a_values": a_values, "b_values": b_values, "scores": scores}


# How many pair heat-grids sweep.txt draws for a wide grid. The .txt is read by an
# agent, so this is an attention budget, not a size one: all the pairs are in the
# report either way, and the line under them says how many were left there.
MAX_TXT_FIELDS = 3


def _field_score_range(field):
    """Spread of a field's real scores — how much the pair actually moves the metric.

    Used to order the heat-grids in sweep.txt: a pair whose scores barely change says
    little, a wide range is where the silhouette is discriminating.
    """
    vals = [v for row in (field.get("scores") or []) for v in row if v is not None]
    return (max(vals) - min(vals)) if len(vals) > 1 else 0.0


def score_fields(scored, dof_names, kinds, gate_field, depth_weight=0.0,
                 max_pairs=28):
    """EVERY dense score field this sweep can project, in ONE shape and ONE key.

        {"yaw|pitch": <field>,     # a DOF PAIR: the 2-D matrix
         "yaw":       <field>,     # ONE DOF: the marginal, same shape, 1 column
         ...}

    Both are the same object (`dofs`/`a_values`/`b_values`/`scores`) so a reader has
    one code path, not a pair-vs-marginal branch. A one-axis field is the marginal:
    the best score at each explored value across every other swept DOF, which is the
    1-D case of the exact max-over-the-rest rule the pair field uses.

    This replaced two keys and two shapes — a single `dof_field_2d` written only for
    a grid with exactly 2 lattice DOFs, plus the marginal buried in
    `dof_stats[dof].profile` as `[value, score]` pairs. Anything wider than 2 DOFs
    got neither, so a 4-DOF sweep that scored 12,636 candidates kept no dense score
    at all and a viewer could only draw its --sweep-topk listing.

    `max_pairs` bounds the report: 8 lattice DOFs is 28 pairs. Beyond it the DOFs
    are taken in order and what was dropped is returned separately, so a truncation
    can never read as "nothing was scored". Returns ({}, []) when nothing is dense.
    """
    names = [d for d in dof_names if d in kinds]
    out, omitted = {}, []
    for name in names:
        field = score_field(scored, name, None, kinds, gate_field, depth_weight)
        if field is not None:
            out[name] = field
    n_pairs = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if n_pairs >= max_pairs:
                omitted.append(f"{a}|{b}")
                continue
            field = score_field(scored, a, b, kinds, gate_field, depth_weight)
            if field is not None:
                out[f"{a}|{b}"] = field
                n_pairs += 1
    return out, omitted


def _field_ascii(field):
    """A compact ASCII heat-grid of one PAIR `score_fields` entry, for the .txt
    legend. Rows =
    dof_a (top→bottom = low→high), cols = dof_b (left→right = low→high); each cell
    a single glyph binned by score within the field's own [min,max]. The '*' marks
    the peak cell. Empty (null) cells render as space."""
    a_vals, b_vals = field["a_values"], field["b_values"]
    rows = field["scores"]
    flat = [s for row in rows for s in row if s is not None]
    if not flat:
        return None
    lo, hi = min(flat), max(flat)
    rng = (hi - lo) or 1.0
    ramp = " .:-=+o#%@"       # '*' is reserved for the peak marker (not in the ramp)
    peak = max(((i, j) for i, row in enumerate(rows)
                for j, s in enumerate(row) if s is not None),
               key=lambda ij: rows[ij[0]][ij[1]])
    da, db = field["dofs"]
    lines = [f"2-D SCORE FIELD  {da} (rows, {a_vals[0]:+.3g}..{a_vals[-1]:+.3g}) x "
             f"{db} (cols, {b_vals[0]:+.3g}..{b_vals[-1]:+.3g}); "
             f"glyph = combined in [{lo:.3f}, {hi:.3f}], '@'=high, '*'=peak:"]
    for i, row in enumerate(rows):
        cells = []
        for j, s in enumerate(row):
            if s is None:
                cells.append(" ")
            elif (i, j) == peak:
                cells.append("*")
            else:
                k = int((s - lo) / rng * (len(ramp) - 1))
                cells.append(ramp[min(max(k, 0), len(ramp) - 1)])
        lines.append(f"  {a_vals[i]:+7.3g} |{''.join(cells)}")
    peak_a, peak_b = a_vals[peak[0]], b_vals[peak[1]]
    lines.append(f"  peak at {da}={peak_a:+.4g}, {db}={peak_b:+.4g} "
                 f"(combined {rows[peak[0]][peak[1]]:.4f}).")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# landscape-stats legend lines
# --------------------------------------------------------------------------- #
def dof_stats_lines(dof_stats_map, kinds):
    """Compact grid/candidate landscape-stat lines for the .txt legend (or None).

    One line per swept DOF (there is no ok/low verdict — the reader judges the
    numbers). `kinds` labels each DOF pose/joint. DE reports deliberately omit
    these statistics because continuous samples do not form a per-axis landscape.
    The full per-value `profile` stays in grid/candidate JSON."""
    stats = {n: c for n, c in (dof_stats_map or {}).items() if c}
    if not stats:
        return None
    head = ("DOF LANDSCAPE (how well the silhouette pins each swept DOF — small "
            "dyn_range/score_std or a wide plateau_width => NOT pinned; a "
            "near-symmetric object / a small part that hides can tie on IoU):")
    cols = (f"{'dof':>14}  {'kind':>5}  {'dyn_range':>9}  {'score_std':>9}  "
            f"{'plateau_w':>9}  {'best':>10}  boundary")
    lines = [head, cols]
    for n, c in stats.items():
        kind = "pose" if kinds.get(n) == dof_space.POSE else "joint"
        dr, sd = c.get("dyn_range"), c.get("score_std")
        pw, bv = c.get("plateau_width"), c.get("best_value")
        row = (f"{n:>14}  {kind:>5}  "
               f"{('n/a' if dr is None else f'{dr:.4f}'):>9}  "
               f"{('n/a' if sd is None else f'{sd:.4f}'):>9}  "
               f"{('n/a' if pw is None else f'{pw:.4f}'):>9}  "
               f"{('n/a' if bv is None else f'{bv:.4f}'):>10}  "
               f"{'YES' if c.get('at_boundary') else 'no':>8}")
        lines.append(row)
    tail = ("For a NOT-pinned DOF (barely-moving or a wide plateau) do NOT paste its "
            "swept value as ground truth — set it from the photo and visual "
            "directional read and use the sweep only to refine within the range the "
            "photo supports.")
    lines.append(tail)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# rotation / paste helpers (pure — no mathutils)
# --------------------------------------------------------------------------- #
def _quat_of(p):
    """Scalar-first (w,x,y,z) quaternion of a placement.

    Every placement reaching here is built by `dof_space.placement_dict` or the shared
    engine's `_placement`, and both always write `quaternion` — so a missing key is a
    broken producer, not an identity pose. It raises rather than defaulting: silently
    substituting identity would publish a report whose `pose` is a rotation nobody
    scored (conventions/state_json.md §1 — the quaternion is authoritative)."""
    q = p.get("quaternion")
    if q is None:
        raise KeyError(
            "placement has no 'quaternion'; a pose's rotation is authoritative and "
            "cannot be defaulted to identity (see conventions/state_json.md §1)")
    return [float(v) for v in q]


def pose_out(p):
    """JSON form of a candidate placement, per conventions/state_json.md: the
    shared wire payload plus a read-only `scale` (informational — it always echoes
    the shared top-level SCALE, since the pose order is rigid and no verb may
    search scale). A report entry is a self-contained placement, so unlike a
    fragment pose it does carry scale."""
    return state_json.round_pose({
        "quaternion": _quat_of(p),
        "translation": p["translation"],
        "scale": lie.scalar_scale(p["scale"]),
    })


def pose_pystr(p):
    """Exact scene.py FRAMES[frame]['pose'] block for the best pose (scalar-first
    quaternion; NO scale — scale is the shared top-level SCALE). Printed at the
    report's own precision (state_json.md §5) so pasting it reproduces the pose
    that was scored."""
    q = ", ".join(state_json.paste_literal(v) for v in _quat_of(p))
    t = ", ".join(state_json.paste_literal(v) for v in p["translation"])
    return ('"pose": {\n'
            f"    \"quaternion\": ({q}),\n"
            f"    \"translation\": ({t}),\n"
            "}")


def _joints_pystr(states):
    """Exact scene.py FRAMES joints block for the best articulation."""
    lines = ['"joints": {']
    for n, v in (states or {}).items():
        lines.append(f'    "{n}": {state_json.paste_literal(v)},')
    lines.append("}")
    return "\n".join(lines)


def _order_str(order, pose_dofs):
    """Compact one-line view of a candidate's swept pose orders (skips those held
    at 0). '(base)' when all zero / none."""
    if not order:
        return "(base)"
    parts = [f"{d}={float(order[d]):+.4g}" for d in pose_dofs
             if abs(float(order.get(d, 0.0))) > 1e-9]
    return "  ".join(parts) if parts else "(base)"


# --------------------------------------------------------------------------- #
# the ONE unified report writer (replaces the three views' _write_legend)
# --------------------------------------------------------------------------- #
def _rec_out(rec, has_pose, has_joint):
    """A ranked-entry dict for sweep.json. Emits a `pose` block iff pose DOFs were
    swept, plus the always-present IoU/depth fields and the pose `order` (which-way
    the search moved).

    `joints` is the candidate's FULL absolute state — every declared joint, whether
    the grid varied it or held it — and is written whenever the scene articulates at
    all, not only when joint DOFs were swept. `Space.realize` already returns
    base_states + swept, so this is what was RENDERED. Writing only the swept subset
    made every entry a half-state that a reader had to complete from
    `meta.held_joints`, and a reader that forgot straightened the rest (states are
    absolute: omitted is not "unchanged"). One entry, one complete state, no union."""
    out = {
        "iou_raw": round(rec["iou_raw"], 4),
        "iou_visible": (round(rec["iou_visible"], 4)
                        if rec.get("iou_visible") is not None else None),
        "depth_mae": (round(rec["depth_mae"], 5)
                      if rec.get("depth_mae") is not None else None),
        "depth_canon": (round(rec["depth_canon"], 4)
                        if rec.get("depth_canon") is not None else None),
        "behind_camera": bool(rec.get("behind_camera")),
    }
    if rec.get("candidate_id") is not None:
        out["candidate_id"] = str(rec["candidate_id"])
    if rec.get("hypothesis_ids"):
        out["hypothesis_ids"] = [str(v) for v in rec["hypothesis_ids"]]
    if rec.get("_restart") is not None:
        out["restart"] = int(rec["_restart"])
    if rec.get("_seed") is not None:
        out["seed"] = int(rec["_seed"])
    # `rank` alone. There was a second key `score_rank` carrying the identical
    # number: `rank` once meant the ROW INDEX of the listing, so the true ranking
    # needed its own name, and after `rank` took over the ranking both were emitted.
    # Two spellings of one integer is what made "which one names a candidate?"
    # ambiguous, so only `rank` is written — see write_report, which stamps it.
    if rec.get("_pass") is not None:
        # WHICH lattice this candidate came from — index into meta.passes. A refined
        # entry sits off the coarse grid on purpose; without this the reader can see
        # that it does but not which grid it is on.
        out["pass"] = int(rec["_pass"])
    if has_pose:
        out["pose"] = pose_out(rec["placement"])
        order = rec.get("order")
        out["order"] = ({k: round(float(v), 6) for k, v in order.items()}
                        if order else None)
    if rec.get("joints"):
        out["joints"] = state_json.round_joints(rec["joints"])
    return out


def depth_line(best, depth_weight):
    """One human-readable line on the winner's depth error (None without
    supervision)."""
    if not depth_weight:
        return None
    canon = best.get("depth_canon")
    if canon is None:
        return ("depth: UNSCORED (no scorable depth pixels for the winner) — "
                "verify depth with analysis.scorers.depth before trusting this fit.")
    return (f"depth: best depth error = {canon:.3f} of the object's size "
            f"(raw mae {best['depth_mae']:.4f})")


def _panel_legend_line(meta):
    """The legend sentence explaining how PANELS were chosen — it must match the
    mechanism that actually ran (meta.panels.selection), never assert binning on an
    order that had no lattice to bin against."""
    selection = (meta.get("panels") or {}).get("selection", BY_CELL)
    tail = {
        BY_CELL: ("Rendered PANELS are the best candidate in each distinct cell of "
                  "the grid you ordered, so a row has an image only if it won its "
                  "cell"),
        BY_SCORE: ("This order supplied explicit candidates, so there is no ordered "
                   "grid to bin against: rendered PANELS are simply the top of the "
                   "score order"),
        BY_RESTART: ("DE ran independent seeded restarts: each listed candidate is "
                     "one restart's winner, and rendered PANELS come from those "
                     "restart winners"),
        EVERY: "EVERY scored candidate was rendered",
    }.get(selection, "")
    listing = ("The listing is restart-winner score order"
               if selection == BY_RESTART else "The listing is plain score order")
    return (listing + ". " + tail
            + ". The `rank` column is the candidate's position in the numeric "
            "ranking and is how you NAME it (it matches ranked[i].rank in the "
            "json, and `reseed --from sweep.json#rank=N`); `row` is only the line "
            "number. They differ where the listing is gappy — topk cuts it, then "
            "panelled candidates from below the cut are appended keeping their "
            "true rank.")


_NOTE = (
    "IoU here is a numpy proxy mirroring silhouette.py (iou_raw / iou_visible). "
    "Re-verify the best with the authoritative cv2 silhouette.py on match.png at full "
    "resolution, and check the turntable (IoU is depth-blind). The winning pose's "
    "quaternion + translation paste into FRAMES[frame]['pose'] and joint states into "
    "FRAMES[frame]['joints']. SCALE is never touched: the pose order is rigid and "
    "scale is global (owned by the built object), so no verb may search it — resize "
    "via the top-level SCALE in scene.py. Search DOFs are unified: camera-frame pose "
    "ORDERS (roll/yaw/pitch deg, dpx/dpy frame-fraction, tz depth-fraction) "
    "and/or absolute JOINT states, freely mixed. With --tracking, ranking is DEPTH-AWARE: "
    "combined = gate IoU - depth_weight * min(depth_canon, 1), depth_canon the RAW "
    "depth error as a fraction of the object's longest dimension (no scale fit — a "
    "fitted scale can hide a pose error)."
)


def next_hop_block(out_dir, out_stem, frame, is_apply):
    """The order-shaped NEXT HOP block for `<stem>.txt`.

    Refining a frame is coordinate descent — sweep 2-3 DOFs, pick by eye, sweep the
    next few STARTING FROM THE PICK — and the report already ends with a paste block
    for the BEST candidate's pose and joints. That block serves committing a frame,
    not hopping: it hands the agent ~10 float literals to retype into the next
    order, and because placement and articulation ride in DIFFERENT order keys
    (`sweep.pose_start` vs top-level `joints`), the observed failure is carrying one and
    dropping the other — the joint then reverts to the frame's committed state while
    the pose carries, and the render still looks plausible.

    So the report also prints the hop as an ORDER: `"seed": "<this
    report>#rank=<N>"`, which the manager expands into both keys at claim time
    (pool.reseed.resolve_order_seed). Nothing to transcribe, so nothing to drop.

    The rank is left as a PLACEHOLDER rather than pre-baked with the winner. The
    workflow is built on overruling the numeric winner — IoU is a documented proxy
    and a silhouette-best candidate is routinely an edge-on trap; in one 53-frame
    run 5 frames accepted a non-winner and 17 notes describe rejecting a
    higher-IoU candidate. A block that read `#best` would quietly recommend the one
    choice the agent is supposed to be making for itself.
    """
    if is_apply:
        # `apply` scored ONE explicit config — there are no candidates to choose
        # between, so there is no pick to descend from. The paste block above is
        # the whole answer.
        return []
    report = os.path.join(out_dir, f"{out_stem}.json")
    order = {
        "id": "<new-unique-id>",
        "views": "sweep",
        **({"frame": frame} if frame is not None else {}),
        "seed": f"{report}#rank=<the rank you chose>",
        "sweep": {"ranges": "<the next DOFs, e.g. yaw:-6,6,7>"},
    }
    return ["",
            "NEXT HOP — descend from the candidate YOU picked (paste as a pool "
            "order):",
            "",
            json.dumps(order, indent=2),
            "",
            "`seed` fills BOTH the placement (sweep.pose_start) and the FULL joint state "
            "from that candidate — the two ride in different keys, and carrying one "
            "without the other is the classic silent revert. Use the `rank` column "
            "above (not the `row`); `#best` and `#candidate_id=<id>` also resolve. "
            "You will often not pick rank 0: IoU is a proxy and the top candidate "
            "is regularly an edge-on trap, so judge the strips by eye."]


def write_report(out_dir, ranked, base, best, meta, topk, gate_field, depth_weight,
                 space_kinds, dump_images=(), out_stem="sweep", view="sweep"):
    """Write <out_stem>.json + <out_stem>.txt for a completed run (any space).

    `space_kinds` is {dof: POSE|JOINT} over the SWEPT DOFs — it decides which paste
    blocks appear. `meta` carries the frame/resolution/method/stats/field blocks the
    engine assembled. `base` is the base placement dict. `out_stem`/`view` name the
    outputs + label the legend so the imperative `apply` view (a single-candidate
    run of the same engine) writes apply.json / apply.txt / apply_best.png with an
    apply-flavored legend, while sweep keeps its sweep.* names — one writer, no drift.
    """
    has_pose = any(k == dof_space.POSE for k in space_kinds.values())
    has_joint = any(k == dof_space.JOINT for k in space_kinds.values())
    top = ranked[:max(1, int(topk))]
    # A panelled candidate must always be listed, even when it scores below the
    # topk cut — otherwise an image on disk has no row explaining what it is.
    listed = {id(r) for r in top}
    top = top + [r for r in ranked
                 if r.get("_panel_image") and id(r) not in listed]
    frame = meta.get("frame")
    is_apply = view == "apply"

    manifest = {
        "view": view,
        "camera": "camera0 (fixed); object re-posed/articulated"
                  + ("" if is_apply else " to maximize silhouette IoU"),
        "note": _NOTE,
        **meta,
        "pose_start": pose_out(base),
        "best": {**_rec_out(best, has_pose, has_joint),
                 "image": f"{out_stem}_best.png",
                 **({"depth": best["_best_depth"]}
                    if best.get("_best_depth") else {}),
                 "combined": round(score_of(best, gate_field, depth_weight), 4)},
        # The listing is score ordered. For DE its inputs are already restricted to
        # one winner per restart; other methods pass the complete candidate order.
        # A row carries "image" only when that candidate was panelled.
        "ranked": [{"rank": r.get("_rank", i),
                    **_rec_out(r, has_pose, has_joint),
                    "combined": round(score_of(r, gate_field, depth_weight), 4),
                    **({"image": r["_panel_image"]} if r.get("_panel_image")
                       else {}),
                    **({"depth": r["_panel_depth"]} if r.get("_panel_depth")
                       else {})}
                   for i, r in enumerate(top)],
    }
    with open(os.path.join(out_dir, f"{out_stem}.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    depth_on = bool(depth_weight)
    pose_dofs = [d for d, k in space_kinds.items() if k == dof_space.POSE]
    swept_joints = [d for d, k in space_kinds.items() if k == dof_space.JOINT]
    if is_apply:
        lines = [
            f"Apply legend (the applied config scored at "
            f"{meta['sweep_resolution'][0]}x{meta['sweep_resolution'][1]}, "
            f"gate={gate_field}"
            + (f", combined = {gate_field} - {depth_weight:g}*depth_canon"
               if depth_on else "")
            + ") — camera 0 fixed; the directional order was applied, rendered, and "
            "scored (no search).",
            "IoU is a fast numpy proxy; re-verify with silhouette.py on the render.",
        ]
    else:
        lines = [
            f"Sweep legend ({meta.get('n_candidates', 0)} candidates scored at "
            f"{meta['sweep_resolution'][0]}x{meta['sweep_resolution'][1]}, "
            f"gate={gate_field}"
            + (f", ranked by combined = {gate_field} - {depth_weight:g}*depth_canon"
               if depth_on else "")
            + f", method={meta.get('method', 'grid')}) — camera 0 fixed; object "
            "re-posed/articulated.",
            "IoU is a fast numpy proxy; re-verify the best with silhouette.py on "
            "match.png.",
            _panel_legend_line(meta),
        ]
        # Only when refine actually ran: it is what makes `n_candidates` a MULTIPLE of
        # the grid product, which reads as a bug until the passes are spelled out.
        ps = passes_summary(meta.get("passes"))
        if ps:
            lines.append(ps)
    held = meta.get("held_joints") or {}
    # UNCONDITIONAL when the scene articulates at all — a POSE-ONLY sweep holds every
    # joint, and the value it holds is exactly what a reseeding agent most needs to
    # check. `base`/`pose` carry placement only (normalize_placement drops a `joints`
    # key), so the held state comes from the frame's committed joints — a sweep never
    # writes its winner back. Two calls in a row therefore hold the frame's value, not
    # the previous winner, and gating this line on `swept_joints` left the pose-only
    # second call silent about the one number that had reverted.
    if swept_joints or held:
        # a joint can be in the SPACE but in no range — the grid pins it at its base
        # state, so it is held, not swept. Subtract, or the two lists would overlap
        # and the line would claim a search that did not happen.
        actually_swept = [d for d in swept_joints if d not in held]
        lines.append(f"Swept joints: {', '.join(actually_swept) or 'none'}. "
                     f"Held: {held or '{}'}.")
    if not is_apply:
        stats = dof_stats_lines(meta.get("dof_stats"), space_kinds)
        if stats:
            lines.append(stats)
        # The heat-grids, PAIRS only: a one-DOF field would just redraw the marginal
        # the stats table above already prints. A pair is what the table cannot show
        # — a marginal collapses a diagonal valley to two flat axes, so coupling is
        # legible only as the joint field.
        pairs = [f for f in (meta.get("score_fields") or {}).values()
                 if len(f.get("dofs") or []) == 2]
        # strongest coupling first: the widest score range is the most informative
        # pair to spend the agent's attention on.
        pairs.sort(key=_field_score_range, reverse=True)
        for fld in pairs[:MAX_TXT_FIELDS]:
            ascii_field = _field_ascii(fld)
            if ascii_field:
                lines.append(ascii_field)
        extra = len(pairs) - len(pairs[:MAX_TXT_FIELDS])
        if extra > 0:
            lines.append(f"({extra} more DOF pair field(s) in the report's "
                         "`score_fields`, ordered here by score range.)")
    dl = depth_line(best, depth_weight)
    if dl:
        lines.append(dl)
    if dump_images:
        pan = meta.get("panels") or {}
        asked = int(pan.get("asked") or 0)
        cells = pan.get("cells_occupied")
        selection = pan.get("selection", BY_CELL)
        how = (
            "best candidate per ordered grid cell" if selection == BY_CELL else
            "best candidate from each DE restart" if selection == BY_RESTART else
            "top of the score order")
        if len(dump_images) >= asked:
            short = ""
        elif cells is not None:
            short = (f" Asked {asked}, but the whole sweep occupies only {cells} "
                     "distinct ordered cell(s) — the rest of the grid scored nothing "
                     "different, so no filler panels were rendered.")
        else:
            short = (f" Asked {asked}, but only {len(dump_images)} candidate(s) "
                     "were available to render.")
        if pan.get("budget_error"):
            short += (" visuals=all was refused by the visual budget, so this fell "
                      f"back to the auto level: {pan['budget_error']}")
        lines.append(f"{len(dump_images)} pose(s) also rendered to "
                     f"{out_stem}_top_00.png .. "
                     f"{out_stem}_top_{len(dump_images) - 1:02d}.png "
                     f"({how}; 00 = the numeric winner).{short}")

    if is_apply:
        # single applied config: one compact score line, no ranked table.
        b = best
        vis = "n/a" if b.get("iou_visible") is None else f"{b['iou_visible']:.4f}"
        dc = ("n/a" if b.get("depth_canon") is None else f"{b['depth_canon']:.4f}")
        comb = f"{score_of(b, gate_field, depth_weight):.4f}"
        lines += ["",
                  f"iou_raw={b['iou_raw']:.4f}  iou_vis={vis}  dp_canon={dc}  "
                  f"combined={comb}"]
    else:
        # ranked table
        col = ("pose order | joints" if (has_pose and has_joint)
               else ("joints" if has_joint else
                     "order (roll/yaw/pitch deg, dpx/dpy frac, tz)"))
        lines += ["",
                  # `row` is just the line number; `rank` is the candidate's
                  # position in the numeric ranking and is THE addressable id —
                  # it matches `ranked[i].rank` in the json, so "rank 14" means
                  # the same candidate in both files and in `reseed --from
                  # ...#rank=14`. These were spelled the other way round (`rank`
                  # for the row, `score#` for the ranking), which made an agent's
                  # "rank 10" note resolve to a DIFFERENT candidate: the listing
                  # is gappy (topk cuts at 10, then panelled candidates below the
                  # cut are appended keeping their true rank), so row N and rank N
                  # diverge exactly where a panel was worth looking at.
                  f"{'row':>4}  {'rank':>6}  {'combined':>8}  "
                  f"{'iou_raw':>8}  {'iou_vis':>8}  "
                  f"{'dp_canon':>8}  {col}"]
        for i, r in enumerate(top):
            vis = "n/a" if r.get("iou_visible") is None else f"{r['iou_visible']:.4f}"
            dc = ("n/a" if r.get("depth_canon") is None else f"{r['depth_canon']:.4f}")
            comb = f"{score_of(r, gate_field, depth_weight):.4f}"
            cells = []
            if has_pose:
                cells.append(_order_str(r.get("order"), pose_dofs))
            if has_joint:
                # DISPLAY only, and only over `swept_joints` — a joint the grid swept
                # always has a value in every candidate, so this cannot stand in for a
                # missing state the way a 0.0 fill on the RENDER path would. A hole
                # here is a report bug, so it prints as one instead of as a plausible
                # "0.0000" the reader would take for a real straightened joint.
                jr = r.get("joints") or {}
                jv = "  ".join(
                    f"{d}=" + (f"{float(jr[d]):+.4f}" if d in jr else "  ??????")
                    for d in swept_joints)
                cells.append(jv)
            rank = int(r.get("_rank", i))
            lines.append(f"{i:>4}  {rank:>6}  {comb:>8}  "
                         f"{r['iou_raw']:>8.4f}  {vis:>8}  "
                         f"{dc:>8}  {' | '.join(cells)}")

    # paste block(s)
    target = f"FRAMES[{frame!r}]" if frame is not None else "the frame"
    label = "APPLIED — paste into" if is_apply else "BEST — paste into"
    lines += ["", f"{label} {target}:"]
    if has_pose:
        lines += ["", pose_pystr(best["placement"])]
    if best.get("joints"):
        # the FULL state, and printed whenever the scene articulates — a FRAMES
        # `joints` block must name every joint (FK reads a missing one as 0.0), so a
        # swept-only block was a paste that straightened whatever it omitted.
        lines += ["", _joints_pystr(best["joints"])]
    # No SCALE line: the order is RIGID, so a candidate's scale always equals the
    # base SCALE (scale is global, owned by the built object — see
    # dof_space.REMOVED_POSE_DOFS).
    lines += next_hop_block(out_dir, out_stem, frame, is_apply)
    with open(os.path.join(out_dir, f"{out_stem}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    bv = "n/a" if best.get("iou_visible") is None else f"{best['iou_visible']:.4f}"
    bd = ("" if best.get("depth_canon") is None
          else f" depth_canon={best['depth_canon']:.4f}")
    label = "applied" if is_apply else "best"
    print(f"[render_wrapper] wrote {os.path.join(out_dir, out_stem + '.json')} + "
          f"{out_stem}.txt ({label} iou_raw={best['iou_raw']:.4f} "
          f"iou_visible={bv}{bd})")


def write_empty_report(out_dir, base, meta, out_stem="sweep"):
    """Minimal <out_stem>.json + <out_stem>.txt when no candidate scored."""
    view = meta.get("view", "sweep")
    manifest = {
        "view": view,
        "camera": "camera0 (fixed); object re-posed/articulated to maximize "
                  "silhouette IoU",
        "note": "No candidate finished scoring (see 'status'); base pose echoed for "
                "reference. Re-run with a larger --sweep-timeout / smaller grid.",
        **meta,
        "pose_start": pose_out(base),
        "best": None,
        "ranked": [],
    }
    with open(os.path.join(out_dir, f"{out_stem}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_dir, f"{out_stem}.txt"), "w") as f:
        f.write(f"{view}: no candidates scored (status={meta.get('status')}). "
                f"Base pose kept; nothing to paste.\n")
    print(f"[render_wrapper] wrote {os.path.join(out_dir, out_stem + '.json')} + "
          f"{out_stem}.txt (empty, status={meta.get('status')})")
