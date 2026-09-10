#!/usr/bin/env python3
"""self_intersection.py — per-frame part-pair overlap volumes. A WARNING, not a gate.

WHAT IT ANSWERS. "Did the lid end up inside the body at frame 000120, and is that
even plausible?" — the temporal report says how far things MOVED; this says whether
the configuration they moved INTO is physically possible. Two watertight parts of a
real object cannot share volume, so a part-pair overlap is direct evidence the pose
or the articulation is wrong — but small overlaps are also how touching parts are
MODELLED (a label sunk into the bottle so the boolean has something to cut, ribs
riding in a collar), so the number is reported and marked, never judged: this tool
has no exit-code gate and no threshold that decides anything (`!!` is a DISPLAY
mark with its floor printed in the header — same contract as the temporal report's
seam marks).

THE FRAME IT COMPUTES IN, and why that makes it cheap. Every part shares ONE rigid
base pose + ONE uniform scale per frame (POSE.md §3), and rigid+scale transforms
preserve volume ratios and relative configuration — so self-intersection is
INVARIANT to the base pose and is computed in the CANONICAL frame, using only the
frame's JOINT STATES (rig.fk.forward_kinematics at identity base). Volumes and
extents therefore come out in canonical units (fractions of the object's longest
dimension — the same `_canon` unit as the temporal report), directly, with no
per-frame mesh re-transform of anything that did not articulate. Consequences:

  * a run with NO joints has ONE configuration; the whole report is the rest pose;
  * frames with identical joint states are the SAME configuration — computed once
    and cached, so a 30-frame run with 8 distinct configurations pays for 8;
  * pairs INSIDE one rigid group (both parts under the same joint, or both on the
    base) can never change their overlap: those are the object's DESIGN CONTACTS,
    computed once and listed once in the header, not repeated per frame. The
    per-frame table carries only the pairs articulation can actually change.

MACHINERY. Exact mesh booleans via manifold3d (already trimesh 4.x's boolean
backend): part -> Manifold once, per configuration a rigid `transform` (~1us) + an
AABB prune (most pairs never come near each other) + `A ^ B` on the survivors
(~0.5ms each on real runs — measured against `min_gap` at ~33ms and vertex
containment, which also MISSES edge-through-face intersections where no vertex is
inside; the boolean is both the fastest and the only exact option, and it returns
the overlap's volume and bounds rather than a bare yes/no). No convex
decomposition: that is the price of PHYSICS-ENGINE penetration queries (FCL), not
of booleans. The price HERE is watertightness, which analysis.check_watertight
already gates — a non-watertight part cannot hold a volume, so it is EXCLUDED from
every pair and named in `warnings` (its overlaps are unknown, not zero).

AND THE VERDICT MUST CARRY THAT PRICE. `unknown_pairs` counts the articulating
pairs with an excluded part on one side, and while it is non-zero the footer
refuses to print "NO articulating-pair overlaps" — silence about an unchecked
pair reads as a clean bill of health.

WHAT A ROW MEANS. For each configuration, for each articulating pair that survives
the prune: `volume` (canonical units^3), `frac_smaller` = volume / min(part
volumes) — the "is this plausible" number: 0.2% is a graze, 5% is a lid inside a
body — and `size` = the overlap's CUBE-ROOT volume, its equivalent-cube edge
(canonical units, so it reads against the temporal report's translation column;
rotation-invariant, and a thin graze cannot inflate it the way a bounding box
would). ONSETS are the temporal
tie-in: a pair whose overlap APPEARS between two frames names the step that drove
it there — usually the same step the temporal report shows as an outlier.

THE TABLE IS THE REPORT, in the temporal report's shape: one row per frame in
temporal order, every value one scalar, '-' for "no overlap", `!!` as a margin
mark — so the two reports read side by side, row for row. `--pairs` is the
focused per-PAIR view (worst overlap + frame spans); the JSON carries every
per-frame per-pair row at full precision.

Runs in the artscript env, from harness/:
  micromamba run -n artscript python -m analysis.self_intersection \
      --pose-json RUN/mesh/pose.json --out /tmp/self_intersection.json
(the GLB defaults to object.glb beside the pose.json)
"""

import argparse
import os

import numpy as np
import manifold3d

from analysis import frames as frames_lib
from analysis.lib import glb as glb_lib
from analysis.lib import pose_read
from analysis.lib.io import write_json
from core import joints as joints_core
from rig import fk


SCHEMA_VERSION = 1

UNITS = {
    "volume": "canonical units^3 (the canonical frame's own units; the object's "
              "longest dimension is ~1)",
    "frac_smaller": "overlap volume / the SMALLER part's volume — dimensionless",
    "size": "the overlap volume's cube root — the edge of an equivalent cube, "
            "canonical units (comparable with the temporal report's "
            "translation_canon; rotation-invariant, sliver-proof)",
}

# the DISPLAY mark's floor (frac_smaller). Not a judgement and not a gate — the
# same contract as the temporal report's marks: every overlap is in the report at
# full precision, the floor only decides which rows get `!!` in the table, and it
# is printed in the header so a reader knows what the mark means. 1% of the
# smaller part is comfortably above modelling-contact noise on real runs (labels
# sunk into a body measure ~10% by DESIGN, but those are static pairs and listed
# as design contacts, not marked).
MARK_FLOOR = 0.01


# --------------------------------------------------------------------------- #
# ingest — GLB -> canonical part meshes -> manifolds
# --------------------------------------------------------------------------- #
# The GLB -> canonical-frame ingest is analysis.lib.glb.load_parts — the ONE
# home of the glTF Y-up -> canonical Z-up seam. It matters HERE because the
# joint origins/axes this tool poses parts with are canonical; FK about a
# glTF-frame mesh swings the lid about the wrong hinge line.
load_parts = glb_lib.load_parts


def build_manifolds(parts):
    """(manifolds, skipped) — a manifold3d.Manifold per WATERTIGHT part.

    A non-watertight part cannot hold a volume, so a boolean against it is
    undefined; it is skipped and NAMED, because "no overlap reported" must not
    read as "no overlap" for a part that was never checked. `skipped` is a list
    of {name, faces, why} — the why is check_watertight's diagnosis (open edge
    count, winding), so the reader knows what to fix without a second tool run.
    """
    manifolds, skipped = {}, []
    for name, mesh in parts.items():
        if not mesh.is_watertight:
            skipped.append({"name": name, "faces": int(len(mesh.faces)),
                            "why": _watertight_why(mesh)})
            continue
        man = manifold3d.Manifold(manifold3d.Mesh(
            mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32)))
        if man.status() != manifold3d.Error.NoError:
            skipped.append({"name": name, "faces": int(len(mesh.faces)),
                            "why": f"manifold3d: {man.status().name}"})
            continue
        if man.is_empty():
            skipped.append({"name": name, "faces": int(len(mesh.faces)),
                            "why": "empty manifold (zero volume)"})
            continue
        manifolds[name] = man
    return manifolds, skipped


def _watertight_why(mesh):
    """Why trimesh refuses this mesh, as one actionable phrase: open edges are
    holes to close; over-shared edges are non-manifold junctions; a closed but
    inconsistently wound surface has flipped faces."""
    edges = mesh.edges_sorted
    if len(edges):
        _, counts = np.unique(edges, axis=0, return_counts=True)
        open_edges = int((counts == 1).sum())
        overshared = int((counts > 2).sum())
    else:
        open_edges = overshared = 0
    if open_edges:
        return f"{open_edges} open edge(s)"
    if overshared:
        return f"{overshared} edge(s) shared by >2 faces (non-manifold)"
    if not mesh.is_winding_consistent:
        return "inconsistent winding (flipped faces)"
    return "fails trimesh watertight check"


# --------------------------------------------------------------------------- #
# configuration — joint states -> per-part canonical transform
# --------------------------------------------------------------------------- #
def _rigid_groups(joint_defs, part_names):
    """{part_name: group_key} — which rigid body each part rides.

    A part in some joint's child group moves with that joint (the group key is
    the joint's name); every other part is the BASE. Two parts with the same key
    can never move relative to each other, which is what makes their overlap a
    design contact rather than a per-frame quantity."""
    owner = {}
    for j in joint_defs:
        for part in fk.child_names(j):
            owner[part] = j.get("name")
    return {p: owner.get(p) for p in part_names}


def _config_key(joint_defs, states):
    """A hashable identity for one articulated configuration (rounded so fp echo
    of the same authored state doesn't defeat the cache)."""
    return tuple(round(float(states.get(n, 0.0)), 6)
                 for n in sorted(j.get("name") for j in joint_defs))


def _pair_overlaps(manifolds, part_volumes, groups, joint_defs, states):
    """All articulating-pair overlaps for ONE configuration.

    FK at IDENTITY base gives each part's pure canonical-frame transform (parts
    under no joint get identity). The AABB prune runs on transformed bounds;
    the boolean runs only on box-overlapping pairs from DIFFERENT rigid groups —
    same-group pairs are the static design contacts, computed elsewhere once.
    """
    placed = fk.forward_kinematics(joint_defs, None, states)
    names = sorted(manifolds)
    moved, boxes = {}, {}
    for n in names:
        W = placed.get(n)
        moved[n] = manifolds[n] if W is None else manifolds[n].transform(
            np.asarray(W, dtype=np.float32)[:3, :4])
        bb = moved[n].bounding_box()
        boxes[n] = (np.asarray(bb[:3]), np.asarray(bb[3:]))
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if groups[a] == groups[b]:
                continue                      # a design contact, not a frame fact
            lo = np.maximum(boxes[a][0], boxes[b][0])
            hi = np.minimum(boxes[a][1], boxes[b][1])
            if (hi <= lo).any():
                continue                      # AABB prune: cannot intersect
            inter = moved[a] ^ moved[b]
            vol = float(inter.volume())
            if vol <= 0.0:
                continue
            out.append({
                "pair": [a, b],
                "volume": round(vol, 9),
                "frac_smaller": round(
                    vol / min(part_volumes[a], part_volumes[b]), 6),
                # the overlap's characteristic linear size: cube-root volume.
                # NOT a bounding-box measure — a box is axis-dependent and a
                # thin graze along a diagonal face inflates its diagonal to
                # part-scale while the actual shared volume stays negligible.
                "size": round(vol ** (1.0 / 3.0), 6),
            })
    out.sort(key=lambda r: -r["frac_smaller"])
    return out


def _unknown_articulating_pairs(all_groups, skipped_names):
    """How many articulating (different-rigid-group) pairs involve a part that
    was EXCLUDED for not being watertight — i.e. pairs whose overlap is unknown.
    Printed beside the verdict so "no overlaps found" is never read as "none
    exist" while a pair went unchecked."""
    skipped = set(skipped_names)
    if not skipped:
        return 0
    names = sorted(all_groups)
    n = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if all_groups[a] == all_groups[b]:
                continue                  # a design contact, not an articulating pair
            if a in skipped or b in skipped:
                n += 1
    return n


def _static_contacts(manifolds, part_volumes, groups):
    """Overlaps between parts of the SAME rigid group — the object's design
    contacts (a label sunk into the body, ribs riding in a collar). Constant at
    every frame by construction, so computed once at rest and stated once."""
    names = sorted(manifolds)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if groups[a] != groups[b]:
                continue
            inter = manifolds[a] ^ manifolds[b]
            vol = float(inter.volume())
            if vol <= 0.0:
                continue
            out.append({
                "pair": [a, b],
                "volume": round(vol, 9),
                "frac_smaller": round(
                    vol / min(part_volumes[a], part_volumes[b]), 6),
            })
    out.sort(key=lambda r: -r["frac_smaller"])
    return out


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def build(pose_json, glb=None, order=None):
    """The self-intersection report for one run.

    `pose_json` — a loaded pose.json dict or its path; `glb` defaults to
    object.glb beside it. `order` — explicit frame order (default: temporal by
    frame number, analysis.frames).

    Returns {schema_version, units, mark_floor, parts, skipped_parts,
    static_contacts, frames, onsets, unknown_pairs, warnings}:

      * `parts`           — {name: {volume, size, group}} for every checked part;
      * `skipped_parts`   — [{name, faces, why}] for non-watertight parts,
                            EXCLUDED from every pair (their overlaps are unknown,
                            not zero); `why` is the geometry diagnosis;
      * `static_contacts` — same-rigid-group overlaps, constant by construction,
                            stated once (module doc);
      * `frames`          — one row per frame: {frame, joints, overlaps} where
                            `overlaps` is the articulating-pair list, biggest
                            frac_smaller first (empty = clean);
      * `onsets`          — {pair: [frames where its overlap APPEARED]}: the step
                            INTO that frame drove the parts together — the row to
                            read beside the temporal report.
      * `unknown_pairs`   — how many articulating pairs involve an EXCLUDED
                            non-watertight part, i.e. how many overlaps this
                            report cannot speak to. Printed beside the verdict
                            so "no overlaps found" is never read as "none
                            exist" when a joint child was never checked.

    Joint-state completeness is enforced per frame (JOINTS.md §3, same rule and
    same shared check as the temporal report): a defaulted state would measure a
    plausible-looking WRONG configuration.
    """
    pj = pose_read.load_pose_json(pose_json)
    if glb is None:
        if not isinstance(pose_json, str):
            raise ValueError("glb path required when pose_json is not a path")
        glb = os.path.join(os.path.dirname(os.path.abspath(pose_json)),
                           "object.glb")
    joint_defs = [j for j in (pj.get("joint_defs") or pj.get("JOINTS") or [])
                  if j.get("name")]
    order = list(order) if order else frames_lib.order_frames(
        pose_read.frame_names(pj))

    parts = load_parts(glb)
    manifolds, skipped = build_manifolds(parts)
    part_volumes = {n: float(m.volume()) for n, m in manifolds.items()}
    groups = _rigid_groups(joint_defs, manifolds)

    # groups over EVERY part, watertight or not — the unknown-pair count has to
    # see the excluded ones to be able to count them.
    all_groups = _rigid_groups(joint_defs, parts)
    unknown_pairs = _unknown_articulating_pairs(
        all_groups, [s["name"] for s in skipped])

    warnings = []
    if skipped:
        # WHICH excluded parts articulate is the actionable half: a non-watertight
        # sticker on the base is cosmetic, a non-watertight joint CHILD means the
        # pair that would expose a wrong hinge is the pair nobody checked.
        art = sorted(s["name"] for s in skipped if all_groups.get(s["name"]))
        warnings.append(
            f"!! {len(skipped)} of {len(parts)} part(s) NOT watertight, "
            "excluded from every pair — their overlaps are UNKNOWN, not zero "
            "(per-part diagnosis in the geometry table below; fix via "
            "analysis.check_watertight).")
        if unknown_pairs:
            detail = (f" {len(art)} of them ride a joint ({', '.join(art[:4])}"
                      f"{f', +{len(art) - 4} more' if len(art) > 4 else ''})."
                      if art else "")
            warnings.append(
                f"!! {unknown_pairs} articulating pair(s) could NOT be checked "
                f"at all because one side is excluded.{detail} This report's "
                "'no overlap' rows and its final verdict cover only the "
                "checked pairs — they are NOT a clean bill of health for the "
                "mechanism until every articulating part is watertight.")
    empty = [n for n, v in part_volumes.items() if v <= 0.0]
    if empty:
        warnings.append(f"!! zero-volume part(s): {', '.join(sorted(empty))}")

    static = _static_contacts(manifolds, part_volumes, groups)

    # one boolean pass per DISTINCT configuration; frames sharing joint states
    # share the result (the module doc's invariance argument).
    cache = {}
    rows = []
    for frame in order:
        _, states, _ = pose_read.frame_entry(pj, frame)
        joints_core.require_complete_states(
            joint_defs, states, f"self-intersection report, frame {frame!r}")
        key = _config_key(joint_defs, states)
        if key not in cache:
            cache[key] = _pair_overlaps(manifolds, part_volumes, groups,
                                        joint_defs, states)
        rows.append({"frame": frame,
                     "joints": {k: round(float(v), 6)
                                for k, v in (states or {}).items()},
                     "overlaps": cache[key]})

    # onsets: the frames where a pair's overlap APPEARED (prev frame had none).
    onsets = {}
    prev = set()
    for row in rows:
        now = {tuple(o["pair"]) for o in row["overlaps"]}
        for pair in sorted(now - prev):
            onsets.setdefault(" ∩ ".join(pair), []).append(row["frame"])
        prev = now

    return {
        "schema_version": SCHEMA_VERSION,
        "units": UNITS,
        "mark_floor": MARK_FLOOR,
        "glb": glb,
        # `size` is the part's cube-root volume — its equivalent-cube side, the
        # same measure as an overlap's `size`, so the three lengths (overlap,
        # part, part) compare in ONE unit without any ratio arithmetic.
        "parts": {n: {"volume": round(part_volumes[n], 9),
                      "size": round(part_volumes[n] ** (1.0 / 3.0), 6),
                      "group": groups[n]}
                  for n in sorted(manifolds)},
        "skipped_parts": sorted(skipped, key=lambda s: s["name"]),
        "static_contacts": static,
        "frames": rows,
        "onsets": onsets,
        "unknown_pairs": unknown_pairs,
        "warnings": warnings,
        "distinct_configurations": len(cache),
    }


# --------------------------------------------------------------------------- #
# human-readable table
# --------------------------------------------------------------------------- #
def _spans(order, hit_frames):
    """`hit_frames` compressed to contiguous spans over `order`, as text:
    '000100..000190, 000290' — contiguity meaning CONSECUTIVE IN THE REPORT,
    whatever the capture's stride. The compression a reader wants: a lid that is
    wrong for ten straight frames is one fact, not ten rows."""
    idx = {f: i for i, f in enumerate(order)}
    hits = sorted(hit_frames, key=lambda f: idx[f])
    spans, start, prev = [], None, None
    for f in hits:
        if start is None:
            start = prev = f
        elif idx[f] == idx[prev] + 1:
            prev = f
        else:
            spans.append((start, prev))
            start = prev = f
    if start is not None:
        spans.append((start, prev))

    def stem(f):
        return os.path.splitext(f)[0]

    return ", ".join(stem(a) if a == b else f"{stem(a)}..{stem(b)}"
                     for a, b in spans)


def _cell(value, width, fmt=".4f"):
    """One right-aligned cell of `width`, or a dash of the same width when there
    is nothing to report — padded so every row stays aligned (same helper, same
    reason, as temporal.report._cell)."""
    if value is None:
        return "-".rjust(width)
    return f"{value:{fmt}}".rjust(width)


def table_lines(report, pairs_view=False):
    """The report as aligned text — the same numbers, for a terminal.

    THE DEFAULT IS THE TEMPORAL REPORT'S SHAPE: one row per frame, in temporal
    order, every value one scalar — so the two reports read side by side, row
    for row. Per frame: `new` (pairs whose overlap APPEARS here — the onset;
    the step INTO this frame drove parts together, and it is the row to read
    in the temporal report), `pairs` (how many articulating pairs overlap at
    all), `worst%`/`size` (the single worst pair's numbers), and WHICH pair
    that is. '-' = no overlap at this frame, never 0. `!!` in the margin marks
    a row whose worst overlap >= mark_floor — a DISPLAY mark, floor in the
    header, nothing decided by it (the temporal report's `>>` contract).

    `pairs_view` is the focused per-PAIR summary instead: worst overlap + the
    frame spans where the pair overlaps at all — a persistent overlap is one
    fact about a span, and this is the view that says it once. The full
    per-frame per-pair rows are always in the JSON.
    """
    lines = []
    for w in report.get("warnings") or []:
        lines += [w, ""]

    # the geometry table: every EXCLUDED part with its diagnosis, so "why is
    # this part absent from every pair" is answered here, not in another tool.
    skipped = report.get("skipped_parts") or []
    if skipped:
        name_w = max(len("part (excluded)"),
                     max(len(s["name"]) for s in skipped))
        ghead = f"  {'part (excluded)':<{name_w}} {'faces':>6}  geometry"
        lines += [ghead, "  " + "-" * (len(ghead) - 2)]
        for s in skipped:
            lines.append(f"!!{s['name']:<{name_w}} {s.get('faces', 0):>6}  "
                         f"{s.get('why', 'NOT watertight')}")
        lines.append("")
    n_frames = len(report["frames"])
    order = [r["frame"] for r in report["frames"]]
    onset_pairs = {f: [] for f in order}
    for pair, frames in report["onsets"].items():
        for f in frames:
            onset_pairs[f].append(pair)

    part_size = {n: p["size"] for n, p in report["parts"].items()}

    def pair_label(a, b):
        """`nozzle(.09) ∩ lid(.13)` — each part CARRIES its own side, so the
        overlap side beside it compares in one unit, no ratio to invert."""
        return (f"{a}({part_size[a]:.2f}) ∩ {b}({part_size[b]:.2f})")

    lines += [
        f"self-intersection report — {len(report['parts'])} watertight "
        f"part(s), {n_frames} frames, {report['distinct_configurations']} "
        "distinct joint configuration(s)",
        "  every length is an equivalent-cube SIDE in object units (cube-root "
        "volume): ovl is the overlap region's side, the (n.nn) after each "
        "part name is that part's side — read ovl against the smaller part's "
        "side, and against the temporal report's translation column.",
        f"  worst% = overlap volume / smaller part's volume; !! marks worst% "
        f">= {report['mark_floor']:.0%} — a display mark, nothing gates on "
        "it. 'new' counts pairs whose overlap APPEARS at this frame: the "
        "step INTO that frame drove the parts together.",
    ]
    static = report["static_contacts"]
    if static:
        worst = ", ".join(
            f"{a}∩{b} {r['frac_smaller']:.1%}"
            for r in static[:4] for a, b in [r["pair"]])
        more = f" (+{len(static) - 4} more)" if len(static) > 4 else ""
        lines.append(
            f"  {len(static)} design contact(s) inside rigid groups, constant "
            f"at every frame, not listed per frame: {worst}{more}")

    # per-pair aggregation (the footer and the pairs view both read it)
    pair_rec = {}
    clean = 0
    for row in report["frames"]:
        if not row["overlaps"]:
            clean += 1
        for o in row["overlaps"]:
            rec = pair_rec.setdefault(tuple(o["pair"]),
                                      {"frames": [], "worst": o})
            rec["frames"].append(row["frame"])
            if o["frac_smaller"] > rec["worst"]["frac_smaller"]:
                rec["worst"] = o

    if pairs_view:
        head = (f"{'':>2}{'pair (side)':<46} {'ovl':>6} {'worst%':>7} "
                f"{'frames':>6}  overlapping at")
        lines += ["", head, "-" * len(head)]
        for pair, rec in sorted(
                pair_rec.items(),
                key=lambda kv: -kv[1]["worst"]["frac_smaller"]):
            w = rec["worst"]
            mark = "!!" if w["frac_smaller"] >= report["mark_floor"] else "  "
            lines.append(
                f"{mark}{pair_label(*pair):<46}"
                f" {_cell(w['size'], 6, '.3f')}"
                f" {_cell(100 * w['frac_smaller'], 7, '.2f')}"
                f" {len(rec['frames']):>6}  {_spans(order, rec['frames'])}")
    else:
        head = (f"{'':>2}{'frame':>13} {'new':>4} {'pairs':>5} "
                f"{'ovl':>6} {'worst%':>7}  worst pair (side)")
        lines += ["", head, "-" * len(head)]
        for row in report["frames"]:
            over = row["overlaps"]
            w = over[0] if over else None       # sorted worst-first in build()
            mark = ("!!" if w and w["frac_smaller"] >= report["mark_floor"]
                    else "  ")
            new = len(onset_pairs[row["frame"]])
            lines.append(
                f"{mark}{row['frame']:>13} {_cell(new or None, 4, 'd')} "
                f"{_cell(len(over) or None, 5, 'd')} "
                f"{_cell(w['size'] if w else None, 6, '.3f')} "
                f"{_cell(100 * w['frac_smaller'] if w else None, 7, '.2f')}  "
                + (pair_label(*w["pair"]) if w else "-"))
    lines.append("-" * len(head))

    unknown = int(report.get("unknown_pairs") or 0)
    if not pair_rec:
        # NEVER an unqualified all-clear while a pair went unchecked.
        if unknown:
            lines.append(
                f"  No overlaps among the CHECKED pairs in any of the "
                f"{n_frames} frame(s) — but {unknown} articulating pair(s) "
                "were UNCHECKED (a non-watertight part on one side), so this "
                "is NOT a clean mechanism verdict. Fix those parts and rerun.")
        else:
            lines.append(
                f"  NO articulating-pair overlaps in any of the {n_frames} "
                "frame(s). (Design contacts above, if any, are modelling "
                "choices inside one rigid group, not pose errors.)")
    else:
        deepest = max(pair_rec.items(),
                      key=lambda kv: kv[1]["worst"]["frac_smaller"])
        w = deepest[1]["worst"]
        lines += [
            f"  {len(pair_rec)} overlapping pair(s) across "
            f"{n_frames - clean} frame(s); {clean} frame(s) clean",
            f"  deepest: {deepest[0][0]} ∩ {deepest[0][1]} at "
            f"{w['frac_smaller']:.2%} of the smaller part "
            f"({_spans(order, deepest[1]['frames'])})",
            "  " + ("per-frame view: drop --pairs" if pairs_view else
                    "per-pair spans: --pairs; every per-frame per-pair row "
                    "is in the JSON"),
        ]
    return lines


def main():
    p = argparse.ArgumentParser(
        description="per-frame part-pair overlap volumes via exact mesh "
                    "booleans — a consistency WARNING beside the temporal "
                    "report, never a gate (exit code is always 0 when the "
                    "report was produced; no threshold decides anything)")
    p.add_argument("--pose-json", required=True, help="a run's mesh/pose.json")
    p.add_argument("--glb", default=None,
                   help="the exported canonical GLB (default: object.glb "
                        "beside the pose.json)")
    p.add_argument("--order", default="", help="comma-separated frame order "
                   "(default: temporal order by frame number)")
    p.add_argument("--pairs", action="store_true",
                   help="the focused per-PAIR view: one line per overlapping "
                        "pair with its worst overlap and frame spans, instead "
                        "of the default one-row-per-frame table")
    p.add_argument("--out", default="", help="write the JSON report here")
    p.add_argument("--out-table", default="",
                   help="also write the printed table to this .txt")
    args = p.parse_args()

    order = [s for s in args.order.split(",") if s] or None
    report = build(args.pose_json, glb=args.glb, order=order)
    text = "\n".join(table_lines(report, pairs_view=args.pairs))
    print(text)
    if args.out_table:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_table)),
                    exist_ok=True)
        with open(args.out_table, "w") as f:
            f.write(text + "\n")
        print(f"\n[self-intersection] wrote {args.out_table}")
    if args.out:
        write_json(args.out, report)
        print(f"[self-intersection] wrote {args.out}")


if __name__ == "__main__":
    main()
