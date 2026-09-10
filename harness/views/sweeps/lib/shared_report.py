"""shared_report.py — the two report writers for the CANONICAL verbs.

`osweep`, `oapply` and `oapply_all` search a rotation in the object's own frame;
`shared_engine` drives the Blender render/score loop, and everything here turns the
result into files: `<stem>.json` (the manifest), `<stem>.txt` (the legend + per-frame
table + paste blocks), and `<stem>_poses.json` (the merge fragment).

Why it is its own module: this is ~560 lines of TEXT and JSON formatting that never
touches `bpy`. It is `bpy`-free by contract, so the report shape is checkable in
the analysis env.

The SHARED and PER-FRAME writers stay separate on purpose: they implement different
contracts (one rotation plus seeded-frame inheritance vs N frame-local rotations with
no inheritance), and merging them would mean a writer full of `if per_frame`. What
they genuinely share are the four formatting mechanics — `_rot_label`,
`_legend_head`, `_frame_metrics_block`, `_frame_table` — each of which is the ONE
copy, because a duplicated formatter drifts silently into two report dialects with
one of them stale.
"""

import hashlib
import json
import os

import numpy as np

from core import state_json
from core.centre_calculation import (
    SOURCE_CANONICAL_ORIGIN as PIVOT_SOURCE_CANONICAL_ORIGIN,
    SOURCE_FK_AABB as PIVOT_SOURCE_FK_AABB)
from rig import lie
from views.sweeps.lib import metrics
from views.sweeps.lib.metrics import pivot_out as _pivot_out
from views.sweeps.lib.planner import CANON_DOFS
from views.sweeps.lib.shared_placement import (canon_dict as _canon_dict,
                                           canon_quat as _canon_quat,
                                           placement as _placement)


def _gate_of(rec, gate_field):
    """The frame's gate IoU (iou_visible when hand-gated, else iou_raw)."""
    return (rec.get("iou_visible") if gate_field == "iou_visible"
            else rec.get("iou_raw"))


def _scene_sha1(scene_path):
    try:
        with open(scene_path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()
    except (OSError, TypeError):
        return None


def _write_fragment(out_dir, stem, view, preps, q_best, scene_path, rot_label,
                    per_frame=None):
    """<stem>_poses.json — the winner's corrected per-frame poses in the
    multiagent.windows FRAGMENT shape (conventions/state_json.md §6), so an orchestrator
    merges the result through `multiagent.windows merge/apply` (single-writer, every
    frame baked to authored) instead of hand-pasting — the safe channel for a
    SUBSET run, where pasting the reference block would leak the rotation to every
    seeded frame outside the subset. Joints are intentionally OMITTED: merge's
    overlay keeps each frame's committed joint states.

    `window_id` / `model_feedback` are omitted too: this is a SWEEP fragment, not a
    refiner's window report. `merge` tolerates their absence (it takes the window
    id from its own plan), so the file drops into a window dir as-is.

    `per_frame` ({frame: (quaternion, label)}) switches to PER-FRAME mode, where
    each frame carries its OWN rotation. The fragment is written in that mode too
    — unlike `sweep`/`apply`, which write none — because a per-frame osweep is
    inherently multi-frame, and hand-pasting N blocks is the error-prone path the
    fragment exists to remove."""
    frames = {}
    for p in preps:
        name = str(p.fs.name)
        if per_frame is not None:
            q, label = per_frame[p.fs.name]
            note = f"{view}: per-frame canonical rotation {label}"
        else:
            q, note = q_best, f"{view}: shared canonical rotation {rot_label}"
        # `centre` comes from the prep so this pose IS the scored render's (the
        # frame's FK-AABB pivot in per-frame mode, None -> canonical origin in
        # shared mode) — a fragment composed about a different pivot than the
        # render would propose a pose that was never scored.
        P = lie.right_reorient(p.P0, q, centre=p.c_canon)
        # No `scale`: a fragment pose is a PROPOSAL, and omitting scale means a
        # merge cannot be handed a scale change (conventions/state_json.md §4).
        frames[name] = {
            "pose": state_json.round_pose({"quaternion": P.q, "translation": P.t}),
            "notes": note,
        }
    doc = {"schema_version": state_json.FRAGMENT_SCHEMA_VERSION,
           "scene_sha1": _scene_sha1(scene_path),
           "view": view, "frames": frames}
    path = os.path.join(out_dir, f"{stem}_poses.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return os.path.basename(path)


# --------------------------------------------------------------------------- #
# report mechanics shared by the two writers
# --------------------------------------------------------------------------- #
# The SHARED and PER-FRAME writers below stay separate on purpose: they implement
# different contracts (one rotation + inheritance vs N frame-local rotations with no
# inheritance), and merging them would mean a writer full of `if per_frame`. But the
# four mechanics here were duplicated between them, and a duplicated FORMATTER drifts
# silently — two report readers, one of them stale. Each of these is the one copy.
def _rot_label(cand, labels=None):
    """One candidate -> its human label: the named set's label if it has one, else
    the rx/ry/rz degrees. The label is what the paste blocks and the fragment note
    say, so both writers must spell it the same way."""
    return (labels or {}).get(cand) or ", ".join(
        f"{d}={v:+.4g}" for d, v in zip(CANON_DOFS, cand))


def _legend_head(view, contract, sweep_res, n_frames, depth_weight):
    """The first three lines of every <stem>.txt: what was rotated, at what
    resolution over how many frames, how `combined` is formed, and the standing
    instruction to re-verify with silhouette.py.

    `contract` is the one clause that differs between the modes. The rest is
    identical by contract, not by coincidence: an agent reading either report must
    find the scoring provenance in the same place."""
    return [
        f"{view} legend — camera 0 fixed; {contract}",
        f"scored at {sweep_res[0]}x{sweep_res[1]} across {n_frames} frame(s)"
        + (f", combined = gate IoU - {depth_weight:g}*depth_canon" if depth_weight
           else ", gate = silhouette IoU") + ".",
        "IoU is a fast numpy proxy; re-verify with silhouette.py on the match render.",
    ]


def _frame_metrics_block(name, rec, prep, **authorship):
    """The metric fields every frames[] entry carries, in both modes: the gate IoU
    (already resolved through the frame's own gate_field), the two raw IoUs, the
    canonical depth error, and the depth verdict. `None` where a metric does not
    apply (no hand mask -> no iou_visible; no --tracking -> no depth), which is what the
    table renders as "n/a".

    `authorship` (`authored=`, and per-frame's `was_seeded=`) is spliced in right
    after `frame`, where each mode's manifest already put it: a human scanning
    frames[] reads WHOSE answer this is before reading how good it is.

    `joints` is the frame's COMMITTED articulation — the state this run rendered at.
    These verbs search no joints, but the entry still has to name the state, because a
    report is a reseed source: `pose` + no joints is a half-record, and the next hop
    either straightens the object or is only saved by merging onto a committed state
    that may since have moved. `pose_frame` already refused an incomplete dict before
    any of these pixels existed, so this is complete by construction."""
    gate = _gate_of(rec, prep.gate_field)
    return {
        "frame": name,
        **authorship,
        "joints": state_json.round_joints(prep.fs.joint_states),
        "gate_field": prep.gate_field,
        "gate_iou": (round(gate, 4) if gate is not None else None),
        "iou_raw": round(rec["iou_raw"], 4),
        "iou_visible": (round(rec["iou_visible"], 4)
                        if rec.get("iou_visible") is not None else None),
        "depth_canon": (round(rec["depth_canon"], 4)
                        if rec.get("depth_canon") is not None else None),
    }


def _frame_table(frames_out, lead=None, trail=None):
    """The per-frame gate/iou_raw/iou_vis/dp_canon table: a header row + one row per
    frame, with `None` printed as "n/a" so a missing metric never reads as 0.0000.

    `lead`/`trail` are optional `(header, fb -> cell)` pairs for ONE extra column,
    right after the frame name or at the end of the row. Each mode has exactly one
    such column and it is the thing that mode is ABOUT: SHARED leads with `auth`
    (whether the frame INHERITS the rotation or needs its own paste block),
    PER-FRAME trails with `rotation` (each frame's own answer). Both stay in the
    column they already occupied, so this refactor changes no report byte."""
    def row(name, lead_cell, gate, raw, vis, dc, trail_cell):
        return (f"{name:>18}" + (f"  {lead_cell}" if lead else "")
                + f"  {gate:>6}  {raw}  {vis:>7}  {dc:>8}"
                + (f"   {trail_cell}" if trail else ""))
    rows = [row("frame", lead[0] if lead else "", "gate", f"{'iou_raw':>7}",
                "iou_vis", "dp_canon", trail[0] if trail else "")]
    for fb in frames_out:
        vis = "n/a" if fb["iou_visible"] is None else f"{fb['iou_visible']:.4f}"
        dc = "n/a" if fb["depth_canon"] is None else f"{fb['depth_canon']:.4f}"
        gate = "n/a" if fb["gate_iou"] is None else f"{fb['gate_iou']:.4f}"
        rows.append(row(fb["frame"], lead[1](fb) if lead else "", gate,
                        f"{fb['iou_raw']:>7.4f}", vis, dc,
                        trail[1](fb) if trail else ""))
    return rows


def _paste_blocks(frames_out, prep_by_name, quat_of, scale, ref_name,
                  seeded_key="authored", label_of=None):
    """The `# <frame>` + paste-ready pose block for each frame, in report order.

    `quat_of(frame)` is the correction to apply (one shared quaternion, or that
    frame's own); `label_of`, when given, appends the rotation to the comment line.
    `seeded_key` names the frames_out field that says whether the frame was seeded:
    the SHARED writer marks a frame it BAKED to authored ("authored" false), the
    per-frame writer authors every frame and records the fact separately
    ("was_seeded"). Both mark the reference, because pasting order matters there."""
    lines = []
    for fb in frames_out:
        was_seeded = (not fb["authored"] if seeded_key == "authored"
                      else bool(fb.get(seeded_key)))
        marks = [m for m in (" (reference)" if fb["frame"] == ref_name else "",
                             " (was seeded)" if was_seeded else "") if m]
        tail = f"  {label_of(fb['frame'])}" if label_of else ""
        prep = prep_by_name[fb["frame"]]
        # centre from the prep, like every composition site: the pasted pose must
        # BE the scored render's (None -> canonical origin in shared mode).
        lines += ["", f"# {fb['frame']}{''.join(marks)}{tail}",
                  metrics.pose_pystr(_placement(
                      prep.P0, quat_of(fb["frame"]), scale,
                      centre=prep.c_canon))]
    return lines


def write_report(out_dir, stem, view, mode, preps, ref_name, ranked, best_c, best,
                  best_images, depth_weight, sweep_res, match_res, timed_out,
                  labels=None, cand_images=None, scene_path=None,
                  requested_frames=None, scene_frames=None, passes=None):
    """Write <stem>.json + <stem>.txt for a completed osweep/oapply run.

    `mode` is the value recorded in the manifest — pass `cfg.mode` rather than
    letting this writer assume SHARED, so the report cannot be wrong about itself.

    `passes` is the search's pass list (`metrics.pass_record`): pass 0's coarse
    lattice plus one entry per refine pass, so the fine grid a winner came from is
    reconstructable from disk instead of only inferrable from an off-grid value.

    `labels` ({cand: label}) is set for a NAMED candidate set (oapply flip
    panels): the manifest then ranks by label and the .txt gets a frames x
    candidates gate-IoU matrix. A SUBSET run (`requested_frames` names fewer
    frames than `scene_frames` declares) switches the paste contract: every
    scored frame gets a paste block (seeded ones too — they can no longer
    inherit, since pasting the reference would flip the out-of-subset frames)."""
    is_apply = view == "oapply"
    scale = preps[0].fs.scale
    q_best = _canon_quat(best_c)
    prep_by_name = {p.fs.name: p for p in preps}
    scored_names = {p.fs.name for p in preps}
    requested = set(requested_frames if requested_frames is not None
                    else scored_names)
    subset = bool(scene_frames) and requested < set(scene_frames)
    # the leak warning is about PASTING the reference's block, so it keys off
    # the frames that actually got one (scored), not the requested list.
    ref_in_subset = subset and ref_name in scored_names
    cand_images = cand_images or {}

    def frame_block(name, rec):
        prep = prep_by_name[name]
        return {
            # inheritance is this mode's contract, so whether the frame is AUTHORED
            # decides whether it needs a paste block at all.
            **_frame_metrics_block(name, rec, prep, authored=prep.authored),
            # centre is None in shared mode (canonical-origin pivot — see the
            # manifest's `pivot` field), passed anyway so every composition site
            # reads the prep, not a mode assumption.
            "pose": metrics.pose_out(_placement(prep.P0, q_best, scale,
                                                centre=prep.c_canon)),
            "image": best_images.get(name),
        }

    frames_out = [frame_block(p.fs.name, best["frames"][p.fs.name])
                  for p in preps if p.fs.name in best["frames"]]

    # the fragment is the machine channel for the winner's poses — on a subset
    # run it is THE safe way to commit (multiagent.windows merge/apply bakes every
    # frame to authored, so nothing leaks past the subset).
    rot_label = _rot_label(best_c, labels)
    fragment_file = _write_fragment(out_dir, stem, view, preps, q_best,
                                    scene_path, rot_label)

    manifest = {
        "view": view,
        # "shared" vs "per_frame" is the FIRST thing a reader needs: it says whether
        # `shared_rotation` or each frames[] entry's own `rotation` is the answer.
        # Passed in rather than hardcoded SHARED: only the shared path calls this
        # writer, so the two agree today, but a hardcoded mode is a report that
        # cannot be wrong about itself for the wrong reason.
        "mode": mode,
        "camera": "camera0 (fixed); ONE shared canonical rotation right-multiplied "
                  "onto every frame's pose"
                  + ("" if is_apply else " to maximize mean silhouette IoU"),
        "note": _NOTE,
        "shared_rotation": {**_canon_dict(best_c),
                            "quaternion": [round(float(v), 6) for v in q_best],
                            **({"label": labels[best_c]} if labels else {})},
        # SHARED mode keeps the canonical ORIGIN pivot (M' = M @ R_extra, t and s
        # untouched) DELIBERATELY, unlike the per-frame verbs (which turn about
        # each frame's FK-AABB centre): one shared right-factor is what lets
        # seeded frames inherit the reference's paste, and it is the correction a
        # rotated build() would be. Stated rather than omitted, because the other
        # reports carry a `pivot` and a missing field would read as "forgotten"
        # instead of "chosen".
        "pivot": {"centre": None, "source": PIVOT_SOURCE_CANONICAL_ORIGIN},
        "mean_gate_iou": round(best["mean"], 4),
        "min_gate_iou": round(best["min"], 4),
        "depth_weight": depth_weight or None,
        "sweep_resolution": sweep_res,
        "match_resolution": match_res,
        "n_frames": len(frames_out),
        "n_candidates": len(ranked),
        # what the search actually scored, pass by pass. `n_candidates` above is the
        # SUM over these, which is why it exceeds the coarse grid product when refine
        # ran — a multiple that reads as a bug until the passes are on the record.
        "passes": list(passes or []),
        "reference_frame": ref_name,
        "subset": subset,
        "frames_scored": sorted(str(n) for n in scored_names),
        "frames_in_scene": (len(scene_frames) if scene_frames else None),
        "poses_fragment": fragment_file,
        "status": "timed_out" if timed_out else "ok",
        "frames": frames_out,
    }
    if ref_in_subset:
        manifest["warning"] = _REF_IN_SUBSET_WARNING
    if not is_apply:
        # grid mode lists the score-order top 10, and a row carries `images` when
        # that cell was PANELLED — an image on disk must never lack a row that
        # explains what it is. Panelled cells below the cut are appended, keeping
        # their TRUE score rank (a panel's row is wherever it actually ranks).
        listed = ([(i, c, v) for i, (c, v) in enumerate(ranked[:10])]
                  + [(i + 10, c, v) for i, (c, v) in enumerate(ranked[10:])
                     if c in cand_images])
        manifest["ranked"] = [
            {"rank": i, **_canon_dict(c), "mean_gate_iou": round(v["mean"], 4),
             "min_gate_iou": round(v["min"], 4),
             **({"images": dict(cand_images[c])} if c in cand_images else {})}
            for i, c, v in listed]
    if labels:
        # a named set is small — rank ALL of it, with labels + per-frame images.
        manifest["ranked"] = [
            {"rank": i, "label": labels[c], **_canon_dict(c),
             "mean_gate_iou": round(v["mean"], 4),
             "min_gate_iou": round(v["min"], 4),
             **({"images": dict(cand_images[c])} if c in cand_images else {})}
            for i, (c, v) in enumerate(ranked)]
        # per-frame per-candidate gate IoUs — which frames prefer which basin.
        matrix = {}
        for c, v in ranked:
            row = {}
            for p in preps:
                rec = v["frames"].get(p.fs.name)
                if rec is None:
                    continue
                g = _gate_of(rec, p.gate_field)
                row[str(p.fs.name)] = round(g, 4) if g is not None else None
            matrix[labels[c]] = row
        manifest["candidate_frame_gate_iou"] = matrix
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- the .txt legend + per-frame table + paste blocks ------------------- #
    rot = ", ".join(f"{d}={v:+.4g}" for d, v in zip(CANON_DOFS, best_c))
    if labels:
        rot = f"{labels[best_c]}  ({rot})"
    qs = ", ".join(state_json.paste_literal(v) for v in q_best)
    verb = "applied" if is_apply else "search winner"
    lines = _legend_head(
        view, "ONE SHARED canonical object rotation right-multiplied onto EVERY "
              f"frame's pose ({verb}).",
        sweep_res, len(frames_out), depth_weight)
    if subset:
        lines.append(
            f"SUBSET run: {len(frames_out)} of {len(scene_frames)} scene frames "
            "scored; the rotation applies to ONLY these frames.")
    ps = metrics.passes_summary(passes)
    if ps:
        lines.append(ps)
    lines += [
        "",
        f"shared rotation:  {rot}   (quaternion ({qs}))",
        f"mean gate IoU {best['mean']:.4f}   min (worst frame) {best['min']:.4f}",
        "",
    ]
    # the `auth` column is this mode's own: it says which frames INHERIT and so need
    # no paste block.
    lines += _frame_table(
        frames_out,
        lead=(f"{'auth':>4}",
              lambda fb: f"{'yes' if fb['authored'] else 'no':>4}"))

    if labels:
        # frames x candidates gate-IoU matrix: one glance says which frames
        # prefer which basin (a seam reads as a column flip mid-table). '*'
        # marks each frame's best candidate.
        lines += ["", "CANDIDATE MATRIX — gate IoU per (frame, candidate); "
                  "'*' = the frame's own best. Flip twins tying on IoU are "
                  "settled by LOOKING at the per-candidate renders "
                  f"({stem}_<candidate>_<frame>.png), never by score:"]
        cols = [labels[c] for c, _ in ranked]
        by_cand = {labels[c]: v["frames"] for c, v in ranked}
        lines.append("  " + f"{'frame':>18}  "
                     + "  ".join(f"{col:>12}" for col in cols))
        for p in preps:
            name = p.fs.name
            gates = {}
            for col in cols:
                rec = by_cand[col].get(name)
                gates[col] = _gate_of(rec, p.gate_field) if rec else None
            known = {c: g for c, g in gates.items() if g is not None}
            top = max(known, key=known.get) if known else None
            cells = []
            for col in cols:
                g = gates[col]
                mark = "*" if col == top else " "
                cells.append(f"{'n/a':>11}{mark}" if g is None
                             else f"{g:>11.4f}{mark}")
            lines.append(f"  {str(name):>18}  " + "  ".join(cells))
        # the MEAN row is the candidate's RANKING score (mean combined = gate -
        # depth penalty), not the column mean of the gate cells above — label it
        # so the two never read as the same number.
        mean_row = "  ".join(
            f"{v['mean']:>11.4f}{'*' if c == best_c else ' '}"
            for c, v in ranked)
        lines.append("  " + f"{'MEAN combined':>18}  " + mean_row)

    authored = [fb for fb in frames_out if fb["authored"]]
    seeded = [fb for fb in frames_out if not fb["authored"]]
    if subset:
        # Subset contract: the camera re-seed derives seeded frames from the
        # REFERENCE, so "paste the reference and the rest inherit" would apply
        # the rotation to every out-of-subset seeded frame too. Instead EVERY
        # subset frame gets its own block (baking seeded ones to authored),
        # and the fragment/merge path is pointed to as the safe channel.
        lines += ["",
                  "PASTE (SUBSET) — this rotation is scoped to the frames above, "
                  "so every one of them needs its OWN authored pose block below "
                  "(seeded frames become authored; they can no longer inherit). "
                  f"Prefer merging {stem}_poses.json via multiagent.windows over "
                  "hand-pasting."]
        if ref_in_subset:
            lines += ["", "WARNING: " + _REF_IN_SUBSET_WARNING]
        lines += _paste_blocks(frames_out, prep_by_name, lambda n: q_best, scale,
                               ref_name)
    else:
        lines += ["",
                  "PASTE — copy each block into the matching FRAMES entry, leaving "
                  'its "moved"/"joints" untouched. Seeded frames (no authored '
                  '"pose") INHERIT this rotation automatically via the camera '
                  "re-seed once the reference is pasted — they need NO edit."]
        # only the AUTHORED frames get a block here — that is the inheritance
        # contract, and it is the one difference from the subset branch above.
        lines += _paste_blocks(authored, prep_by_name, lambda n: q_best, scale,
                               ref_name)
        if seeded:
            lines += ["", "# seeded (inherit automatically, no paste needed): "
                      + ", ".join(fb["frame"] for fb in seeded)]
    with open(os.path.join(out_dir, f"{stem}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    if ref_in_subset:
        print(f"[render_wrapper] {view} WARNING: " + _REF_IN_SUBSET_WARNING)
    print(f"[render_wrapper] wrote {os.path.join(out_dir, stem + '.json')} + "
          f"{stem}.txt + {fragment_file} (mean gate IoU {best['mean']:.4f}, "
          f"min {best['min']:.4f} over {len(frames_out)} frame(s)"
          + (", SUBSET" if subset else "") + ")")


def write_per_frame_report(out_dir, stem, view, mode, preps, ref_name, raw,
                            winners, rows, best_images, depth_weight, sweep_res,
                            match_res, timed_out, labels=None, scene_path=None,
                            requested_frames=None, scene_frames=None,
                            frame_passes=None):
    """Write <stem>.json + <stem>.txt for a PER-FRAME osweep/oapply run.

    The rule this writer implements: a per-frame run over N frames is N independent
    `sweep`/`apply` runs. So there is no `shared_rotation` and no inheritance
    language — each `frames[]` entry carries its OWN rotation and its own paste
    block, every frame authored. `mean_gate_iou`/`min_gate_iou` are kept as a
    summary over the per-frame WINNERS (cheap, and the rollup/brief layers read
    them), and no cross-frame agreement statistic is invented: the per-frame
    rotation column already shows which frame is the outlier, which a scalar loses.

    `ranked` stays a FLAT top-level list, one entry per panelled (frame, candidate)
    pair, because `analysis.viz.sweep_sides.is_object_report` detects an object
    report by that SHAPE (a `frames` list plus ranked entries carrying `images`) and
    `report_rows` yields one row per image with no change to its loop.

    `frame_passes` is {frame: [pass_record]} — PER FRAME, unlike the shared writer's
    flat list, because each frame refined into its own window. Those records are what
    make each frame's fine lattice reconstructable; a top-level `passes` here would
    have to pick one frame's windows and misdescribe the rest."""
    scale = preps[0].fs.scale
    prep_by_name = {p.fs.name: p for p in preps}
    scored_names = {p.fs.name for p in preps}
    requested = set(requested_frames if requested_frames is not None
                    else scored_names)
    subset = bool(scene_frames) and requested < set(scene_frames)
    quats = {n: _canon_quat(c) for n, c in winners.items()}

    def rot_block(name):
        c = winners[name]
        return {**_canon_dict(c),
                "quaternion": [round(float(v), 6) for v in quats[name]],
                **({"label": labels[c]} if labels else {})}

    def rot_label(name):
        return _rot_label(winners[name], labels)

    frames_out, gates = [], []
    for p in preps:
        name = p.fs.name
        rec = raw[winners[name]][name]
        gates.append(metrics.score_of(rec, p.gate_field, depth_weight))
        frames_out.append({
            # every frame is AUTHORED after a per-frame run: its rotation is
            # frame-local, so nothing can inherit it.
            **_frame_metrics_block(name, rec, p, authored=True,
                                   was_seeded=not p.authored),
            "rotation": rot_block(name),
            # THIS frame's pivot: the rotation turned about the frame's own
            # FK-AABB centre (measured at its articulation — lib/pivot.py), so the
            # angle is only interpretable next to the point it turned about.
            "pivot": _pivot_out(p.pivot),
            "n_candidates": len(
                [1 for recs in raw.values() if name in recs]),
            # this frame's own search: the coarse lattice plus the windows IT
            # descended through. The counts here sum to `n_candidates` above.
            "passes": list((frame_passes or {}).get(name) or []),
            "pose": metrics.pose_out(
                _placement(p.P0, quats[name], scale, centre=p.c_canon)),
            "image": best_images.get(name),
        })

    fragment_file = _write_fragment(
        out_dir, stem, view, preps, None, scene_path, None,
        per_frame={p.fs.name: (quats[p.fs.name], rot_label(p.fs.name))
                   for p in preps})

    # flat ranked: one entry per panelled (frame, candidate) pair, each carrying
    # `images` so is_object_report stays true and every image on disk has a row.
    ranked_out = []
    for r in rows:
        name, c = r["frame"], r["cand"]
        rec = raw[c][name]
        g = _gate_of(rec, prep_by_name[name].gate_field)
        ranked_out.append({
            "rank": int(r["rank"]), "frame": name, **_canon_dict(c),
            **({"label": labels[c]} if labels else {}),
            "gate_iou": (round(g, 4) if g is not None else None),
            "winner": c == winners[name],
            "images": {name: r["image"]},
        })

    manifest = {
        "view": view,
        # symmetric with _write_report: the mode comes from the config, not from
        # which writer happens to be running.
        "mode": mode,
        "camera": "camera0 (fixed); ONE canonical rotation PER FRAME, right-"
                  "multiplied onto that frame's pose about that frame's own "
                  "FK-AABB centre — N independent fits",
        "note": _PER_FRAME_NOTE,
        # PER FRAME the rotation turns about each frame's own FK-AABB pivot
        # (measured at ITS articulation), so the top level can only point at the
        # per-frame records — a single centre here would misdescribe every frame
        # whose articulation differs. Old reports carry source
        # "canonical_origin" here, which is what keeps the two conventions
        # distinguishable on disk (core/centre_calculation.py, SOURCE_*).
        "pivot": {"per_frame": True, "source": PIVOT_SOURCE_FK_AABB},
        "mean_gate_iou": round(float(np.mean(gates)), 4) if gates else None,
        "min_gate_iou": round(float(np.min(gates)), 4) if gates else None,
        "depth_weight": depth_weight or None,
        "sweep_resolution": sweep_res,
        "match_resolution": match_res,
        "n_frames": len(frames_out),
        "reference_frame": ref_name,
        "subset": subset,
        "frames_scored": sorted(str(n) for n in scored_names),
        "frames_in_scene": (len(scene_frames) if scene_frames else None),
        "poses_fragment": fragment_file,
        "status": "timed_out" if timed_out else "ok",
        "frames": frames_out,
        "ranked": ranked_out,
    }
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- .txt: per-frame table with a ROTATION column, then N paste blocks --- #
    lines = _legend_head(
        view, "ONE canonical object rotation PER FRAME, right-multiplied onto that "
              "frame's pose. This is N independent fits: nothing is shared, nothing "
              "is inherited.",
        sweep_res, len(frames_out), depth_weight)
    lines += [
        "",
        f"mean gate IoU over the per-frame winners "
        f"{manifest['mean_gate_iou'] if gates else float('nan'):.4f}   "
        f"min (worst frame) "
        f"{manifest['min_gate_iou'] if gates else float('nan'):.4f}",
        "Read the rotation column: whether the frames AGREE is the thing to look at "
        "(a lone outlier means that frame, not the object, is the problem).",
    ]
    # One line, not N: each frame refined into its OWN window, so the bands differ per
    # frame and only the json can carry them all (frames[].passes). What the .txt owes
    # the reader is that refine RAN — otherwise a winner off the coarse grid, and a
    # per-frame n_candidates above the grid product, both read as bugs.
    n_refine = max((len(v) - 1 for v in (frame_passes or {}).values()), default=0)
    if n_refine > 0:
        lines.append(
            f"SEARCH PASSES — the coarse lattice plus {n_refine} refine pass(es) PER "
            "FRAME, each frame re-centering on its own best and shrinking. So a "
            "frame's winner sits off the coarse grid BY DESIGN, and its "
            "n_candidates exceeds the grid product. Each frame's own bands + centers "
            "are in the json under frames[].passes.")
    lines.append("")
    # the `rotation` column is this mode's own: N answers, one per frame, and their
    # AGREEMENT is the thing a reader is checking.
    lines += _frame_table(frames_out,
                          trail=("rotation", lambda fb: rot_label(fb["frame"])))

    lines += ["",
              "PASTE — every frame below is AUTHORED by this run: its rotation is "
              "frame-local, so there is no inheritance and no reference to paste "
              "first. Copy each block into the matching FRAMES entry, leaving its "
              '"moved"/"joints" untouched. Prefer merging '
              f"{stem}_poses.json via multiagent.windows (single-writer) over "
              f"hand-pasting {len(frames_out)} blocks."]
    lines += _paste_blocks(frames_out, prep_by_name, lambda n: quats[n], scale,
                           ref_name, seeded_key="was_seeded", label_of=rot_label)
    with open(os.path.join(out_dir, f"{stem}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[render_wrapper] wrote {os.path.join(out_dir, stem + '.json')} + "
          f"{stem}.txt + {fragment_file} (PER-FRAME: mean gate IoU "
          f"{manifest['mean_gate_iou']}, min {manifest['min_gate_iou']} over "
          f"{len(frames_out)} frame(s))")


_REF_IN_SUBSET_WARNING = (
    "the REFERENCE frame is inside this subset. Pasting its new pose into "
    "scene.py re-seeds every NON-authored frame OUTSIDE the subset from it, "
    "leaking the rotation past the subset boundary. Commit through the "
    "poses fragment (multiagent.windows merge/apply bakes all frames to authored), "
    "or paste an authored pose for every out-of-subset seeded frame first."
)


_NOTE = (
    "ONE shared rotation in the object's CANONICAL frame — the rotation vector "
    "(rx, ry, rz), degrees about canonical +X right / +Y front / +Z up (axis = "
    "direction, angle = norm; NOT the camera-frame yaw/pitch of sweep/apply) — is "
    "right-multiplied onto EVERY frame's pose (M' = M @ R_extra), so it re-orients "
    "the built geometry coherently across all frames — the fix for an object built "
    "mis-oriented (e.g. flipped). It CANNOT "
    "fix shape. Ranked by MEAN gate IoU across frames; min flags the worst frame. "
    "Paste the per-frame quaternion+translation into each AUTHORED FRAMES[...]['pose'] "
    "(reference + any 'moved'/authored frame); seeded frames inherit via the camera "
    "re-seed. On a SUBSET run (--frames names fewer frames than the scene) that "
    "inheritance contract is OFF: every scored frame gets its own paste block, and "
    "the <stem>_poses.json fragment (multiagent.windows merge/apply) is the safe commit "
    "channel. With --tracking, ranking is depth-aware (combined = gate IoU - "
    "depth_weight * min(depth_canon, 1))."
)


_PER_FRAME_NOTE = (
    "ONE rotation PER FRAME in the object's CANONICAL frame — the rotation vector "
    "(rx, ry, rz), degrees about canonical +X right / +Y front / +Z up (axis = "
    "direction, angle = norm; NOT the camera-frame yaw/pitch of sweep/apply) — "
    "right-multiplied onto THAT frame's pose about the frame's own FK-AABB centre "
    "(M' = M @ T(c)R_extraT(-c), each frame's `pivot`), so the object spins in "
    "place on screen; translation carries only the pivot correction. This is N "
    "INDEPENDENT fits: each frame picks its own winner by its own gate IoU, and "
    "nothing is shared or inherited — the object-centric equivalent of running "
    "sweep/apply on each frame. Use it when a frame reads wrong in a way that is "
    "easier to name in the object's own axes ('this one came out facing backwards') "
    "than as a camera-frame yaw. It CANNOT fix shape, and it is the WRONG verb for a "
    "build-orientation error (the object is mis-oriented in every frame): that is "
    "ONE decision, so use oapply_all. EVERY frame below is authored by this run: "
    "paste each frame's own quaternion+translation into its FRAMES[...]['pose'], or "
    "merge the <stem>_poses.json fragment through multiagent.windows (single-writer). "
    "Whether the per-frame rotations AGREE is the thing to read off the table — a "
    "lone outlier means that frame is the problem, not the object. With --tracking, "
    "ranking is depth-aware (combined = gate IoU - depth_weight * "
    "min(depth_canon, 1))."
)
