"""
aggregate.py — roll up per-frame metrics into one multi-frame report.

The single-frame tools (silhouette.py, depth.py) already score one image/one view
each; a multi-frame run invokes them once per frame. This tool collects those
per-frame JSONs (+ the run's pose.json) into a single report.json + a human
table, and adds CROSS-FRAME checks that only exist with multiple frames:

  * moved-flag sanity — a frame declared moved=False uses its camera-seeded base
    pose verbatim (all cross-frame change is meant to be camera + joints). If
    such a frame still has a poor IoU/depth, either the object really did move
    (flag it moved=True and author a pose) or a joint state is wrong. This is a
    CROSS-CHECK on the agent's per-frame call, not the decision itself.

NO TEMPORAL CHECK LIVES HERE. Residuals BETWEEN consecutive frames are a separate
read with its own entry point — analysis.temporal.report, run on the poses a window
owns — because the one thing those numbers cannot distinguish (a real half-turn from
a spurious ~180 basin flip) is exactly what the IMAGES decide, so a gate here could
only restate the measurement.

Runs in the 'artscript' env, from harness/:
  micromamba run -n artscript python -m analysis.rollup.aggregate \
      --run-dir RUN_DIR [--pose RUN_DIR/mesh/pose.json] \
      [--pass 0012 | --views-dir RUN_DIR/views/0012] [--out .../report.json]

Per-frame inputs are discovered by frame name from the selected pass's pose.json (the filename
convention lives in analysis.lib.io):
  <views>/metrics_<stem>.json      (silhouette IoU; optional)
  <views>/depth_<stem>.json        (depth agreement; optional)
where <stem> is the frame name without extension (e.g. 000080 for 000080.jpg).
"""

import argparse
import os

from core import joints as joints_core
from analysis.lib.io import (depth_json, frame_stem, metrics_json, read_json,
                             write_json)


def _pass_dirs(views_root):
    """Ordered pass subdirectories, including legacy labeled passes."""
    if not views_root or not os.path.isdir(views_root):
        return []
    return sorted(
        name for name in os.listdir(views_root)
        if not name.startswith("_")
        and os.path.isdir(os.path.join(views_root, name))
    )


def resolve_views_dir(run_dir, views_dir="", pass_id=""):
    """Resolve one exact metrics directory without guessing among passes."""
    if views_dir and pass_id:
        raise ValueError("--views-dir and --pass are mutually exclusive")
    if pass_id:
        if not run_dir:
            raise ValueError("--pass requires --run-dir")
        if not str(pass_id).isdigit():
            raise ValueError("--pass must be a numeric pass id such as 0012")
        name = f"{int(pass_id):04d}"
        selected = os.path.join(run_dir, "views", name)
        if not os.path.isdir(selected):
            raise ValueError(f"pass {name} not found at {selected}")
        return selected
    if views_dir:
        if not os.path.isdir(views_dir):
            raise ValueError(f"views directory not found: {views_dir}")
        return views_dir
    if not run_dir:
        return "."

    root = os.path.join(run_dir, "views")
    passes = _pass_dirs(root)
    if len(passes) == 1:
        return os.path.join(root, passes[0])
    if len(passes) > 1:
        raise ValueError(
            "multiple render passes exist; select one with --pass ID or "
            f"--views-dir PATH (available: {', '.join(passes)})")
    return root


def resolve_pose_path(run_dir, views_dir, pose_path=""):
    """Bind aggregation to the selected pass pose, with explicit legacy escape."""
    if pose_path:
        return pose_path
    local = os.path.join(views_dir, "pose.json")
    if os.path.isfile(local):
        return local
    if run_dir:
        root = os.path.join(run_dir, "views")
        if os.path.abspath(views_dir) == os.path.abspath(root) and not _pass_dirs(root):
            legacy = os.path.join(run_dir, "mesh", "pose.json")
            if os.path.isfile(legacy):
                return legacy
    raise ValueError(
        f"selected views directory has no pose.json: {views_dir}; "
        "pass --pose PATH for a legacy pass")


def _gate_iou(m):
    """The gate IoU a metrics.json reports (iou_visible when a hand mask was
    used, else iou_raw)."""
    if m is None:
        return None
    gf = m.get("gate_field", "iou_raw")
    return m.get(gf, m.get("iou_raw"))


def build_report(pose, views_dir):
    frames = pose.get("frames", {})
    rows = []
    for name, f in frames.items():
        stem = frame_stem(name)
        m = read_json(os.path.join(views_dir, metrics_json(stem)))
        d = read_json(os.path.join(views_dir, depth_json(stem)))
        rows.append({
            "frame": name,
            "view_index": f.get("view_index"),
            "moved": f.get("moved", False),
            "joints": f.get("joints", {}),
            "gate_field": (m or {}).get("gate_field"),
            "iou": _gate_iou(m),
            "iou_raw": (m or {}).get("iou_raw"),
            "iou_pass": (m or {}).get("pass"),
            "depth_mae": (d or {}).get("depth_mae"),
            "depth_mae_canon": (d or {}).get("depth_mae_canon"),
            "depth_bias_canon": (d or {}).get("depth_bias_canon"),
            "depth_coverage": (d or {}).get("depth_coverage"),
            "has_metrics": m is not None,
            "has_depth": d is not None,
        })
    return rows


def moved_flag_sanity(rows, iou_floor=0.75, canon_ceiling=0.10):
    """Frames declared not-moved (pose = camera seed) that still score poorly —
    surface them so the agent revisits the call (object moved? joint wrong?).
    `canon_ceiling` bounds `depth_mae_canon` (the RAW depth error as a fraction
    of the object's longest dimension)."""
    flagged = []
    for r in rows:
        if r["moved"]:
            continue
        iou_bad = r["iou"] is not None and r["iou"] < iou_floor
        canon = r["depth_mae_canon"]
        depth_bad = canon is not None and canon > canon_ceiling
        if iou_bad or depth_bad:
            flagged.append({
                "frame": r["frame"], "iou": r["iou"],
                "depth_mae_canon": canon,
                "reason": "seeded pose (moved=False) but "
                          + ("low IoU" if iou_bad else "")
                          + (" and " if iou_bad and depth_bad else "")
                          + ("high depth error" if depth_bad else "")
                          + " — object may have moved (set moved=True + author a "
                            "pose) or a joint state is off.",
            })
    return {"iou_floor": iou_floor, "canon_ceiling": canon_ceiling,
            "flagged": flagged}


# joint limits: tolerant parse shared via bpy-free core.joints
_joint_limit = joints_core.joint_limit


def joint_ranges(rows, joint_defs, eps=1e-6):
    """Per DOF joint, the OBSERVED state range across frames vs the DECLARED
    valid range (`limit`) — the predict→correct signal for joint limits.

    For each revolute/prismatic joint, collect its per-frame states from `rows`
    (which already carry each frame's `joints`), report observed [min, max], the
    declared limit if any, and a verdict:
      * no_state     — the joint never appears in any frame's states.
      * no_limit     — states seen but no `limit` declared (add one to bound it).
      * within       — every observed state lies inside the declared limit.
      * exceeds_limit — an observed state falls outside the declared limit (an
                        impossible config the harness had to clamp, OR the limit
                        is too tight — reconcile in scene.py).
    """
    out = []
    for jd in joint_defs:
        name = jd.get("name")
        jtype = jd.get("type", "fixed")
        if not name or jtype == "fixed":
            continue
        vals = [float(r["joints"][name]) for r in rows
                if r.get("joints") and name in r["joints"]]
        limit = _joint_limit(jd)
        rec = {
            "name": name, "type": jtype,
            "observed_min": round(min(vals), 4) if vals else None,
            "observed_max": round(max(vals), 4) if vals else None,
            "n_states": len(vals),
            "limit": list(limit) if limit else None,
        }
        if not vals:
            rec["verdict"] = "no_state"
        elif limit is None:
            rec["verdict"] = "no_limit"
        elif min(vals) < limit[0] - eps or max(vals) > limit[1] + eps:
            rec["verdict"] = "exceeds_limit"
        else:
            rec["verdict"] = "within"
        out.append(rec)
    return out


def _fmt(v, nd=3):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def write_table(out_path, rows, moved_chk, pose, joint_rng=None):
    lines = [
        f"Multi-frame report — {len(rows)} frame(s), "
        f"reference={pose.get('reference_frame')}, scale={pose.get('scale')}",
        f"parts={len(pose.get('parts', []))}  "
        f"joints={[j.get('name') for j in pose.get('joint_defs', [])]}",
        "",
        f"{'frame':>12}  {'view':>4}  {'moved':>5}  {'iou':>6}  "
        f"{'dcanon':>6}  {'dbias':>7}  joints",
    ]
    for r in rows:
        js = ", ".join(f"{k}={_fmt(v)}" for k, v in (r["joints"] or {}).items()) or "—"
        lines.append(
            f"{r['frame']:>12}  {_fmt(r['view_index']):>4}  "
            f"{str(r['moved']):>5}  {_fmt(r['iou']):>6}  "
            f"{_fmt(r['depth_mae_canon']):>6}  "
            f"{_fmt(r['depth_bias_canon']):>7}  {js}")
    lines.append("")
    if joint_rng:
        lines.append("")
        exceeds = [j for j in joint_rng if j["verdict"] == "exceeds_limit"]
        verdict = "EXCEEDS LIMIT" if exceeds else "OK"
        lines.append(f"joint valid-range [{verdict}] — observed state range vs "
                     "declared limit:")
        for j in joint_rng:
            obs = (f"[{_fmt(j['observed_min'])}, {_fmt(j['observed_max'])}]"
                   if j["observed_min"] is not None else "(no states)")
            lim = (f"limit [{_fmt(j['limit'][0])}, {_fmt(j['limit'][1])}]"
                   if j["limit"] else "no limit declared")
            lines.append(f"  {j['name']} ({j['type']}): observed {obs}  {lim}  "
                         f"-> {j['verdict']}")
        if exceeds:
            lines.append("  -> a state fell OUTSIDE its declared limit (the harness "
                         "clamped it). Reconcile in scene.py: widen the joint's "
                         "`limit` if that config is real, or fix the frame's joint "
                         "state.")
        no_limit = [j["name"] for j in joint_rng if j["verdict"] == "no_limit"]
        if no_limit:
            lines.append(f"  -> {', '.join(no_limit)}: no `limit` declared — add one "
                         "to bound the DOF so impossible configs can't ship.")
    if moved_chk["flagged"]:
        lines.append("")
        lines.append("moved-flag sanity — revisit these frames:")
        for f in moved_chk["flagged"]:
            lines.append(f"  {f['frame']}: iou={_fmt(f['iou'])} "
                         f"depth_canon={_fmt(f['depth_mae_canon'])} — {f['reason']}")
    else:
        lines.append("moved-flag sanity: OK (no not-moved frame scores poorly)")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="",
                    help="run dir containing views/ and mesh/")
    ap.add_argument("--pose", default="", help="path to pose.json")
    selected = ap.add_mutually_exclusive_group()
    selected.add_argument("--views-dir", default="",
                          help="exact dir with per-frame *.json")
    selected.add_argument("--pass", dest="pass_id", default="",
                          help="numeric pass id under RUN_DIR/views (for example 0012)")
    ap.add_argument("--out", default="", help="output report.json path")
    a = ap.parse_args()

    run = a.run_dir
    try:
        views_dir = resolve_views_dir(run, a.views_dir, a.pass_id)
        pose_path = resolve_pose_path(run, views_dir, a.pose)
    except ValueError as exc:
        raise SystemExit(f"aggregate: {exc}")
    out_path = a.out or os.path.join(views_dir, "report.json")

    pose = read_json(pose_path)
    if pose is None:
        raise SystemExit(f"aggregate: pose.json not found ({pose_path!r})")

    rows = build_report(pose, views_dir)
    moved_chk = moved_flag_sanity(rows)
    joint_rng = joint_ranges(rows, pose.get("joint_defs", []))

    report = {
        "reference_frame": pose.get("reference_frame"),
        "scale": pose.get("scale"),
        "n_frames": len(rows),
        "parts": pose.get("parts", []),
        "joint_defs": pose.get("joint_defs", []),
        "frames": rows,
        "moved_flag_sanity": moved_chk,
        "joint_ranges": joint_rng,
    }
    write_json(out_path, report)
    print(f"[aggregate] wrote {out_path}\n")
    write_table(os.path.splitext(out_path)[0] + ".txt", rows,
                moved_chk, pose, joint_rng)


if __name__ == "__main__":
    main()
