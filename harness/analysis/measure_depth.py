"""
measure_depth.py — measure tentative per-frame object poses off the observed pointmap.

depth.py *scores* a pose you already committed to; this tool *measures* rough
ones to start from. The capture ships per-pixel camera-frame 3-D points
(`depth/<stem>.npz`) plus the per-view cameras, so instead of guessing pose by
eye you can read the object's placement straight out of the cloud — for EVERY
frame, not just the reference:

  * object center in frame k          -> FRAMES[k]["pose"]["translation"]  (measured)
  * observed relative camera pose     -> FRAMES[k]["pose"]["quaternion"]   (rotation
      inv(c2w[k]) @ c2w[ref]             seeded from camera motion; ref = identity)
  * oriented-box extents (fused)      -> the shared SCALE (longest ~ 1 unit)
                                         AND the canonical W:H:D proportions for build()
  * observed intrinsics (to mask res) -> the exact `--intrinsics` string to render with

Per-frame pose model. We anchor the REFERENCE frame's object rotation at identity
and read every OTHER frame's rotation off the known tracking camera motion: the object
sits still in the world and the camera orbits, so its rotation in frame k's camera
is exactly the relative camera pose `rot(inv(c2w[k]) @ c2w[ref])`. Translation is
MEASURED independently per frame (the object's centroid in that view), so a frame
where the object was re-gripped still gets a correct translation. This is a rough
INIT everywhere; refine rotation with the `sweep` and verify with `depth.py`.

We take each frame's confident points (mask ∩ observed keep ∩ conf>=thr ∩ finite Z,
minus any hand occluder), compute their centroid (the per-frame translation), and
fit an oriented bounding box by PCA (SVD of the centered points) for the shared
SCALE + proportions. We deliberately DO NOT derive rotation from the PCA axes —
turning axes on a partial front-surface cloud into a rotation is fragile — the
per-frame rotation comes from the camera motion instead; `obb_axes` is only a
which-way hint. The observed-depth loaders live in `analysis.lib.depth_obs`; mask I/O in
`analysis.lib.rasters`; the rotation->quaternion in `rig.lie` (pure numpy).

Frames of reference. The harness renders every frame from a fixed camera 0
(identity extrinsic, OpenCV) and MOVES the object into frame k's camera; the observed
per-view `local` pointmap lives in that same frame-k camera frame, and we adopt
observed scale as truth (see depth.py). So each frame's `center` is already in the
units its `FRAMES[k]["pose"]["translation"]` expects — paste it straight in.

Runs in the 'artscript' micromamba env (NOT Blender's python), from harness/:
  micromamba run -n artscript python -m analysis.measure_depth \
      --tracking CAPTURE/tracking                  \
      --mask   MASK.png                            \
      [--hand-mask HAND.png] [--hand-dilate 0.0]   \
      [--view N | --frame-name 000040.jpg | --source IMAGE.png] \
      [--conf-thr 0.1] [--min-points 50]           \
      [--out RUN_DIR/views/measure_depth.json]

Multi-frame mode (OPT-IN). Pass `--run-dir RUN_DIR` (reads layout.json) OR
`--frames "a.jpg,b.jpg" --masks-dir DIR [--hand-masks-dir DIR] --ref-frame NAME`
to measure EVERY keyframe on its own mask: it emits a paste-ready per-frame
`seed_pose` (measured translation + camera-motion rotation) and a `pose_snippet`
(one `"pose": {...}` fragment per frame to drop INTO each existing FRAMES entry —
NOT a full FRAMES block, so it can't clobber a frame's `moved`/`joints`), and
fuses the per-frame extents into one shared base shape (SCALE +
canonical proportions). Single-view (above) stays the default.

Hand occlusion. Pass `--hand-mask` (the same occluder mask silhouette.py/depth.py
take) and the hand region is excluded from the fitted points — otherwise the
hand's (nearer) surface pulls the center/extents off the object.
"""

import argparse
import json
import os

import numpy as np

from analysis.lib import depth_obs, rasters
from analysis.lib.io import write_json
from core import captures, run_layout
from rig.lie import matrix_to_quat


# --------------------------------------------------------------------------- #
# OBB via PCA
# --------------------------------------------------------------------------- #
def fit_obb(pts, weights=None):
    """Fit an oriented bounding box to (N,3) points by PCA.

    Returns (center (3,), axes (3,3) rows=unit axes sorted by extent desc,
    extents (3,) full width along each axis sorted desc). `weights` (N,) if given
    weight the centroid (confidence-weighted) — the axes/extents stay unweighted
    (a robust extent, not a variance).
    """
    if weights is not None and float(weights.sum()) > 0:
        center = np.average(pts, axis=0, weights=weights)
    else:
        center = pts.mean(axis=0)
    centered = pts - center
    # principal axes = right-singular vectors; rows of Vt are unit axes.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt.T                       # (N,3) coords along each axis
    extents = proj.max(axis=0) - proj.min(axis=0)
    order = np.argsort(extents)[::-1]            # longest first
    return center, vt[order], extents[order]


def aabb_extents(pts):
    """Axis-aligned (camera-0 frame) full extents — a sanity fallback."""
    return pts.max(axis=0) - pts.min(axis=0)


# --------------------------------------------------------------------------- #
# single-view fit (shared by the single-view CLI and the multi-frame driver)
# --------------------------------------------------------------------------- #
_NOTES = (
    "Paste suggested_translation -> FRAMES[REFERENCE_FRAME]['pose']"
    "['translation'] and suggested_scale -> the shared top-level SCALE "
    "(canonical longest dim ~1; scale is shared across frames, never "
    "per-frame); proportion build() by canonical_ratios (longest:mid:"
    "short). Pass intrinsics_for_render to render.sh --intrinsics. "
    "rotation_euler is NOT estimated — orient by hand; obb_axes (camera-0 "
    "frame, longest first) is a hint for which way the box points. Verify "
    "with depth.py: aim for a small depth_mae_canon (the RAW residual as a "
    "fraction of the object's size — a strong-but-noisy guide, not a gate); "
    "depth_bias_canon says which way translation-Z is off (+ = too far, "
    "- = too near).")


def measure_one(tracking_dir, mask_path, *, view=None, frame_name="", source="",
                conf_thr=0.1, hand_mask="", hand_dilate=0.0, min_points=50):
    """Fit the pose anchors for ONE tracking view against `mask_path`. Pure (no I/O).

    Returns the full report dict, or — when fewer than `min_points` confident
    object points survive — the error dict (still carrying view / frame_name /
    n_points / intrinsics_for_render). Shared by the single-view CLI (main) and
    the multi-frame driver (run_multi) so the per-view fit lives in one place.
    """
    v, fname = depth_obs.resolve_view(tracking_dir, view, frame_name, source)
    obs_xyz, obs_depth, conf, keep, K_grid = depth_obs.load_view(tracking_dir, v)
    mask = rasters.load_mask(mask_path)

    # Everything fits on the MASK grid: resample observed (its grid) onto it and
    # rescale K to match, so the reported intrinsics render at the mask/source
    # resolution the agent works in.
    H, W = mask.shape[:2]
    obs_xyz, obs_depth, conf, keep, K = depth_obs.resample_to(
        (H, W), obs_xyz, obs_depth, conf, keep, K_grid)

    # hand keep-region = ¬hand at the mask grid (reuse rasters.build_keep).
    hand_keep = None
    if hand_mask:
        hand = rasters.load_mask(hand_mask)
        hand_keep = rasters.build_keep(hand, (H, W), hand_dilate)

    sel = (mask > 0) & keep & (conf >= conf_thr) & np.isfinite(obs_depth)
    if hand_keep is not None:
        sel = sel & (hand_keep > 0)
    n = int(sel.sum())

    intr_str = (f"{K[0,0]:.4f},{K[1,1]:.4f},{K[0,2]:.4f},{K[1,2]:.4f},{W},{H}")

    if n < min_points:
        return {
            "view": v, "frame_name": fname, "n_points": n,
            "error": f"only {n} confident object points (< --min-points "
                     f"{min_points}); check mask/view/conf-thr",
            "intrinsics_for_render": intr_str,
        }

    pts = obs_xyz[sel].astype(np.float64)        # (N,3), camera-0 frame
    w = conf[sel].astype(np.float64)

    center, axes, extents = fit_obb(pts, weights=w)
    aabb = aabb_extents(pts)
    e0 = float(extents[0])

    return {
        "view": v,
        "frame_name": fname,
        "n_points": n,
        "mean_conf": round(float(w.mean()), 4),
        "center": [round(float(c), 5) for c in center],
        "obb_extents": [round(float(e), 5) for e in extents],
        "obb_axes": [[round(float(a), 5) for a in row] for row in axes],
        "aabb_extents": [round(float(e), 5) for e in aabb],
        "suggested_translation": [round(float(c), 5) for c in center],
        "suggested_scale": round(e0, 5),
        "canonical_ratios": [round(float(e / e0), 4) for e in extents]
        if e0 > 0 else None,
        "intrinsics_for_render": intr_str,
        "notes": _NOTES,
    }


# --------------------------------------------------------------------------- #
# multi-frame: layout resolution + driver
# --------------------------------------------------------------------------- #
_MULTI_NOTES = (
    "PASTE each frame's measured `pose` (from pose_snippet) INTO its existing "
    "scene.py FRAMES entry, leaving that entry's `moved`/`joints` as authored (the "
    "snippet is pose-only fragments, NOT a full FRAMES block, so it can't drop "
    "them). Each frame gets a measured per-frame pose = {quaternion (from the observed "
    "camera motion, ref = identity), translation (the object's measured centroid "
    "in that frame)}. This "
    "is a rough INIT for every frame; refine rotation with the sweep and verify "
    "with depth.py. SEED the shared base shape too: base.scale -> the shared "
    "top-level SCALE; base.canonical_ratios -> build() proportions "
    "(longest:mid:short — the SHORT/depth axis is a LOWER bound taken from the "
    "deepest single view, widen toward canonical_ratios_spread.short.max if "
    "depth reads thin). base.intrinsics_for_render -> render.sh --intrinsics "
    "(or just pass --tracking, which applies each frame's K automatically). "
    "This tool measures POSE only and does NOT decide moved — the VLM critic "
    "(critic.py) owns the moved-vs-articulated call; trust it. A frame's seed_pose "
    "ROTATION is a camera-motion guess that a genuine re-grip may invalidate — the "
    "translation is still measured (correct), so keep it and re-orient with the "
    "sweep; if the critic says articulation, model it with a JOINT (the base pose "
    "stays). SCALE + canonical_ratios are invariant to rigid motion, so "
    "moved/articulated frames stay in the fuse. Verify with depth.py per frame + "
    "aggregate.py's cross-frame report.")


def resolve_layout(args):
    """Resolve the multi-frame layout from --run-dir/layout.json (via
    core.run_layout.load_run_layout) + CLI overrides (CLI wins). Returns a dict
    with tracking (the capture's tracking dir), masks_dir, hand_masks_dir, frames
    (list), ref_frame."""
    manifest = {}
    if args.run_dir:
        manifest = run_layout.load_run_layout(args.run_dir)
        if manifest is None:
            raise SystemExit(f"[measure_depth] no layout.json in {args.run_dir}; "
                             "pass --frames/--masks-dir/--tracking explicitly")
        if manifest.get("kind") == "single":
            raise SystemExit(
                f"[measure_depth] {args.run_dir}/layout.json is a single-image run "
                "(kind=single); multi-frame mode needs keyframes. Use the "
                "single-view form (--mask/--view).")

    def pick(cli, key):
        return cli if cli else manifest.get(key, "")

    layout = {
        "tracking": args.tracking or manifest.get("tracking", ""),
        "masks_dir": pick(args.masks_dir, "masks_dir"),
        "hand_masks_dir": pick(args.hand_masks_dir, "hand_masks_dir"),
        "ref_frame": pick(args.ref_frame, "ref_frame"),
    }
    if args.frames:
        layout["frames"] = [f.strip() for f in args.frames.split(",") if f.strip()]
    else:
        layout["frames"] = list(manifest.get("frames", []))
    if not layout["ref_frame"] and layout["frames"]:
        layout["ref_frame"] = layout["frames"][0]

    missing = [k for k in ("tracking", "masks_dir", "ref_frame") if not layout[k]]
    if not layout["frames"]:
        missing.append("frames")
    if missing:
        raise SystemExit(
            f"[measure_depth] layout incomplete (missing: {', '.join(missing)}). "
            "Pass the corresponding --flag or fix layout.json.")
    if layout["ref_frame"] not in layout["frames"]:
        raise SystemExit(
            f"[measure_depth] --ref-frame {layout['ref_frame']!r} is not in "
            f"frames {layout['frames']}")

    # Absolutize dir paths CLI overrides may have introduced (manifest paths
    # come back absolute; relative flags resolve against the invoking cwd).
    for k in ("tracking", "masks_dir", "hand_masks_dir"):
        if layout[k]:
            layout[k] = os.path.abspath(layout[k])
    return layout


def _fuse_base(good, ref_report, ref_ok, ref_name):
    """Fuse per-frame OK reports into one base-shape seed. SCALE and ratios are
    invariant to rigid object motion, so moved frames are kept in the fuse.

    scale = median longest extent; short/depth ratio = MAX across views (a single
    front-surface view underestimates depth). ref_translation / intrinsics come
    solely from the reference frame (a moved non-ref frame cannot corrupt them).
    """
    base = {"ref_frame": ref_name}
    if good:
        longs = [r["obb_extents"][0] for r in good]
        mids = [r["canonical_ratios"][1] for r in good]
        shorts = [r["canonical_ratios"][2] for r in good]
        scale = float(np.median(longs))
        lo, hi = float(min(longs)), float(max(longs))
        base["scale"] = round(scale, 5)
        base["scale_spread"] = {
            "min": round(lo, 5), "max": round(hi, 5),
            "median": round(scale, 5),
            "relative_spread": round((hi - lo) / scale, 4) if scale else None,
            "n": len(longs)}
        base["canonical_ratios"] = [1.0, round(float(np.median(mids)), 4),
                                    round(float(max(shorts)), 4)]
        base["canonical_ratios_spread"] = {
            "mid": {"min": round(float(min(mids)), 4),
                    "median": round(float(np.median(mids)), 4),
                    "max": round(float(max(mids)), 4)},
            "short": {"min": round(float(min(shorts)), 4),
                      "median": round(float(np.median(shorts)), 4),
                      "max": round(float(max(shorts)), 4)}}
        base["depth_axis_policy"] = (
            "short/depth ratio = MAX across views: a single front-surface view "
            "underestimates depth, so the deepest view is the best single-view "
            "proxy (a LOWER bound on true depth). See canonical_ratios_spread.short.")
    else:
        scale = None
        base["scale"] = None
    base["ref_translation"] = ref_report["center"] if ref_ok else None
    base["intrinsics_for_render"] = (ref_report["intrinsics_for_render"]
                                     if ref_ok else None)
    return base, scale


def _seed_poses(per_frame, ref_report, tracking_dir):
    """Attach a paste-ready per-frame `seed_pose` to every OK report and return a
    `pose_snippet` string: one `"pose": {...}` fragment per frame, to be pasted
    INTO the matching FRAMES entry (NOT a full `FRAMES = {}` block).

    Emitting fragments — not a whole FRAMES assignment — is deliberate: a wholesale
    paste over an existing FRAMES would drop each entry's `moved`/`joints` keys
    (they revert to their defaults), silently flipping a hand-authored `moved=True`
    back to False. This tool measures POSE only and never decides `moved`, so the
    artifact it emits must not be able to clobber that decision either. Copy each
    `"pose": {...}` into its frame and leave `moved`/`joints` as authored.

    Each frame's pose is {quaternion, translation}:
      * translation = that frame's MEASURED object centroid (`center`), already in
        the frame-k camera units FRAMES[k]['pose']['translation'] expects.
      * quaternion  = rot(inv(c2w[k]) @ c2w[ref]) — the object's rotation in frame
        k when the reference rotation is anchored at identity (the object is still
        in the world; the camera orbits). The reference frame yields (1,0,0,0).

    Mutates the OK reports in place (so per_frame carries seed_pose) and returns
    the snippet (or "" when no cameras / no ref view)."""
    try:
        c2w = depth_obs.load_c2w(tracking_dir)       # (K,4,4)
    except Exception:
        c2w = None
    ref_view = ref_report.get("view") if ref_report else None
    if c2w is None or ref_view is None:
        return ""                                # no camera motion -> no rotation seed

    lines = [
        '# measured per-frame poses — paste each "pose": {...} INTO the matching',
        '# FRAMES entry, leaving that entry\'s "moved" / "joints" untouched. This',
        '# tool measures POSE only; it does NOT decide "moved" (the critic does).',
    ]
    for rep in per_frame:
        name = rep.get("frame_name")
        if rep.get("status") != "ok":
            lines.append(f'# {name!r}: [{rep.get("status")}] no measured pose '
                         f'— {rep.get("error", "")}')
            continue
        rel = depth_obs.relative_pose(c2w, ref_view, rep["view"])   # inv(c2w[k])@c2w[ref]
        quat = matrix_to_quat(rel[:3, :3])
        q = [round(float(v), 6) for v in quat]
        t = [round(float(v), 6) for v in rep["center"]]
        rep["seed_pose"] = {"quaternion": q, "translation": t}
        header = f'# {name}' + (' (reference)' if name == ref_report.get(
            "frame_name") else '')
        qs = ", ".join(f"{v:.6g}" for v in q)
        ts = ", ".join(f"{v:.6g}" for v in t)
        lines.append("")
        lines.append(header)
        lines.append('"pose": {')
        lines.append(f'    "quaternion": ({qs}),')
        lines.append(f'    "translation": ({ts}),')
        lines.append('},')
    return "\n".join(lines)


def run_multi(args):
    """Multi-frame seeding: measure a per-frame pose for every keyframe and fuse a
    shared base shape."""
    layout = resolve_layout(args)
    tracking_dir, frames, ref = (layout["tracking"], layout["frames"],
                                 layout["ref_frame"])

    out = args.out
    if not out and args.run_dir:
        out = os.path.join(args.run_dir, "measure_multi.json")

    # 1) per-frame fits (one bad frame never aborts the batch) ---------------
    per_frame = []
    for name in frames:
        mask = captures.mask_for(layout["masks_dir"], name)
        if not os.path.isfile(mask):
            per_frame.append({"frame_name": name, "status": "skipped_no_mask",
                              "error": f"no object mask {mask}"})
            continue
        hand = ""
        if layout["hand_masks_dir"]:
            h = captures.mask_for(layout["hand_masks_dir"], name)
            hand = h if os.path.isfile(h) else ""
        try:
            rep = measure_one(tracking_dir, mask, frame_name=name, source=name,
                              conf_thr=args.conf_thr, hand_mask=hand,
                              hand_dilate=args.hand_dilate,
                              min_points=args.min_points)
        except SystemExit as e:                  # e.g. no keyframes.json match
            per_frame.append({"frame_name": name, "status": "error",
                              "error": str(e)})
            continue
        rep["status"] = "too_few_points" if "error" in rep else "ok"
        per_frame.append(rep)
        if args.per_frame_out and out:
            write_json(os.path.join(os.path.dirname(os.path.abspath(out)),
                                    f"measure_{stem}.json"), rep)

    good = [r for r in per_frame if r.get("status") == "ok"]
    ref_report = next((r for r in per_frame if r.get("frame_name") == ref), None)
    ref_ok = ref_report is not None and ref_report.get("status") == "ok"

    # 2) fuse the base shape --------------------------------------------------
    base, scale = _fuse_base(good, ref_report, ref_ok, ref)

    # 3) per-frame tentative pose: measured translation + camera-motion rotation
    pose_snippet = _seed_poses(per_frame, ref_report, tracking_dir)

    report = {
        "mode": "multi_frame",
        "tracking": tracking_dir, "ref_frame": ref, "frames": frames,
        "per_frame": per_frame, "base": base,
        "pose_snippet": pose_snippet,
        "notes": _MULTI_NOTES,
    }
    text = json.dumps(report, indent=2)
    if out:
        write_json(out, report)
    print(text)
    _print_multi_summary(report)


def _print_multi_summary(report):
    base = report["base"]
    print(f"[measure_depth] MULTI ref={report['ref_frame']} "
          f"n_frames={len(report['frames'])}  base_scale={base.get('scale')}  "
          f"ratios={base.get('canonical_ratios')}")
    for r in report["per_frame"]:
        if r.get("status") == "ok":
            sp = r.get("seed_pose")
            q = f"  quat={sp['quaternion']}" if sp else ""
            print(f"  {r['frame_name']:>14} view {r['view']:>2}  "
                  f"n={r['n_points']:>6}  center={r['center']}{q}  "
                  f"scale={r['suggested_scale']}  ratios={r['canonical_ratios']}")
        else:
            print(f"  {r['frame_name']:>14} [{r.get('status')}] "
                  f"{r.get('error', '')}")

    # paste-ready per-frame poses (measured translation + camera-motion rotation)
    if report.get("pose_snippet"):
        print("\n[measure_depth] paste each \"pose\" into its scene.py FRAMES entry "
              "(leave moved/joints as authored; ref rotation = identity, others "
              "from tracking camera motion; refine rotation with the sweep):")
        print(report["pose_snippet"])


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="read object pose anchors (translation/scale/proportions/"
                    "intrinsics) off the observed pointmap — one view (default), or "
                    "every keyframe with --run-dir/--frames (multi-frame seed)")
    p.add_argument("--tracking", "--pi3x", dest="tracking", default="",
                   help="capture tracking dir "
                   "(CAPTURE/tracking; the capture root also works); required "
                   "for single-view, or supplied by --run-dir/layout.json. "
                   "--pi3x is the old name, still accepted")
    p.add_argument("--mask", default="", help="object mask png (interior); "
                   "required in single-view mode")
    p.add_argument("--source", default="", help="source frame name to resolve "
                   "the Pi3X view (via keyframes) if --view/--frame-name absent")
    p.add_argument("--view", type=int, default=None,
                   help="Pi3X view index; else resolved from --frame-name/--source")
    p.add_argument("--frame-name", default="",
                   help="source frame filename to match a Pi3X view (via keyframes)")
    p.add_argument("--conf-thr", type=float, default=0.1,
                   help="confidence below this (or Pi3X keep==False) is ignored")
    p.add_argument("--hand-mask", default="", help="binary HAND (occluder) mask "
                   "png; its region is excluded from the fitted points (Pi3X sees "
                   "the hand's surface there, not the object's)")
    p.add_argument("--hand-dilate", type=float, default=0.0,
                   help="grow the hand ignore-region by this fraction of the "
                   "longer side (matches silhouette.py; 0 = use the mask exactly)")
    p.add_argument("--min-points", type=int, default=50,
                   help="fail if fewer confident object points survive selection")
    p.add_argument("--out", default="", help="write JSON report here")

    # --- multi-frame mode (opt-in: triggered by --run-dir or --frames) ------
    g = p.add_argument_group("multi-frame mode (opt-in: --run-dir or --frames)")
    g.add_argument("--run-dir", default="", help="run dir with a layout.json "
                   "(pi3x/masks_dir/hand_masks_dir/frames/ref_frame); enables "
                   "multi-frame seeding across every keyframe")
    g.add_argument("--frames", default="", help="comma-separated frame filenames "
                   "to seed (e.g. '000000.jpg,000070.jpg'); enables multi-frame "
                   "mode without a run dir")
    g.add_argument("--masks-dir", default="", help="dir of per-frame object masks "
                   "<stem>.png (overrides layout.json)")
    g.add_argument("--hand-masks-dir", default="", help="dir of per-frame hand "
                   "(occluder) masks <stem>.png (overrides layout.json)")
    g.add_argument("--frames-dir", default="", help="dir of source frames "
                   "(accepted for parity with layout/shape_pass; not used by the fit)")
    g.add_argument("--ref-frame", default="", help="reference frame name "
                   "(anchors the gauge); default = first of --frames")
    g.add_argument("--per-frame-out", action="store_true", help="also write "
                   "measure_<stem>.json beside --out (one per frame)")
    args = p.parse_args()

    # --- dispatch -----------------------------------------------------------
    multi = bool(args.run_dir or args.frames)
    if multi:
        if args.mask or args.view is not None:
            p.error("multi-frame mode (--run-dir/--frames) picks a view per "
                    "frame — don't combine it with --mask/--view")
        run_multi(args)
        return

    # single-view (default) — behavior unchanged
    if not args.tracking:
        p.error("--tracking is required")
    if not args.mask:
        p.error("--mask is required (or use --run-dir/--frames for multi-frame)")
    report = measure_one(args.tracking, args.mask, view=args.view,
                         frame_name=args.frame_name, source=args.source,
                         conf_thr=args.conf_thr, hand_mask=args.hand_mask,
                         hand_dilate=args.hand_dilate, min_points=args.min_points)
    text = json.dumps(report, indent=2)
    if args.out:
        write_json(args.out, report)
    print(text)
    if "error" in report:
        print(f"[measure_depth] view {report['view']}: too few points "
              f"({report['n_points']})")
    else:
        print(f"[measure_depth] view {report['view']} ({report['frame_name']})  "
              f"n={report['n_points']}  center={report['center']}  "
              f"scale={report['suggested_scale']}  "
              f"ratios={report['canonical_ratios']}  "
              f"intr={report['intrinsics_for_render']}")


if __name__ == "__main__":
    main()
