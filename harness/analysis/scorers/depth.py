"""depth.py — geometric supervision: rendered depth vs the OBSERVED pointmap.

Silhouette IoU (silhouette.py) is DEPTH-BLIND: a part floating toward the camera
overlaps the same 2D silhouette as one welded on, so IoU can't see it. This tool
adds the missing third dimension. It compares the harness's rendered depth (the
`depth` view -> depth.npy, planar camera-space Z) against the depth observed
for the SAME view — the Z channel of the capture's per-frame camera-frame
pointmap (`depth/<stem>.npz`) — inside the object, weighted by the per-pixel
confidence so untrustworthy pixels don't penalise. The loaders live in
`analysis.lib.depth_obs`; the panel drawing in `analysis.lib.panels`.

Frames of reference. The harness camera 0 (identity extrinsic, OpenCV
convention) IS the frame the per-view `local` pointmap lives in, and we adopt the
observed (arbitrary-but-self-consistent) scale as ground truth: the canonical mesh
is unit and the frame's pose + shared SCALE carry it into observed units, so rendered
depth compares to observed depth RAW (no scale solving). This is what generalises to
video — one unit canonical mesh + per-keyframe observed poses. There is
deliberately NO per-view scale fit here: a fitted scale assumes the pose is
right, so a pose error masquerades as "scale" and the fitted residual explains
it away. The signed `depth_bias` says which way to move instead.

Primary term = per-pixel DEPTH error. Secondary = full pointmap XYZ error
(unproject the rendered depth through the observed K and compare 3-D points), which also
catches lateral (X/Y) drift depth alone misses.

Runs in the 'artscript' micromamba env (NOT Blender's python), from harness/:
  micromamba run -n artscript python -m analysis.scorers.depth \
      --tracking    CAPTURE/tracking               \
      --render-depth PASS_DIR/depth_000040.npy    \
      --mask        MASK.png                      \
      --source      IMAGE.png                     \
      [--view N | --frame-name 000040.jpg]        \
      [--conf-thr 0.1] [--out PASS_DIR/depth_000040.json] [--panel composite_depth.png]

Confidence < --conf-thr (default 0.1) or keep==False -> weight 0 (ignored),
mirroring "if it's less than ~0.1 it can't be trusted, don't penalise it".

Hand occlusion. Where a hand occludes the object, the observed depth is the HAND's
surface (nearer), not the object's, while our render shows the object continuing
behind it — so scoring depth there is meaningless. Pass `--hand-mask` (the same
occluder mask silhouette.py takes) and that region is waived from the scored
region, exactly like silhouette.py's iou_visible (reuses rasters.build_keep;
`--hand-dilate` widens it to absorb boundary noise).
"""

import argparse
import json
import os

import cv2
import numpy as np

from core import cam_math

from analysis.lib import depth_obs, panels, rasters
from analysis.lib.io import write_json
from core import state_json
from core.depth_config import resolve_conf_thr
from rig import lie


def resolve_scale(scale_arg, render_depth_path):
    """The object's longest dimension in scene units, for canonical-units errors.

    --scale wins; else read `scale` from the run's pose.json (searched next to
    the render depth: <views>/pose.json, then <views>/../mesh/pose.json). None if
    neither exists — canonical fields are then omitted."""
    if scale_arg is not None:
        return max(lie.scalar_scale(scale_arg, "--scale"), 1e-9)
    base = os.path.dirname(os.path.abspath(render_depth_path))
    s = state_json.first_scale(
        os.path.join(base, "pose.json"),
        os.path.join(os.path.dirname(base), "mesh", "pose.json"))
    return None if s is None else max(lie.scalar_scale(s, "pose.json scale"), 1e-9)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def unproject(depth, K):
    """Back-project a planar-depth map to camera-frame XYZ (OpenCV: +Z forward).

    Thin wrapper over the shared core (same conventions as the render side), kept
    under a local name so the scorer reads self-contained."""
    return cam_math.unproject(depth, K)


def _wmean(values, w):
    tot = float(w.sum())
    return float((values * w).sum() / tot) if tot > 0 else None


def depth_report(render_depth, obs_xyz, obs_depth, conf, keep, K, mask,
                 conf_thr, hand_keep=None, scale=None):
    """Compute the full depth/pointmap agreement report over the object interior.

    Region = object mask ∩ rendered geometry ∩ observed keep ∩ conf>=thr, and (when
    a hand mask is given) ∩ hand_keep = ¬hand. Weight w = conf there, 0
    elsewhere. All errors are conf-weighted over that region.

    `scale` (the object's longest dimension in scene units — the shared SCALE)
    adds `depth_mae_canon` = RAW mae / scale (error as
    a fraction of the object's size — a natural metric prior for any object). RAW on purpose — no
    per-view scale fit: a fitted scale assumes the pose is right, so a pose
    error masquerades as 'scale' and the fitted residual explains it away. The
    signed `depth_bias` carries the direction instead.
    """
    render_valid = np.isfinite(render_depth)
    valid = (mask > 0) & render_valid & keep & (conf >= conf_thr)
    if hand_keep is not None:
        valid = valid & (hand_keep > 0)
    w = np.where(valid, conf, 0.0).astype(np.float32)
    n = int(valid.sum())

    scored_render = np.where(render_valid, render_depth, 0.0).astype(np.float32)
    # coverage = fraction of the SCORABLE mask (object minus waived hand) that our
    # render actually fills, so a waived hand doesn't deflate it.
    mask_region = (mask > 0)
    if hand_keep is not None:
        mask_region = mask_region & (hand_keep > 0)
    denom = float(mask_region.sum())
    inter_mask_render = int((mask_region & render_valid).sum())

    report = {
        "n_scored_px": n,
        "conf_thr": round(float(conf_thr), 4),
        "depth_coverage": round(inter_mask_render / denom, 4) if denom else 0.0,
        "mean_conf_used": round(_wmean(conf, w), 4) if n else None,
    }
    if hand_keep is not None:
        report["hand_coverage"] = round(float((hand_keep == 0).mean()), 4)
    if n == 0:
        report.update({
            "depth_mae": None, "depth_rmse": None,
            "depth_bias": None, "depth_bias_canon": None,
            "depth_mae_canon": None,
            "point_l2_mae": None, "point_l2_rmse": None,
            "_maps": {"render_depth": render_depth, "obs_depth": obs_depth,
                      "valid": valid},
        })
        return report

    signed = scored_render - obs_depth
    err = np.abs(signed)
    mae = _wmean(err, w)
    rmse = float(np.sqrt(_wmean(err ** 2, w)))
    report["depth_mae"] = round(mae, 5)
    report["depth_rmse"] = round(rmse, 5)
    # SIGNED mean residual (render − observed): the DIRECTION to move, not just the
    # magnitude the MAE gives. +bias = render sits BEHIND / too far (pull it in:
    # smaller SCALE or −translation-Z); −bias = render IN FRONT / too near. This
    # is the number that the diverging residual panel visualises pixel-by-pixel.
    bias = _wmean(signed, w)
    report["depth_bias"] = round(bias, 5)

    # RAW error in canonical units (fraction of the object's longest dimension).
    # Raw on purpose (see docstring).
    if scale:
        report["depth_mae_canon"] = round(mae / scale, 4)
        # signed bias in canonical units (fraction of object size, signed like
        # depth_bias): the direction AND how far, on the same scale as the MAE.
        report["depth_bias_canon"] = round(bias / scale, 4)
    else:
        report["depth_bias_canon"] = None

    # secondary: full XYZ pointmap L2 (catches lateral X/Y drift depth misses)
    render_xyz = unproject(scored_render, K)
    l2 = np.linalg.norm(render_xyz - obs_xyz, axis=-1)
    report["point_l2_mae"] = round(_wmean(l2, w), 5)
    report["point_l2_rmse"] = round(float(np.sqrt(_wmean(l2 ** 2, w))), 5)

    report["_maps"] = {"render_depth": render_depth, "obs_depth": obs_depth,
                       "valid": valid}
    return report


# --------------------------------------------------------------------------- #
# panel  [ source | our depth | observed depth | residual ]
# --------------------------------------------------------------------------- #
def depth_panels(source_bgr, maps, report, height):
    """The depth panels, keyed so callers can pick a subset.

    Returns {'source', 'our', 'observed', 'residual'} -> `panels.Panel(image, title,
    sub)`: the PICTURE plus the words that belong over it, each render scaled to
    `height` and NOT yet labeled. This module owns the depth colormaps and knows
    what its own panels should be called; it deliberately does not draw the header
    bar, because whoever assembles the row is the only one who can know whether a
    neighbouring column carries a caption — and every panel in a row must reserve
    the caption line or none of them may (`panels.label`, `viz.rows.build_row`).
    Label them with `panels.label_row`, or let `viz.rows` do it as it does for
    every other column.

    `build_panel` labels and hstacks all four; `composite.py` takes just 'our' +
    'residual' into its own strip. The residual is SIGNED (diverging
    blue/white/red — see `panels.signed_diverging_heat`) so the panel shows which
    WAY depth is off, not just how much.
    """
    h = height
    render_depth = maps["render_depth"]
    obs_depth = maps["obs_depth"]
    valid = maps["valid"]

    # shared range over BOTH maps' valid pixels pooled (percentiles so a few
    # outliers don't wash out the scale), so the two depth panels are directly
    # comparable. Same pooled-percentile kernel the object-units sheet uses for
    # a whole clip (`panels.pct_range`) — one range rule, two media.
    dmin, dmax = panels.pct_range(
        [render_depth[valid & np.isfinite(render_depth)], obs_depth[valid]]
        if valid.any() else [np.array([1.0])])

    # TURBO depth maps over the shared [dmin,dmax] range (black outside valid).
    our = panels.depth_heat(render_depth, dmin, dmax, valid=valid)
    obs = panels.depth_heat(obs_depth, dmin, dmax, valid=valid)

    # SIGNED residual (render − observed): the sign says which way to move, not just
    # how far. blue = render in front (too near), red = render behind (too far),
    # white = agree. Same saturation span the two depth panels share.
    signed = (np.where(np.isfinite(render_depth), render_depth, 0.0)
              - obs_depth)
    rheat = panels.signed_diverging_heat(signed, valid, span=0.5 * (dmax - dmin))

    src = panels.scale_to_height(source_bgr, h)
    our = panels.scale_to_height(our, h)
    obs = panels.scale_to_height(obs, h)
    rheat = panels.scale_to_height(rheat, h)

    mae = report.get("depth_mae")
    canon = report.get("depth_mae_canon")
    bias = report.get("depth_bias")
    dsub = f"range [{dmin:.2f},{dmax:.2f}]"
    # LEAD the sub-line with the actionable signed bias (which way to move), so it
    # survives when a narrow panel truncates the tail. Then MAE/canon.
    if mae is None:
        rsub = "no valid pixels"
    else:
        btag = ""
        if bias is not None:
            btag = (f"bias={bias:+g} " +
                    ("BEHIND/too far  " if bias > 0 else "IN FRONT/too near  "))
        tail = (f"MAE={mae} canon={canon}"
                if canon is not None else f"MAE={mae} (no scale)")
        rsub = f"{btag}{tail}"
    # The NEAR/FAR key is stamped INTO the heat image (it is anchored to the bottom
    # of the picture it describes), which is another reason to hand back the image
    # rather than a labeled panel: this belongs to the picture, the title does not.
    panels.stamp_signed_legend(rheat)           # always-legible colour key
    return {
        "source": panels.Panel(src, "SOURCE"),
        "our": panels.Panel(our, "OUR DEPTH", dsub),
        "observed": panels.Panel(obs, "OBSERVED DEPTH", dsub),
        "residual": panels.Panel(
            rheat, "SIGNED DEPTH RESIDUAL (ours-observed): blue=near red=far", rsub),
    }


def build_panel(source_bgr, maps, report, height):
    """[ source | our depth | observed depth | residual ], renders scaled to `height`."""
    return panels.hstack_panels(panels.label_row(
        depth_panels(source_bgr, maps, report, height)))


def compute(tracking_dir, render_depth_path, mask_path, source="", view=None,
            frame_name="", conf_thr=None, hand_mask="", hand_dilate=0.0,
            scale=None):
    """Run the depth-agreement computation once. Returns (report, maps, view,
    frame_name); `report`'s `_maps` are split out into `maps`. Shared by main()
    and composite.py so the observed load + resample + scoring lives in one place.
    `scale` (explicit value, or None to auto-read the run's pose.json — see
    resolve_scale) enables depth_mae_canon.
    `conf_thr` None auto-resolves the run's depth_config.json floor (walking up
    from the render depth); pass a float to override (see resolve_conf_thr)."""
    conf_thr = resolve_conf_thr(render_depth_path, conf_thr)
    v, fname = depth_obs.resolve_view(tracking_dir, view, frame_name, source)
    obs_xyz, obs_depth, conf, keep, K_grid = depth_obs.load_view(tracking_dir, v)
    render_depth = rasters.load_render_depth(render_depth_path)
    mask = rasters.load_mask(mask_path)

    # everything compares on the RENDER grid; resample observed (its grid) up to it.
    if mask.shape != render_depth.shape:
        mask = cv2.resize(mask, (render_depth.shape[1], render_depth.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    obs_xyz, obs_depth, conf, keep, K = depth_obs.resample_to(
        render_depth.shape, obs_xyz, obs_depth, conf, keep, K_grid)

    # hand keep-region K = ¬hand at the render grid (reuse rasters.build_keep).
    hand_keep = None
    if hand_mask:
        hand = rasters.load_mask(hand_mask)
        hand_keep = rasters.build_keep(hand, render_depth.shape, hand_dilate)

    report = depth_report(render_depth, obs_xyz, obs_depth, conf, keep, K, mask,
                          conf_thr, hand_keep=hand_keep,
                          scale=resolve_scale(scale, render_depth_path))
    maps = report.pop("_maps")
    report = {"view": v, "frame_name": fname, **report}
    return report, maps, v, fname


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="depth agreement: rendered depth vs the observed pointmap")
    p.add_argument("--tracking", "--pi3x", dest="tracking", required=True,
                   help="capture tracking dir (CAPTURE/tracking; the capture "
                        "root also works). --pi3x is the old name, still accepted")
    p.add_argument("--render-depth", required=True,
                   help="depth.npy from the 'depth' view (float32, +inf bg)")
    p.add_argument("--mask", required=True, help="object mask png (interior)")
    p.add_argument("--source", default="", help="source RGB image (for the panel)")
    p.add_argument("--view", type=int, default=None,
                   help="tracking view index; else resolved from --frame-name/--source")
    p.add_argument("--frame-name", default="",
                   help="source frame filename to match a tracking view (via keyframes)")
    p.add_argument("--conf-thr", type=float, default=None,
                   help="confidence below this (or keep==False) is ignored; "
                   "default resolves the run's depth_config.json floor (backend), "
                   "else 0.1")
    p.add_argument("--hand-mask", default="", help="binary HAND (occluder) mask "
                   "png; its region is waived from depth scoring (the object "
                   "continues behind it and the sensor sees the hand, not the object)")
    p.add_argument("--hand-dilate", type=float, default=0.0,
                   help="grow the hand ignore-region by this fraction of the "
                   "longer side (matches silhouette.py; 0 = use the mask exactly)")
    p.add_argument("--scale", type=float, default=None,
                   help="the object's longest dimension in scene units (the "
                   "scene's shared uniform scalar SCALE) — enables "
                   "depth_mae_canon (RAW mae as a fraction of the object's "
                   "size). Default: read `scale` "
                   "from the run's pose.json next to --render-depth")
    p.add_argument("--out", default="", help="write JSON report here")
    p.add_argument("--panel", default="", help="write the 4-panel PNG here")
    p.add_argument("--residual-out", default="", help="also write the SIGNED "
                   "DEPTH-RESIDUAL panel (ours - observed gt; blue=render in front/too "
                   "near, red=behind/too far, white=agree, + the MAE/bias) as "
                   "its OWN png here. The individual image the visual judge reads to "
                   "see WHERE depth disagrees most AND which way to move (mirrors "
                   "composite's --overlap-out for silhouette)")
    p.add_argument("--timeline-dir", default="", help="also archive the depth panel "
                   "into <dir>/<frame>/<NNN>_depth.png (one entry per step), beside "
                   "the composite timeline; needs --frame or --frame-name")
    p.add_argument("--frame", default="", help="frame name for the --timeline-dir "
                   "subfolder (default: --frame-name)")
    p.add_argument("--tag", default="", help="timeline entry name (e.g. iter03); "
                   "default is a zero-padded auto-increment per frame")
    p.add_argument("--height", type=int, default=512)
    args = p.parse_args()

    report, maps, v, frame_name = compute(
        args.tracking, args.render_depth, args.mask, source=args.source,
        view=args.view, frame_name=args.frame_name, conf_thr=args.conf_thr,
        hand_mask=args.hand_mask, hand_dilate=args.hand_dilate,
        scale=args.scale)

    text = json.dumps(report, indent=2)
    if args.out:
        write_json(args.out, report)
    print(text)

    if args.panel or args.timeline_dir or args.residual_out:
        src = (rasters.load_bgr(args.source) if args.source
               else np.zeros((maps["render_depth"].shape[0],
                              maps["render_depth"].shape[1], 3), np.uint8))
        parts = depth_panels(src, maps, report, args.height)

        # standalone DEPTH-RESIDUAL png — the individual image the judge reads to
        # see WHERE depth disagrees most. Header-bar free, like composite's
        # --overlap-out: the numbers are in the JSON, the picture carries its own
        # NEAR/FAR key, and this is one of several images shown side by side.
        if args.residual_out:
            rp = parts["residual"].image
            os.makedirs(os.path.dirname(os.path.abspath(args.residual_out)),
                        exist_ok=True)
            cv2.imwrite(args.residual_out, rp)
            print(f"[depth] wrote {args.residual_out}  ({rp.shape[1]}x{rp.shape[0]})")

        if args.panel or args.timeline_dir:
            panel = panels.hstack_panels(panels.label_row(parts))
            if args.panel:
                os.makedirs(os.path.dirname(os.path.abspath(args.panel)),
                            exist_ok=True)
                cv2.imwrite(args.panel, panel)
                print(f"[depth] wrote {args.panel}  "
                      f"({panel.shape[1]}x{panel.shape[0]})")
            panels.timeline_save(panel, args.timeline_dir,
                                 args.frame or args.frame_name or frame_name,
                                 "depth", tag=args.tag)

    if report.get("depth_mae") is not None:
        canon = report.get("depth_mae_canon")
        head = f"canon={canon}  " if canon is not None else ""
        bias = report.get("depth_bias")
        btag = (f"bias={bias:+g} "
                f"({'BEHIND/too far' if bias > 0 else 'IN FRONT/too near'})  "
                if bias is not None else "")
        print(f"[depth] view {v} ({frame_name})  {head}{btag}"
              f"depth_mae={report['depth_mae']}  "
              f"point_l2_mae={report['point_l2_mae']}  "
              f"coverage={report['depth_coverage']}")
    else:
        print(f"[depth] view {v}: no valid pixels to score "
              f"(coverage={report['depth_coverage']})")


if __name__ == "__main__":
    main()
