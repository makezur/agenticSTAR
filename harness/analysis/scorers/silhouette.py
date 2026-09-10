"""
silhouette.py — quantitative silhouette agreement between render and source mask.

The rendered silhouette (from the render's alpha channel — the harness renders
on a transparent film) should COINCIDE with the provided object mask. This
module scores that agreement and is both a CLI and an importable library
(composite.py annotates its panel with the IoU from here). The image/mask I/O it
uses lives in `analysis.lib.rasters`.

The gate IoU.
  Every frame renders from the FIXED camera 0 (identity extrinsic + real
  intrinsics) with the object posed at true scale in front of it (the frame's
  pose + shared SCALE), so the render overlaps the photo 1:1. The gate metric is
  therefore the raw pixel IoU (`iou_raw`) — placement, scale, and proportions all
  count. IoU is silhouette-only and depth-blind, so it does NOT tell you whether a
  low score is a shape or a pose error: that call is the agent's, from the
  turntable (coherence), the VLM critic's tagged fixes, and `aspect_ratio_source`
  vs `aspect_ratio_render` (proportion) — see AGENT_TASK.md "Shape vs. pose vs.
  articulation". (`aspect_ratio` is reported as a proportion hint; there is no
  bbox-normalized "shape-only" IoU — a normalized number invited deciding shape
  from a metric instead of from the turntable + critic.)

CLI (pass --source to also get the RGB color residual):
  micromamba run -n artscript python -m analysis.scorers.silhouette \
      --mask MASK.png --render RENDER.png [--source IMAGE.png] [--out metrics.json]
Exit code 0 if raw IoU >= --pass-iou (default 0.98), else 1.
"""

import argparse
import json
import sys

import cv2
import numpy as np

from analysis.lib import rasters
from analysis.lib.io import write_json


# --------------------------------------------------------------------------- #
# normalization + IoU
# --------------------------------------------------------------------------- #
def tight_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1  # x0,y0,x1,y1


def rgb_residual(source_bgr, source_mask, render_bgr, render_mask,
                 keep_mask=None):
    """FEATURE-ALIGNED per-pixel RGB difference between source and render.

    The harness fixes the camera (camera-0 identity extrinsic + real
    intrinsics) and the agent builds the object at true scale in front of it, so
    the render overlaps the photo 1:1 at native resolution — no bbox
    normalization needed. We compare colors directly, at the source resolution,
    only where BOTH silhouettes agree (a shape mismatch is scored by the raw IoU,
    not here). The render color/mask are resized to the source resolution if they
    differ (they shouldn't when rendered with --match-res IMAGE).

    `keep_mask` (0/1, source resolution) is the non-hand keep-region K: when
    given, BOTH the overlap numerator and the union denominator are restricted to
    K, so occluded (hand) pixels neither contribute color error nor deflate
    `overlap_frac`. The source colors under the hand are the hand's, not the
    object's, so scoring them would be meaningless anyway.

    Returns a report dict plus, under "_maps", arrays for the visual panel:
      residual : float32 (H,W) mean-abs-RGB error in 0..255, 0 outside overlap
      overlap  : uint8 0/1 where both silhouettes cover the object (source res)

    Scores (over the overlap region):
      rgb_mae        : mean abs error per channel, 0..255 (lower = closer color)
      rgb_mae_norm   : rgb_mae / 255, 0..1
      rgb_rmse       : root-mean-square error, 0..255
      color_score    : 1 - rgb_mae_norm, 0..1 (higher = better; mirrors IoU sense)
      overlap_frac   : overlap pixels / union pixels (how much of the object the
                       color score actually covers — low means shapes disagree)
    """
    H, W = source_mask.shape[:2]
    if render_bgr.shape[:2] != (H, W):
        render_bgr = cv2.resize(render_bgr, (W, H), interpolation=cv2.INTER_AREA)
    if render_mask.shape[:2] != (H, W):
        render_mask = cv2.resize(render_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    s_mask = source_mask > 0
    r_mask = render_mask > 0
    if keep_mask is not None:
        keep = keep_mask > 0
        s_mask = s_mask & keep
        r_mask = r_mask & keep
    overlap = (s_mask & r_mask).astype(np.uint8)
    union = int((s_mask | r_mask).sum())

    diff = np.abs(source_bgr.astype(np.float32) - render_bgr.astype(np.float32))
    residual = diff.mean(axis=2)          # mean over channels -> (H,W)
    residual_masked = residual * overlap  # zero outside overlap for the panel

    n = int(overlap.sum())
    if n == 0:
        report = {
            "rgb_mae": None, "rgb_mae_norm": None, "rgb_rmse": None,
            "color_score": None, "overlap_frac": 0.0,
        }
    else:
        sel = overlap > 0
        mae = float(residual[sel].mean())
        rmse = float(np.sqrt((diff[sel] ** 2).mean()))
        report = {
            "rgb_mae": round(mae, 2),
            "rgb_mae_norm": round(mae / 255.0, 4),
            "rgb_rmse": round(rmse, 2),
            "color_score": round(1.0 - mae / 255.0, 4),
            "overlap_frac": round(float(n / union), 4) if union else 0.0,
        }
    report["_maps"] = {"residual": residual_masked, "overlap": overlap}
    return report


def iou(a, b):
    a = a > 0
    b = b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def aspect_ratio(mask):
    bb = tight_bbox(mask)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    h = max(1, y1 - y0)
    return float((x1 - x0) / h)  # width / height


def silhouette_iou(mask, render_mask,
                   source_bgr=None, render_bgr=None,
                   hand_mask=None, hand_dilate=0.0):
    """Full comparison report between a source mask and a render silhouette.

    If `source_bgr` and `render_bgr` are also given, the color-agreement fields
    from `rgb_residual` are merged in (an RGB residual over the silhouette
    overlap — the "what to recolor" signal complementing the shape IoU). The
    per-pixel residual arrays are NOT included here so the report stays
    JSON-serializable; call `rgb_residual` directly for the visual maps.

    Hand occlusion. If `hand_mask` (a binary occluder mask) is given, the object
    legitimately continues BEHIND the hand, so the hand region is scored as a
    "don't care": every metric is computed over the keep-region K = ¬hand only
    (see `rasters.build_keep`; `hand_dilate`, 0 by default, would widen K to
    absorb boundary noise at the cost of scoring fewer pixels).
    This adds `iou_visible` — the PRIMARY gate when a hand is present, recorded in
    `gate_field` — alongside the full-image `iou_raw` (kept as a diagnostic).
    The RGB residual and the aspect-ratio hints are likewise restricted to K so
    every signal stays fair under occlusion. SAM labels occluded object pixels as
    hand, so the object mask is visible-only and ignoring hand pixels is exactly
    "no penalty and no reward for what the render does behind the hand."
    """
    # Render silhouette aligned to the source-mask grid — every metric below
    # compares on this common grid.
    rm = cv2.resize((render_mask > 0).astype(np.uint8),
                    (mask.shape[1], mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    m = (mask > 0).astype(np.uint8)

    # Keep-region K = ¬hand at the scoring grid (None when no hand mask).
    keep = rasters.build_keep(hand_mask, m.shape, hand_dilate) if hand_mask is not None else None

    # Masked copies for the hand-ignored metrics (identity when keep is None).
    m_k = m * keep if keep is not None else m
    rm_k = rm * keep if keep is not None else rm

    ar_src = aspect_ratio(m_k * 255)
    ar_rnd = aspect_ratio(rm_k * 255)
    report = {
        # Raw 1:1 IoU over the FULL image (original masks, unaffected by the
        # hand). The gate when no hand is present; a diagnostic once a hand is
        # (iou_visible becomes the gate then). IoU is depth-blind and does not
        # distinguish a shape error from a pose error — read it beside the
        # turntable + the VLM critic + the aspect-ratio hint (below) to make that
        # call (see AGENT_TASK.md "Shape vs. pose vs. articulation").
        "iou_raw": round(iou(m, rm), 4),
        # Proportion hint: width/height of each tight silhouette bbox. A source
        # vs render mismatch is a proportion (usually shape/build()) issue that a
        # uniform SCALE can't fix. Restricted to K under a hand mask so it stays
        # fair. This is a HINT, not a shape-only score.
        "aspect_ratio_source": round(ar_src, 3) if ar_src else None,
        "aspect_ratio_render": round(ar_rnd, 3) if ar_rnd else None,
        "source_coverage": round(float((mask > 0).mean()), 4),
        "render_coverage": round(float((render_mask > 0).mean()), 4),
        "gate_field": "iou_raw",
    }
    if keep is not None:
        # PRIMARY gate under occlusion: IoU over non-hand pixels only.
        report["gate_field"] = "iou_visible"
        report["hand_coverage"] = round(float((keep == 0).mean()), 4)
        union_k = int(np.logical_or(m_k > 0, rm_k > 0).sum())
        if union_k == 0:
            # The whole scored region is ignored (object fully under the hand, or
            # K emptied by over-dilation / a garbage hand mask). iou()'s
            # empty-union convention returns 1.0 — a silent false pass — so flag
            # it and report no score instead.
            report["iou_visible"] = None
            report["visible_region_empty"] = True
        else:
            report["iou_visible"] = round(iou(m_k, rm_k), 4)
    if source_bgr is not None and render_bgr is not None:
        res = rgb_residual(source_bgr, mask, render_bgr, render_mask,
                           keep_mask=keep)
        res.pop("_maps", None)
        report.update(res)
    return report


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="silhouette IoU: render vs source mask")
    p.add_argument("--mask", required=True, help="source object mask (png)")
    p.add_argument("--render", required=True, help="render png (alpha silhouette)")
    p.add_argument("--source", default="", help="source RGB image; enables the "
                   "RGB residual (color-agreement) fields")
    p.add_argument("--out", default="", help="write JSON report here")
    p.add_argument("--hand-mask", default="", help="binary HAND (occluder) mask "
                   "png; when given, its region is scored as don't-care and "
                   "iou_visible (the gate) replaces iou_raw as the primary metric")
    p.add_argument("--hand-dilate", type=float, default=0.0,
                   help="grow the hand ignore-region by this fraction of the "
                   "longer image side before ignoring (0 = the default: use the "
                   "hand mask exactly. Dilating absorbs occlusion-edge noise but "
                   "only ever REMOVES pixels from scoring, and it eats agreement "
                   "faster than error, so it can LOWER iou_visible)")
    p.add_argument("--pass-iou", type=float, default=0.98,
                   help="gate-IoU threshold for exit-code pass (applied to "
                   "iou_visible when a hand mask is given, else iou_raw)")
    args = p.parse_args()

    mask = rasters.load_mask(args.mask)
    rmask = rasters.render_silhouette(args.render)
    src_bgr = rasters.load_bgr(args.source) if args.source else None
    # the render presented on the shared opaque backdrop, so this CLI's rgb_mae is
    # the same number composite.py reports for the same pair (see rasters.load_render)
    rnd_bgr = rasters.load_render(args.render) if args.source else None
    hand = rasters.load_mask(args.hand_mask) if args.hand_mask else None
    report = silhouette_iou(mask, rmask,
                            source_bgr=src_bgr, render_bgr=rnd_bgr,
                            hand_mask=hand, hand_dilate=args.hand_dilate)
    report["pass_iou"] = args.pass_iou
    # Gate on whichever IoU is primary (iou_visible under a hand mask, else
    # iou_raw). A None gate value (fully-occluded / empty keep-region) is a fail,
    # never a silent pass.
    gate = report.get(report["gate_field"])
    report["pass"] = gate is not None and gate >= args.pass_iou

    text = json.dumps(report, indent=2)
    if args.out:
        write_json(args.out, report)
    print(text)
    gate_str = f"{gate:.3f}" if gate is not None else "n/a (visible region empty)"
    # Only show the raw IoU separately when it isn't already the gate (i.e. under a
    # hand mask, where iou_visible is the gate and iou_raw is the diagnostic).
    raw_note = ("" if report["gate_field"] == "iou_raw"
                else f" (iou_raw={report['iou_raw']:.3f})")
    msg = (f"[silhouette] {report['gate_field']}={gate_str}{raw_note}  "
           f"PASS={report['pass']}")
    if report.get("color_score") is not None:
        msg += (f"  |  color_score={report['color_score']:.3f} "
                f"(rgb_mae={report['rgb_mae']:.1f}/255)")
    print(msg)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
