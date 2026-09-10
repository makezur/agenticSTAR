"""Build sheets of source frames with the posed render and base axes overlaid.

Each tile contains exactly one image: the source frame, its opaque
``match_<frame>.png`` render, and the projected canonical base triad. X is red,
Y green, and Z blue. The triad origin is the object's base translation; its
three directions are the object's base rotation, so the glyph exposes all six
base-pose degrees of freedom without adding a second residual convention.
Shaded shafts, depth luminance, and end-on dot/cross badges provide shape cues.

The render, pose, and effective intrinsics must come from the same harness pass.
Current passes record all three in ``PASS_DIR``. A resolution mismatch fails
instead of resizing either image because a resized match render is no longer the
1:1 camera projection the overlay claims to show.

Runs from harness/::

    micromamba run -n artscript python -m analysis.viz.pose_overlay_sheet \
        --run-dir RUN_DIR --pass-dir RUN_DIR/views/0012 \
        --frames 000040.jpg,000048.jpg
"""

import argparse
import json
import os

import cv2
import numpy as np

from analysis.lib import pose_read, rasters
from analysis.lib.io import write_json
from analysis.viz import frame_sheet
from rig import lie


AXES = (
    ("X", (0, 0, 255)),
    ("Y", (0, 210, 0)),
    ("Z", (255, 0, 0)),
)
DEFAULT_OPACITY = 1.0
DEFAULT_AXIS_LENGTH = 0.35
END_ON_COMPONENT = 0.9
AXIS_SHEET_TITLE = "CIRCLE = END-ON   DOT = TOWARD   CROSS = AWAY"


def _intrinsics(frame_record, frame_name):
    intr = frame_record.get("intrinsics")
    required = ("fx", "fy", "cx", "cy", "width", "height")
    if not isinstance(intr, dict) or any(intr.get(key) is None for key in required):
        raise ValueError(
            f"frame {frame_name!r} has no complete effective intrinsics in "
            "pose.json; render a current harness pass so the axes use the exact "
            "camera that produced match_<frame>.png")
    return intr


def project_base_axes_with_depth(object_pose, scale, intrinsics,
                                 axis_length=DEFAULT_AXIS_LENGTH):
    """Project the base triad and report each positive axis's view component.

    The view component is the camera-space axis direction dotted with camera
    +Z: positive points away from the viewer and negative points toward them.
    It is the missing depth cue when a 3D axis foreshortens to a point in 2D.
    """
    length = float(axis_length)
    if length <= 0.0:
        raise ValueError("--axis-length must be positive")
    matrix = np.asarray(lie.object_pose_matrix(object_pose, scale), dtype=float)
    canonical = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [length, 0.0, 0.0, 1.0],
        [0.0, length, 0.0, 1.0],
        [0.0, 0.0, length, 1.0],
    ])
    camera = (matrix @ canonical.T).T[:, :3]
    directions = camera[1:] - camera[0]
    norms = np.linalg.norm(directions, axis=1)
    components = {
        name: float(direction[2] / max(norm, 1e-12))
        for name, direction, norm in zip(("X", "Y", "Z"), directions, norms)
    }
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    projected = {}
    for name, point in zip(("origin", "X", "Y", "Z"), camera):
        if not np.isfinite(point).all() or point[2] <= 1e-8:
            projected[name] = None
            continue
        projected[name] = (
            float(fx * point[0] / point[2] + cx),
            float(fy * point[1] / point[2] + cy),
        )
    return projected, components


def project_base_axes(object_pose, scale, intrinsics,
                      axis_length=DEFAULT_AXIS_LENGTH):
    """Project the base origin and canonical +X/+Y/+Z endpoints.

    Returns ``{"origin"|"X"|"Y"|"Z": (x, y) | None}``. A point at or behind
    the camera is ``None``. ``axis_length`` is in canonical object units and is
    transformed by the pose's shared scale with the rest of the object.
    """
    projected, _components = project_base_axes_with_depth(
        object_pose, scale, intrinsics, axis_length)
    return projected


def blend_render(source, render_bgr, render_alpha, opacity=DEFAULT_OPACITY):
    """Alpha-composite a harness render over its source image."""
    opacity = float(opacity)
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("--opacity must be between 0 and 1")
    if source.shape[:2] != render_bgr.shape[:2] \
            or source.shape[:2] != render_alpha.shape[:2]:
        raise ValueError(
            f"source/render resolution mismatch: source "
            f"{source.shape[1]}x{source.shape[0]}, render "
            f"{render_bgr.shape[1]}x{render_bgr.shape[0]}; rerender match at "
            "source resolution with --match-res")
    alpha = render_alpha.astype(np.float32)[:, :, None] * (opacity / 255.0)
    mixed = source.astype(np.float32) * (1.0 - alpha)
    mixed += render_bgr.astype(np.float32) * alpha
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def _pixel(point):
    if point is None or not np.isfinite(point).all():
        return None
    # Keep malformed/extreme projections away from OpenCV's int32 drawing path.
    if max(abs(float(point[0])), abs(float(point[1]))) > 1_000_000:
        return None
    return int(round(point[0])), int(round(point[1]))


def _color_scale(color, factor):
    return tuple(int(np.clip(channel * factor, 0, 255)) for channel in color)


def _blend_white(color, amount):
    amount = float(np.clip(amount, 0.0, 1.0))
    return tuple(
        int(round(channel + (255 - channel) * amount))
        for channel in color)


def _axis_gradient_color(color, view_component, position):
    """Relative shaft color: near shifts toward white and far darkens."""
    component = float(np.clip(view_component, -1.0, 1.0))
    position = float(np.clip(position, 0.0, 1.0))
    # Positive means this point is nearer than the shaft midpoint. Camera +Z
    # points away; only relative position matters, never camera distance.
    nearness = -2.0 * component * (position - 0.5)
    if nearness >= 0.0:
        return _blend_white(color, 0.22 * nearness)
    return _color_scale(color, 1.0 + 0.25 * nearness)


def _axis_depth_legend():
    """RGB depth-gradient strip for the compact sheet-header legend."""
    width = 180
    band_height = 5
    strip = np.empty((band_height * len(AXES), width, 3), dtype=np.uint8)
    weights = np.linspace(0.0, 1.0, width)[:, None]
    for index, (_label, color) in enumerate(AXES):
        far = np.asarray(
            _axis_gradient_color(color, 1.0, 1.0), dtype=float)
        near = np.asarray(
            _axis_gradient_color(color, 1.0, 0.0), dtype=float)
        gradient = np.rint(
            far[None, :] * (1.0 - weights) + near[None, :] * weights)
        y0 = index * band_height
        strip[y0:y0 + band_height] = gradient.astype(np.uint8)[None, :, :]
    return strip, [(0.0, "FAR + DARK"), (1.0, "NEAR + LIGHT")]


def _draw_axis_shaft(
        image, origin, endpoint, color, thickness, view_component):
    """Draw a relative depth-gradient shaft and filled arrowhead."""
    start = np.asarray(origin, dtype=float)
    end = np.asarray(endpoint, dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    endpoint_color = _axis_gradient_color(
        color, view_component, 1.0)
    if length < 2.0:
        return length, endpoint_color
    direction = delta / length
    perpendicular = np.array([-direction[1], direction[0]])
    head_length = min(max(9.0, thickness * 4.5), length * 0.42)
    head_half = max(5.0, thickness * 2.3)
    neck = end - direction * head_length

    cv2.line(
        image, tuple(np.rint(start).astype(int)),
        tuple(np.rint(neck).astype(int)), (10, 10, 10),
        thickness + 5, cv2.LINE_AA)
    shaft_delta = neck - start
    segments = max(2, min(20, int(np.ceil(np.linalg.norm(shaft_delta) / 8.0))))
    for index in range(segments):
        t0, t1 = index / segments, (index + 1) / segments
        p0 = start + shaft_delta * t0
        p1 = start + shaft_delta * t1
        segment_color = _axis_gradient_color(
            color, view_component, (t0 + t1) * 0.5)
        cv2.line(
            image, tuple(np.rint(p0).astype(int)),
            tuple(np.rint(p1).astype(int)), segment_color,
            thickness + 2, cv2.LINE_AA)

    highlight_offset = perpendicular * max(1.0, thickness * 0.35)
    for index in range(segments):
        t0, t1 = index / segments, (index + 1) / segments
        p0 = start + shaft_delta * t0 + highlight_offset
        p1 = start + shaft_delta * t1 + highlight_offset
        segment_color = _axis_gradient_color(
            color, view_component, (t0 + t1) * 0.5)
        cv2.line(
            image, tuple(np.rint(p0).astype(int)),
            tuple(np.rint(p1).astype(int)),
            _blend_white(segment_color, 0.32),
            max(1, thickness // 2), cv2.LINE_AA)

    head = np.array([
        end,
        neck + perpendicular * head_half,
        neck - perpendicular * head_half,
    ])
    outline = np.rint(head).astype(np.int32)
    cv2.fillConvexPoly(image, outline, (10, 10, 10), cv2.LINE_AA)
    inset = np.array([
        end - direction * 2.0,
        neck + direction * 2.0 + perpendicular * (head_half - 2.5),
        neck + direction * 2.0 - perpendicular * (head_half - 2.5),
    ])
    cv2.fillConvexPoly(
        image, np.rint(inset).astype(np.int32), endpoint_color, cv2.LINE_AA)
    cv2.line(
        image, tuple(np.rint(inset[0]).astype(int)),
        tuple(np.rint(inset[1]).astype(int)),
        _blend_white(endpoint_color, 0.35),
        max(1, thickness // 2), cv2.LINE_AA)
    return length, endpoint_color


def _draw_end_on_badge(image, endpoint, color, view_component, thickness):
    """Draw conventional dot-toward / cross-away notation at an endpoint."""
    center = tuple(np.rint(endpoint).astype(int))
    radius = max(7, thickness * 2 + 1)
    cv2.circle(image, center, radius + 3, (10, 10, 10), -1, cv2.LINE_AA)
    cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
    ink = (15, 15, 15)
    if view_component < 0.0:
        cv2.circle(
            image, center, max(2, radius // 3), ink, -1, cv2.LINE_AA)
    else:
        arm = max(3, int(round(radius * 0.55)))
        width = max(2, thickness)
        cv2.line(
            image, (center[0] - arm, center[1] - arm),
            (center[0] + arm, center[1] + arm), ink, width, cv2.LINE_AA)
        cv2.line(
            image, (center[0] - arm, center[1] + arm),
            (center[0] + arm, center[1] - arm), ink, width, cv2.LINE_AA)
    return radius


def _axis_thickness(image):
    """Resolution-relative base stroke using the image's shorter dimension."""
    return max(3, int(round(min(image.shape[:2]) / 160.0)))


def draw_base_axes(image, projected, view_components=None):
    """Draw a shaded, depth-aware XYZ triad onto ``image`` in place."""
    origin = _pixel(projected.get("origin"))
    if origin is None:
        return image
    view_components = view_components or {}
    h, w = image.shape[:2]
    thickness = _axis_thickness(image)
    shaft_thickness = thickness + 1
    font_scale = max(0.58, min(1.0, max(h, w) / 950.0))

    # A lit sphere makes the shared origin read as a 3D anchor. Draw it first
    # so shafts and depth badges remain legible when an axis is foreshortened.
    if 0 <= origin[0] < w and 0 <= origin[1] < h:
        radius = max(6, thickness + 4)
        cv2.circle(image, origin, radius + 3, (10, 10, 10), -1, cv2.LINE_AA)
        cv2.circle(image, origin, radius, (125, 125, 125), -1, cv2.LINE_AA)
        cv2.circle(
            image, (origin[0] - radius // 3, origin[1] - radius // 3),
            max(2, radius // 3), (250, 250, 250), -1, cv2.LINE_AA)

    # Far endpoints first, near endpoints last: their overlap now follows depth.
    ordered_axes = sorted(
        AXES, key=lambda item: view_components.get(item[0], 0.0),
        reverse=True)
    for label, color in ordered_axes:
        endpoint = _pixel(projected.get(label))
        if endpoint is None:
            continue
        view_component = float(view_components.get(label, 0.0))
        _length, endpoint_color = _draw_axis_shaft(
            image, origin, endpoint, color, shaft_thickness, view_component)
        badge_radius = 0
        if abs(view_component) >= END_ON_COMPONENT:
            badge_radius = _draw_end_on_badge(
                image, endpoint, endpoint_color, view_component, thickness)
        if 0 <= endpoint[0] < w and 0 <= endpoint[1] < h:
            label_offset = max(8, badge_radius + 5)
            text_at = (min(w - 14, max(2, endpoint[0] + label_offset)),
                       min(h - 4, max(14, endpoint[1] - 8)))
            cv2.putText(image, label, text_at, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (12, 12, 12),
                        thickness + 2, cv2.LINE_AA)
            cv2.putText(image, label, text_at, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, endpoint_color, thickness, cv2.LINE_AA)
    return image


def _match_path(pass_dir, frame_name):
    stem = os.path.splitext(os.path.basename(frame_name))[0]
    return os.path.join(pass_dir, f"match_{stem}.png")


def build_overlay(source_path, render_path, pose_json, frame_name,
                  opacity=DEFAULT_OPACITY, axis_length=DEFAULT_AXIS_LENGTH):
    """Build one native-resolution source + render + base-axis overlay."""
    frames = pose_json.get("frames") or pose_json.get("FRAMES") or {}
    if frame_name not in frames:
        raise ValueError(f"frame {frame_name!r} is absent from pass pose.json")
    object_pose, _, _ = pose_read.frame_entry(pose_json, frame_name)
    intr = _intrinsics(frames[frame_name], frame_name)
    source = rasters.load_bgr(source_path)
    render_bgr, render_alpha = rasters.load_rgba(render_path)
    expected = int(intr["width"]), int(intr["height"])
    actual = source.shape[1], source.shape[0]
    if expected != actual:
        raise ValueError(
            f"frame {frame_name!r} intrinsics are for "
            f"{expected[0]}x{expected[1]} but its source is "
            f"{actual[0]}x{actual[1]}; rerender match at source resolution "
            "with --match-res")
    overlay = blend_render(source, render_bgr, render_alpha, opacity)
    projected, view_components = project_base_axes_with_depth(
        object_pose, pose_json.get("scale", 1.0), intr, axis_length)
    return draw_base_axes(
        overlay, projected, view_components=view_components), projected


def write_pose_overlay_sheets(
        run_dir, pass_dir, out_dir="", frame_specs=(), every=1,
        opacity=DEFAULT_OPACITY, axis_length=DEFAULT_AXIS_LENGTH,
        frames_per_page=12, columns=4, tile_height=300, pack=True):
    """Write overlay assets, paginated sheets, and an enriched sheet manifest."""
    run_dir = os.path.abspath(run_dir)
    pass_dir = os.path.abspath(pass_dir)
    pose_path = os.path.join(pass_dir, "pose.json")
    if not os.path.isfile(pose_path):
        raise ValueError(f"pass has no pose.json: {pose_path}")
    pose_json = pose_read.load_pose_json(pose_path)
    frames_dir, names, _ = frame_sheet.resolve_source(run_dir=run_dir)
    selected = frame_sheet.select_frames(
        names, frames_dir, specs=frame_specs, every=every)

    pose_frames = pose_json.get("frames") or pose_json.get("FRAMES") or {}
    missing_pose = [name for name in selected if name not in pose_frames]
    if missing_pose:
        raise ValueError(
            f"pass pose.json is missing {len(missing_pose)} selected frame(s): "
            + ", ".join(missing_pose[:8]))
    # Validate document shape and camera metadata before writing any output.
    for name in selected:
        pose_read.frame_entry(pose_json, name)
        _intrinsics(pose_frames[name], name)

    missing = [
        name for name in selected
        if not os.path.isfile(_match_path(pass_dir, name))
    ]
    if missing:
        raise ValueError(
            f"pass is missing match renders for {len(missing)} selected frame(s): "
            + ", ".join(missing[:8]))

    out_dir = os.path.abspath(
        out_dir or os.path.join(run_dir, "pose_overlay_sheets"))
    overlays_dir = os.path.join(out_dir, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)

    frame_meta = {}
    for name in selected:
        source_path = os.path.join(frames_dir, name)
        render_path = _match_path(pass_dir, name)
        overlay, projected = build_overlay(
            source_path, render_path, pose_json, name,
            opacity=opacity, axis_length=axis_length)
        overlay_path = os.path.join(overlays_dir, name)
        if not cv2.imwrite(overlay_path, overlay):
            raise OSError(f"failed to write {overlay_path}")
        frame_meta[name] = {
            "source_path": source_path,
            "render_path": render_path,
            "overlay_path": overlay_path,
            "projected_axes": {
                key: (None if value is None else [float(v) for v in value])
                for key, value in projected.items()
            },
        }

    pages, manifest_path = frame_sheet.write_sheets(
        frames_dir=overlays_dir, out_dir=out_dir, frame_specs=selected,
        frames_per_page=frames_per_page, columns=columns,
        tile_height=tile_height, pack=pack,
        title=AXIS_SHEET_TITLE, stem="pose_overlay_sheet",
        ramp=_axis_depth_legend())
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["schema_version"] = 1
    manifest["kind"] = "pose_overlay_sheet"
    manifest["run_dir"] = run_dir
    manifest["pass_dir"] = pass_dir
    manifest["pose_json"] = pose_path
    manifest["opacity"] = float(opacity)
    manifest["axis_length_canon"] = float(axis_length)
    manifest["axis_colors_bgr"] = {
        label: list(color) for label, color in AXES}
    for entry in manifest["frames"]:
        entry.update(frame_meta[entry["frame_id"]])
    write_json(manifest_path, manifest)
    return pages, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="sheet any temporal frame subset with only the opaque "
                    "match render and projected object-base XYZ axes")
    parser.add_argument("--run-dir", required=True,
                        help="run whose layout.json resolves source frames")
    parser.add_argument("--pass-dir", required=True,
                        help="one coherent harness pass containing pose.json and "
                             "match_<frame>.png renders")
    parser.add_argument(
        "--frames", action="append", default=[],
        help="subset: repeatable names, comma lists, globs, or an inclusive "
             "first-last range; default is every run frame")
    parser.add_argument("--every", type=int, default=1,
                        help="keep every Nth selected frame (default: 1)")
    parser.add_argument("--opacity", type=float, default=DEFAULT_OPACITY,
                        help="render opacity over the source, 0..1 (default: 1.0)")
    parser.add_argument(
        "--axis-length", type=float, default=DEFAULT_AXIS_LENGTH,
        help="base-axis length in canonical object units (default: 0.35)")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--frames-per-page", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-height", type=int, default=300)
    parser.add_argument("--no-pack", dest="pack", action="store_false")
    args = parser.parse_args()
    try:
        write_pose_overlay_sheets(
            args.run_dir, args.pass_dir, out_dir=args.out_dir,
            frame_specs=args.frames, every=args.every, opacity=args.opacity,
            axis_length=args.axis_length,
            frames_per_page=args.frames_per_page, columns=args.columns,
            tile_height=args.tile_height, pack=args.pack)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
