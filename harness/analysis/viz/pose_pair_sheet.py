"""Standalone render + adjacent-pair sheets for committed object poses.

The command consumes a run's canonical ``mesh/object.glb`` and
``mesh/pose.json`` directly.  It software-renders every selected frame once,
draws the opaque render and object-base XYZ axes once, then sheets every
adjacent pair.  Render cache keys include the complete GLB and pose documents,
so an old render can never be mistaken for a fresh one.

Runs from the repository root::

    PYTHONPATH=harness python -m analysis.viz.pose_pair_sheet \
        --run-dir RUN_DIR --out-dir OUT_DIR
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np

from analysis.lib import pose_read, raster
from analysis.lib.glb import load_parts
from analysis.lib.io import write_json
from analysis.viz import frame_sheet
from analysis.viz.pose_overlay_sheet import (
    AXIS_SHEET_TITLE,
    DEFAULT_AXIS_LENGTH,
    DEFAULT_OPACITY,
    _axis_depth_legend,
    build_overlay,
)
from bookkeeping import ledger as bookkeeping
from core import filehash, joints as joints_core
from rig import fk, lie

RENDER_SCHEMA = 1


def _defaults(run_dir, pose_json="", object_glb=""):
    run_dir = os.path.abspath(run_dir)
    pose_json = os.path.abspath(
        pose_json or os.path.join(run_dir, "mesh", "pose.json"))
    object_glb = os.path.abspath(
        object_glb or os.path.join(run_dir, "mesh", "object.glb"))
    for label, path in (("pose JSON", pose_json), ("object GLB", object_glb)):
        if not os.path.isfile(path):
            raise ValueError(f"{label} does not exist: {path}")
    return run_dir, pose_json, object_glb


def render_fingerprint(pose_json, object_glb):
    """Content identity for renders; paths and mtimes intentionally do not count."""
    payload = json.dumps({
        "schema": RENDER_SCHEMA,
        "pose_sha1": filehash.file_sha1(pose_json),
        "object_glb_sha1": filehash.file_sha1(object_glb),
    }, sort_keys=True).encode("utf-8")
    return filehash.bytes_sha1(payload)[:16]


def _part_color(mesh):
    material = getattr(getattr(mesh, "visual", None), "material", None)
    color = getattr(material, "main_color", None)
    if color is None or len(color) < 3:
        return np.array([145, 160, 168], dtype=np.float64)
    return np.asarray(color[:3], dtype=np.float64)


def _child_transforms(joint_defs, states):
    accumulated = fk.accumulate_canonical(joint_defs, states)
    transforms = {}
    for joint in joint_defs:
        for child in fk.child_names(joint):
            transforms[child] = accumulated[joint["name"]]
    return transforms


def render_frame(parts, pose_doc, frame_name):
    """Render one posed canonical GLB to transparent BGRA."""
    frames = pose_doc.get("frames") or pose_doc.get("FRAMES") or {}
    record = frames.get(frame_name)
    if record is None:
        raise ValueError(f"frame {frame_name!r} is absent from pose.json")
    intr = record.get("intrinsics") or {}
    required = ("fx", "fy", "cx", "cy", "width", "height")
    if any(intr.get(key) is None for key in required):
        raise ValueError(f"frame {frame_name!r} has incomplete intrinsics")
    width, height = int(intr["width"]), int(intr["height"])
    object_pose, states, _ = pose_read.frame_entry(pose_doc, frame_name)
    base = np.asarray(
        lie.object_pose_matrix(object_pose, pose_doc.get("scale")), dtype=float)
    joint_defs = pose_doc.get("joint_defs") or []
    joints_core.require_complete_states(
        joint_defs, states, f"pose-pair render frame {frame_name!r}")
    child_tf = _child_transforms(joint_defs, states)

    entities = []
    triangle_colors = []
    for seg_id, (name, mesh) in enumerate(parts.items(), 1):
        if seg_id > 255:
            raise ValueError("software renderer supports at most 255 GLB parts")
        local = child_tf.get(name, np.eye(4))
        vertices = np.column_stack(
            [np.asarray(mesh.vertices, dtype=float), np.ones(len(mesh.vertices))])
        camera = (base @ local @ vertices.T).T[:, :3]
        z = camera[:, 2]
        uv = np.column_stack([
            float(intr["fx"]) * camera[:, 0] / z + float(intr["cx"]),
            float(intr["fy"]) * camera[:, 1] / z + float(intr["cy"]),
        ])
        faces = np.asarray(mesh.faces, dtype=np.int64)
        valid = raster.valid_faces(uv, z, faces, width, height)
        if len(valid):
            edges_a = camera[valid[:, 1]] - camera[valid[:, 0]]
            edges_b = camera[valid[:, 2]] - camera[valid[:, 0]]
            normals = np.cross(edges_a, edges_b)
            norms = np.linalg.norm(normals, axis=1)
            normals /= np.maximum(norms[:, None], 1e-12)
            light = np.array([0.35, -0.45, -0.82])
            light /= np.linalg.norm(light)
            shade = 0.48 + 0.52 * np.abs(normals @ light)
            rgb = np.clip(
                _part_color(mesh)[None, :] * shade[:, None], 0, 255)
            triangle_colors.append(rgb[:, ::-1].astype(np.uint8))
        entities.append((seg_id, uv, z, faces))

    seg, _depth, tri = raster.rasterize(entities, width, height)
    image = np.zeros((height, width, 4), dtype=np.uint8)
    hit = seg > 0
    if hit.any():
        colors = np.concatenate(triangle_colors) if triangle_colors else \
            np.zeros((0, 3), dtype=np.uint8)
        image[hit, :3] = colors[tri[hit]]
        image[hit, 3] = 255
    return image


def ensure_renders(pose_json, object_glb, frames, cache_root, force=False):
    """Return a complete content-addressed render directory."""
    fingerprint = render_fingerprint(pose_json, object_glb)
    render_dir = os.path.join(cache_root, fingerprint)
    manifest_path = os.path.join(render_dir, "render_manifest.json")
    expected = [
        os.path.join(render_dir, f"match_{os.path.splitext(name)[0]}.png")
        for name in frames
    ]
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if not force and manifest.get("fingerprint") == fingerprint \
                and all(os.path.isfile(p) for p in expected):
            return render_dir, manifest, True

    os.makedirs(render_dir, exist_ok=True)
    missing = list(zip(frames, expected)) if force else [
        (name, path) for name, path in zip(frames, expected)
        if not os.path.isfile(path)]
    if missing:
        parts = load_parts(object_glb)
        if not parts:
            raise ValueError(
                f"object GLB contains no renderable parts: {object_glb}")
        with open(pose_json) as f:
            pose_doc = json.load(f)
    for index, (name, path) in enumerate(missing, 1):
        image = render_frame(parts, pose_doc, name)
        if not cv2.imwrite(path, image):
            raise OSError(f"failed to write {path}")
        print(
            f"[pose-pairs] rendered {index}/{len(missing)} {name}", flush=True)
    shutil.copy2(pose_json, os.path.join(render_dir, "pose.json"))
    prior_frames = manifest.get("frames") or []
    cached_frames = list(dict.fromkeys([*prior_frames, *frames]))
    manifest = {
        "schema_version": RENDER_SCHEMA,
        "fingerprint": fingerprint,
        "pose_json": pose_json,
        "pose_sha1": filehash.file_sha1(pose_json),
        "object_glb": object_glb,
        "object_glb_sha1": filehash.file_sha1(object_glb),
        "frames": cached_frames,
    }
    write_json(manifest_path, manifest)
    return render_dir, manifest, False


def write_pair_sheets(
        run_dir, out_dir="", pose_json="", object_glb="", frames_dir="",
        frame_specs=(), every=1, opacity=DEFAULT_OPACITY,
        axis_length=DEFAULT_AXIS_LENGTH, tile_height=300, force_render=False):
    """Render selected frames once and write one sheet per adjacent pair."""
    run_dir, pose_json, object_glb = _defaults(
        run_dir, pose_json, object_glb)
    frames_dir, names, _ = frame_sheet.resolve_source(
        run_dir=run_dir, frames_dir=frames_dir)
    selected = frame_sheet.select_frames(
        names, frames_dir, specs=frame_specs, every=every)
    if len(selected) < 2:
        raise ValueError("pair sheets require at least two selected frames")
    out_dir = bookkeeping.artifact_dir(
        run_dir, "pose_pair_sheets", requested=out_dir)
    renders_root = os.path.join(out_dir, "renders")
    render_dir, render_manifest, cache_hit = ensure_renders(
        pose_json, object_glb, selected, renders_root, force=force_render)

    overlays_dir = os.path.join(out_dir, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)
    pose_doc = pose_read.load_pose_json(pose_json)
    overlay_meta = {}
    for index, name in enumerate(selected, 1):
        overlay, projected = build_overlay(
            os.path.join(frames_dir, name),
            os.path.join(
                render_dir, f"match_{os.path.splitext(name)[0]}.png"),
            pose_doc, name, opacity=opacity, axis_length=axis_length)
        overlay_path = os.path.join(overlays_dir, name)
        if not cv2.imwrite(overlay_path, overlay):
            raise OSError(f"failed to write {overlay_path}")
        overlay_meta[name] = {
            key: None if point is None else [float(v) for v in point]
            for key, point in projected.items()
        }
        print(f"[pose-pairs] overlay {index}/{len(selected)} {name}", flush=True)

    pairs_root = os.path.join(out_dir, "pairs")
    pair_records = []
    for index, (first, second) in enumerate(zip(selected, selected[1:])):
        first_stem = os.path.splitext(first)[0]
        second_stem = os.path.splitext(second)[0]
        pair_dir = os.path.join(
            pairs_root, f"{index:04d}_{first_stem}_{second_stem}")
        pages, manifest_path = frame_sheet.write_sheets(
            frames_dir=overlays_dir, out_dir=pair_dir,
            frame_specs=[first, second], frames_per_page=2, columns=2,
            tile_height=tile_height, pack=True,
            title=AXIS_SHEET_TITLE, stem="pose_overlay_sheet",
            ramp=_axis_depth_legend())
        pair_records.append({
            "index": index,
            "frames": [first, second],
            "sheet": pages[0],
            "manifest": manifest_path,
        })
        print(
            f"[pose-pairs] pair {index + 1}/{len(selected) - 1} "
            f"{first} -> {second}", flush=True)

    manifest = {
        "schema_version": 1,
        "kind": "pose_pair_sheets",
        "run_dir": run_dir,
        "frames_dir": frames_dir,
        "selected_frames": selected,
        "pair_count": len(pair_records),
        "pairs": pair_records,
        "opacity": float(opacity),
        "axis_length_canon": float(axis_length),
        "render_cache_hit": cache_hit,
        "render": render_manifest,
        "projected_axes": overlay_meta,
    }
    manifest_path = os.path.join(out_dir, "pose_pair_manifest.json")
    write_json(manifest_path, manifest)
    return pair_records, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="render committed poses and sheet every adjacent frame pair")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pose-json", default="",
                        help="default: RUN_DIR/mesh/pose.json")
    parser.add_argument("--object-glb", default="",
                        help="default: RUN_DIR/mesh/object.glb")
    parser.add_argument("--out-dir", default="",
                        help="default: active iteration's pose_pair_sheets/ "
                             "for managed runs; RUN_DIR/pose_pair_sheets "
                             "otherwise")
    parser.add_argument(
        "--frames-dir", default="",
        help="source frames override for a run whose layout path moved")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--frames", action="append", default=[],
        help="repeatable names, comma lists, globs, or ranges")
    selection.add_argument(
        "--step", metavar="A:B",
        help="one temporal pair, using the same syntax as gap_sheet")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--opacity", type=float, default=DEFAULT_OPACITY)
    parser.add_argument("--axis-length", type=float, default=DEFAULT_AXIS_LENGTH)
    parser.add_argument("--tile-height", type=int, default=300)
    parser.add_argument(
        "--force-render", action="store_true",
        help="rerender every selected frame even when its content cache is complete")
    args = parser.parse_args()
    try:
        frame_specs = args.frames
        if args.step:
            _frames_dir, names, _ = frame_sheet.resolve_source(
                run_dir=args.run_dir, frames_dir=args.frames_dir)
            frame_specs = list(frame_sheet.parse_steps([args.step], names)[0])
        _pairs, manifest = write_pair_sheets(
            args.run_dir, out_dir=args.out_dir, pose_json=args.pose_json,
            object_glb=args.object_glb, frames_dir=args.frames_dir,
            frame_specs=frame_specs,
            every=args.every, opacity=args.opacity,
            axis_length=args.axis_length, tile_height=args.tile_height,
            force_render=args.force_render)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"[pose-pairs] manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
