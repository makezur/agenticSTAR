"""camera.py — bounding box, camera placement, intrinsics, Pi3X cameras.

The 'match' camera is always at the world origin with an identity extrinsic in
the OpenCV/SfM convention (+X right, -Y up, +Z into the scene) — the fixed
camera-0 gauge. We render every frame from this one camera and MOVE THE OBJECT
into each frame's camera (M_k = T_{k<-ref} @ M_ref); only the per-frame
INTRINSICS change (Pi3X K[k], or an iPhone-13 placeholder when no Pi3X). This
module owns the camera itself (framing orbit, intrinsics) and loads the per-frame
Pi3X cameras (extrinsics + intrinsics) that drive the object-move.
"""

import json
import math
import os

import bpy
from mathutils import Matrix, Vector

from core import cam_math


# --------------------------------------------------------------------------- #
# bounding box
# --------------------------------------------------------------------------- #
def scene_aabb():
    """World-space (mins, maxs) of all mesh objects, as Vectors.

    The raw box, exposed because a caller accumulating bounds over SEVERAL poses
    (views/mechanism.py, framing one camera for a whole joint sweep) must union
    boxes, not spheres: re-boxing a bounding sphere and re-sphering that box
    inflates the radius by up to sqrt(3) per round trip, which shrinks the object
    to a corner of the frame. Returns None when the scene holds no mesh."""
    mins = Vector((math.inf,) * 3)
    maxs = Vector((-math.inf,) * 3)
    found = False
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        found = True
        for corner in obj.bound_box:
            wc = obj.matrix_world @ Vector(corner)
            mins = Vector(map(min, mins, wc))
            maxs = Vector(map(max, maxs, wc))
    return (mins, maxs) if found else None


def bbox_to_sphere(mins, maxs):
    """(center, radius) of the bounding SPHERE of an AABB — the framing pair
    `place_camera` takes. The ONE place that conversion is spelled."""
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).length / 2.0, 1e-4)
    return center, radius


def scene_bbox():
    """World-space (center, radius) of all mesh objects. radius = bounding sphere."""
    box = scene_aabb()
    if box is None:
        return Vector((0, 0, 0)), 1.0
    return bbox_to_sphere(*box)


def make_camera(fov_deg, ortho):
    cam_data = bpy.data.cameras.new("rig_cam")
    if ortho:
        cam_data.type = "ORTHO"
    else:
        cam_data.lens_unit = "FOV"
        cam_data.angle = math.radians(fov_deg)
    cam_obj = bpy.data.objects.new("rig_cam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


# In the camera-0 (OpenCV) world frame, image-up is world -Y and the object
# sits in front of the camera along +Z. The turntable and lighting orbit around
# this frame's vertical axis (-Y up) so views/lights read right-side up.
CAM0_UP = Vector((0.0, -1.0, 0.0))  # world direction that appears "up" in-frame


def orbit_offset(dist, azimuth_deg, elevation_deg):
    """Offset (in the camera-0 frame) of a viewpoint at the given azimuth/
    elevation, distance `dist` from the target. Vertical axis is -Y (CAM0_UP);
    azimuth=0/elevation=0 is the camera-0 front view (on the -Z side, looking
    +Z), azimuth sweeps around the -Y axis, positive elevation lifts toward -Y.
    """
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return Vector((
        dist * math.cos(el) * math.sin(az),   # +X = right
        -dist * math.sin(el),                 # up is -Y -> lift toward -Y
        -dist * math.cos(el) * math.cos(az),  # az=0 -> -Z (camera-0 side)
    ))


def aim_camera(cam_obj, loc, target, up=CAM0_UP, ortho=False, ortho_scale=None):
    """Point the camera at `target` from `loc` with a chosen world up-vector.

    Built manually (not to_track_quat, which hardcodes world +Z as up) so the
    turntable respects the camera-0 frame's -Y up."""
    fwd = (target - loc)
    if fwd.length < 1e-9:
        fwd = Vector((0.0, 0.0, 1.0))
    fwd.normalize()
    zc = -fwd                       # Blender camera looks down local -Z
    xc = up.cross(zc)
    if xc.length < 1e-9:            # up parallel to view dir; pick any right
        xc = Vector((1.0, 0.0, 0.0))
    xc.normalize()
    yc = zc.cross(xc)               # local +Y (image up)
    rot = Matrix((xc, yc, zc)).transposed().to_4x4()  # columns = basis axes
    cam_obj.matrix_world = Matrix.Translation(loc) @ rot
    if ortho and ortho_scale is not None:
        cam_obj.data.ortho_scale = ortho_scale


def place_camera(cam_obj, center, radius, azimuth, elevation, cam_radius, ortho):
    """Position camera around `center` (camera-0 frame, -Y up) and aim at it.

    'auto' distance fits the bounding sphere in BOTH image axes at the camera's
    own FOV and the scene's current render resolution (cam_math.fit_distance) —
    a flat `radius * k` cannot, because k would have to depend on the aspect
    ratio and the FOV."""
    r = bpy.context.scene.render
    if cam_radius == "auto" or cam_radius is None:
        fov = math.degrees(cam_obj.data.angle)
        dist = cam_math.fit_distance(radius, fov, r.resolution_x, r.resolution_y)
    else:
        dist = float(cam_radius)
    loc = center + orbit_offset(dist, azimuth, elevation)
    aim_camera(cam_obj, loc, center, up=CAM0_UP, ortho=ortho,
               ortho_scale=cam_math.fit_ortho_scale(
                   radius, r.resolution_x, r.resolution_y))


# --------------------------------------------------------------------------- #
# explicit camera pose + intrinsics (for known / measured / video cameras)
# --------------------------------------------------------------------------- #
# The object is built at the world origin, so "pose wrt object" == pose in the
# world frame. We accept the pose in common external conventions and convert to
# Blender's (camera looks down -Z, +Y up).
#
# opencv  : camera +X right, +Y down, +Z forward   (COLMAP / OpenCV / most SfM)
# opengl  : camera +X right, +Y up,   -Z forward    (== Blender)
# blender : alias of opengl
_CV_TO_BLENDER = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))  # flip Y and Z axes


def parse_pose(spec):
    """Accept a 4x4 (list of rows), a flat list/str of 16 numbers -> Matrix."""
    if isinstance(spec, str):
        nums = [float(x) for x in spec.replace(";", ",").split(",") if x.strip() != ""]
        if len(nums) != 16:
            raise ValueError(f"--pose needs 16 numbers, got {len(nums)}")
        rows = [nums[0:4], nums[4:8], nums[8:12], nums[12:16]]
    elif len(spec) == 16:  # flat sequence
        s = [float(x) for x in spec]
        rows = [s[0:4], s[4:8], s[8:12], s[12:16]]
    else:  # 4x4 nested
        rows = [[float(x) for x in row] for row in spec]
        if len(rows) != 4 or any(len(r) != 4 for r in rows):
            raise ValueError("pose must be a 4x4 matrix")
    return Matrix(rows)


def pose_to_blender_c2w(mat, pose_type="c2w", convention="opencv"):
    """Normalize any accepted pose to a Blender camera-to-world matrix."""
    c2w = mat if pose_type == "c2w" else mat.inverted()
    if convention == "opencv":
        c2w = c2w @ _CV_TO_BLENDER  # re-express camera axes in Blender's frame
    return c2w


def _view_fac_in_px(cam_data, pax, pay, width, height):
    """Blender's 'view factor' in px (mirrors BlenderProc). Depends on the
    effective sensor fit — the axis the sensor size is measured along."""
    fit = cam_data.sensor_fit
    if fit == "AUTO":
        fit = "HORIZONTAL" if pax * width >= pay * height else "VERTICAL"
    if fit == "HORIZONTAL":
        return float(width)
    return (pay / pax) * height  # VERTICAL


def apply_K(cam_data, fx, fy, cx, cy, width, height):
    """Configure a perspective camera to a pinhole K stated at (width, height).

    Faithful port of BlenderProc's set_intrinsics_from_K_matrix: fx!=fy is
    encoded via render pixel_aspect (kept >= 1, which Blender requires); the
    principal point via lens shift normalized by the correct view factor. No
    skew (Blender can't represent it).

    RESOLUTION-AGNOSTIC: this configures lens/shift/pixel_aspect for the given
    (width, height) but DOES NOT touch scene.render.resolution — the caller owns
    resolution and must have set it to (width, height) already. Pass a K that
    was re-absolutized to the resolution actually being rendered (see
    NormIntrinsics.at). Verify with --debug-project.
    """
    width, height = int(width), int(height)
    cam_data.type = "PERSP"

    # fx != fy -> pixel aspect (Blender clamps these to >= 1, so put the >1
    # factor on whichever axis needs it)
    pax = pay = 1.0
    if fx > fy:
        pay = fx / fy
    elif fx < fy:
        pax = fy / fx
    pixel_aspect_ratio = pay / pax

    view_fac = _view_fac_in_px(cam_data, pax, pay, width, height)
    sensor = cam_data.sensor_width if cam_data.sensor_fit != "VERTICAL" else cam_data.sensor_height

    cam_data.lens = fx * sensor / view_fac
    cam_data.shift_x = (cx - (width - 1) / 2.0) / -view_fac
    cam_data.shift_y = (cy - (height - 1) / 2.0) / view_fac * pixel_aspect_ratio

    scene = bpy.context.scene
    scene.render.pixel_aspect_x = pax
    scene.render.pixel_aspect_y = pay
    print(f"[render_wrapper] intrinsics fx={fx} fy={fy} cx={cx} cy={cy} "
          f"{width}x{height} -> lens={cam_data.lens:.4f}mm "
          f"shift=({cam_data.shift_x:.4f},{cam_data.shift_y:.4f}) "
          f"pixel_aspect=({pax:.4f},{pay:.4f})")


def apply_intrinsics(cam_data, intr):
    """Apply a pixel-space intrinsics dict {fx,fy,cx,cy,width,height} — the
    caller must have set scene.render.resolution to (width,height) first."""
    apply_K(cam_data, intr["fx"], intr["fy"], intr["cx"], intr["cy"],
            intr["width"], intr["height"])


# pure-numpy K parser lives in core.cam_math; re-exported for existing callers.
parse_intrinsics_str = cam_math.parse_intrinsics_str


# iPhone-13-wide PLACEHOLDER intrinsics — the pinhole model + the constant live
# in core.cam_math (pure numpy, resolution-synthesized). Re-exported here for the
# existing rig-side callers.
IPHONE13_LONGSIDE_FOV_DEG = cam_math.IPHONE13_LONGSIDE_FOV_DEG
iphone13_intrinsics = cam_math.iphone13_intrinsics


def identity_c2w():
    """The fixed camera-0 extrinsic: identity in OpenCV, re-expressed in Blender
    axes. Every frame renders from this camera (the object is moved instead)."""
    return pose_to_blender_c2w(Matrix.Identity(4), pose_type="c2w",
                               convention="opencv")


def resolve_camera(args, frame_intr=None):
    """The 'match' camera spec for a frame: identity extrinsic (camera 0) + the
    frame's intrinsics (a resolution-agnostic NormIntrinsics if known, else the
    iPhone-13 placeholder synthesized at render resolution).

    `frame_intr` is a cam_math.NormIntrinsics (explicit K or Pi3X K[k]) or None.
    Returns {'c2w': Matrix, 'intr': NormIntrinsics|None, 'intr_default': bool}.
    When 'intr' is None and 'intr_default' is True, place_match_camera
    synthesizes the iPhone-13 K from the current render resolution."""
    intr = frame_intr
    return {"c2w": identity_c2w(), "intr": intr, "intr_default": intr is None}


# --------------------------------------------------------------------------- #
# Pi3X per-frame cameras (extrinsics + intrinsics) — the object-move source
# --------------------------------------------------------------------------- #
def load_tracking_cameras(tracking_dir):
    """Load a capture tracking dir's cameras.npz -> {'c2w': (K,4,4),
    'intrinsics': (K,3,3), 'size': (W, H)}. Delegates to the shared capture
    reader (numpy-only — datasets.common.capture works inside Blender); the
    local name is kept for the existing rig-side callers."""
    from core import captures
    return captures.load_cameras(tracking_dir)


def relative_extrinsic(c2w, ref, k):
    """The rigid point-transform carrying the object from the reference camera
    frame into view k's: inv(c2w[k]) @ c2w[ref] (OpenCV). ref==k -> identity."""
    import numpy as np
    rel = np.linalg.inv(c2w[k]) @ c2w[ref]
    return Matrix([[float(v) for v in row] for row in rel])


def norm_intrinsics(K, obs_wh):
    """Normalize a Pi3X 3x3 K (stated on its (Wp, Hp) grid) into a resolution-
    agnostic cam_math.NormIntrinsics — re-absolutized to the render grid at apply
    time, so the sweep's low-res scoring raster is honored (no resolution clobber)."""
    Wp, Hp = obs_wh
    return cam_math.NormIntrinsics.from_K(K, Wp, Hp)


def debug_project(cam_obj, point_str):
    """Print where a world point lands in pixels for the current camera."""
    from bpy_extras.object_utils import world_to_camera_view
    scene = bpy.context.scene
    x, y, z = [float(v) for v in point_str.split(",")]
    ndc = world_to_camera_view(scene, cam_obj, Vector((x, y, z)))
    W = scene.render.resolution_x
    H = scene.render.resolution_y
    px = ndc.x * W
    py = (1.0 - ndc.y) * H  # image y-down
    print(f"[debug-project] world({x},{y},{z}) -> pixel({px:.2f},{py:.2f}) "
          f"depth={ndc.z:.4f} {'(in front)' if ndc.z > 0 else '(BEHIND camera)'}")


# --------------------------------------------------------------------------- #
# overscan intrinsics — shared by the crop and visibility views
# --------------------------------------------------------------------------- #
def effective_intrinsics(cam_obj, place_result, scene):
    """The intrinsics dict for a posed match camera, with a FOV fallback.

    `place_result` is what render.place_match_camera returned. Normally it carries
    an 'intr_effective' dict; when the camera was placed from an explicit pose with
    no intrinsics, synthesize an intrinsics dict from the camera's FOV + current
    render resolution (focal from the angle, centered principal point). Shared by
    the crop and visibility views (both need real pixel intrinsics to overscan)."""
    intr = place_result.get("intr_effective")
    if intr is not None:
        return intr
    W0, H0 = scene.render.resolution_x, scene.render.resolution_y
    f = cam_math.focal_from_fov(math.degrees(cam_obj.data.angle), max(W0, H0))
    return {"fx": f, "fy": f, "cx": (W0 - 1) / 2.0, "cy": (H0 - 1) / 2.0,
            "width": W0, "height": H0}


# overscan widening is pure px math — lives in core.cam_math, re-exported here
# for the crop and visibility views.
overscan_intrinsics = cam_math.overscan_intrinsics
