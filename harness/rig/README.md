# `rig/` — the fixed Blender-side rig (agent never edits this)

Shared infrastructure that `harness/render_wrapper.py` (a thin dispatcher) and
the per-view modules in `../views/` build on. It runs **inside Blender**
(`import bpy`). The agent authors `scene.py`; it does **not** touch anything
here.

| module | owns |
|---|---|
| `args.py` | CLI parsing (`parse_args`). Each flag's `help=` is the ground truth for that flag. |
| `scene.py` | clear scene, exec the agent's `scene.py` `build()` → a `SceneSpec` (SCALE/REFERENCE_FRAME/JOINTS/FRAMES), `pose_frame` (per-frame base pose + FK), snapshot/restore, write `pose.json`. |
| `camera.py` | bbox, camera creation + framing orbit, pinhole intrinsics, the fixed camera-0 extrinsic, Pi3X per-frame cameras (`load_tracking_cameras`/`relative_extrinsic`/`norm_intrinsics`), overscan intrinsics, `--debug-project`. |
| `lighting.py` | neutral 3-light rig + fallback material for unmateraled parts. |
| `render.py` | resolution + engine config, the render call, match-camera placement (per-frame intrinsics), the `Context` bundle (carries the current `frame`, `spec`, `canonical`), `requested_views()`. |
| `imaging.py` | numpy pixel helpers shared by ids/crop/visibility: PNG load, box draw, ID palette + emission materials + color-management state, nearest-palette part assignment. |
| `export.py` | optional voxel remesh + canonical GLB export. |
| `lie.py` | **pure-numpy** SO(3)/Sim(3) core (no `bpy`/`mathutils`/`scipy`, so it imports inside Blender AND the analysis env): quaternion algebra, exp/log maps, `Pose` (s,q,t) compose/inverse/apply, and the sweep's camera-frame "order" increments (`roll`/`yaw`/`pitch` about the object centre, `scale_delta`, pixel-aligned `pixel_translation_delta`, `apply_order`). |
| `transforms.py` | Blender-facing pose math over `lie` (numpy↔`mathutils` at the boundary): `compose_pose` (quaternion OR legacy euler), `compose_placement_with_scale`, `rotation_matrix_of`, `reseat_pose` (camera-motion seed, emits quaternion), `frame_pose_to_dict` (pose.json: `quaternion`+`translation`+`scale`; no matrix echo), and live articulation FK (`forward_kinematics`/`joint_transform`, with parent-chain + child-group support). |

## Boundaries a refactor must keep
- `transforms.py` is a **library** imported inside Blender; it stays in this
  package next to its importers (`scene.py`, `../views/sweeps/sweep.py`).
- `render_wrapper.py` inserts its own dir on `sys.path`, so `import rig` /
  `import views` resolve as packages. Keep it in `harness/`.
