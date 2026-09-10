"""rig — the fixed Blender-side reconstruction rig (agent never edits this).

Shared infrastructure:
  args      — CLI parsing
  scene     — clear / exec scene.py build() / pose apply+unapply / pose.json
  camera    — bbox, camera placement, intrinsics, explicit-pose handling
  lighting  — neutral 3-light rig + fallback material
  render    — resolution, engine config, the render call, match-camera + Context
  imaging   — pixel helpers shared by the ids/crop/visibility views
  export    — voxel remesh + GLB export
  transforms— pose math (compose/apply/orient + reserved articulation hook)

The per-view render code lives in the sibling `views` package.
"""
