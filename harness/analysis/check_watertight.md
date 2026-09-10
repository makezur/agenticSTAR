# `check_watertight.py` — the geometry gate

Per-part watertight/manifold check on the exported GLB (via trimesh). The
exported GLB is the **canonical (rest-pose)** mesh — pose is a rigid+scale
transform and doesn't affect manifoldness, so checking the canonical geometry is
correct.

## Run (artscript env)
```bash
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.check_watertight \
    RUN_DIR/mesh/object.glb --out RUN_DIR/mesh/watertight_report.json
```
Exit code `0` and `"pass": true` means every part is watertight.
Parts are keyed by **scene-graph node name** — the `scene.py` part names that
`JOINTS` and the self-intersection report speak, not the exporter's datablock
names (`Cube.004`) — and `bounds` reads in **canonical (+Z up) axes**: the
ingest is the shared `analysis.lib.glb.load_parts`, which owns the glTF-Y-up →
canonical seam.
It does **not** certify collision readiness, clearance through a joint's full
travel, or freedom from self-collision; those require downstream collision
geometry and validation.

## If `"pass": false`
1. First fix it in `scene.py` — usually a boolean left a non-manifold edge; make
   cutters fully protrude through surfaces, or keep the part as separate manifold
   primitives instead of one boolean.
2. If a part is inherently hard to keep manifold, re-render with a voxel remesh
   to force a watertight surface, then re-check:
   ```bash
   harness/render.sh RUN_DIR/scene.py RUN_DIR/views \
       --views match --match-res IMAGE \
       --export RUN_DIR/mesh/object.glb --remesh voxel:0.01
   ```
   (Smaller voxel size = finer detail. Remesh fuses parts, so prefer the
   `scene.py` fix when part separation matters.)
