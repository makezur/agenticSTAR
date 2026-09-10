# `pose_pair_sheet.py` - standalone pose-pair diagnostics

This command renders the committed canonical object for each selected frame,
overlays that render opaquely and draws the object-base XYZ triad, then writes one
sheet for every adjacent temporal pair.

```bash
PYTHONPATH=harness python -m analysis.viz.pose_pair_sheet \
  --run-dir RUN_DIR --step FRAME_A:FRAME_B
```

It needs `RUN_DIR/mesh/object.glb`, `RUN_DIR/mesh/pose.json`, and a resolvable
`layout.json`. Use `--object-glb` or `--pose-json` to override those defaults.
Use `--frames-dir` when an archived layout points to a capture location that
has moved.
`--step A:B` selects one pair, matching `gap_sheet`. `--frames` accepts the same
names, comma lists, ranges, and globs as
`frame_sheet`; `--every N` subsamples before pairing.

For a bookkeeping-managed run, the active iteration owns the output at
`RUN_DIR/iterations/NNNNNN/pose_pair_sheets/`. The command requires an open
iteration and rejects `--out-dir` paths outside it. For an unmanaged standalone
run, the default remains `RUN_DIR/pose_pair_sheets/`.

The command writes:

- `renders/<fingerprint>/`: one transparent object render per unique frame.
  The fingerprint hashes the complete GLB and pose document, so stale renders
  are not reused.
- `overlays/`: one source + render + axis image per frame.
- `pairs/NNNN_first_second/`: one two-frame sheet for each adjacent pair.
- `pose_pair_manifest.json`: source paths, hashes, selected frames, projected
  axes, and every generated sheet.

Rendering uses the shared canonical GLB loader, pose/FK implementation, and
software z-buffer.

The triad keeps fixed red `X`, green `Y`, and blue `Z` identities. Each shaft
uses a relative depth gradient: its nearer end is brighter and its farther end
is darker, independent of the object's camera distance. An axis within 25.8
degrees of the view direction gets conventional end-on notation: a dot points
toward the viewer and a cross points away.

The sheet header replaces the generic title with that encoding: `CIRCLE =
END-ON`, `DOT = TOWARD`, `CROSS = AWAY`, and an RGB strip from `FAR + DARK` to
`NEAR + LIGHT`.
