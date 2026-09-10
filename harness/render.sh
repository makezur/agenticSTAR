#!/usr/bin/env bash
# render.sh — thin wrapper around the Blender binary for headless rendering.
#
# Usage:
#   render.sh <scene.py> <out_dir> [extra render_wrapper.py args...]
#
# <out_dir> is the RENDERS ROOT. For a bookkeeping-managed run it must be the
# active iteration's renders/ directory; use `python -m bookkeeping run`.
# Unmanaged standalone calls may choose any output root. Each invocation
# writes its outputs into a fresh numbered pass subdir (<out_dir>/0001, /0002,
# ...) and PRINTS that pass dir at the end — read this run's renders from it. A
# view NOT rendered this pass is simply absent from its dir (no stale leftovers
# under a live-looking name). --pass-label is optional manifest metadata and
# never changes the numeric directory name.
#
# The camera is ALWAYS the fixed camera-0 (identity extrinsic); the object is
# posed in front of it. Only the pinhole INTRINSICS K are configurable, in
# priority order:
#   1) --intrinsics "fx,fy,cx,cy,W,H"  or  --camera-json cam.json (its
#      'intrinsics' block) — an explicit measured K, applied to all frames.
#   2) --tracking DIR — per-frame K from the known cameras (cameras.npz).
#   3) iPhone-13 PLACEHOLDER (a warning is printed) when neither is given.
# K is resolution-agnostic: its stated W,H is only the grid it was measured on;
# it is rescaled to whatever resolution is actually rendered. The 'turntable'
# always uses the spherical --fov orbit — it's a 3D sanity check, not a match.
#
# Examples:
#   harness/render.sh runs/mug/scene.py runs/mug/views \
#       --views match,turntable --fov 45 --export runs/mug/mesh/object.glb
#
#   harness/render.sh runs/mug/scene.py runs/mug/views \
#       --views match --intrinsics 900,900,400,300,800,600
#
#   # sanity-check the intrinsics: where does the object origin project?
#   harness/render.sh runs/mug/scene.py runs/mug/views --views match \
#       --camera-json cam.json --debug-project 0,0,0
set -euo pipefail

BLENDER="${BLENDER:-/home/ubuntu/blender/blender}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Blender normally discovers its bundled resources (scripts/, python/) by walking
# up from its own executable. In some environments that walk fails ("could not
# get a list of mounted file-systems"), leaving bpy unimportable. When
# BLENDER_SYSTEM_RESOURCES is unset, derive it from the configured executable:
# the adjacent <major>.<minor>/ dir carries those resources.
if [ -z "${BLENDER_SYSTEM_RESOURCES:-}" ]; then
  for _cand in "$(dirname "$BLENDER")"/[0-9]*.[0-9]*; do
    if [ -d "$_cand/scripts" ] && [ -d "$_cand/python" ]; then
      export BLENDER_SYSTEM_RESOURCES="$_cand"
      break
    fi
  done
fi

if [ "$#" -lt 2 ]; then
  echo "usage: render.sh <scene.py> <out_dir> [extra args...]" >&2
  exit 2
fi

SCENE="$1"; OUT="$2"; shift 2

# Unified-bookkeeping runs may not revive the legacy top-level views/ tree.
# Reject it before mkdir and before Blender so an obsolete command cannot leave
# an orphan pass that is absent from iteration.json.
RUN_DIR="$(cd "$(dirname "$SCENE")" && pwd)"
if [ -f "$RUN_DIR/.harness/bookkeeping.json" ]; then
  LEGACY_OUT="$RUN_DIR/views"
  if [ "$(readlink -m "$OUT")" = "$(readlink -m "$LEGACY_OUT")" ]; then
    echo "render.sh: refusing unbookkept output for managed run: $LEGACY_OUT" >&2
    echo "use 'python -m bookkeeping run --record-only' and" >&2
    echo '  "$HARNESS_ITERATION_DIR/renders"' >&2
    exit 2
  fi
fi

mkdir -p "$OUT"

# Optional hard backstop: set RENDER_TIMEOUT (seconds) to bound the WHOLE process
# (build + all views + export). This is the only thing that can reap a render
# WEDGED inside a single GPU call — the in-process --sweep-timeout can only stop
# BETWEEN candidates. 'timeout' forks Blender as its child, forwards TERM, then
# SIGKILLs after --kill-after; 'exec' leaves no intermediate shell to orphan it.
# Unset (the default) -> behavior is identical to a bare 'exec blender'.
if [ -n "${RENDER_TIMEOUT:-}" ]; then
  exec timeout --signal=TERM --kill-after=15s "$RENDER_TIMEOUT" \
    "$BLENDER" --background --factory-startup \
    --python "$HERE/render_wrapper.py" -- \
    --scene "$SCENE" --out "$OUT" "$@"
else
  exec "$BLENDER" --background --factory-startup \
    --python "$HERE/render_wrapper.py" -- \
    --scene "$SCENE" --out "$OUT" "$@"
fi
