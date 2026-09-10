#!/usr/bin/env bash
# shape_pass.sh — render and score the ONE committed state on every frame.
#
# Usage:
#   harness/utils/shape_pass.sh RUN_DIR [--frames a.jpg,b.jpg] [--critic]
#                    [--engine CYCLES|BLENDER_EEVEE_NEXT] [--samples N]
#                    [--no-aggregate]
#                    [--tracking DIR] [--frames-dir DIR] [--masks-dir DIR]
#                    [--hand-masks-dir DIR] [--ref-frame f.jpg]
#
# The main beat of the SHAPE loop (author scene.py -> this -> look), and the
# verification pass a windows round runs at its close (merge -> apply -> this ->
# adjudicate).
#
# Runs render.sh (match,turntable,depth) then, per frame, composite.py + depth.py
# (+ optional critic.py), then aggregate.py — keyed off the pass dir render.sh
# prints, so the pass number is never hand-typed into six paths. Layout comes from
# RUN_DIR/layout.json (written by run.sh from a CAPTURE dir — see
# datasets/common/capture.py); see harness/utils/shape_pass.md.
#
# render.sh launches Blender itself; the analysis runs in the 'artscript' env, so
# this wrapper drives shape_pass.py inside that env.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec micromamba run -n artscript python "$HERE/shape_pass.py" "$@"
