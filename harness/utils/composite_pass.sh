#!/usr/bin/env bash
# Render match views and build review composites for an open pose iteration.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec micromamba run -n artscript python "$HERE/composite_pass.py" "$@"
