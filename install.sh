#!/usr/bin/env bash
#
# install.sh — host setup for the harness, idempotent. Runs, in order:
#   1. the `artscript` micromamba env (environment.yml)
#   2. Blender 4.2.5 LTS into $BLENDER_DIR (default ~/blender)   [install_blender.sh]
#   3. the `pi3x` micromamba env + yyfz/Pi3 checkout + weights     [tools/install_pi3x.sh]
#
# Then, once, with your API key exported, the agent launcher you want:
#   ANTHROPIC_API_KEY=... tools/install_claude_bwrap.sh     # Claude Code
#   OPENAI_API_KEY=...    tools/install_codex_bwrap.sh      # Codex
# (kept out of this script: they store the key and may need sudo for bubblewrap).
#
# Usage:
#   ./install.sh               # everything above
#   ./install.sh --no-pi3x     # skip the Pi3X env (you already have captures)
#   ./install.sh --check       # report what is present, change nothing
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_DIR="${BLENDER_DIR:-$HOME/blender}"
export BLENDER_DIR

WITH_PI3X=1
CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-pi3x) WITH_PI3X=0; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) printf 'install.sh: unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

log() { printf '==> %s\n' "$*"; }
ok() { printf '    (ok) %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

for tool in micromamba git curl tar xz tmux bwrap python3; do
  command -v "$tool" >/dev/null 2>&1 || die "'$tool' is not on PATH (see README § Host prerequisites)"
done
command -v nvidia-smi >/dev/null 2>&1 || echo "warning: nvidia-smi not found; EEVEE rendering and Pi3X need an NVIDIA GPU" >&2

have_env() { micromamba env list | awk '{print $1}' | grep -qx "$1"; }

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  have_env artscript && ok "artscript env" || echo "  MISSING artscript env"
  micromamba run -n artscript python -c "import trimesh, cv2, PIL, manifold3d" >/dev/null 2>&1 \
    && ok "artscript imports" || echo "  MISSING artscript imports (trimesh cv2 PIL manifold3d)"
  [[ -x "$BLENDER_DIR/blender" ]] && ok "Blender at $BLENDER_DIR/blender ($("$BLENDER_DIR/blender" --version 2>/dev/null | head -1))" \
    || echo "  MISSING Blender at $BLENDER_DIR/blender"
  [[ "$WITH_PI3X" -eq 0 ]] || "$HERE/tools/install_pi3x.sh" --check || true
  for agent in claude codex; do
    [[ -f "${XDG_CONFIG_HOME:-$HOME/.config}/$agent-articulated/paths.conf" ]] \
      && ok "$agent-articulated launcher installed" || echo "  (not installed) $agent-articulated launcher — tools/install_${agent}_bwrap.sh"
  done
  exit 0
fi

if have_env artscript; then
  ok "artscript env exists"
else
  log "Creating the artscript env"
  micromamba create -y -f "$HERE/environment.yml"
fi
micromamba run -n artscript python -c "import trimesh, cv2, PIL, manifold3d; print('    (ok) artscript imports')"

log "Blender"
"$HERE/install_blender.sh"

if [[ "$WITH_PI3X" -eq 1 ]]; then
  log "Pi3X env"
  "$HERE/tools/install_pi3x.sh"
fi

cat <<EOF

Host setup done. Next:
  export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY for Codex
  tools/install_claude_bwrap.sh  # or tools/install_codex_bwrap.sh
  tools/run_kf.sh --agent claude examples/garden_shears/capture:0
Remember to export BLENDER=$BLENDER_DIR/blender for host-side render.sh calls.
EOF
