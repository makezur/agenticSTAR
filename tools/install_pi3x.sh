#!/usr/bin/env bash
#
# install_pi3x.sh — idempotently provision the Pi3X environment tools/make_capture.py
# runs in: a dedicated micromamba env `pi3x` (torch + CUDA wheels), the official
# yyfz/Pi3 checkout under third_party/Pi3, and the yyfz233/Pi3X weights in the
# Hugging Face cache (public repo — no token needed).
#
# Usage:
#   tools/install_pi3x.sh              # install what is missing
#   tools/install_pi3x.sh --check      # report, change nothing
#   tools/install_pi3x.sh --update     # also git pull the Pi3 checkout
#   tools/install_pi3x.sh --skip-weights
#
# Knobs: PI3_REPO_DIR (default REPO/third_party/Pi3), PI3X_ENV_NAME (pi3x),
# PYTORCH_INDEX_URL (cu128 wheels), TORCH_VERSION (2.10.0).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ENV_NAME="${PI3X_ENV_NAME:-pi3x}"
PI3_DIR="${PI3_REPO_DIR:-$REPO/third_party/Pi3}"
PI3_URL="https://github.com/yyfz/Pi3.git"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"

CHECK_ONLY=0
UPDATE=0
SKIP_WEIGHTS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --update) UPDATE=1; shift ;;
    --skip-weights) SKIP_WEIGHTS=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) printf 'install_pi3x.sh: unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

log() { printf '==> %s\n' "$*"; }
ok() { printf '    (ok) %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v micromamba >/dev/null 2>&1 || die "micromamba is not on PATH"
MAMBA="$(command -v micromamba)"
env_run() { "$MAMBA" run -n "$ENV_NAME" "$@"; }
env_exists() { "$MAMBA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; }

check_install() {
  local rc=0 name command
  for spec in \
    "Pi3 checkout:test -d '$PI3_DIR/.git'" \
    "micromamba env $ENV_NAME:env_exists" \
    "torch $TORCH_VERSION:env_run python -c 'import torch; assert torch.__version__.split(\"+\")[0] == \"$TORCH_VERSION\"'" \
    "CUDA torch:env_run python -c 'import torch; assert torch.cuda.is_available()'" \
    "Pi3X import:env_run env PI3_REPO_DIR='$PI3_DIR' python -c 'import os, sys; sys.path.insert(0, os.environ[\"PI3_REPO_DIR\"]); from pi3.models.pi3x import Pi3X'" \
    "converter deps:env_run python -c 'import cv2, numpy, PIL, plyfile, safetensors, einops'" \
    "Pi3X weights:env_run python -c \"from huggingface_hub import try_to_load_from_cache as f; assert isinstance(f('yyfz233/Pi3X', 'model.safetensors'), str)\""; do
    name="${spec%%:*}"; command="${spec#*:}"
    if eval "$command" >/dev/null 2>&1; then
      printf '  ok      %s\n' "$name"
    else
      printf '  MISSING %s\n' "$name"; rc=1
    fi
  done
  return "$rc"
}

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  check_install
  exit $?
fi

command -v git >/dev/null 2>&1 || die "git is not installed"
if [[ -d "$PI3_DIR/.git" ]]; then
  origin="$(git -C "$PI3_DIR" remote get-url origin 2>/dev/null || true)"
  [[ "$origin" == "$PI3_URL" ]] || die "$PI3_DIR exists but origin is '$origin' (expected $PI3_URL)"
  if [[ "$UPDATE" -eq 1 ]]; then
    log "Updating Pi3 in $PI3_DIR"; git -C "$PI3_DIR" pull --ff-only
  else
    ok "Pi3 checkout exists at $PI3_DIR"
  fi
else
  [[ ! -e "$PI3_DIR" ]] || die "$PI3_DIR exists but is not a git checkout"
  log "Cloning Pi3 to $PI3_DIR"
  mkdir -p "$(dirname "$PI3_DIR")"
  git clone "$PI3_URL" "$PI3_DIR"
fi

if env_exists; then
  ok "micromamba environment '$ENV_NAME' exists"
else
  log "Creating micromamba environment '$ENV_NAME' from tools/pi3x_env.yml"
  "$MAMBA" create -y -n "$ENV_NAME" -f "$HERE/pi3x_env.yml"
fi

if env_run python -c "import torch, torchvision; assert torch.__version__.split('+')[0] == '$TORCH_VERSION'" >/dev/null 2>&1; then
  ok "PyTorch $TORCH_VERSION is installed"
else
  log "Installing PyTorch $TORCH_VERSION (+ torchvision) from $PYTORCH_INDEX_URL"
  env_run python -m pip install "torch==$TORCH_VERSION" torchvision --index-url "$PYTORCH_INDEX_URL"
fi

log "Installing Pi3 (no deps — its pins target an older torch; the env above supplies them)"
env_run python -m pip install --no-deps -e "$PI3_DIR"

if [[ "$SKIP_WEIGHTS" -eq 1 ]]; then
  ok "weights prefetch skipped (downloaded on first make_capture.py run)"
else
  log "Prefetching yyfz233/Pi3X weights into the Hugging Face cache"
  env_run python -c "from huggingface_hub import hf_hub_download; hf_hub_download('yyfz233/Pi3X', 'model.safetensors'); hf_hub_download('yyfz233/Pi3X', 'config.json')"
fi

log "Verifying installation"
check_install
log "Pi3X environment is ready. Run:"
printf '  micromamba run -n %s python tools/make_capture.py --src examples/garden_shears/input --out captures/garden_shears\n' "$ENV_NAME"
