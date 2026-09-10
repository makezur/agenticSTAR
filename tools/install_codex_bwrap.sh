#!/usr/bin/env bash
# Install Bubblewrap and register the codex-articulated launcher.
#
# Also provisions the dedicated Codex state (~/.local/state/codex-articulated/
# codex-home): config.toml (OpenAI API provider, model, trusted sandbox paths,
# web_search disabled) and, from $OPENAI_API_KEY exported once, the API-key login
# Codex itself writes into auth.json (mode 0600). The key is never written into
# TOML, the checkout, or the bwrap environment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/codex-providers.sh"
PROJECT_DIR="$(cd "$HERE/.." && pwd)"
BLENDER_DIR="${BLENDER_DIR:-$HOME/blender}"
# Empty by default; auto-detected after argument parsing unless --mamba-root is
# given.
MAMBA_ROOT="${MAMBA_ROOT:-}"
CODEX_MODEL="${ARTSCRIPT_CODEX_MODEL:-$CODEX_OPENAI_MODEL_DEFAULT}"
CODEX_STATE="${XDG_STATE_HOME:-$HOME/.local/state}/codex-articulated/codex-home"
INSTALL_DIR="${HOME}/.local/bin"

usage() {
  cat <<'EOF'
Usage: install_codex_bwrap.sh [options]

Options:
  --project DIR      Host project directory to expose read/write
  --blender DIR      Host Blender directory to expose read-only
  --mamba-root DIR   Host micromamba root to expose read-only
  --codex-state DIR  Dedicated persistent Codex state directory
  --model ID         Codex model (default: gpt-5.6-sol)
  --install-dir DIR  Launcher destination (default: ~/.local/bin)
  --check            Validate prerequisites without installing anything
  -h, --help         Show this help
EOF
}

CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT_DIR="${2:?missing value for --project}"; shift 2 ;;
    --blender) BLENDER_DIR="${2:?missing value for --blender}"; shift 2 ;;
    --mamba-root) MAMBA_ROOT="${2:?missing value for --mamba-root}"; shift 2 ;;
    --codex-state) CODEX_STATE="${2:?missing value for --codex-state}"; shift 2 ;;
    --model) CODEX_MODEL="${2:?missing value for --model}"; shift 2 ;;
    --install-dir) INSTALL_DIR="${2:?missing value for --install-dir}"; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

# Locate the micromamba/conda root that holds the artscript environment. Prefer
# an explicit value, then MAMBA_ROOT_PREFIX (exported by micromamba shell init),
# then `micromamba info`, then a scan of common install locations.
detect_mamba_root() {
  local candidate
  if [[ -n "${MAMBA_ROOT_PREFIX:-}" && -d "$MAMBA_ROOT_PREFIX/envs/artscript" ]]; then
    printf '%s\n' "$MAMBA_ROOT_PREFIX"; return 0
  fi
  if command -v micromamba >/dev/null 2>&1; then
    candidate="$(micromamba info 2>/dev/null |
      sed -n 's/.*base environment : *//p' | head -n1)"
    if [[ -n "$candidate" && -d "$candidate/envs/artscript" ]]; then
      printf '%s\n' "$candidate"; return 0
    fi
  fi
  for candidate in "$HOME/micromamba" "$HOME/.local/share/mamba" \
                   "$HOME/miniforge3" "$HOME/mambaforge" \
                   "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [[ -d "$candidate/envs/artscript" ]]; then
      printf '%s\n' "$candidate"; return 0
    fi
  done
  return 1
}

if [[ -z "$MAMBA_ROOT" ]]; then
  MAMBA_ROOT="$(detect_mamba_root || true)"
  [[ -n "$MAMBA_ROOT" ]] || {
    echo "Could not locate a mamba/conda root containing envs/artscript." >&2
    echo "Pass one explicitly with --mamba-root DIR." >&2
    exit 1
  }
fi

PROJECT_DIR="$(realpath "$PROJECT_DIR")"
BLENDER_DIR="$(realpath "$BLENDER_DIR")"
MAMBA_ROOT="$(realpath "$MAMBA_ROOT")"
CODEX_STATE="$(realpath -m "$CODEX_STATE")"
CODEX_COMMAND="$(command -v codex 2>/dev/null || true)"
[[ -n "$CODEX_COMMAND" ]] || { echo "codex is not installed" >&2; exit 1; }
CODEX_BIN="$(realpath "$CODEX_COMMAND")"

# Codex standalone distributions bundle a static rg in codex-path/. Accept a
# system rg too, but record its real path because the sandbox has a cleared PATH.
RG_BIN="$(command -v rg 2>/dev/null || true)"
if [[ -z "$RG_BIN" ]]; then
  BUNDLED_RG="$(dirname "$(dirname "$CODEX_BIN")")/codex-path/rg"
  [[ -x "$BUNDLED_RG" ]] && RG_BIN="$BUNDLED_RG"
fi
[[ -n "$RG_BIN" ]] ||
  { echo "ripgrep is not installed and was not bundled with Codex" >&2; exit 1; }
RG_BIN="$(realpath "$RG_BIN")"

# Locate the micromamba binary. A conda/miniforge-style install keeps it inside
# the root at bin/micromamba; a standalone install (this host) keeps it on PATH
# at ~/.local/bin. The root itself is not enough -- the launcher must mount the
# actual binary into the sandbox, so record its real path here.
MAMBA_BIN=""
if [[ -x "$MAMBA_ROOT/bin/micromamba" ]]; then
  MAMBA_BIN="$MAMBA_ROOT/bin/micromamba"
elif command -v micromamba >/dev/null 2>&1; then
  MAMBA_BIN="$(command -v micromamba)"
fi
[[ -n "$MAMBA_BIN" ]] || { echo "micromamba binary not found on PATH or in $MAMBA_ROOT/bin" >&2; exit 1; }
MAMBA_BIN="$(realpath "$MAMBA_BIN")"

[[ -d "$PROJECT_DIR" ]] || { echo "Project not found: $PROJECT_DIR" >&2; exit 1; }
[[ -x "$BLENDER_DIR/blender" ]] ||
  { echo "Blender not found: $BLENDER_DIR/blender" >&2; exit 1; }
[[ -d "$MAMBA_ROOT/envs/artscript" ]] ||
  { echo "artscript environment not found: $MAMBA_ROOT/envs/artscript" >&2; exit 1; }
[[ -x "$MAMBA_BIN" ]] ||
  { echo "micromamba binary not executable: $MAMBA_BIN" >&2; exit 1; }
[[ -n "$CODEX_BIN" && -x "$CODEX_BIN" ]] ||
  { echo "codex is not installed" >&2; exit 1; }
[[ -x "$RG_BIN" ]] ||
  { echo "rg binary not executable: $RG_BIN" >&2; exit 1; }

# Run the given command as root: directly when already root, otherwise via sudo.
as_root() {
  if [[ "$EUID" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "Installing bubblewrap requires root or sudo." >&2
    exit 1
  fi
}

# Confirm bwrap can actually create a user namespace and map uids. On Ubuntu
# 23.10+ the AppArmor mitigation kernel.apparmor_restrict_unprivileged_userns
# blocks this for unconfined binaries, so bwrap fails with "setting up uid map:
# Permission denied" even though it is installed. Returns 0 when sandboxing works.
# /usr plus the usr-merge symlinks are bound so the dynamic linker can resolve
# and actually exec a binary -- otherwise a working namespace fails on execvp.
bwrap_userns_works() {
  bwrap --unshare-user --uid 0 --gid 0 \
    --ro-bind /usr /usr \
    --symlink usr/bin /bin --symlink usr/sbin /sbin \
    --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
    --tmpfs /tmp /usr/bin/true >/dev/null 2>&1
}

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  command -v bwrap >/dev/null 2>&1 ||
    { echo "bubblewrap is not installed" >&2; exit 1; }
  bwrap_userns_works ||
    { echo "bwrap cannot create a user namespace (AppArmor userns restriction);" \
           "rerun without --check to install the bwrap AppArmor profile" >&2
      exit 1; }
  echo "All prerequisites are present."
  exit 0
fi

if ! command -v bwrap >/dev/null 2>&1; then
  # The package is named 'bubblewrap' on every distro family below.
  if command -v apt-get >/dev/null 2>&1; then
    as_root apt-get update
    as_root apt-get install -y bubblewrap
  elif command -v dnf >/dev/null 2>&1; then
    as_root dnf install -y bubblewrap
  elif command -v yum >/dev/null 2>&1; then
    as_root yum install -y bubblewrap
  elif command -v zypper >/dev/null 2>&1; then
    as_root zypper install -y bubblewrap
  elif command -v pacman >/dev/null 2>&1; then
    as_root pacman -S --noconfirm bubblewrap
  elif command -v apk >/dev/null 2>&1; then
    as_root apk add bubblewrap
  else
    echo "Unsupported package manager; install 'bubblewrap' and rerun." >&2
    exit 1
  fi
fi

APPARMOR_PROFILE_PATH="/etc/apparmor.d/codex-bwrap"

# Install a bwrap-scoped AppArmor profile granting the `userns` permission. This
# is narrower than disabling the sysctl host-wide: only the bwrap binary gains
# the exemption; the mitigation stays active for every other unconfined program.
install_bwrap_apparmor_profile() {
  local bwrap_path
  bwrap_path="$(command -v bwrap)"
  bwrap_path="$(realpath "$bwrap_path")"
  as_root tee "$APPARMOR_PROFILE_PATH" >/dev/null <<EOF
# Managed by tools/install_codex_bwrap.sh. Grants unprivileged user-namespace
# creation to bwrap only, so the codex-articulated sandbox can start on hosts
# with kernel.apparmor_restrict_unprivileged_userns=1 (Ubuntu 23.10+).
abi <abi/4.0>,
include <tunables/global>

profile codex-bwrap $bwrap_path flags=(unconfined) {
  userns,

  # Site-specific additions and overrides.
  include if exists <local/codex-bwrap>
}
EOF
  if command -v apparmor_parser >/dev/null 2>&1; then
    as_root apparmor_parser -r -W "$APPARMOR_PROFILE_PATH"
  elif command -v systemctl >/dev/null 2>&1; then
    as_root systemctl reload apparmor
  else
    echo "Installed $APPARMOR_PROFILE_PATH but could not reload AppArmor." >&2
    echo "Reload it manually, then rerun this installer." >&2
    return 1
  fi
}

USERNS_FIXED=0
if ! bwrap_userns_works; then
  echo "bwrap cannot create a user namespace (AppArmor userns restriction)."
  echo "Installing a bwrap-scoped AppArmor profile: $APPARMOR_PROFILE_PATH"
  if [[ ! -e /sys/kernel/security/apparmor ]]; then
    echo "AppArmor is not active on this host; cannot install the profile." >&2
    echo "Enable user namespaces for bwrap manually, then rerun." >&2
    exit 1
  fi
  install_bwrap_apparmor_profile
  if bwrap_userns_works; then
    USERNS_FIXED=1
    echo "AppArmor profile loaded; bwrap user namespaces work now."
  else
    echo "AppArmor profile installed but bwrap still cannot create a namespace." >&2
    echo "User namespaces may be disabled at the kernel level" \
         "(user.max_user_namespaces=0); enable them, then rerun." >&2
    exit 1
  fi
fi

mkdir -p "$INSTALL_DIR"
install -m 0755 "$HERE/codex-articulated" "$INSTALL_DIR/codex-articulated"
install -m 0644 "$HERE/codex-providers.sh" "$INSTALL_DIR/codex-providers.sh"
# The launcher looks for the egress relay beside itself.
install -m 0644 "$HERE/egress_broker.py" "$INSTALL_DIR/egress_broker.py"

mkdir -p "$CODEX_STATE"
chmod 0700 "$CODEX_STATE"
codex_provider_write_config "$CODEX_STATE" "$CODEX_MODEL" "$PROJECT_DIR"

# Codex owns the auth-file format: hand the key to its own login command.
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  printf '%s\n' "$OPENAI_API_KEY" |
    CODEX_HOME="$CODEX_STATE" "$CODEX_BIN" login --with-api-key \
      -c 'forced_login_method="api"' \
      -c 'cli_auth_credentials_store="file"' >/dev/null ||
    { echo "Codex API-key login failed for $CODEX_STATE" >&2; exit 1; }
  [[ ! -f "$CODEX_STATE/auth.json" ]] || chmod 0600 "$CODEX_STATE/auth.json"
elif ! grep -q '"OPENAI_API_KEY"' "$CODEX_STATE/auth.json" 2>/dev/null; then
  echo "warning: no OPENAI_API_KEY exported and no API-key login in" \
       "$CODEX_STATE/auth.json yet; the launcher will refuse to start." >&2
fi
# The key is provisioning input, not runtime configuration.
unset OPENAI_API_KEY

CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/codex-articulated"
mkdir -p "$CONFIG_DIR"
{
  printf 'ARTICULATED_DIR=%q\n' "$PROJECT_DIR"
  printf 'BLENDER_DIR=%q\n' "$BLENDER_DIR"
  printf 'MAMBA_ROOT=%q\n' "$MAMBA_ROOT"
  printf 'HOST_MAMBA_BIN=%q\n' "$MAMBA_BIN"
  printf 'HOST_CODEX_HOME=%q\n' "$CODEX_STATE"
  printf 'HOST_CODEX_BIN=%q\n' "$CODEX_BIN"
  printf 'HOST_RG_BIN=%q\n' "$RG_BIN"
} > "$CONFIG_DIR/paths.conf"
chmod 0600 "$CONFIG_DIR/paths.conf"

echo "Installed launcher: $INSTALL_DIR/codex-articulated"
echo "Project mount:      $PROJECT_DIR -> /home/ubuntu/articulated_script (read/write)"
echo "Conventions overlay: $PROJECT_DIR/conventions (read-only)"
echo "Harness overlay:    $PROJECT_DIR/harness (read-only)"
echo "Blender mount:      $BLENDER_DIR -> /home/ubuntu/blender (read-only)"
echo "Micromamba:         $MAMBA_ROOT (root) + $MAMBA_BIN (binary)"
echo "Ripgrep:            $RG_BIN -> /opt/codex/rg"
echo "Temporary storage:  isolated tmpfs"
echo "Codex state:        $CODEX_STATE (dedicated)"
echo "Provider / model:   openai / $CODEX_MODEL"
[[ "$USERNS_FIXED" -eq 1 ]] &&
  echo "AppArmor profile:   $APPARMOR_PROFILE_PATH (bwrap userns exemption)"
echo
echo "Run: codex-articulated"
