#!/usr/bin/env bash
# Install Bubblewrap and register the claude-articulated launcher.
#
# Mirror of install_codex_bwrap.sh for the Claude Code backend. It writes its own
# paths.conf (~/.config/claude-articulated/) and its own dedicated state directory
# (~/.local/state/claude-articulated/claude-home: settings.json + .claude.json +
# the per-session transcripts), so the Codex installation is never touched and
# both launchers can be installed on one host.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/claude-providers.sh"
PROJECT_DIR="$(cd "$HERE/.." && pwd)"
BLENDER_DIR="${BLENDER_DIR:-$HOME/blender}"
# Empty by default; auto-detected after argument parsing unless --mamba-root is
# given.
MAMBA_ROOT="${MAMBA_ROOT:-}"
CLAUDE_STATE="${XDG_STATE_HOME:-$HOME/.local/state}/claude-articulated/claude-home"
INSTALL_DIR="${HOME}/.local/bin"
# Model for the generated settings.json (the provider is the Anthropic API).
CLAUDE_PROVIDER="${ARTSCRIPT_CLAUDE_PROVIDER:-$CLAUDE_PROVIDER_DEFAULT}"
CLAUDE_MODEL="${ARTSCRIPT_CLAUDE_MODEL:-$CLAUDE_ANTHROPIC_MODEL_DEFAULT}"
REWRITE_SETTINGS=0

usage() {
  cat <<'EOF'
Usage: install_claude_bwrap.sh [options]

Options:
  --project DIR       Host project directory to expose read/write
  --blender DIR       Host Blender directory to expose read-only
  --mamba-root DIR    Host micromamba root to expose read-only
  --claude-state DIR  Dedicated persistent Claude Code state directory
  --install-dir DIR   Launcher destination (default: ~/.local/bin)
  --model ID          Claude model (default: claude-fable-5[1m])
  --rewrite-settings  Regenerate settings.json even if one exists
  --check             Validate prerequisites without installing anything
  -h, --help          Show this help

Export ANTHROPIC_API_KEY once before running this; it is stored in the state
directory (mode 0600) and never written into the checkout. Export OPENAI_API_KEY
as well to let the VLM critic run on OpenAI inside Claude runs
(harness/analysis/scorers/critic/critic.md).
EOF
}

CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT_DIR="${2:?missing value for --project}"; shift 2 ;;
    --blender) BLENDER_DIR="${2:?missing value for --blender}"; shift 2 ;;
    --mamba-root) MAMBA_ROOT="${2:?missing value for --mamba-root}"; shift 2 ;;
    --claude-state) CLAUDE_STATE="${2:?missing value for --claude-state}"; shift 2 ;;
    --install-dir) INSTALL_DIR="${2:?missing value for --install-dir}"; shift 2 ;;
    --model) CLAUDE_MODEL="${2:?missing value for --model}"; shift 2 ;;
    --rewrite-settings) REWRITE_SETTINGS=1; shift ;;
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
CLAUDE_STATE="$(realpath -m "$CLAUDE_STATE")"
CLAUDE_COMMAND="$(command -v claude 2>/dev/null || true)"
[[ -n "$CLAUDE_COMMAND" ]] || { echo "claude is not installed" >&2; exit 1; }
# Record the stable shim (~/.local/bin/claude), NOT its realpath. The native
# installer keeps one directory per version and repoints the shim on `claude
# update`; a resolved path here pinned the sandbox to a stale 2.1.252 after the
# host had moved on to 2.1.259 (Fable needs >= 2.1.255). The launcher resolves
# the shim at every launch, so the sandbox follows host updates automatically.
CLAUDE_BIN="$CLAUDE_COMMAND"
CLAUDE_PROVIDER="$(claude_provider_validate "$CLAUDE_PROVIDER" || true)"
[[ -n "$CLAUDE_PROVIDER" ]] ||
  { echo "ARTSCRIPT_CLAUDE_PROVIDER must be anthropic" >&2; exit 1; }

# Claude Code embeds its own ripgrep. A system rg is bound for the agent's shell
# commands when present; record its real path because the sandbox clears PATH.
RG_BIN="$(command -v rg 2>/dev/null || true)"
[[ -z "$RG_BIN" ]] || RG_BIN="$(realpath "$RG_BIN")"

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
[[ -n "$CLAUDE_BIN" && -x "$CLAUDE_BIN" ]] ||
  { echo "claude is not installed" >&2; exit 1; }
[[ -z "$RG_BIN" || -x "$RG_BIN" ]] ||
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
# Same profile file and content as install_codex_bwrap.sh writes -- it is about
# the bwrap binary, not about which agent runs inside, so the two installers
# share it rather than fight over it.
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
install -m 0755 "$HERE/claude-articulated" "$INSTALL_DIR/claude-articulated"
install -m 0644 "$HERE/claude-providers.sh" "$INSTALL_DIR/claude-providers.sh"
# The launcher looks for the egress relay beside itself.
install -m 0644 "$HERE/egress_broker.py" "$INSTALL_DIR/egress_broker.py"

mkdir -p "$CLAUDE_STATE"
chmod 0700 "$CLAUDE_STATE"
if [[ "$REWRITE_SETTINGS" -eq 1 || ! -e "$CLAUDE_STATE/settings.json" ]]; then
  claude_provider_write_settings "$CLAUDE_STATE" "$CLAUDE_PROVIDER" "$CLAUDE_MODEL"
fi
# Trust is keyed on the sandbox paths the launcher mounts the checkout at, so the
# first unattended launch asks nothing.
claude_provider_write_global_config "$CLAUDE_STATE" \
  /home/ubuntu /home/ubuntu/articulated_script /home/ubuntu/articulated_script/runs
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  umask 077
  printf '%s\n' "$ANTHROPIC_API_KEY" > "$CLAUDE_STATE/$CLAUDE_API_KEY_FILE"
  umask 022
elif [[ ! -s "$CLAUDE_STATE/$CLAUDE_API_KEY_FILE" ]]; then
  echo "warning: no ANTHROPIC_API_KEY exported and no" \
       "$CLAUDE_STATE/$CLAUDE_API_KEY_FILE yet; the launcher will refuse to start." >&2
fi
# The key is provisioning input, not runtime configuration.
unset ANTHROPIC_API_KEY

# Optional: an OpenAI key for the harness's VLM critic (--critic-provider openai).
# Stored beside the Anthropic key; the launcher exports it into the sandbox and
# allows api.openai.com only when this file exists.
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  umask 077
  printf '%s\n' "$OPENAI_API_KEY" > "$CLAUDE_STATE/openai_api_key"
  umask 022
  echo "Stored an OpenAI key for the VLM critic: $CLAUDE_STATE/openai_api_key"
fi
unset OPENAI_API_KEY

CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/claude-articulated"
mkdir -p "$CONFIG_DIR"
{
  printf 'ARTICULATED_DIR=%q\n' "$PROJECT_DIR"
  printf 'BLENDER_DIR=%q\n' "$BLENDER_DIR"
  printf 'MAMBA_ROOT=%q\n' "$MAMBA_ROOT"
  printf 'HOST_MAMBA_BIN=%q\n' "$MAMBA_BIN"
  printf 'HOST_CLAUDE_HOME=%q\n' "$CLAUDE_STATE"
  printf 'HOST_CLAUDE_BIN=%q\n' "$CLAUDE_BIN"
  printf 'HOST_RG_BIN=%q\n' "$RG_BIN"
} > "$CONFIG_DIR/paths.conf"
chmod 0600 "$CONFIG_DIR/paths.conf"

echo "Installed launcher: $INSTALL_DIR/claude-articulated"
echo "Project mount:      $PROJECT_DIR -> /home/ubuntu/articulated_script (read/write)"
echo "Conventions overlay: $PROJECT_DIR/conventions (read-only)"
echo "Harness overlay:    $PROJECT_DIR/harness (read-only)"
echo "Blender mount:      $BLENDER_DIR -> /home/ubuntu/blender (read-only)"
echo "Micromamba:         $MAMBA_ROOT (root) + $MAMBA_BIN (binary)"
echo "Ripgrep:            ${RG_BIN:-<none; Claude Code embeds its own>} -> /opt/claude/rg"
echo "Temporary storage:  isolated tmpfs"
echo "Claude state:       $CLAUDE_STATE (dedicated)"
echo "Claude binary:      $CLAUDE_BIN -> $(realpath "$CLAUDE_BIN") (resolved at each launch)"
echo "Provider / model:   $CLAUDE_PROVIDER / $CLAUDE_MODEL"
[[ "$USERNS_FIXED" -eq 1 ]] &&
  echo "AppArmor profile:   $APPARMOR_PROFILE_PATH (bwrap userns exemption)"
echo
echo "Run: claude-articulated"
