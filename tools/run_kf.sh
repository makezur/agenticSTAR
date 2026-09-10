#!/usr/bin/env bash
# run_kf.sh — one detached, supervised run per capture, each on its own GPU.
#
#   tools/run_kf.sh examples/garden_shears/capture:0
#   tools/run_kf.sh --agent claude --timeout-hours 4 captures/knife:2 captures/mp:3
#
# Each CAPTURE:GPU pair becomes:
#   ./run.sh --capture CAPTURE <name>_kf --frames <the tracking keyframes> \
#            --gpu GPU --no-depth-cost --no-depth-report --detach
#
# --frames is the capture's tracking/keyframes.json in view order, so the run
# uses exactly the frames Pi3X solved a camera for. Depth cost + report are off
# (the depth render stays on). Attach to a launched run later with
#   tools/agent-supervisor attach --run-dir <runs-root>/<name>
#   tools/claude-supervisor attach --run-dir <runs-root>/<name>   # --agent claude
#
# --agent codex|claude (default codex, or $ARTSCRIPT_AGENT) picks the coding agent
# for every run. Everything else passes straight through to run.sh.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat <<EOF
usage: $0 [OPTIONS] CAPTURE:GPU [CAPTURE:GPU ...]

options:
  --runs-root DIR         create runs beneath DIR instead of REPO/runs
  --agent NAME            coding agent: codex (default) or claude
  --agent-retries N       agent restarts after the initial attempt
  --codex-provider NAME   Codex provider (openai)
  --claude-provider NAME  Claude provider (anthropic)
  --critic-provider NAME  VLM critic backend: auto (default), anthropic or openai
  --enable MODULE         turn an optional harness module on for every run
  --disable MODULE        turn an optional harness module off (e.g. mechanism)
  --candidate-sheet-max-dimension N
                          target candidate-sheet pages (0 = no resize; floor 0.5x)
  --side-by-side-max-dimension N
                          target review strips (0 = no resize; floor 0.5x)
  --keep-spools N         retain newest N spool archives (default 8; 0 = all)
  --timeout-hours HOURS  stop each run after this duration (default 8)
  --no-timeout           retain unlimited run duration
EOF
}

RUNS_ROOT=""
AGENT_RETRIES=""
CODEX_PROVIDER="${ARTSCRIPT_CODEX_PROVIDER:-}"
AGENT="${ARTSCRIPT_AGENT:-}"
CLAUDE_PROVIDER="${ARTSCRIPT_CLAUDE_PROVIDER:-}"
CRITIC_PROVIDER="${ARTSCRIPT_CRITIC_PROVIDER:-}"
ENABLE_MODULES=()
DISABLE_MODULES=()
CANDIDATE_SHEET_MAX_DIMENSION=""
SIDE_BY_SIDE_MAX_DIMENSION=""
KEEP_SPOOLS="8"
TIMEOUT_HOURS="8"
PAIRS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --runs-root)
      [ "$#" -ge 2 ] || { echo "error: --runs-root requires a directory" >&2; exit 2; }
      [ -n "$2" ] || { echo "error: --runs-root requires a directory" >&2; exit 2; }
      RUNS_ROOT="$2"
      shift 2
      ;;
    --agent-retries)
      [ "$#" -ge 2 ] || { echo "error: --agent-retries requires an integer" >&2; exit 2; }
      AGENT_RETRIES="$2"
      shift 2
      ;;
    --codex-provider)
      [ "$#" -ge 2 ] || { echo "error: --codex-provider requires openai" >&2; exit 2; }
      CODEX_PROVIDER="$2"
      shift 2
      ;;
    --enable)
      [ "$#" -ge 2 ] || { echo "error: --enable requires a module name" >&2; exit 2; }
      ENABLE_MODULES+=("$2")
      shift 2
      ;;
    --disable)
      [ "$#" -ge 2 ] || { echo "error: --disable requires a module name" >&2; exit 2; }
      DISABLE_MODULES+=("$2")
      shift 2
      ;;
    --agent)
      [ "$#" -ge 2 ] || { echo "error: --agent requires codex or claude" >&2; exit 2; }
      AGENT="$2"
      shift 2
      ;;
    --claude-provider)
      [ "$#" -ge 2 ] || { echo "error: --claude-provider requires anthropic" >&2; exit 2; }
      CLAUDE_PROVIDER="$2"
      shift 2
      ;;
    --critic-provider)
      [ "$#" -ge 2 ] || { echo "error: --critic-provider requires auto, anthropic or openai" >&2; exit 2; }
      CRITIC_PROVIDER="$2"
      shift 2
      ;;
    --candidate-sheet-max-dimension)
      [ "$#" -ge 2 ] || { echo "error: --candidate-sheet-max-dimension requires an integer" >&2; exit 2; }
      CANDIDATE_SHEET_MAX_DIMENSION="$2"
      shift 2
      ;;
    --side-by-side-max-dimension)
      [ "$#" -ge 2 ] || { echo "error: --side-by-side-max-dimension requires an integer" >&2; exit 2; }
      SIDE_BY_SIDE_MAX_DIMENSION="$2"
      shift 2
      ;;
    --keep-spools)
      [ "$#" -ge 2 ] || { echo "error: --keep-spools requires an integer" >&2; exit 2; }
      KEEP_SPOOLS="$2"
      shift 2
      ;;
    --timeout-hours)
      [ "$#" -ge 2 ] || { echo "error: --timeout-hours requires a number" >&2; exit 2; }
      TIMEOUT_HOURS="$2"
      shift 2
      ;;
    --no-timeout)
      TIMEOUT_HOURS=""
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      PAIRS+=("$@")
      break
      ;;
    -*)
      echo "error: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
    *)
      PAIRS+=("$1")
      shift
      ;;
  esac
done

[ "${#PAIRS[@]}" -gt 0 ] || { usage >&2; exit 2; }

if [ -n "$AGENT_RETRIES" ]; then
  case "$AGENT_RETRIES" in
    *[!0-9]*|'')
      echo "error: --agent-retries must be a non-negative integer, got '$AGENT_RETRIES'" >&2
      exit 2
      ;;
  esac
fi

case "$CODEX_PROVIDER" in
  "" | openai) ;;
  *)
    echo "error: --codex-provider must be openai, got '$CODEX_PROVIDER'" >&2
    exit 2
    ;;
esac
case "$AGENT" in
  "" | codex | claude) ;;
  *)
    echo "error: --agent must be codex or claude, got '$AGENT'" >&2
    exit 2
    ;;
esac
case "$CLAUDE_PROVIDER" in
  "" | anthropic) ;;
  *)
    echo "error: --claude-provider must be anthropic, got '$CLAUDE_PROVIDER'" >&2
    exit 2
    ;;
esac

if [ -n "$CANDIDATE_SHEET_MAX_DIMENSION" ]; then
  case "$CANDIDATE_SHEET_MAX_DIMENSION" in
    *[!0-9]*|'')
      echo "error: --candidate-sheet-max-dimension must be a non-negative integer, got '$CANDIDATE_SHEET_MAX_DIMENSION'" >&2
      exit 2
      ;;
  esac
fi

if [ -n "$SIDE_BY_SIDE_MAX_DIMENSION" ]; then
  case "$SIDE_BY_SIDE_MAX_DIMENSION" in
    *[!0-9]*|'')
      echo "error: --side-by-side-max-dimension must be a non-negative integer, got '$SIDE_BY_SIDE_MAX_DIMENSION'" >&2
      exit 2
      ;;
  esac
fi

case "$KEEP_SPOOLS" in
  *[!0-9]*|'')
    echo "error: --keep-spools must be a non-negative integer, got '$KEEP_SPOOLS'" >&2
    exit 2
    ;;
esac

if [ -n "$TIMEOUT_HOURS" ]; then
  case "$TIMEOUT_HOURS" in
    ''|*[!0-9.]*|*.*.*)
      echo "error: --timeout-hours must be a positive finite number, got '$TIMEOUT_HOURS'" >&2
      exit 2
      ;;
  esac
  awk -v value="$TIMEOUT_HOURS" 'BEGIN { exit !(value + 0 > 0) }' || {
    echo "error: --timeout-hours must be a positive finite number, got '$TIMEOUT_HOURS'" >&2
    exit 2
  }
fi

for pair in "${PAIRS[@]}"; do
  capture="${pair%:*}"
  gpu="${pair##*:}"
  [ "$capture" != "$pair" ] || { echo "error: expected CAPTURE:GPU, got '$pair'" >&2; exit 2; }
  [ -f "$capture/tracking/keyframes.json" ] || {
    echo "error: no tracking/keyframes.json in $capture" >&2; exit 2; }

  frames="$(python3 - "$capture" <<'PY'
import json, sys
kf = json.load(open(f"{sys.argv[1]}/tracking/keyframes.json"))
print(",".join(k["frame_name"] for k in sorted(kf, key=lambda k: int(k["view"]))))
PY
)"

  echo "== $(basename "$capture") -> gpu $gpu, $(awk -F, '{print NF}' <<<"$frames") keyframes"
  agent_args=()
  [ -z "$RUNS_ROOT" ] || agent_args+=(--runs-root "$RUNS_ROOT")
  [ -z "$AGENT_RETRIES" ] || agent_args+=(--agent-retries "$AGENT_RETRIES")
  [ -z "$CODEX_PROVIDER" ] || agent_args+=(--codex-provider "$CODEX_PROVIDER")
  [ -z "$AGENT" ] || agent_args+=(--agent "$AGENT")
  [ -z "$CLAUDE_PROVIDER" ] || agent_args+=(--claude-provider "$CLAUDE_PROVIDER")
  [ -z "$CRITIC_PROVIDER" ] || agent_args+=(--critic-provider "$CRITIC_PROVIDER")
  for name in ${ENABLE_MODULES[@]+"${ENABLE_MODULES[@]}"}; do agent_args+=(--enable "$name"); done
  for name in ${DISABLE_MODULES[@]+"${DISABLE_MODULES[@]}"}; do agent_args+=(--disable "$name"); done
  [ -z "$CANDIDATE_SHEET_MAX_DIMENSION" ] || agent_args+=(--candidate-sheet-max-dimension "$CANDIDATE_SHEET_MAX_DIMENSION")
  [ -z "$SIDE_BY_SIDE_MAX_DIMENSION" ] || agent_args+=(--side-by-side-max-dimension "$SIDE_BY_SIDE_MAX_DIMENSION")
  agent_args+=(--keep-spools "$KEEP_SPOOLS")
  [ -z "$TIMEOUT_HOURS" ] || agent_args+=(--timeout-hours "$TIMEOUT_HOURS")
  "$HERE/run.sh" --capture "$capture" "$(basename "$capture")_kf" \
    --frames "$frames" --gpu "$gpu" \
    --no-depth-cost --no-depth-report "${agent_args[@]}" --detach
done
