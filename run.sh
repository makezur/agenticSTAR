#!/usr/bin/env bash
# run.sh — scaffold a fresh run directory for a reconstruction task and print
# the launch command for the agent.
#
# Multi-frame (preferred — a CAPTURE drives the per-frame known cameras):
#   ./run.sh --capture DIR --frames "000000.jpg,000080.jpg" [run_name] \
#            [--runs-root DIR] \
#            [--no-depth-cost] [--no-depth-report] [--no-depth-render] \
#            [--no-timestamp] [--prompt-only]
#
# Existing runs can be resumed without re-running any scaffold step:
#   ./run.sh --resume-run runs/<name> [--detach]
#
# A capture is the canonical materialized sequence every dataset backend
# produces (datasets/common/capture.py): frames/ + mask_object/ [+ mask_hand/]
# + tracking/ (cameras) [+ depth/]. Produce one from frames + masks with
#   micromamba run -n pi3x python tools/make_capture.py --src DIR --out captures/NAME
# The depth backend (pi3x | none) comes from the capture's manifest — captures
# without depth (the converter's default) run masks-only automatically.
#
# --no-depth-cost / --no-depth-report / --no-depth-render scaffold
# RUN_DIR/depth_config.json with the depth supervision (sweep penalty), the depth
# reporting (scorer + panels), and/or the depth RENDER itself turned OFF. All three
# default on and are independent; camera poses + measure_depth are unaffected by any
# of them. Note --no-depth-report leaves the render ON (the object-units DEPTH panel
# and depth sheet need no GT); --no-depth-render is what stops paying for depth.
#
# --enable NAME / --disable NAME (repeatable, comma lists; env
# ARTSCRIPT_ENABLE_MODULES / ARTSCRIPT_DISABLE_MODULES) pick the run's optional
# harness MODULES — today `mechanism`, the blind A/B joint-axis gate. The choice
# is scaffolded into RUN_DIR/modules.json, which the sandbox launcher mounts
# READ-ONLY over the run so the agent can read but never change it, and it also
# selects which sections of the task document the run is given. Unset modules
# take harness/core/modules.py's DEFAULTS (default: mechanism OFF). Fixed at
# scaffold: --resume-run refuses these flags.
#
#   ...or sample every Nth frame instead of listing them (stride by frame number):
#   ./run.sh --capture DIR --every 10 [run_name] [--start N] [--stop N] [...]
#   (--every expands against the capture's frames dir via analysis.frames sample)
#
# --workers N / --ncpu N size the two concurrency knobs in RUN_DIR/run_config.json:
#   workers = number of resident EEVEE Blender pool processes (render fan-out);
#             default 16 (the pool clamps to 16 per visible GPU)
#   ncpu    = CPU count handed to the solver/CPU work; default 16
# Both ignored on a --no-timestamp re-run that reuses an existing run_config.json
# (that file is kept, not rewritten).
#
# --candidate-sheet-max-dimension N targets derived candidate-sheet previews
# (0 = no preview). Canonical pages stay native; previews never downsample below
# 0.5x native linear resolution. Recorded before the agent starts.
#
# --side-by-side-max-dimension N applies the same policy to derived previews of
# recurring side_by_side_*.png strips. Canonical strips always stay native.
#
# --keep-spools N controls archived render-pool retention for this run. The
# default keeps the newest 8 archives; 0 keeps every archive indefinitely.
#
# --agent-retries N allows N supervisor restarts after the initial Codex
# attempt. Without it, tools/agent-supervisor keeps its configured/default
# total-attempt limit. The value is passed explicitly into the tmux process, so
# it does not depend on a pre-existing tmux server inheriting an environment
# variable.
#
# --timeout-hours HOURS asks the supervisor to stop the run after a fixed
# wall-clock duration. It is primarily useful for detached batches; omitted
# keeps run.sh's historic unlimited behavior.
#
# --agent codex|claude selects the coding agent (default codex, or $ARTSCRIPT_AGENT).
# codex uses the tools/codex-articulated + tools/agent-supervisor pair (OpenAI API,
# OPENAI_API_KEY provisioned once by tools/install_codex_bwrap.sh); claude uses
# their mirrors tools/claude-articulated + tools/claude-supervisor (Anthropic API,
# ANTHROPIC_API_KEY provisioned once by tools/install_claude_bwrap.sh) with the
# same sandbox, egress confinement and supervisor/state.json contract.
# --codex-provider openai / --claude-provider anthropic name the model transport
# explicitly (each is the only choice; the supervisor persists it so restarts and
# resumes stay on it).
#
# --gpu 1 assigns one GPU to this run; --gpus "1,2" assigns a GPU subset. The
# sandbox launcher binds only those /dev/nvidiaN nodes (EEVEE workers physically
# cannot land elsewhere — the only pinning that works for EEVEE, which ignores
# CUDA_VISIBLE_DEVICES), and run_config.json's pool.gpus mirrors it. Default:
# GPU 0 in pool.gpus, all devices visible (previous behavior). Size workers ~16
# per GPU (the pool clamps).
#
# --skip-frames N drops the first N frames of the resolved keyframe list (0 =
# default, skip nothing); --max-kfs N caps it to at most N frames by TRUNCATING
# the tail (keeps the first N, drops the rest). Both compose with --frames /
# --every / the default all-frames set; skip is applied first. Default -1 (max-kfs)
# = no cap. E.g. --every 5 --skip-frames 20 --max-kfs 30 strides by 5, drops the
# first 20 of those keyframes, then keeps the next 30.
#
# Creates RUNS_ROOT/<run_name>_<YYYYmmdd-HHMMSS>/ with iteration and mesh
# subdirectories and prints the AGENT_TASK.md context to hand to a coding agent.
# RUNS_ROOT defaults to REPO/runs and can be changed with --runs-root. The run name is
# timestamped by default so a re-run never clobbers a prior one (--no-timestamp to
# opt out). The agent authors <run_dir>/scene.py (FRAMES/JOINTS/SCALE — see examples/).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Every host-side step below shells out to `micromamba run -n artscript`, which
# needs MAMBA_ROOT_PREFIX; resolve it here so the launcher does not depend on
# which shell started it. Precedence matches tools/install_codex_bwrap.sh's
# detect_mamba_root: an inherited value first, then `micromamba info`, then the
# common install locations -- each accepted only if it actually holds the env.
resolve_mamba_root() {
  local candidate
  if [ -n "${MAMBA_ROOT_PREFIX:-}" ] && [ -d "$MAMBA_ROOT_PREFIX/envs/artscript" ]; then
    printf '%s\n' "$MAMBA_ROOT_PREFIX"; return 0
  fi
  candidate="$(micromamba info 2>/dev/null |
    sed -n 's/.*base environment : *//p' | head -n1)"
  if [ -n "$candidate" ] && [ -d "$candidate/envs/artscript" ]; then
    printf '%s\n' "$candidate"; return 0
  fi
  for candidate in "$HOME/.local/share/mamba" "$HOME/micromamba" \
                   "$HOME/miniforge3" "$HOME/mambaforge" \
                   "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [ -d "$candidate/envs/artscript" ]; then
      printf '%s\n' "$candidate"; return 0
    fi
  done
  return 1
}

command -v micromamba >/dev/null 2>&1 || {
  echo "error: micromamba is not on PATH; run.sh needs the 'artscript' env" >&2
  exit 1
}
MAMBA_ROOT_PREFIX="$(resolve_mamba_root || true)"
[ -n "$MAMBA_ROOT_PREFIX" ] || {
  echo "error: no mamba root containing envs/artscript found." >&2
  echo "  looked at: \$MAMBA_ROOT_PREFIX, 'micromamba info', ~/.local/share/mamba," >&2
  echo "             ~/micromamba, ~/miniforge3, ~/mambaforge, ~/miniconda3, ~/anaconda3" >&2
  echo "  create it with:  micromamba create -y -f $HERE/environment.yml" >&2
  echo "  or point at an existing one:  export MAMBA_ROOT_PREFIX=/path/to/root" >&2
  exit 1
}
export MAMBA_ROOT_PREFIX

CAPTURE_DIR=""
FRAMES=""
EVERY=""
START=""
STOP=""
MAX_KFS="-1"
SKIP_FRAMES="0"
WORKERS=""
NCPU=""
CANDIDATE_SHEET_MAX_DIMENSION="0"
SIDE_BY_SIDE_MAX_DIMENSION="0"
KEEP_SPOOLS="8"
GPUS=""
TIMESTAMP=1
DEPTH_COST=1
DEPTH_REPORT=1
DEPTH_RENDER=1
LAUNCH_CODEX=1
SUPERVISED=1
DETACH=0
RESUME_RUN=""
RUNS_ROOT="${ARTSCRIPT_RUNS_ROOT:-$HERE/runs}"
AGENT_RETRIES=""
TIMEOUT_HOURS=""
CODEX_PROVIDER="${ARTSCRIPT_CODEX_PROVIDER:-}"
AGENT="${ARTSCRIPT_AGENT:-codex}"
CLAUDE_PROVIDER="${ARTSCRIPT_CLAUDE_PROVIDER:-}"
ENABLE_MODULES=()
DISABLE_MODULES=()
[ -z "${ARTSCRIPT_ENABLE_MODULES:-}" ] || ENABLE_MODULES+=("$ARTSCRIPT_ENABLE_MODULES")
[ -z "${ARTSCRIPT_DISABLE_MODULES:-}" ] || DISABLE_MODULES+=("$ARTSCRIPT_DISABLE_MODULES")
POS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --capture) CAPTURE_DIR="${2:-}"; shift 2 ;;
    --frames) FRAMES="${2:-}"; shift 2 ;;
    --every) EVERY="${2:-}"; shift 2 ;;
    --max-kfs) MAX_KFS="${2:-}"; shift 2 ;;
    --skip-frames) SKIP_FRAMES="${2:-}"; shift 2 ;;
    --workers) WORKERS="${2:-}"; shift 2 ;;
    --ncpu) NCPU="${2:-}"; shift 2 ;;
    --candidate-sheet-max-dimension)
      CANDIDATE_SHEET_MAX_DIMENSION="${2:-}"
      shift 2
      ;;
    --side-by-side-max-dimension)
      SIDE_BY_SIDE_MAX_DIMENSION="${2:-}"
      shift 2
      ;;
    --keep-spools)
      [ "$#" -ge 2 ] || { echo "error: --keep-spools requires an integer" >&2; exit 2; }
      KEEP_SPOOLS="$2"
      shift 2
      ;;
    --gpu)
      [ "$#" -ge 2 ] || { echo "error: --gpu requires a GPU id" >&2; exit 2; }
      [ -z "$GPUS" ] || { echo "error: pass --gpu/--gpus only once" >&2; exit 2; }
      case "$2" in
        ''|*[!0-9]*) echo "error: --gpu must be a non-negative integer, got '$2'" >&2; exit 2 ;;
      esac
      GPUS="$2"
      shift 2
      ;;
    --gpus)
      [ "$#" -ge 2 ] || { echo "error: --gpus requires comma-separated GPU ids" >&2; exit 2; }
      [ -z "$GPUS" ] || { echo "error: pass --gpu/--gpus only once" >&2; exit 2; }
      [ -n "$2" ] || { echo "error: --gpus requires comma-separated GPU ids" >&2; exit 2; }
      GPUS="$2"
      shift 2
      ;;
    --start) START="${2:-}"; shift 2 ;;
    --stop) STOP="${2:-}"; shift 2 ;;
    --no-depth-cost) DEPTH_COST=0; shift ;;
    --no-depth-report) DEPTH_REPORT=0; shift ;;
    --no-depth-render) DEPTH_RENDER=0; shift ;;
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
    --no-timestamp) TIMESTAMP=0; shift ;;
    --prompt-only) LAUNCH_CODEX=0; shift ;;
    --detach) DETACH=1; shift ;;
    --direct) SUPERVISED=0; shift ;;
    --agent-retries)
      [ "$#" -ge 2 ] || { echo "error: --agent-retries requires an integer" >&2; exit 2; }
      AGENT_RETRIES="$2"
      shift 2
      ;;
    --timeout-hours)
      [ "$#" -ge 2 ] || { echo "error: --timeout-hours requires a number" >&2; exit 2; }
      TIMEOUT_HOURS="$2"
      shift 2
      ;;
    --codex-provider)
      [ "$#" -ge 2 ] || { echo "error: --codex-provider requires openai" >&2; exit 2; }
      CODEX_PROVIDER="$2"
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
    --resume-run)
      [ "$#" -ge 2 ] || { echo "error: --resume-run requires a run directory" >&2; exit 2; }
      RESUME_RUN="$2"
      shift 2
      ;;
    --runs-root)
      [ "$#" -ge 2 ] || { echo "error: --runs-root requires a directory" >&2; exit 2; }
      [ -n "$2" ] || { echo "error: --runs-root requires a directory" >&2; exit 2; }
      RUNS_ROOT="$2"
      shift 2
      ;;
    *) POS+=("$1"); shift ;;
  esac
done
set -- ${POS[@]+"${POS[@]}"}

case "$CODEX_PROVIDER" in
  "" | openai) ;;
  *) echo "error: --codex-provider must be openai, got '$CODEX_PROVIDER'" >&2; exit 2 ;;
esac
case "$AGENT" in
  codex | claude) ;;
  *) echo "error: --agent must be codex or claude, got '$AGENT'" >&2; exit 2 ;;
esac
case "$CLAUDE_PROVIDER" in
  "" | anthropic) ;;
  *) echo "error: --claude-provider must be anthropic, got '$CLAUDE_PROVIDER'" >&2; exit 2 ;;
esac
# Module names are validated against the registry up front, so a typo fails
# here rather than scaffolding a run with the wrong gate set.
module_flags() {
  local name
  for name in ${ENABLE_MODULES[@]+"${ENABLE_MODULES[@]}"}; do printf '%s\n' --enable "$name"; done
  for name in ${DISABLE_MODULES[@]+"${DISABLE_MODULES[@]}"}; do printf '%s\n' --disable "$name"; done
}
if [ "${#ENABLE_MODULES[@]}" -gt 0 ] || [ "${#DISABLE_MODULES[@]}" -gt 0 ]; then
  micromamba run -n artscript env PYTHONPATH="$HERE/harness" \
    python -m core.modules validate ${ENABLE_MODULES[@]+"${ENABLE_MODULES[@]}"} \
                                    ${DISABLE_MODULES[@]+"${DISABLE_MODULES[@]}"} || exit 2
fi

case "$CANDIDATE_SHEET_MAX_DIMENSION" in
  ''|*[!0-9]*)
    echo "error: --candidate-sheet-max-dimension must be a non-negative integer, got '$CANDIDATE_SHEET_MAX_DIMENSION'" >&2
    exit 2
    ;;
esac

case "$SIDE_BY_SIDE_MAX_DIMENSION" in
  ''|*[!0-9]*)
    echo "error: --side-by-side-max-dimension must be a non-negative integer, got '$SIDE_BY_SIDE_MAX_DIMENSION'" >&2
    exit 2
    ;;
esac

case "$KEEP_SPOOLS" in
  ''|*[!0-9]*)
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

# The agent pair. Everything below refers to these two paths plus the three
# agent_* values, so the codex branch stays byte-for-byte what it always was.
if [ "$AGENT" = "claude" ]; then
  launcher="$HERE/tools/claude-articulated"
  supervisor="$HERE/tools/claude-supervisor"
  agent_label="Claude Code"
  agent_installer="tools/install_claude_bwrap.sh"
  agent_goal_prefix=""             # a plain positional prompt starts the turn
else
  launcher="$HERE/tools/codex-articulated"
  supervisor="$HERE/tools/agent-supervisor"
  agent_label="Codex"
  agent_installer="tools/install_codex_bwrap.sh"
  agent_goal_prefix="/goal "       # Codex goal mode
fi
# The provider flag each agent's launcher and supervisor understand; empty when
# the caller left the choice to the supervisor's persisted state (resume).
agent_provider_args() {
  if [ "$AGENT" = "claude" ]; then
    [ -z "$CLAUDE_PROVIDER" ] || printf '%s\n' --claude-provider "$CLAUDE_PROVIDER"
  else
    [ -z "$CODEX_PROVIDER" ] || printf '%s\n' --codex-provider "$CODEX_PROVIDER"
  fi
}

mkdir -p "$RUNS_ROOT"
RUNS_ROOT="$(cd "$RUNS_ROOT" && pwd)"

AGENT_MAX_ATTEMPTS=""
AGENT_MAX_ROTATIONS=""
if [ -n "$AGENT_RETRIES" ]; then
  case "$AGENT_RETRIES" in
    *[!0-9]*|'')
      echo "error: --agent-retries must be a non-negative integer, got '$AGENT_RETRIES'" >&2
      exit 2
      ;;
  esac
  AGENT_MAX_ATTEMPTS="$((AGENT_RETRIES + 1))"
  AGENT_MAX_ROTATIONS="$AGENT_RETRIES"
fi

# Resume is deliberately handled before capture validation or scaffold writes.
# This path may only read existing run metadata and create supervisor/ runtime
# state, so it cannot overwrite NOTES.md, layout.json, scene.py, or reports.
if [ -n "$RESUME_RUN" ]; then
  [ "$#" -eq 0 ] || { echo "error: --resume-run does not accept a run name" >&2; exit 2; }
  [ "$LAUNCH_CODEX" -eq 1 ] || { echo "error: --prompt-only cannot be used with --resume-run" >&2; exit 2; }
  [ "$SUPERVISED" -eq 1 ] || { echo "error: --direct cannot safely resume an existing run" >&2; exit 2; }
  [ -d "$RESUME_RUN" ] || { echo "error: run directory not found: $RESUME_RUN" >&2; exit 2; }
  RESUME_RUN="$(cd "$RESUME_RUN" && pwd)"
  [ -f "$RESUME_RUN/layout.json" ] || {
    echo "error: not a reconstruction run (layout.json missing): $RESUME_RUN" >&2
    exit 2
  }
  [ -x "$launcher" ] || { echo "error: isolated $agent_label launcher is missing: $launcher" >&2; exit 1; }
  [ -x "$supervisor" ] || { echo "error: agent supervisor is missing: $supervisor" >&2; exit 1; }
  # The module set is part of the run (its task document was assembled from
  # it, and the gates it ran under are what its results mean): a resume cannot
  # change it.
  [ "${#ENABLE_MODULES[@]}" -eq 0 ] && [ "${#DISABLE_MODULES[@]}" -eq 0 ] || {
    echo "error: --enable/--disable are fixed at scaffold; a resume keeps $RESUME_RUN/modules.json" >&2
    exit 2
  }
  resume_args=(resume --run-dir "$RESUME_RUN" --launcher "$launcher")
  mapfile -t provider_args < <(agent_provider_args)
  resume_args+=(${provider_args[@]+"${provider_args[@]}"})
  [ -z "$AGENT_MAX_ATTEMPTS" ] || resume_args+=(--max-attempts "$AGENT_MAX_ATTEMPTS")
  [ -z "$AGENT_MAX_ROTATIONS" ] || resume_args+=(--max-rotations "$AGENT_MAX_ROTATIONS")
  [ -z "$TIMEOUT_HOURS" ] || resume_args+=(--timeout-hours "$TIMEOUT_HOURS")
  [ "$DETACH" -eq 1 ] && resume_args+=(--detach)
  exec "$supervisor" "${resume_args[@]}"
fi

CODEX_PROVIDER="${CODEX_PROVIDER:-openai}"
CLAUDE_PROVIDER="${CLAUDE_PROVIDER:-anthropic}"

[ "$DETACH" -eq 0 ] || [ "$LAUNCH_CODEX" -eq 1 ] || {
  echo "error: --detach requires an agent launch" >&2
  exit 2
}
[ "$DETACH" -eq 0 ] || [ "$SUPERVISED" -eq 1 ] || {
  echo "error: --detach cannot be used with --direct" >&2
  exit 2
}
[ -z "$AGENT_MAX_ATTEMPTS" ] || [ "$LAUNCH_CODEX" -eq 1 ] || {
  echo "error: --agent-retries requires an agent launch" >&2
  exit 2
}
[ -z "$AGENT_MAX_ATTEMPTS" ] || [ "$SUPERVISED" -eq 1 ] || {
  echo "error: --agent-retries requires the supervisor (omit --direct)" >&2
  exit 2
}

# Normalize and validate the multi-GPU form before creating any run files.
if [ -n "$GPUS" ]; then
  GPUS="${GPUS//[[:space:]]/}"
  if [[ ! "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "error: --gpus must be comma-separated GPU ids, got '$GPUS'" >&2
    exit 2
  fi
fi

# The Bubblewrap launcher exposes this repo at this fixed in-sandbox path (must
# match SANDBOX_PROJECT in tools/codex-articulated — the launcher owns the mount).
SANDBOX_PROJECT="/home/ubuntu/articulated_script"
SANDBOX_RUNS="$SANDBOX_PROJECT/runs"

# Printed prompts retain host paths; directly launched prompts use sandbox paths.
prompt_path() {
  local path="$1"
  if [ "$LAUNCH_CODEX" -eq 1 ]; then
    case "$path" in
      "$RUN_DIR") echo "$SANDBOX_RUNS/$(basename "$RUN_DIR")" ;;
      "$RUN_DIR"/*)
        echo "$SANDBOX_RUNS/$(basename "$RUN_DIR")/${path#"$RUN_DIR"/}"
        ;;
      "$HERE") echo "$SANDBOX_PROJECT" ;;
      "$HERE"/*) echo "$SANDBOX_PROJECT/${path#"$HERE"/}" ;;
      *)
        echo "error: isolated $agent_label cannot expose a path outside the project: $path" >&2
        exit 1
        ;;
    esac
  else
    echo "$path"
  fi
}

emit_or_launch_prompt() {
  local prompt="$1"
  if [ "$LAUNCH_CODEX" -eq 1 ]; then
    [ -x "$launcher" ] || {
      echo "error: isolated $agent_label launcher is missing: $launcher" >&2
      echo "run $agent_installer first" >&2
      exit 1
    }
    echo "Launching isolated $agent_label for $RUN_DIR"
    mapfile -t provider_args < <(agent_provider_args)
    # --gpus narrows the sandbox to that /dev/nvidiaN subset (SANDBOX_GPUS is
    # the launcher's knob). EEVEE inside the sandbox uses the first bound GPU,
    # so one GPU per run is the effective grain — run several runs on
    # different --gpus to use the whole box.
    if [ "$SUPERVISED" -eq 1 ]; then
      [ -x "$supervisor" ] || {
        echo "error: agent supervisor is missing: $supervisor" >&2
        exit 1
      }
      local supervisor_args=(start --run-dir "$RUN_DIR" --launcher "$launcher" --prompt "$prompt")
      supervisor_args+=("${provider_args[@]}")
      [ -n "$GPUS" ] && supervisor_args+=(--gpus "$GPUS")
      [ -z "$AGENT_MAX_ATTEMPTS" ] || supervisor_args+=(--max-attempts "$AGENT_MAX_ATTEMPTS")
      [ -z "$AGENT_MAX_ROTATIONS" ] || supervisor_args+=(--max-rotations "$AGENT_MAX_ROTATIONS")
      [ -z "$TIMEOUT_HOURS" ] || supervisor_args+=(--timeout-hours "$TIMEOUT_HOURS")
      [ "$DETACH" -eq 1 ] && supervisor_args+=(--detach)
      exec "$supervisor" "${supervisor_args[@]}"
    fi
    if [ -n "$GPUS" ]; then
      exec env SANDBOX_GPUS="$GPUS" "$launcher" --run-dir "$RUN_DIR" \
        "${provider_args[@]}" "${agent_goal_prefix}${prompt}"
    fi
    exec "$launcher" --run-dir "$RUN_DIR" \
      "${provider_args[@]}" "${agent_goal_prefix}${prompt}"
  fi

  echo "Copy-paste this prompt to launch a reconstruction agent:"
  echo "================================================================"
  printf '%s\n' "$prompt"
  echo "================================================================"
}

# Append a _YYYYmmdd-HHMMSS stamp to the run name so a re-run never clobbers a
# prior one and runs/ sorts chronologically. Opt out with --no-timestamp.
stamp_run_name() {
  if [ "$TIMESTAMP" -eq 1 ]; then echo "$1_$(date +%Y%m%d-%H%M%S)"; else echo "$1"; fi
}

# Resolve the depth backend from the CAPTURE MANIFEST (its "depth" field:
# pi3x | none), look up that backend's confidence FLOOR in
# harness/depth_config.json, and write RUN_DIR/depth_config.json
# ({backend, conf_thr, cost, report, render}); the depth scorers + the sweep
# auto-resolve that per-run file (walking up from the render path).
# `cost`/`report`/`render` are the three depth flip switches (default on;
# --no-depth-cost / --no-depth-report / --no-depth-render scaffold them off).
# A depth-less capture (depth: none) forces cost+report OFF — masks-only mode —
# but NOT `render`: our own depth render needs no GT, and the monocular case is
# exactly where the object-units DEPTH panel / depth sheet earn their keep.
# --no-depth-render is the knob for a run that should not pay for depth at all.
# stdout is the resolved backend name (for the launch prompt); the human
# summary goes to stderr.
write_depth_config() {
  local run_dir="$1"
  local cfg="$HERE/harness/depth_config.json"
  micromamba run -n artscript python - "$cfg" "$CAPTURE_DIR" \
      "$run_dir/depth_config.json" "$DEPTH_COST" "$DEPTH_REPORT" \
      "$DEPTH_RENDER" <<'PY'
import json, sys
cfg_path, capture_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
cost, report = sys.argv[4] == "1", sys.argv[5] == "1"
render = sys.argv[6] == "1"
backend = json.load(open(f"{capture_dir}/capture.json")).get("depth", "none")
backends = json.load(open(cfg_path)).get("backends", {})
weight = None
if backend == "none":
    # no observed depth to compare against -> no cost, no scoring. `render` is
    # deliberately NOT forced off: it feeds the GT-free object-units visuals,
    # which are the whole point of the monocular case.
    cost = report = False
    conf_thr = 0.0
elif backend in backends:
    b = backends[backend]
    # object form {conf_thr, weight}; a bare number is the legacy floor-only form
    if isinstance(b, dict):
        conf_thr = b.get("conf_thr", 0.1)
        weight = b.get("weight")
    else:
        conf_thr = b
else:
    sys.exit(f"error: capture depth backend '{backend}' unknown; "
             f"valid: none, {', '.join(sorted(backends))}")
out = {"backend": backend, "conf_thr": conf_thr,
       "cost": cost, "report": report, "render": render}
if weight is not None:
    # the backend's default sweep depth-supervision weight; an explicit
    # --sweep-depth-weight / order depth_weight still wins (core.depth_config)
    out["weight"] = weight
json.dump(out, open(out_path, "w"), indent=2)
print(f"depth backend: {backend} (conf_thr={conf_thr}, weight={weight}, "
      f"cost={cost}, report={report}, render={render})", file=sys.stderr)
print(backend)
PY
}

# Scaffold RUN_DIR/run_config.json — a starter config for shape_pass.sh's G/N
# concurrency knobs + visual presentation (precedence: CLI flag > this json >
# built-in default; see harness/utils/shape_pass.md). The production profile itself
# (pool on, 16 workers/ncpu per GPU, alpha-backed panels) lives beside the schema in
# harness/utils/_run_config.py:scaffold_config — NOT here — so the scaffold and
# the config reader can't drift apart. Guarded on existence: a --no-timestamp
# re-run never clobbers a hand-edited config (unlike layout.json/
# depth_config.json, which are derived from run.sh's args and so are always
# rewritten).
write_run_config() {
  local run_dir="$1"
  micromamba run -n artscript env PYTHONPATH="$HERE/harness" \
    python -m utils._run_config "$run_dir" \
    --workers "${WORKERS:-16}" --ncpu "${NCPU:-16}" --gpus "${GPUS:-0}" \
    --candidate-sheet-max-dimension "$CANDIDATE_SHEET_MAX_DIMENSION" \
    --side-by-side-max-dimension "$SIDE_BY_SIDE_MAX_DIMENSION" \
    --keep-spools "$KEEP_SPOOLS"
}

# ---- multi-frame (capture) path -----------------------------------------------
if [ -n "$CAPTURE_DIR" ]; then
  if [ -n "$FRAMES" ] && [ -n "$EVERY" ]; then
    echo "error: pass either --frames or --every, not both" >&2
    exit 2
  fi
  CAPTURE_DIR="$(cd "$CAPTURE_DIR" && pwd)"

  # Validate the capture first — a broken capture should fail HERE, not three
  # tools deep into a shape pass.
  micromamba run -n artscript env PYTHONPATH="$HERE" \
    python -m datasets.common.validate "$CAPTURE_DIR"

  FRAMES_DIR="$CAPTURE_DIR/frames"
  MASKS_DIR="$CAPTURE_DIR/mask_object"
  HAND_MASKS_DIR="$CAPTURE_DIR/mask_hand"
  # --frames/--every default: ALL frames from the manifest
  if [ -z "$FRAMES" ] && [ -z "$EVERY" ]; then
    FRAMES="$(micromamba run -n artscript env PYTHONPATH="$HERE" python - "$CAPTURE_DIR" <<'PY'
import json, sys
print(",".join(json.load(open(f"{sys.argv[1]}/capture.json"))["frames"]))
PY
)"
  fi

  # --every N: sample every Nth frame (stride by frame number) from FRAMES_DIR,
  # via analysis.frames — the one place that owns the frame-naming convention.
  if [ -n "$EVERY" ]; then
    sample_args=(--frames-dir "$FRAMES_DIR" --every "$EVERY" --emit names)
    [ -n "$START" ] && sample_args+=(--start "$START")
    [ -n "$STOP" ] && sample_args+=(--stop "$STOP")
    FRAMES="$(micromamba run -n artscript env PYTHONPATH="$HERE/harness" \
      python -m analysis.frames sample "${sample_args[@]}")" || {
        echo "error: --every $EVERY sampled no frames from $FRAMES_DIR" >&2; exit 2; }
    echo "Sampled every ${EVERY}th frame: $FRAMES"
  fi

  # --skip-frames N: drop the first N frames from the resolved keyframe list
  # (--skip-frames 100 skips the first 100). Applied BEFORE --max-kfs, so
  # "skip first N, then keep at most M" composes. 0 (default) = skip nothing.
  if [ "$SKIP_FRAMES" != "0" ]; then
    case "$SKIP_FRAMES" in
      ''|*[!0-9]*) echo "error: --skip-frames must be >= 0, got '$SKIP_FRAMES'" >&2; exit 2 ;;
    esac
    IFS=',' read -r -a _SKIP_ARR <<< "$FRAMES"
    _SKIP_ARR=("${_SKIP_ARR[@]:$SKIP_FRAMES}")
    if [ "${#_SKIP_ARR[@]}" -eq 0 ]; then
      echo "error: --skip-frames $SKIP_FRAMES dropped every resolved frame" >&2
      exit 2
    fi
    FRAMES="$(IFS=','; echo "${_SKIP_ARR[*]}")"
    echo "Skipped first ${SKIP_FRAMES}: ${#_SKIP_ARR[@]} frames remain -> $FRAMES"
  fi

  # --max-kfs N: cap the resolved keyframe list to at most N frames by TRUNCATING
  # the tail — keep the first N, drop the rest — so it composes with --every /
  # --frames / the default all-frames set. -1 (default) = no cap.
  if [ "$MAX_KFS" != "-1" ]; then
    case "$MAX_KFS" in
      ''|*[!0-9]*|0) echo "error: --max-kfs must be >= 1 (or -1 for no cap), got '$MAX_KFS'" >&2; exit 2 ;;
    esac
    IFS=',' read -r -a _CAP_ARR <<< "$FRAMES"
    _CAP_ARR=("${_CAP_ARR[@]:0:$MAX_KFS}")
    FRAMES="$(IFS=','; echo "${_CAP_ARR[*]}")"
    echo "Capped to --max-kfs ${MAX_KFS}: ${#_CAP_ARR[@]} frames -> $FRAMES"
  fi

  RUN_NAME="${1:-}"
  if [ -z "$RUN_NAME" ]; then RUN_NAME="$(basename "$CAPTURE_DIR")_multiview"; fi
  RUN_NAME="$(stamp_run_name "$RUN_NAME")"

  RUN_DIR="$RUNS_ROOT/$RUN_NAME"
  mkdir -p "$RUN_DIR/mesh"
  RUN_DIR="$(cd "$RUN_DIR" && pwd)"
  echo "# run: $RUN_NAME (multi-frame)" > "$RUN_DIR/NOTES.md"

  IFS=',' read -r -a FRAME_ARR <<< "$FRAMES"
  echo "Run directory: $RUN_DIR"
  echo "CAPTURE:       $CAPTURE_DIR"
  echo "FRAMES:        $FRAMES"
  echo
  echo "Per-frame inputs (matched by basename):"
  printf '%-14s %-40s %-40s %s\n' "frame" "image" "object mask" "hand mask"
  # A frame without its image or object mask is a hard error, not a <MISSING>
  # table cell an unattended launch scrolls past: a detached agent would burn
  # its whole budget against inputs that were never there. Hand masks stay
  # optional (<none>) — captures without them are valid.
  MISSING_INPUTS=()
  for fr in "${FRAME_ARR[@]}"; do
    stem="${fr%.*}"
    img="$FRAMES_DIR/$fr"
    msk="$MASKS_DIR/${stem}.png"
    hnd="$HAND_MASKS_DIR/${stem}.png"
    printf '%-14s %-40s %-40s %s\n' "$fr" \
      "$([ -f "$img" ] && echo "$img" || echo "<MISSING>")" \
      "$([ -f "$msk" ] && echo "$msk" || echo "<MISSING>")" \
      "$([ -f "$hnd" ] && echo "$hnd" || echo "<none>")"
    [ -f "$img" ] && [ -f "$msk" ] || MISSING_INPUTS+=("$fr")
  done
  if [ "${#MISSING_INPUTS[@]}" -gt 0 ]; then
    echo "error: ${#MISSING_INPUTS[@]} frame(s) lack an image or object mask" \
         "in $CAPTURE_DIR: ${MISSING_INPUTS[*]}" >&2
    exit 2
  fi
  REF_FRAME="${FRAME_ARR[0]}"
  REF_IMG="$FRAMES_DIR/$REF_FRAME"

  # Persist the resolved layout so harness/utils/shape_pass.sh can read it once instead
  # of re-deriving the frame/mask paths every pass (single source of truth).
  # "capture" is the run's input root; its tracking/ subdir carries cameras.npz +
  # keyframes.json and depth/ the per-frame
  # pointmaps. NOTE: this is the RUN layout (layout.json), distinct from the
  # per-pass MANIFEST.json written inside iterations/<NNNNNN>/renders/<NNNN>/.
  frames_json=""
  for fr in "${FRAME_ARR[@]}"; do
    [ -n "$frames_json" ] && frames_json="$frames_json, "
    frames_json="$frames_json\"$fr\""
  done

  # layout.json is consumed FROM WITHIN the sandbox (shape_pass.sh et al. read it at
  # /home/ubuntu/articulated_script/runs/...), so its path VALUES must be the
  # paths as seen there — same host->sandbox rewrite prompt_path applies to the
  # prompt. On a --prompt-only host run these stay host paths (identity map).
  PROMPT_HERE="$(prompt_path "$HERE")"
  PROMPT_CAPTURE="$(prompt_path "$CAPTURE_DIR")"
  PROMPT_FRAMES_DIR="$(prompt_path "$FRAMES_DIR")"
  PROMPT_MASKS_DIR="$(prompt_path "$MASKS_DIR")"
  PROMPT_HAND_MASKS_DIR="$(prompt_path "$HAND_MASKS_DIR")"
  PROMPT_RUN_DIR="$(prompt_path "$RUN_DIR")"

  cat > "$RUN_DIR/layout.json" <<EOF
{
  "kind": "multiview",
  "capture": "$PROMPT_CAPTURE",
  "frames_dir": "$PROMPT_FRAMES_DIR",
  "masks_dir": "$PROMPT_MASKS_DIR",
  "hand_masks_dir": "$PROMPT_HAND_MASKS_DIR",
  "ref_frame": "$REF_FRAME",
  "frames": [$frames_json]
}
EOF

  DEPTH_BACKEND="$(write_depth_config "$RUN_DIR")"
  write_run_config "$RUN_DIR"
  # modules.json: which optional harness modules this run carries. Always
  # rewritten like layout.json (derived from run.sh's args); the launcher mounts
  # it read-only inside the sandbox.
  mapfile -t module_args < <(module_flags)
  micromamba run -n artscript env PYTHONPATH="$HERE/harness" \
    python -m core.modules scaffold "$RUN_DIR" ${module_args[@]+"${module_args[@]}"}
  # The run's task document: AGENT_TASK.md at the project root is the TEMPLATE;
  # the agent is pointed at this assembled copy (also read-only in the sandbox),
  # which carries only the sections of the modules the run has.
  micromamba run -n artscript env PYTHONPATH="$HERE/harness" \
    python -m core.task_doc --template "$HERE/AGENT_TASK.md" --run-dir "$RUN_DIR"
  MODULES_SUMMARY="$(micromamba run -n artscript env PYTHONPATH="$HERE/harness" \
    python -m core.modules show "$RUN_DIR")"
  micromamba run -n artscript env PYTHONPATH="$HERE/harness" \
    python -m bookkeeping init --run-dir "$RUN_DIR"

  # measure_depth reads the capture's observed pointmaps; a cameras-only capture
  # (depth: none) has nothing for it to measure, so the hint is dropped.
  MEASURE_HINT=""
  if [ "$DEPTH_BACKEND" != "none" ]; then
    MEASURE_HINT="
  # (optional, first pass) seed poses + shared SCALE off the tracking pointmap:
  micromamba run -n artscript env PYTHONPATH=harness python -m analysis.measure_depth \\
      --run-dir $PROMPT_RUN_DIR --out $PROMPT_RUN_DIR/measure_multi.json"
  fi

  PROMPT="$(cat <<PROMPT_EOF
Follow the task in $PROMPT_RUN_DIR/AGENT_TASK.md (its Working conventions are always in
force — reuse existing tools, record any wished-for tool in RUN_DIR/wish.md, and
reconstruct fresh without consulting other runs/).

Work from project root: $PROMPT_HERE
CAPTURE=$PROMPT_CAPTURE
FRAMES=$FRAMES        # ref = $REF_FRAME
FRAMES_DIR=$PROMPT_FRAMES_DIR
MASKS_DIR=$PROMPT_MASKS_DIR            # object mask
HAND_MASKS_DIR=$PROMPT_HAND_MASKS_DIR  # hand occluder -> --hand-mask
RUN_DIR=$PROMPT_RUN_DIR
DEPTH_BACKEND=$DEPTH_BACKEND
MODULES=$MODULES_SUMMARY   # this run's optional harness modules (RUN_DIR/modules.json, read-only)

Iterate with one command per pass (the harness allocates the iteration number):
  harness/utils/shape_pass.sh $PROMPT_RUN_DIR --frames $FRAMES
  # Never step blind: every pass open >=1 side_by_side_<frame>.png and confirm
  # identity + pose before proceeding.
$MEASURE_HINT
PROMPT_EOF
)"
  echo
  emit_or_launch_prompt "$PROMPT"
  exit 0
fi

echo "usage: run.sh --capture DIR [--agent codex|claude] [--codex-provider openai] [--claude-provider anthropic] [--runs-root DIR] [--frames \"a.jpg,b.jpg\" | --every 10] [--skip-frames N] [--max-kfs N] [--workers N] [--ncpu N] [--candidate-sheet-max-dimension N] [--side-by-side-max-dimension N] [--timeout-hours HOURS] [--gpu N | --gpus \"1,2\"] [run_name] [--start N] [--stop N] [--no-depth-cost] [--no-depth-report] [--no-depth-render] [--enable MODULE] [--disable MODULE] [--no-timestamp] [--prompt-only] [--detach] [--direct]" >&2
echo "       run.sh --resume-run RUN_DIR [--detach]" >&2
echo "(build a capture from frames + masks with tools/make_capture.py)" >&2
exit 2
