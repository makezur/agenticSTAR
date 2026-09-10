"""modules.py — the ONE reader for a run's modules.json: which optional harness
modules (gates + their views + their instructions) this run has.

A "module" is a piece of the harness that a run can carry or not: today only
`mechanism` (the blind A/B joint-axis pick — its `mechanism` view, the
`analysis.mechanism_calls` gate, the `multiagent.windows plan` refusal, and the
finalization-lock line). A module is an ablation asked with a per-run switch,
so the harness and the task document stay ONE codebase and the run records what
it ran with.

    RUN_DIR/modules.json     {"schema": 1, "modules": {"mechanism": false}}

`run.sh` writes it at scaffold time (`--enable NAME` / `--disable NAME`, else
DEFAULTS), and the sandbox launcher mounts it READ-ONLY over the writable run
bind — so the agent inside can read its module set but not change it (and the
harness that reads it is itself read-only in there). There is deliberately NO
environment override in this module: `ARTSCRIPT_X=0 python -m ...` from inside
the sandbox must not be a way to switch a gate off. Resumes never rewrite it.

Readers: `enabled(run_dir, name)` when the caller knows RUN_DIR (shape_pass,
windows, mechanism_calls); `find_modules(start_path)` walks UP like
core.depth_config / core.run_layout for the render side, which only holds an
out path. An ABSENT file resolves to DEFAULTS (a run scaffolded before this
existed, or a --prompt-only host run). A file that exists but is malformed is an
error, never a silent default: the harness wrote it and the agent cannot edit
it, so a bad file is a bug, not a state.

bpy-free: imported from Blender's python (render_wrapper) and the analysis env.
"""

import argparse
import json
import os
import sys

FILE_NAME = "modules.json"
SCHEMA = 1

# name -> what switching it off removes. The registry: an unknown name is an
# error everywhere (run.sh, this CLI), so a typo cannot silently run
# with the gate on.
MODULES = {
    "mechanism": ("the blind A/B joint-axis pick: the `mechanism` view + sheets, "
                  "the analysis.mechanism_calls gate, the multiagent.windows plan "
                  "refusal, and the finalization-lock line"),
}

# What a run gets when run.sh is given neither --enable nor --disable for a
# module, AND what an absent modules.json means. The default is OFF; enable
# with `run.sh --enable mechanism`.
DEFAULTS = {
    "mechanism": False,
}


class ModulesError(ValueError):
    """An unknown module name or an unreadable modules.json."""


def path_for(run_dir):
    return os.path.join(os.path.abspath(run_dir), FILE_NAME)


def validate_names(names):
    """The names as a list, or ModulesError naming the first unknown one."""
    out = []
    for name in names or ():
        name = str(name).strip()
        if not name:
            continue
        if name not in MODULES:
            raise ModulesError(
                f"unknown module {name!r} (known: {', '.join(sorted(MODULES))})")
        out.append(name)
    return out


def parse_spec(spec):
    """'a,b' / ['a', 'b,c'] / None -> validated ['a', 'b', 'c']."""
    if spec is None:
        return []
    if isinstance(spec, str):
        spec = [spec]
    names = []
    for item in spec:
        names.extend(part for part in str(item).split(",") if part.strip())
    return validate_names(names)


def resolve(enable=(), disable=(), defaults=DEFAULTS):
    """The full {name: bool} for a scaffold: DEFAULTS, then --enable/--disable.
    A name in both is an error — there is no sensible precedence."""
    enable, disable = validate_names(enable), validate_names(disable)
    both = sorted(set(enable) & set(disable))
    if both:
        raise ModulesError(f"module(s) both enabled and disabled: {', '.join(both)}")
    modules = dict(defaults)
    for name in enable:
        modules[name] = True
    for name in disable:
        modules[name] = False
    return modules


def _coerce(doc, where):
    """A parsed modules.json -> {name: bool} over EVERY registered module
    (unlisted names take DEFAULTS), or ModulesError."""
    if not isinstance(doc, dict) or not isinstance(doc.get("modules"), dict):
        raise ModulesError(f"{where}: expected {{\"schema\": {SCHEMA}, "
                           "\"modules\": {name: bool}}")
    if doc.get("schema") != SCHEMA:
        raise ModulesError(f"{where}: schema {doc.get('schema')!r}, "
                           f"expected {SCHEMA}")
    modules = dict(DEFAULTS)
    for name, value in doc["modules"].items():
        if name not in MODULES:
            raise ModulesError(f"{where}: unknown module {name!r}")
        if isinstance(value, dict):        # {"enabled": bool, "reason": ...}
            value = value.get("enabled")
        if not isinstance(value, bool):
            raise ModulesError(f"{where}: module {name!r} must be true/false")
        modules[name] = value
    return modules


def load(run_dir):
    """RUN_DIR's module set as {name: bool} over every registered module.
    Absent file -> DEFAULTS; present-but-bad file -> ModulesError."""
    path = path_for(run_dir)
    try:
        with open(path) as f:
            doc = json.load(f)
    except FileNotFoundError:
        return dict(DEFAULTS)
    except (OSError, ValueError) as e:
        raise ModulesError(f"{path}: unreadable ({e})")
    return _coerce(doc, path)


def enabled(run_dir, name):
    """Whether module `name` is on for RUN_DIR (see `load`)."""
    validate_names([name])
    return load(run_dir)[name]


def find_modules(start_path):
    """The module set of the run that owns `start_path` (a render out path /
    pass dir), walking UP to the nearest modules.json. Same walk as
    core.run_layout.find_layout; no file anywhere -> DEFAULTS."""
    d = os.path.abspath(start_path)
    if not os.path.isdir(d):
        d = os.path.dirname(d)
    prev = None
    while d and d != prev:
        if os.path.isfile(os.path.join(d, FILE_NAME)):
            return load(d)
        prev, d = d, os.path.dirname(d)
    return dict(DEFAULTS)


def enabled_at(start_path, name):
    """`enabled` for a caller that only holds an output path (render side)."""
    validate_names([name])
    return find_modules(start_path)[name]


def write(run_dir, modules, reason=""):
    """Write RUN_DIR/modules.json. Every registered module is listed
    explicitly, so the file reads as the run's full contract without knowing
    the harness's DEFAULTS."""
    validate_names(modules)
    doc = {"schema": SCHEMA,
           "modules": {name: bool(modules.get(name, DEFAULTS[name]))
                       for name in sorted(MODULES)}}
    if reason:
        doc["reason"] = str(reason)
    path = path_for(run_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return path


def disabled_line(name):
    """The one sentence every tool prints when it skips a disabled module —
    the same words everywhere so the agent recognizes the state, not a bug."""
    return (f"module '{name}' is DISABLED for this run ({FILE_NAME}): "
            f"{MODULES[name]} — nothing is owed here")


def summary(modules):
    """'mechanism=off' style one-liner for logs and prompts."""
    return ", ".join(f"{n}={'on' if modules[n] else 'off'}"
                     for n in sorted(MODULES))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("scaffold", help="write RUN_DIR/modules.json from "
                        "DEFAULTS + --enable/--disable (run.sh calls this)")
    sw.add_argument("run_dir")
    sw.add_argument("--enable", action="append", default=[],
                    metavar="NAME[,NAME]")
    sw.add_argument("--disable", action="append", default=[],
                    metavar="NAME[,NAME]")
    sw.add_argument("--reason", default="")
    ss = sub.add_parser("show", help="print RUN_DIR's module set, one per line")
    ss.add_argument("run_dir")
    ss.add_argument("--json", action="store_true")
    sv = sub.add_parser("validate", help="exit non-zero on an unknown name "
                        "(launchers validate flags before scaffolding)")
    sv.add_argument("names", nargs="*", metavar="NAME[,NAME]")
    a = p.parse_args(argv)
    try:
        if a.cmd == "scaffold":
            modules = resolve(parse_spec(a.enable), parse_spec(a.disable))
            path = write(a.run_dir, modules, reason=a.reason)
            print(f"[modules] {path}: {summary(modules)}")
        elif a.cmd == "show":
            modules = load(a.run_dir)
            if a.json:
                print(json.dumps(modules, indent=2, sort_keys=True))
            else:
                print(summary(modules))
        else:
            parse_spec(a.names)
    except ModulesError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
