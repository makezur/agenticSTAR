"""task_doc.py — assemble a run's task document from the AGENT_TASK.md template
and the run's module set (core/modules.py).

AGENT_TASK.md at the project root is a TEMPLATE: prose that belongs to an
optional module sits between HTML-comment markers, each on its own line —

    <!-- if module:mechanism -->
    ...the lines a run WITH the module reads...
    <!-- else -->
    ...what a run WITHOUT it reads instead (optional arm)...
    <!-- endif -->

`render` keeps the arm the run's modules.json selects and drops the marker lines,
so with every module on the output is the template minus its markers — the
document every run read before modules existed. The `else` arm exists so a
disabled module leaves no dangling reference (a numbered step that vanished, a
"see Mechanism below" pointing at nothing): it says in one line that the module
is off, so the agent stops looking for sheets this run never renders.

run.sh renders RUN_DIR/AGENT_TASK.md at scaffold time and the sandbox launcher
mounts it READ-ONLY beside modules.json: the agent reads the document it was
given and cannot edit its own instructions. Blocks do not nest — a module is a
flat on/off, and nesting would make the template unreadable for the humans who
maintain it.

bpy-free; stdlib only.
"""

import argparse
import os
import re
import sys

from core import modules as modules_lib

_IF = re.compile(r"^\s*<!--\s*if\s+module:([A-Za-z0-9_]+)\s*-->\s*$")
_ELSE = re.compile(r"^\s*<!--\s*else\s*-->\s*$")
_ENDIF = re.compile(r"^\s*<!--\s*endif\s*-->\s*$")

TEMPLATE_NAME = "AGENT_TASK.md"


class TemplateError(ValueError):
    """Unbalanced or nested markers, or a module the registry does not know."""


def _blocks(lines):
    """Yield (kind, module, line) per template line: kind is 'text' with the
    module whose arm the line is in (None outside any block, '!name' in an
    else-arm), or 'marker'. Raises TemplateError on malformed structure."""
    current = None          # module name of the open block
    in_else = False
    for n, line in enumerate(lines, 1):
        m = _IF.match(line)
        if m:
            if current is not None:
                raise TemplateError(f"line {n}: nested 'if module' inside "
                                    f"module:{current}")
            name = m.group(1)
            if name not in modules_lib.MODULES:
                raise TemplateError(f"line {n}: unknown module {name!r} (known: "
                                    f"{', '.join(sorted(modules_lib.MODULES))})")
            current, in_else = name, False
            yield "marker", None, line
            continue
        if _ELSE.match(line):
            if current is None or in_else:
                raise TemplateError(f"line {n}: 'else' without an open "
                                    "'if module' (or a second else)")
            in_else = True
            yield "marker", None, line
            continue
        if _ENDIF.match(line):
            if current is None:
                raise TemplateError(f"line {n}: 'endif' without an open "
                                    "'if module'")
            current, in_else = None, False
            yield "marker", None, line
            continue
        if current is None:
            yield "text", None, line
        else:
            yield "text", ("!" + current if in_else else current), line
    if current is not None:
        raise TemplateError(f"unterminated 'if module:{current}' block")


def module_names(template_text):
    """Every module the template conditions on (validation for the tests)."""
    names = set()
    for kind, module, _ in _blocks(template_text.splitlines(keepends=True)):
        if module is not None:
            names.add(module.lstrip("!"))
    return names


def render(template_text, modules, header=None):
    """The document for a run with module set `modules` ({name: bool}).
    `header` (a list of lines, or None) is prepended as an HTML comment."""
    out = []
    if header:
        out.append("<!--\n")
        out.extend(line.rstrip("\n") + "\n" for line in header)
        out.append("-->\n")
    for kind, module, line in _blocks(template_text.splitlines(keepends=True)):
        if kind == "marker":
            continue
        if module is None:
            out.append(line)
        elif module.startswith("!"):
            if not modules.get(module[1:], True):
                out.append(line)
        elif modules.get(module, True):
            out.append(line)
    return "".join(out)


def strip_markers(template_text):
    """The template with every marker line removed and every arm kept —
    i.e. `render` with all modules on AND the else-arms too. Only for tests
    that reason about the markup itself; not a document anyone reads."""
    return "".join(
        line for kind, _, line in _blocks(template_text.splitlines(keepends=True))
        if kind != "marker")


def assemble(template_path, run_dir, out_path=None):
    """Render `template_path` for the run at `run_dir` (its modules.json, or
    DEFAULTS) into `out_path` (default RUN_DIR/AGENT_TASK.md). Returns the path."""
    run_dir = os.path.abspath(run_dir)
    modules = modules_lib.load(run_dir)
    with open(template_path) as f:
        template = f.read()
    header = [
        f"Assembled by harness/core/task_doc.py from {os.path.basename(template_path)} "
        "for this run.",
        f"Modules: {modules_lib.summary(modules)} (RUN_DIR/{modules_lib.FILE_NAME}).",
        "This file is the run's task document and is mounted read-only in the "
        "sandbox; the template at the project root is what to edit.",
    ]
    text = render(template, modules, header=header)
    out_path = out_path or os.path.join(run_dir, TEMPLATE_NAME)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, out_path)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template", required=True, help="the AGENT_TASK.md template")
    p.add_argument("--run-dir", required=True,
                   help="the run whose modules.json selects the arms")
    p.add_argument("--out", default=None,
                   help="output path (default RUN_DIR/AGENT_TASK.md)")
    a = p.parse_args(argv)
    try:
        path = assemble(a.template, a.run_dir, a.out)
    except (TemplateError, modules_lib.ModulesError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"[task-doc] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
