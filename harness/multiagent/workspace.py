"""Resolve the ``windows/`` workspace owned by a unified run iteration."""

import os

from analysis.lib.io import read_json
from bookkeeping import ledger as bookkeeping

WINDOWS_DIRNAME = bookkeeping.ITERATIONS_DIR
ITERATION_PREFIX = ""


PLAN_DERIVED_FILES = (
    "plan.json",
    "merged_poses.json",
    "merge_report.json",
    "paste_frames.py",
    "apply_report.json",
    "seed_reconciliation.json",
)


def _iteration_workspaces(run_dir):
    """Sorted ``[(number, path)]`` iteration workspaces for a run."""
    root = os.path.join(os.path.abspath(run_dir), bookkeeping.ITERATIONS_DIR)
    if not os.path.isdir(root):
        return []
    found = []
    for name in os.listdir(root):
        if len(name) != 6 or not name.isdigit():
            continue
        path = os.path.join(root, name, "windows")
        if os.path.isdir(path):
            found.append((int(name), path))
    return sorted(found)


def _active_workspace(run_dir, required=True):
    """Newest global iteration that owns a windows workspace."""
    workspaces = _iteration_workspaces(run_dir)
    if workspaces:
        return workspaces[-1][1]
    if required:
        raise ValueError(f"no pose iteration under "
                         f"{os.path.join(run_dir, bookkeeping.ITERATIONS_DIR)} "
                         "- run `plan` first")
    return None


def _create_workspace(run_dir):
    """Open one pose iteration and create its owned ``windows/`` workspace."""
    opened = bookkeeping.begin(
        run_dir, "pose", attach_to=("pose",))
    number = opened["iteration"]
    wdir = os.path.join(opened["path"], "windows")
    if os.path.exists(wdir):
        raise ValueError(
            f"pose iteration {opened['name']} already has a window plan at "
            f"{wdir}; complete its verification/adjudication or abort it")
    os.makedirs(wdir)
    bookkeeping.record_window_plan(run_dir, number, wdir)
    return number, wdir


def _read_json_lenient(path):
    """read_json, but an UNPARSEABLE file reads as absent rather than raising.

    Only for the `plan` gate, which inspects a PREVIOUS round's artifacts. A
    truncated or hand-mangled merge_report there is not this round's problem, and
    a gate that dies on it blocks all further work with a JSONDecodeError that
    names neither the file nor the fix — strictly worse than declining to gate
    (whoever needs the verdicts will meet `adjudicate`, which does raise). Every
    command that OWNS a file still reads it strictly."""
    try:
        return read_json(path)
    except ValueError:
        return None
