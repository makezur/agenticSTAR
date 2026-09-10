"""multiagent — the harness side of orchestrator/refiner delegation.

The harness never spawns an agent. This package writes the ARTIFACTS the
delegation runs on and validates what comes back:

  windows.py   plan / status / merge / apply / selfcheck over per-window pose
               refinement: split the trajectory into non-overlapping frame
               windows, write one self-contained brief per window, merge the
               refiners' pose fragments single-writer.
  briefs/      the authored brief source. core.md is the ONE template: the
               shared contract + reference material, rendered with the
               run-specific facts and the round's --guidance. A rendered
               iterations/NNNNNN/windows/<wid>/brief.md is the ENTIRE interface to a
               refiner subagent.

The orchestrator (whatever agent runtime runs it) spawns one refiner per
window pointed at its brief, and closes them after merge. Pure
`artscript`-env python, invoked as modules from harness/:

    micromamba run -n artscript env PYTHONPATH=harness python -m multiagent.windows ...
"""
