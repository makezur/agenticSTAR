"""critic — an independent VLM judge for REALISM + IDENTITY (the second opinion).

Run as a module (unchanged entry point):
    micromamba run -n artscript env PYTHONPATH=harness \
        python -m analysis.scorers.critic --source ... --render ... --out ...

The package is split by concern so the prompt is plain, editable text and "the
operations" stay separate from "the CLI":

  * `wrapper.py`          — the CLI: argparse surface + orchestration (`main()`).
  * `core.py`             — every operation: prompt loading, the swappable VLM
                            backends, response parsing, numeric grounding, and the
                            image/turntable helpers.
  * `system_prompt.txt`   — the rubric + JSON output contract (the SYSTEM_PROMPT).
  * `grounding_prompt.txt`— the numeric-context appendix (the GROUNDING_PROMPT),
                            appended only when the caller passes --metrics/--depth/
                            --sweep.
  * `__main__.py`         — makes `python -m analysis.scorers.critic` call main().

See critic.md for the full rubric + the JSON contract + how to add a provider.
"""
