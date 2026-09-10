"""utils — run-level orchestration over a RUN_DIR.

Tools here don't score or render anything themselves: they read a run's
layout.json / pose.json and DRIVE the analysis tools over it (the way
utils/shape_pass.sh drives a render+score pass). Pure `artscript`-env python,
invoked as modules from harness/:

    micromamba run -n artscript env PYTHONPATH=harness python -m utils.<tool> ...
"""
