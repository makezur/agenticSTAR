"""views.sweeps — the five sweep-family VERBS. The machinery is `lib/`.

This directory holds only what an agent invokes: five thin views and their five
`.md` pages, one pair per verb. Two SEARCHED and three IMPERATIVE, in two search
FRAMES:

  * `sweep`  — a UNIFIED camera-frame DOF vector: any mix of pose "order"
               increments (roll/yaw/pitch/dpx/dpy/tz) and absolute joint states,
               so it covers pose-only, joint-only, AND coupled fits that once
               needed three separate views. ONE frame, candidates from a full
               Cartesian `grid`, with optional coarse->fine refinement.
  * `apply`  — the same camera frame, ONE named order instead of a grid.
  * `osweep` — canonical object-frame rotation (rx/ry/rz, right-multiplied),
               searched PER FRAME across a frame set.
  * `oapply` / `oapply_all` — the same canonical rotation from a NAMED candidate
               set, chosen per frame or once for the whole run.

Every one of them is ~50 lines: resolve `ctx.args` into a `lib.config` dataclass,
call its engine. That is the whole view. The imperative three are thin for a
different reason — the agent already knows the answer and only wants it rendered
and scored — so each is a front-end over the same engine its searched twin drives.

The split is verbs here, machinery in `lib/`, because these two
kinds of file are read for different reasons: a caller opens this directory to find
out what a verb DOES, and `lib/` to find out how. `analysis/` is laid out the same
way (`frames.py` + `frames.md` beside `analysis/lib/`). `views/README.md` lists all
five alongside the plain renderers, which is the right view for a caller.

Start at `sweep.md` (the camera-frame family) or `osweep.md` (the canonical one).
"""
