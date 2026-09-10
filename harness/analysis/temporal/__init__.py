"""temporal — how an articulated object moves THROUGH TIME (artscript env).

analysis.pose_diff owns the per-frame LIFT (`lift_placement`: `W = R_cam R_obj` and
`u = R_cam tau`) and THE residual between two of them. This package is what reasons
about a SEQUENCE of frames:

  * `report`      — THE ENTRY POINT. One row per frame, every value a single scalar:
                    velocity and acceleration of the base pose (rotation,
                    translation, radial) and of each joint. Prints an aligned table
                    and writes the same numbers as JSON. Run it on the poses a
                    window owns, read it against the video, repeat.
  * `sequence`    — the producer under it: per-step residuals along the timeline plus
                    the per-frame placement track, in the shape the maths comes out
                    in (vectors and all).
  * `derivatives` — velocity, acceleration and their radial component, over that
                    track. The second derivative is the part a pairwise residual
                    cannot express.
  * `seams`       — which refiner posed which frame, and where two of them meet.
                    Pure ownership bookkeeping; consults no magnitude, because a seam
                    matters for WHO posed it, not how big it is.

NO THRESHOLDS IN THIS PACKAGE. Nothing here decides that a step is too big. A large
rotation is a trustworthy measurement whose ONE ambiguity — a real half-turn versus a
spurious basin flip — only the frames settle, so the tooling measures and the agent
that can look decides.

Pure numpy + stdlib on top of rig.lie / core.joints / analysis.frames; file IO only
in CLI main()s. Invoked as modules from harness/:
    micromamba run -n artscript python -m analysis.temporal.report --pose-json ...
    micromamba run -n artscript python -m analysis.temporal.sequence --pose-json ...
"""

