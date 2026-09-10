"""sweep_family.py — the ONE list of views that produce judgeable candidates.

Five verbs write a self-describing `<stem>.json` report over rendered candidates that
an agent judges by eye: `sweep`/`apply` (camera-frame orders) and
`osweep`/`oapply`/`oapply_all` (canonical object rotations). `pool/panels.py`,
`analysis/viz/sweep_sides.py`, `pool/serve.py`, and `views/__init__.py` all import
this list instead of spelling the set out themselves, so it cannot drift.

ORDER IS LOAD-BEARING for `find_report`, which walks it to pick one report out of a dir
holding several: the camera-frame verbs first, then the object ones.

PURITY: no imports at all, so `pool.manager` (artscript env, no bpy), the analysis-env
sheet builders, and the Blender-side dispatcher share one list.
"""

SWEEP_VIEWS = ("sweep", "apply", "oapply", "oapply_all", "osweep")

# The views that read the `sweep.pose_start` KEY — WHERE THE GRID STARTS, i.e. the
# placement its offsets are measured from. `pool.reseed._placement_key_for` reads this
# to pick WHICH key a seed's placement lands in; it is NOT a list of who may be seeded.
# Every view is seedable through the request's top-level `pose`, which `serve.py`
# applies to the resolved FrameSpec. `sweep.pose_start` is the narrower channel: it
# moves only where the search departs from (`engine._resolve_pose_start`:
# `cfg.pose_start` else `ctx.placement`) and leaves the frame posed where the scene
# put it.
#
# The object verbs pose from the frame's placement too, they just own no `pose_start`
# key for it. Both grammars compose onto the same s*R*p + t, differing in which side
# takes the delta (t' below is the pivot-correction carry — the full maths is
# conventions/POSE.md, "the left/right update"):
#
#     sweep/apply   s * dR*R * p + (t' + dt)   camera-frame increment, LEFT
#     osweep/oapply s * R*dR  * p +  t'        canonical rotation,     RIGHT
#     oapply_all    s * R*dR  * p +  t         canonical rotation,     RIGHT
#
# The per-frame verbs pivot at the frame's FK-AABB centre, so t picks up the
# correction that keeps the object in place on screen; only `oapply_all` keeps the
# canonical-origin pivot (t untouched), by design.
#
# `SharedConfig` has no `pose_start` field and `shared_engine` never reads it — hence
# this tuple. Filling `pose_start` on a view outside it discards the placement while
# the joints still apply, so the frame renders its COMMITTED pose wearing the seed's
# articulation: the half-carried hop, and it renders plausibly.
SWEEP_POSE_START_VIEWS = ("sweep", "apply")

# The views that read the `sweep.mask` KEY — the SCORING mask for ONE frame, which
# both of them REQUIRE. `pool.manager.resolve_sweep_mask` reads this to decide whether
# to fill the key from the run layout.
#
# Identical membership to SWEEP_POSE_START_VIEWS today, and deliberately NOT an alias:
# they answer different questions, and they coincide only because the camera-frame
# engine happens to own both. A third camera-frame verb that searched without a pose
# start (or a pose_start-taking verb scoring N frames) would need exactly one of the
# two updated. One tuple serving two questions is how the next such verb gets silently
# exempted from a required-input fill — and the failure mode of a missing `sweep.mask`
# is `ok: true` with an empty render dir, which nobody reads as a failure.
#
# The object verbs are absent for a real reason, not an oversight: they score N frames
# from a DIRECTORY (`masks_dir`, filled by `resolve_object_masks`), so a single mask
# path is not a thing they can use.
SWEEP_MASK_VIEWS = ("sweep", "apply")
