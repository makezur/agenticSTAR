"""match view — render the posed object from the fixed camera 0 -> match_<frame>.png.

The core comparison render: object posed by this frame's pose (+ shared scale +
joint states) in front of camera 0, at the source resolution (with --match-res)
so it overlays the photo 1:1.
"""

import os

from rig import camera, render


def render_view(ctx):
    a = ctx.args
    render.place_match_camera(a, ctx.cam_obj, ctx.center, ctx.radius,
                              ctx.frame_intr)
    if a.debug_project:
        camera.debug_project(ctx.cam_obj, a.debug_project)
    render.render_to(os.path.join(a.out, f"match{ctx.suffix}.png"))
