"""viz — visualization panels.

`rows` builds THE comparison row (and owns the column registry); `composite`
adds scoring to one, `sweep_sides` emits one per rendered candidate,
`candidate_sheet` paginates many, `frame_sheet` sheets bare source frames,
`pose_overlay_sheet` sheets existing translucent match renders with projected
base axes, `pose_pair_sheet` renders and sheets adjacent pairs end-to-end, and
`seam_sheet` draws one PAIR of frames large (plus their committed renders) for
each cross-window seam.
"""
