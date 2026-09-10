"""gap_sheet — the capture's own frames INSIDE one temporal step.

A run samples its capture (layout.json lists e.g. every 10th frame), so a step in
the temporal report is many unseen capture frames summed into one number — exactly
the fiction analysis.temporal.report's non-consecutive warning describes, made
routine. When the two endpoint keyframes cannot settle whether a residual is real
motion or a mis-posed frame, the dense frames between them can: the capture dir
holds EVERY frame, not just the sampled ones, and this tool sheets the stretch
``--step A:B`` of them.

SOURCE ONLY, on purpose. The in-between frames have no poses and therefore no
renders, so there is nothing to compare them against — this sheet answers "what
did the VIDEO do across this step" (plausibility), not "which basin did the pose
land in" (seam_sheet's question, which needs committed renders).

The drawing is analysis.viz.frame_sheet's, verbatim: this module only selects the
dense frames and delegates to ``write_sheets``, so tiles, packing, pagination, and
the manifest are the same thing a refiner already knows how to read. Each step
gets its own sheet dir (``gap_sheets/<stemA>_<stemB>/``) so ordering two steps
never overwrites evidence.

Density: by default the gap is strided to fit ONE page (both endpoints always
kept) and the chosen stride and dropped count are PRINTED — a sheet that silently
sampled would read as "this is everything the camera saw". ``--every N`` overrides.

Runs in the 'artscript' env, from harness/:
  micromamba run -n artscript python -m analysis.viz.gap_sheet \
      --run-dir RUN_DIR --step 000100.jpg:000110.jpg [--step ...]
"""

import argparse
import json
import math
import os

from analysis import frames as frame_utils
from analysis.lib.io import frame_stem, write_json
from analysis.viz import frame_sheet


def parse_steps(specs, keyframes):
    return frame_sheet.parse_steps(specs, keyframes)


def dense_between(frames_dir, frame_a, frame_b):
    """Every capture frame in [frame_a, frame_b] inclusive, in temporal order.

    Read from the DIRECTORY, not the layout: the layout lists the run's sampled
    keyframes, and the whole point is the frames the run did not sample."""
    lo = frame_utils.frame_number(frame_a)
    hi = frame_utils.frame_number(frame_b)
    if lo is None or hi is None:
        raise ValueError(
            f"step endpoints {frame_a!r}/{frame_b!r} carry no frame number")
    names = [
        name for name in frame_sheet._directory_frames(frames_dir)
        if (n := frame_utils.frame_number(name)) is not None and lo <= n <= hi
    ]
    if not names:
        raise ValueError(
            f"no capture frames between {frame_a} and {frame_b} in {frames_dir}")
    return names


def stride_selection(names, frames_per_page, every=0):
    """(selected, stride): the gap strided to fit one page unless ``every`` is
    given, BOTH endpoints always kept. The endpoints are the step being judged;
    a stride that dropped one would sheet a different step than the one named."""
    if every:
        stride = int(every)
        if stride < 1:
            raise ValueError("--every must be at least 1")
    else:
        stride = max(1, math.ceil(len(names) / frames_per_page))
    selected = names[::stride]
    if names[-1] not in selected:
        selected.append(names[-1])
    return selected, stride


def write_gap_sheets(run_dir, step_specs, out_dir="", every=0,
                     frames_per_page=12, columns=4, tile_height=300):
    """One sheet dir per --step; returns [per-step manifest dicts].

    Endpoint validation, adjacency warning, dense selection and striding happen
    here; every pixel is frame_sheet's.
    """
    frames_dir, keyframes, _ = frame_sheet.resolve_source(run_dir=run_dir)
    steps = parse_steps(step_specs, keyframes)
    root = os.path.abspath(out_dir) if out_dir else os.path.join(
        os.path.abspath(run_dir), "gap_sheets")

    results = []
    for frame_a, frame_b in steps:
        adjacent = keyframes.index(frame_b) - keyframes.index(frame_a) == 1
        if not adjacent:
            between = keyframes.index(frame_b) - keyframes.index(frame_a) - 1
            print(f"!! NOT ONE STEP: {frame_a} -> {frame_b} skips {between} "
                  "keyframe(s) of the run, so this sheet spans several temporal "
                  "steps and its motion is NOT one report row. Proceeding.")
        dense = dense_between(frames_dir, frame_a, frame_b)
        selected, stride = stride_selection(dense, frames_per_page, every=every)
        dropped = len(dense) - len(selected)
        print(f"[gap-sheet] step {frame_a} -> {frame_b}: {len(dense)} capture "
              f"frame(s), showing {len(selected)} (every {stride}), "
              f"{dropped} dropped")

        step_dir = os.path.join(
            root, f"{frame_stem(frame_a)}_{frame_stem(frame_b)}")
        # The endpoints are the run's own keyframes — the step being judged —
        # so they carry the bright rim; everything between is capture-only.
        pages, manifest_path = frame_sheet.write_sheets(
            frames_dir=frames_dir, out_dir=step_dir,
            frame_specs=[",".join(selected)],
            frames_per_page=frames_per_page, columns=columns,
            tile_height=tile_height, highlight_frames=(frame_a, frame_b))

        # Stamp the step's own facts onto the sheet's manifest, so a reader of
        # the evidence can tell what was strided out and whether the stretch
        # really is one report row.
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["step"] = {
            "from": frame_a,
            "to": frame_b,
            "keyframes_adjacent": adjacent,
            "capture_count": len(dense),
            "shown_count": len(selected),
            "stride": stride,
            "dropped": dropped,
        }
        write_json(manifest_path, manifest)
        results.append({"step": manifest["step"], "pages": pages,
                        "manifest": manifest_path})
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True,
                        help="run whose layout.json names the capture and the "
                             "sampled keyframes")
    parser.add_argument(
        "--step", action="append", required=True, metavar="A:B",
        help="a temporal step between two of the run's keyframes, e.g. "
             "000100.jpg:000110.jpg; repeatable, one sheet dir per step. "
             "Non-adjacent keyframes warn loudly but proceed")
    parser.add_argument(
        "--every", type=int, default=0,
        help="show every Nth capture frame (endpoints always kept); default "
             "strides the gap to fit one page and prints what it chose")
    parser.add_argument("--out-dir", default="",
                        help="root for the per-step sheet dirs; defaults to "
                             "RUN_DIR/gap_sheets")
    parser.add_argument("--frames-per-page", type=int, default=12,
                        help="page budget, as in frame_sheet (default: 12)")
    parser.add_argument("--columns", type=int, default=4,
                        help="most columns, as in frame_sheet (default: 4)")
    parser.add_argument("--tile-height", type=int, default=300,
                        help="minimum tile height, as in frame_sheet "
                             "(default: 300)")
    args = parser.parse_args()
    write_gap_sheets(
        args.run_dir, args.step, out_dir=args.out_dir, every=args.every,
        frames_per_page=args.frames_per_page, columns=args.columns,
        tile_height=args.tile_height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
