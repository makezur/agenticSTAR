#!/usr/bin/env python3
"""Render committed pose matches and build per-frame review composites."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HARNESS = os.path.join(REPO, "harness")
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)

from bookkeeping import ledger as bookkeeping  # noqa: E402
from pool.locks import acquire_run_lock  # noqa: E402
from utils import _run_config as run_config  # noqa: E402
from utils import pass_pipeline  # noqa: E402


def log(message):
    print(f"[composite-pass] {message}", flush=True)


def _run_composite(command, stem):
    log(f"composite {stem}: {' '.join(command)}")
    result = subprocess.run(
        command, cwd=HARNESS, capture_output=True, text=True)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            sys.stderr.write(result.stderr)
        raise SystemExit(
            f"[composite-pass] composite {stem} FAILED "
            f"(exit {result.returncode}).")


def build_composites(pass_dir, layout, bg_mode, workers,
                     side_by_side_max_dimension=0):
    def one(frame):
        paths = pass_pipeline.composite_paths(pass_dir, layout, frame)
        command = pass_pipeline.composite_command(
            paths, bg_mode=bg_mode,
            side_by_side_max_dimension=side_by_side_max_dimension)
        _run_composite(command, paths["stem"])
        return paths

    completed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, frame): frame for frame in layout["frames"]}
        for future in as_completed(futures):
            completed.append(future.result())
    missing = [
        path
        for paths in completed
        for key in ("metrics_out", "overlap_out", "composite_out",
                    "side_by_side_out")
        if not os.path.isfile(path := paths[key])
    ]
    if missing:
        raise SystemExit(
            "[composite-pass] analysis completed without expected output(s):\n"
            + "\n".join(f"  {path}" for path in missing))
    return completed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--frames", default="")
    parser.add_argument("--engine", default="BLENDER_EEVEE_NEXT",
                        choices=["CYCLES", "BLENDER_EEVEE_NEXT"])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--pass-label", default="pose-composites")
    parser.add_argument("--ncpu", type=int, default=None)
    parser.add_argument("--bg-mode", choices=["alpha", "black"], default=None)
    parser.add_argument("--tracking", default="")
    parser.add_argument("--frames-dir", default="")
    parser.add_argument("--masks-dir", default="")
    parser.add_argument("--hand-masks-dir", default="")
    parser.add_argument("--ref-frame", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"[composite-pass] no such run dir: {run_dir}")

    lock_handle = acquire_run_lock(run_dir)
    iteration = bookkeeping.active(run_dir, required=True)
    if iteration.get("kind") != "pose":
        raise ValueError(
            f"composite pass requires an open pose iteration; "
            f"{iteration['name']} is {iteration.get('kind')}")

    config = run_config.load_config(run_dir)
    concurrency = run_config.section(config, "concurrency")
    visuals = run_config.section(config, "visuals")
    workers = run_config.pick(
        args.ncpu, concurrency, "ncpu", min(4, os.cpu_count() or 1))
    bg_mode = run_config.pick(args.bg_mode, visuals, "bg_mode", "black")
    if workers < 1:
        raise SystemExit("[composite-pass] --ncpu must be at least 1")

    layout = pass_pipeline.resolve_layout(
        run_dir, args, tool_name="composite-pass")
    pass_pipeline.check_inputs(layout, tool_name="composite-pass")
    log(f"iteration {iteration['name']} (pose): {iteration['path']}")
    log(f"frames: {', '.join(layout['frames'])} (ref {layout['ref_frame']})")

    pass_dir = pass_pipeline.render_pass(
        run_dir, layout, iteration, "match", ",".join(layout["frames"]),
        args.engine, args.samples, pass_label=args.pass_label, log=log,
        run=subprocess.run, tool_name="composite-pass")
    build_composites(
        pass_dir, layout, bg_mode, workers,
        side_by_side_max_dimension=run_config.side_by_side_max_dimension(
            config))
    bookkeeping.record_pass(run_dir, iteration["iteration"], pass_dir)
    log(f"recorded pass for pose iteration {iteration['name']}")
    print(pass_dir)
    lock_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
