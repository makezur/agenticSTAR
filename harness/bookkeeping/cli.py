#!/usr/bin/env python3
"""CLI for inspecting and manually managing unified harness iterations."""

import argparse
import json
import os
import subprocess
import tempfile

from bookkeeping import ledger as bookkeeping
from bookkeeping import pose_history


def _write_json(path, value):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pose-reseed-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "status", "history", "begin", "abort", "bundle",
                 "run", "reseed-pose"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--run-dir", required=True)
        if name in ("begin", "run"):
            cmd.add_argument("--kind", choices=("shape", "pose", "custom"),
                             default="custom")
            cmd.add_argument("--label", default="")
            if name == "run":
                cmd.add_argument(
                    "--record-only", action="store_true",
                    help="record the generated pass without completing the "
                         "iteration")
                cmd.add_argument("argv", nargs=argparse.REMAINDER)
        elif name == "abort":
            cmd.add_argument("--reason", default="")
        elif name == "bundle":
            cmd.add_argument("--out")
        elif name == "reseed-pose":
            cmd.add_argument("--from-iteration", default="previous")
            cmd.add_argument("--frames")
            cmd.add_argument("--out")
            cmd.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "init":
        bookkeeping.ensure_git(args.run_dir)
        result = {"run_dir": args.run_dir, "initialized": True}
    elif args.command == "status":
        result = bookkeeping.status(args.run_dir)
    elif args.command == "history":
        result = bookkeeping.history(args.run_dir)
    elif args.command == "begin":
        result = bookkeeping.begin(args.run_dir, args.kind, args.label,
                                   attach_to=(args.kind,))
    elif args.command == "run":
        argv = list(args.argv)
        if argv[:1] == ["--"]:
            argv = argv[1:]
        if not argv:
            parser.error("run requires a command after `--`")
        opened = bookkeeping.begin(args.run_dir, args.kind, args.label,
                                   attach_to=(args.kind,))
        env = os.environ.copy()
        env["HARNESS_RUN_DIR"] = os.path.abspath(args.run_dir)
        env["HARNESS_ITERATION"] = opened["name"]
        env["HARNESS_ITERATION_DIR"] = opened["path"]
        existing = set()
        for root, _, files in os.walk(opened["path"]):
            if "scene_snapshot.py" in files and "pose.json" in files:
                existing.add(os.path.abspath(root))
        proc = subprocess.run(argv, env=env)
        if proc.returncode:
            return proc.returncode
        candidates = []
        for root, _, files in os.walk(opened["path"]):
            root = os.path.abspath(root)
            if (root not in existing
                    and "scene_snapshot.py" in files
                    and "pose.json" in files):
                candidates.append(root)
        if not candidates:
            raise SystemExit(
                "custom command succeeded but produced no new pass containing "
                "scene_snapshot.py and pose.json under HARNESS_ITERATION_DIR")
        latest = max(candidates, key=os.path.getmtime)
        if args.record_only:
            result = bookkeeping.record_pass(
                args.run_dir, opened["iteration"], latest)
        else:
            result = bookkeeping.complete_pass(
                args.run_dir, opened["iteration"], latest)
    elif args.command == "reseed-pose":
        result = pose_history.reseed_trajectory(
            args.run_dir, args.from_iteration, frames=args.frames)
        out = args.out or os.path.join(
            os.path.abspath(args.run_dir), bookkeeping.STATE_DIR,
            "pose-reseed.json")
        _write_json(out, result)
        result["draft"] = os.path.abspath(out)
        if args.apply:
            from multiagent import scene_source
            backup = os.path.join(
                os.path.abspath(args.run_dir), bookkeeping.STATE_DIR,
                "scene-before-pose-reseed.py")
            rewrite = scene_source.rewrite_frames(
                os.path.join(os.path.abspath(args.run_dir), "scene.py"),
                result["frames"], backup_path=backup)
            result["applied"] = rewrite
    elif args.command == "abort":
        result = bookkeeping.abort(args.run_dir, reason=args.reason)
    else:
        result = {"bundle": bookkeeping.create_bundle(args.run_dir, args.out)}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
