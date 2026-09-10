#!/usr/bin/env python3
"""client.py — the pool manager's file-spool order client.

Drops one orders/<id>.json into a pool spool and blocks until its
results/<id>.json appears (written atomically by the manager, so a completed poll
never reads a partial file). This is the order client the pass tool uses:
`utils/shape_pass.py` (via `composite_pass`) and `pool/session.py` submit their
per-frame renders through `submit`/`wait` here. It stays deliberately thin —
no pose.json/layout extraction, no frame selection — because those decisions
belong to the caller that owns RUN_DIR; the CLI below is for a manual smoke run.

    micromamba run -n artscript env PYTHONPATH=harness python -m pool.client \
        --spool RUN_DIR/spool --frame 000040.jpg --views match,depth --id t1

As a library:
    from pool.client import submit, wait
    submit(spool, {"id": "t1", "frame": "000040.jpg", "views": "match"})
    result = wait(spool, "t1", timeout=120)
"""

import argparse
import errno
import json
import os
import sys
import tempfile
import time

from pool import ledger


def _existing_artifacts(spool, rid):
    """Return spool artifacts proving that `rid` was already allocated."""
    paths = [
        os.path.join(spool, "orders", f"{rid}.json"),
        os.path.join(spool, "claimed", f"{rid}.json"),
        os.path.join(spool, "results", f"{rid}.json"),
        os.path.join(spool, "renders", rid),
    ]
    return [path for path in paths if os.path.exists(path)]


def submit(spool, req):
    """Atomically allocate an immutable order id and write its request.

    Order ids are also result/output identities, so reusing one could make
    wait() return an old result while a new render writes into the old output
    directory. Refuse reuse across every spool lifecycle directory.
    """
    rid = str(req.get("id") or "order")
    req["id"] = rid
    orders = os.path.join(spool, "orders")
    os.makedirs(orders, exist_ok=True)
    final = os.path.join(orders, f"{rid}.json")
    existing = _existing_artifacts(spool, rid)
    if existing:
        raise FileExistsError(
            f"order id {rid!r} was already used; choose a new id "
            f"(found {existing[0]})")

    fd, tmp = tempfile.mkstemp(prefix=f".{rid}.", suffix=".json.tmp",
                               dir=orders, text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(req))
        try:
            # link() is an atomic no-overwrite publish. Unlike replace(), two
            # concurrent clients cannot silently replace the same order id.
            os.link(tmp, final)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise FileExistsError(
                    f"order id {rid!r} was allocated concurrently; "
                    "choose a new id") from exc
            raise
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    # Logged from the CLIENT, after the id is won: the manager only ever sees an
    # order at claim time, so submit->claim queue wait is measurable nowhere else.
    # Best-effort (ledger.append never raises) — a submitted order stands whether
    # or not its line landed.
    ledger.append(spool, "submit", id=rid, frame=req.get("frame"),
                  views=req.get("views"), provenance=req.get("provenance"),
                  pid=os.getpid())
    return rid


def wait(spool, rid, timeout=180.0, poll=0.1):
    """Block until results/<id>.json exists; return the parsed result dict.
    Raises TimeoutError if it doesn't appear in `timeout` seconds."""
    path = os.path.join(spool, "results", f"{rid}.json")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
        time.sleep(poll)
    raise TimeoutError(f"no result for order {rid} within {timeout}s")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spool", required=True)
    p.add_argument("--id", default="order")
    p.add_argument("--frame", default="", help="a scene FRAMES key")
    p.add_argument("--views", default="match")
    p.add_argument("--joints", default="", help="JSON dict of joint overrides")
    p.add_argument("--request", default="",
                   help="a FULL request as JSON (pool/serve.py schema: frame/views/"
                        "pose/joints/sweep/apply) — the general door for sweep "
                        "orders; --frame/--views/--joints fill anything absent")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--no-wait", action="store_true",
                   help="submit and return the id without blocking")
    args = p.parse_args()

    req = json.loads(args.request) if args.request else {}
    if not isinstance(req, dict):
        raise SystemExit("[pool.client] --request must be a JSON object")
    req.setdefault("id", args.id)
    req.setdefault("frame", args.frame)
    req.setdefault("views", args.views)
    if args.joints:
        req.setdefault("joints", json.loads(args.joints))
    rid = submit(args.spool, req)
    print(f"[pool.client] submitted order {rid}", flush=True)
    if args.no_wait:
        return 0
    res = wait(args.spool, rid, timeout=args.timeout)
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
