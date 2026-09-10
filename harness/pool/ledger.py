#!/usr/bin/env python3
"""ledger.py — the spool's append-only order log (`<spool>/ledger.jsonl`).

The spool already keeps WHAT was ordered (`claimed/<id>.json`) and HOW it turned
out (`results/<id>.json`), but nothing on disk records the ORDER LIFECYCLE: when
an order was submitted vs claimed vs finished, which worker slot ran it, whether
a worker death forced a retry, or that a recycle happened mid-batch. The manager
narrates all of that with `log()` — to stdout, which `pool/session.py` leaves
inherited, so for a `shape_pass`/`multiagent.pool_session` pool it lands in whatever
the coordinator's stdout happens to be and is gone once the pool exits.

This module persists those events beside the spool they belong to, so a finished
(or crashed) pool is debuggable from files alone:

    {"t": 1769..., "ts": "2026-07-26T14:20:52.318", "event": "claim", "id": "000100", ...}

**A ledger write NEVER fails an order.** Every append is best-effort: an OSError
(full disk, spool renamed out from under a straggler) is swallowed, because the
render it describes already succeeded and losing a log line must not turn that
into an `ok:false`.

**Appends are one `os.write` on an `O_APPEND` fd**, so the interleaved writers a
spool actually has — G manager slot threads plus one or more separate client
processes — cannot tear each other's lines. The file is opened and closed per
event: a spool is archived by RENAME while a pool may still be running
(`locks.archive_spool`), and reopening by path each time means later events
follow the live path instead of the archived inode.

The ledger lives at the spool ROOT, so `archive_spool` carries a pass's whole
history aside with it, and a fresh pass starts a fresh ledger.

Read it back with the CLI:

    micromamba run -n artscript env PYTHONPATH=harness python -m pool.ledger \
        --spool RUN_DIR/spool                  # per-order timeline + summary
    ... python -m pool.ledger --spool RUN_DIR/spool --failures   # only bad orders
    ... python -m pool.ledger --spool RUN_DIR/spool --raw        # the events
"""

import argparse
import json
import os
import sys
import time

LEDGER_NAME = "ledger.jsonl"

# Events an order passes through, in lifecycle order. `submit` is written by the
# client process, the rest by the manager.
ORDER_EVENTS = ("submit", "claim", "result")


def path_for(spool):
    """The ledger path for `spool` (not created until something is appended)."""
    return os.path.join(spool, LEDGER_NAME)


def _stamp(t):
    """"2026-07-26T14:20:52.318" — local time, ms, sorts the same as `t`."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) \
        + f".{int((t % 1) * 1000):03d}"


def append(spool, event, **fields):
    """Append one event to `<spool>/ledger.jsonl`. Best-effort, never raises.

    Returns True if the line was written, False if it was dropped (which is a
    debugging-data loss, never an order failure — see the module docstring)."""
    if not spool:
        return False
    t = time.time()
    rec = {"t": round(t, 3), "ts": _stamp(t), "event": event}
    rec.update(fields)
    try:
        line = (json.dumps(rec, default=str) + "\n").encode()
    except (TypeError, ValueError):
        return False
    try:
        os.makedirs(spool, exist_ok=True)
        fd = os.open(path_for(spool), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    except OSError:
        return False
    try:
        os.write(fd, line)          # one write on O_APPEND — no torn lines
    except OSError:
        return False
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    return True


def read(spool):
    """Every parseable event in the ledger, in file order. A truncated last line
    (killed mid-append) is skipped rather than raising — the whole point is to be
    readable after a crash. Returns [] when there is no ledger."""
    out = []
    try:
        with open(path_for(spool)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return out
    return out


def orders(spool_or_events):
    """Collapse the ledger into one dict per order id, in first-seen order.

    Each: {"id", "submit"/"claim"/"result": <event>, "events": [...]} plus the
    derived timings a debugging session actually asks for — `queued` (submit ->
    claim), `ran` (claim -> result), `total` — and `deaths` (worker_died events
    charged to this order). Pool-level events (start/recycle/stop) are not
    orders and are skipped."""
    events = spool_or_events if isinstance(spool_or_events, list) \
        else read(spool_or_events)
    by_id = {}
    for rec in events:
        rid = rec.get("id")
        if rid is None:
            continue
        o = by_id.setdefault(str(rid), {"id": str(rid), "events": [], "deaths": 0})
        o["events"].append(rec)
        ev = rec.get("event")
        if ev in ORDER_EVENTS:
            o[ev] = rec
        elif ev == "worker_died":
            o["deaths"] += 1
    for o in by_id.values():
        o["queued"] = _delta(o.get("submit"), o.get("claim"))
        o["ran"] = _delta(o.get("claim"), o.get("result"))
        o["total"] = _delta(o.get("submit") or o.get("claim"), o.get("result"))
        res = o.get("result") or {}
        o["ok"] = res.get("ok")
        o["state"] = "result" if o.get("result") else \
            ("claimed" if o.get("claim") else "submitted")
    return list(by_id.values())


def _delta(a, b):
    """Seconds between two events, or None if either is missing."""
    if not a or not b or a.get("t") is None or b.get("t") is None:
        return None
    return round(b["t"] - a["t"], 3)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _fmt(x, width=6):
    return f"{x:>{width}.2f}" if isinstance(x, (int, float)) else " " * (width - 1) + "-"


def _print_report(spool, failures_only=False):
    events = read(spool)
    if not events:
        print(f"[pool.ledger] no {LEDGER_NAME} under {spool} "
              "(pool predates the ledger, or never ran)")
        return 1
    pool_events = [e for e in events if e.get("id") is None]
    rows = orders(events)
    print(f"[pool.ledger] {spool}")
    for e in pool_events:
        extra = {k: v for k, v in e.items() if k not in ("t", "ts", "event")}
        print(f"  {e['ts']}  {e['event']:<12} "
              + " ".join(f"{k}={v}" for k, v in extra.items()))
    shown = [r for r in rows if not failures_only or r.get("ok") is not True]
    if shown:
        # `render` is the worker's own reported seconds; `ran` is the slot's
        # wall-clock claim->result. render << ran means the time went somewhere
        # else in the slot (panels, a retry, a respawn) — the usual first question.
        print(f"\n  {'id':<24} {'state':<10} {'ok':<5} {'queued':>7} {'ran':>7}"
              f" {'render':>7} {'total':>7}  detail")
        for r in shown:
            ok = "-" if r.get("ok") is None else ("ok" if r["ok"] else "FAIL")
            detail = []
            res = r.get("result") or {}
            if res.get("worker") is not None:
                detail.append(f"w{res['worker']}")
            if r["deaths"]:
                detail.append(f"{r['deaths']} worker death(s)")
            if res.get("error"):
                detail.append(str(res["error"]))
            claim = r.get("claim") or r.get("submit") or {}
            if claim.get("views"):
                detail.insert(0, str(claim["views"]))
            print(f"  {r['id']:<24} {r['state']:<10} {ok:<5} {_fmt(r['queued'], 7)}"
                  f" {_fmt(r['ran'], 7)} {_fmt(res.get('seconds'), 7)}"
                  f" {_fmt(r['total'], 7)}  {'  '.join(detail)}")
    done = [r for r in rows if r.get("result")]
    bad = [r for r in done if not r.get("ok")]
    stuck = [r for r in rows if not r.get("result")]
    ran = [r["ran"] for r in done if r["ran"] is not None]
    print(f"\n  {len(rows)} order(s): {len(done) - len(bad)} ok, {len(bad)} failed,"
          f" {len(stuck)} unfinished"
          + (f"; ran {min(ran):.2f}-{max(ran):.2f}s (mean {sum(ran)/len(ran):.2f})"
             if ran else ""))
    if stuck:
        print("  unfinished: " + ", ".join(f"{r['id']}({r['state']})"
                                          for r in stuck[:12])
              + (" ..." if len(stuck) > 12 else ""))
    return 0


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spool", required=True, help="spool dir holding ledger.jsonl")
    p.add_argument("--raw", action="store_true", help="dump the events, one JSON "
                   "per line, instead of the report")
    p.add_argument("--failures", action="store_true",
                   help="report only orders that failed or never finished")
    args = p.parse_args()
    if args.raw:
        for rec in read(args.spool):
            print(json.dumps(rec))
        return 0
    return _print_report(args.spool, failures_only=args.failures)


if __name__ == "__main__":
    sys.exit(main())
