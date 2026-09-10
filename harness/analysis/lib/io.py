"""io — JSON read/write + the per-frame filename contract.

One owner for two things:

  * JSON I/O: `read_json` (None if absent) and `write_json` (makedirs +
    indent=2).
  * the per-frame filename convention: a frame like `000040.jpg` has stem
    `000040` and its scores live in `metrics_000040.json` and
    `depth_000040.json`. Both the writer (shape_pass.py, via the tools) and the
    reader (aggregate.py)
    go through these builders so the convention is defined ONCE, not re-derived
    on each side.
"""

import json
import os


def frame_stem(name):
    """Frame name without extension: '000040.jpg' -> '000040'."""
    return os.path.splitext(str(name))[0]


def metrics_json(stem):
    return f"metrics_{stem}.json"


def depth_json(stem):
    return f"depth_{stem}.json"


def read_json(path):
    """Parsed JSON at `path`, or None if `path` is empty / missing."""
    if path and os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def write_json(path, obj):
    """Write `obj` as indented JSON to `path`, creating parent dirs. Returns the
    JSON text (so callers can also print it)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = json.dumps(obj, indent=2)
    with open(path, "w") as f:
        f.write(text)
    return text
