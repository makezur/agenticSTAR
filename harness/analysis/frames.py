"""frames.py — order frame ids temporally, and resolve a review window by offset.

Two tiny, pure helpers every caller staging a window of frames kept re-deriving
by hand:

  1. order_frames(names)  — sort frame ids/paths into TEMPORAL order. Frames are
     named by their numeric index (`000040.jpg` -> 40), so "temporal" is just that
     integer, not lexical accident. Mixed widths, bare ids, and full paths all
     sort correctly; a name with no digits falls back to a stable lexical key so
     the sort never throws.

  2. resolve_window(frames_dir, anchor, ...) — given an ANCHOR frame and a set of
     OFFSETS (explicit, or a radius+step), return the existing frame PATHS around
     it in temporal order — the neighborhood you look at when one frame is in
     question: e.g. anchor 000040 with offsets -10,-5,0,5,10 ->
     [000030.jpg, 000035.jpg, 000040.jpg, 000045.jpg, 000050.jpg] (those that
     exist on disk). Offsets are in FRAME-NUMBER units, matching the naming.

  3. sample_frames(frames_dir, step, ...) — pick every STEP-th frame across a dir,
     striding in FRAME-NUMBER units (start, start+step, ...) rather than by position
     in the file list, so gaps don't shift the sampling. This is the "every Nth
     frame" selector `run.sh --every N` feeds into `--frames`: e.g. step 10 over
     000000..000266 -> [000000.jpg, 000010.jpg, ..., 000260.jpg] (those on disk).

Both surfaces also read a run's `layout.json` directly (`--run-dir RUN`): it already
declares the `frames_dir`/`masks_dir` and the ordered `frames` list, so a caller can
point at the run instead of hand-typing paths.

Pure stdlib. Runs in the 'artscript' env, from harness/:

  # order a handful of ids (globs/paths/bare ids all accepted)
  micromamba run -n artscript python -m analysis.frames order \
      --frames 000040.jpg,000000.jpg,000080.jpg
  # ...or just order whatever the run's layout.json lists
  micromamba run -n artscript python -m analysis.frames order --run-dir RUN

  # resolve a 5-frame window around an anchor, emitting ready-to-splice --frames args
  micromamba run -n artscript python -m analysis.frames window \
      --frames-dir RUN/frames --anchor 000040.jpg --radius 10 --step 5 --emit args
  # ...or let layout.json supply frames_dir (or masks_dir with --masks)
  micromamba run -n artscript python -m analysis.frames window \
      --run-dir RUN --anchor 000040.jpg --radius 10 --step 5 --emit args

  # sample every 10th frame across a dir (frame-number stride), as a CSV for --frames
  micromamba run -n artscript python -m analysis.frames sample \
      --frames-dir RUN/frames --every 10 --emit csv
  # ...or let layout.json supply frames_dir
  micromamba run -n artscript python -m analysis.frames sample \
      --run-dir RUN --every 10 --emit csv
"""

import argparse
import glob
import json
import os
import re

from analysis.lib.io import frame_stem
from core import run_layout


# --------------------------------------------------------------------------- #
# temporal key — the numeric frame index, with a safe lexical fallback
# --------------------------------------------------------------------------- #
def frame_number(name):
    """The integer frame index encoded in a frame name/path, or None.

    Uses the LAST run of digits in the stem (so `frame_000040` and `000040.jpg`
    both give 40, and a trailing `_v2`-style suffix wouldn't hijack it). None when
    the stem has no digits at all."""
    m = re.findall(r"\d+", frame_stem(os.path.basename(str(name))))
    return int(m[-1]) if m else None


def _sort_key(name):
    """(has_number, number_or_0, basename) — numbers sort ahead of and among
    themselves before any digit-less name, which then sorts lexically. Total and
    exception-free for any string."""
    n = frame_number(name)
    base = os.path.basename(str(name))
    return (n is None, n if n is not None else 0, base)


def order_frames(names, reverse=False):
    """Frame ids/paths in TEMPORAL order (by numeric index), de-duped, form kept.

    `names` — an iterable of frame names or paths. The returned items are the same
    strings passed in (paths stay paths, bare ids stay bare) — only reordered and
    de-duplicated (first occurrence wins). Digit-less names sort last, lexically.
    """
    seen = set()
    uniq = []
    for n in names or []:
        s = str(n)
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return sorted(uniq, key=_sort_key, reverse=reverse)


# --------------------------------------------------------------------------- #
# window resolution — frame paths around an --anchor
# --------------------------------------------------------------------------- #
def _offsets_from(offsets, radius, step):
    """The list of frame-number offsets to apply around the anchor.

    Explicit `offsets` win; otherwise build a symmetric ±`radius` range stepped by
    `step` (inclusive of both ends and of 0). Always includes 0 (the anchor) and
    is returned sorted/de-duped."""
    if offsets:
        offs = list(offsets)
    else:
        r = int(radius or 0)
        st = max(1, int(step or 1))
        offs = list(range(-r, r + 1, st))
    return sorted(set(offs) | {0})


def resolve_window(frames_dir, anchor, offsets=None, radius=None, step=1,
                   ext=None, existing_only=True):
    """Frame PATHS around `anchor` at the given offsets, in temporal order.

    `frames_dir` — directory the frames live in.
    `anchor`     — the anchor frame name (e.g. '000040.jpg' or bare '000040');
                   its zero-pad WIDTH and extension are inferred from it (override
                   the extension with `ext`, e.g. '.png' for masks).
    `offsets`    — explicit frame-number offsets (e.g. [-10,-5,0,5,10]); if falsy,
                   a symmetric ±`radius` range stepped by `step` is used instead.
    `existing_only` — drop offsets whose file is absent (default). With False the
                   full set of candidate paths is returned regardless.

    Returns the de-duped path list in temporal order (0 = the anchor is always in
    it). Negative resulting frame numbers are skipped."""
    stem = frame_stem(os.path.basename(str(anchor)))
    num = frame_number(anchor)
    if num is None:
        raise ValueError(f"anchor {anchor!r} has no numeric frame index")
    width = len(stem)
    if ext is None:
        ext = os.path.splitext(os.path.basename(str(anchor)))[1] or ".jpg"

    paths = []
    for off in _offsets_from(offsets, radius, step):
        n = num + off
        if n < 0:
            continue
        name = f"{n:0{width}d}{ext}"
        p = os.path.join(frames_dir, name)
        if existing_only and not os.path.isfile(p):
            continue
        if p not in paths:
            paths.append(p)
    return order_frames(paths)


# --------------------------------------------------------------------------- #
# sampling — every Nth frame across a dir, striding by frame number
# --------------------------------------------------------------------------- #
def _dir_frame_paths(frames_dir, ext=None):
    """Existing frame paths in `frames_dir` (numbered ones only), temporal order.

    `ext` filters by extension (e.g. '.jpg'); when None, any file whose stem has a
    numeric index is kept. Digit-less files (no frame number) are ignored."""
    out = []
    for name in os.listdir(frames_dir):
        if ext is not None and os.path.splitext(name)[1].lower() != ext.lower():
            continue
        if frame_number(name) is None:
            continue
        out.append(os.path.join(frames_dir, name))
    return order_frames(out)


def sample_frames(frames_dir, step, start=None, stop=None, ext=None,
                  existing_only=True):
    """Every `step`-th frame in `frames_dir`, striding in FRAME-NUMBER units.

    `frames_dir` — directory the frames live in.
    `step`       — stride in frame-number units (>=1); step 10 selects frame
                   numbers start, start+10, start+20, ... (NOT every 10th file, so
                   gaps in the numbering don't shift the sampling).
    `start`/`stop` — inclusive frame-number bounds; default to the min/max numbered
                   frame present in the dir.
    `ext`        — restrict to this extension (e.g. '.jpg'); inferred from the first
                   numbered frame when None so the pad WIDTH is taken from real files.
    `existing_only` — keep only sampled numbers whose file is on disk (default).
                   With False, every start+k*step in range is emitted regardless.

    Returns the de-duped path list in temporal order. Raises ValueError on step<1 or
    a dir with no numbered frames."""
    if int(step) < 1:
        raise ValueError(f"step must be >= 1, got {step!r}")
    step = int(step)

    all_paths = _dir_frame_paths(frames_dir, ext=ext)
    if not all_paths:
        raise ValueError(f"no numbered frames in {frames_dir!r}"
                         + (f" with ext {ext!r}" if ext else ""))
    nums = [frame_number(p) for p in all_paths]
    # Infer pad width + ext from a real file so emitted names match on-disk naming.
    sample_name = os.path.basename(all_paths[0])
    width = len(frame_stem(sample_name))
    if ext is None:
        ext = os.path.splitext(sample_name)[1] or ".jpg"

    lo = nums[0] if start is None else int(start)
    hi = nums[-1] if stop is None else int(stop)
    present = set(nums)

    paths = []
    for n in range(lo, hi + 1, step):
        if n < 0:
            continue
        if existing_only and n not in present:
            continue
        name = f"{n:0{width}d}{ext}"
        p = os.path.join(frames_dir, name)
        if p not in paths:
            paths.append(p)
    return order_frames(paths)


# --------------------------------------------------------------------------- #
# run-dir layout.json — the frame/mask directories a run already declares
# --------------------------------------------------------------------------- #
def load_layout(run_dir):
    """Read a run dir's `layout.json` via core.run_layout.load_run_layout.

    layout.json is the run's manifest (written by the pipeline): it declares the
    `frames_dir`, `masks_dir`, `hand_masks_dir`, `ref_frame`, and the ordered
    `frames` list — the same contract measure_depth.resolve_layout consumes. Reading
    it here lets `order`/`window` point at a run dir instead of hand-typed paths."""
    layout = run_layout.load_run_layout(run_dir)
    if layout is None:
        raise SystemExit(f"[frames] no layout.json in {run_dir}")
    return layout


# --------------------------------------------------------------------------- #
# CLI — `order` and `window`
# --------------------------------------------------------------------------- #
def _expand(patterns):
    """Flatten repeatable/CSV/glob --frames args into a flat list of tokens
    (globs that match files expand to their hits; everything else passes through)."""
    out = []
    for pat in patterns or []:
        for tok in str(pat).split(","):
            tok = tok.strip()
            if not tok:
                continue
            hits = sorted(glob.glob(tok)) if any(c in tok for c in "*?[") else []
            out.extend(hits or [tok])
    return out


def _emit(paths, emit):
    """Render the resolved list per the --emit format."""
    if emit == "names":           # basenames only, CSV — for run.sh --frames
        return ",".join(os.path.basename(p) for p in paths)
    if emit == "csv":
        return ",".join(paths)
    if emit == "args":            # ready to splice into a --frames-taking command
        return " ".join(f"--frames {p}" for p in paths)
    return "\n".join(paths)       # 'paths' (default): one per line


def main():
    p = argparse.ArgumentParser(
        prog="frames",
        description="order frame ids temporally, or resolve a frame window "
                    "around an --anchor by offset")
    sub = p.add_subparsers(dest="cmd", required=True)

    po = sub.add_parser("order", help="print frame ids/paths in temporal order")
    po.add_argument("--frames", action="append", default=[],
                    help="frame ids/paths; repeatable, and each value may be a "
                         "comma list or a glob. Optional if --run-dir is given "
                         "(then defaults to the layout.json frame list)")
    po.add_argument("--run-dir", default="", help="run dir whose layout.json "
                    "supplies the frame list (when --frames is absent)")
    po.add_argument("--reverse", action="store_true", help="latest first")
    po.add_argument("--emit", choices=["paths", "csv", "args"], default="paths")

    pw = sub.add_parser("window", help="resolve the frames around an anchor")
    pw.add_argument("--frames-dir", default="", help="directory the frames live in "
                    "(or supplied by --run-dir/layout.json's frames_dir)")
    pw.add_argument("--run-dir", default="", help="run dir whose layout.json "
                    "supplies frames_dir (or masks_dir with --masks)")
    pw.add_argument("--masks", action="store_true", help="with --run-dir, resolve "
                    "the window in the mask dir (masks_dir) as .png instead of frames")
    pw.add_argument("--anchor", required=True,
                    help="anchor frame (e.g. 000040.jpg); pad width + ext inferred")
    pw.add_argument("--offsets", default="",
                    help="explicit comma offsets in frame-number units; overrides "
                         "--radius/--step. Use '=' when it starts with a minus so "
                         "argparse doesn't read it as a flag: --offsets=-10,-5,0,5,10")
    pw.add_argument("--radius", type=int, default=10,
                    help="symmetric ± window in frame-number units (default 10)")
    pw.add_argument("--step", type=int, default=5,
                    help="stride within the ±radius window (default 5)")
    pw.add_argument("--ext", default="",
                    help="override the frame extension (e.g. .png for masks)")
    pw.add_argument("--all", action="store_true",
                    help="keep offsets whose file is absent (default: existing only)")
    pw.add_argument("--emit", choices=["paths", "csv", "args"], default="paths")

    ps = sub.add_parser("sample", help="pick every Nth frame across a dir "
                        "(stride by frame number)")
    ps.add_argument("--frames-dir", default="", help="directory the frames live in "
                    "(or supplied by --run-dir/layout.json's frames_dir)")
    ps.add_argument("--run-dir", default="", help="run dir whose layout.json "
                    "supplies frames_dir (or masks_dir with --masks)")
    ps.add_argument("--masks", action="store_true", help="with --run-dir, sample "
                    "the mask dir (masks_dir) as .png instead of frames")
    ps.add_argument("--every", type=int, required=True,
                    help="stride in frame-number units (e.g. 10 = every 10th frame)")
    ps.add_argument("--start", type=int, default=None,
                    help="first frame number to include (default: earliest present)")
    ps.add_argument("--stop", type=int, default=None,
                    help="last frame number to include (default: latest present)")
    ps.add_argument("--ext", default="",
                    help="restrict to this extension (e.g. .png for masks)")
    ps.add_argument("--all", action="store_true",
                    help="emit every strided number even if its file is absent")
    ps.add_argument("--emit", choices=["paths", "csv", "args", "names"],
                    default="paths",
                    help="'names' emits basenames only (for run.sh --frames)")
    args = p.parse_args()

    if args.cmd == "order":
        names = _expand(args.frames)
        if not names and args.run_dir:
            names = list(load_layout(args.run_dir).get("frames") or [])
        if not names:
            p.error("order needs --frames (or --run-dir with a layout.json "
                    "that lists frames)")
        print(_emit(order_frames(names, reverse=args.reverse), args.emit))
        return

    if args.cmd == "sample":
        frames_dir = args.frames_dir
        ext = args.ext or None
        if args.run_dir:
            layout = load_layout(args.run_dir)
            key = "masks_dir" if args.masks else "frames_dir"
            frames_dir = frames_dir or layout.get(key, "")
            if args.masks and not args.ext:
                ext = ".png"
        if not frames_dir:
            p.error("sample needs --frames-dir (or --run-dir with a layout.json)")
        try:
            paths = sample_frames(frames_dir, args.every, start=args.start,
                                  stop=args.stop, ext=ext,
                                  existing_only=not args.all)
        except ValueError as e:
            raise SystemExit(f"[frames] {e}")
        if not paths:
            raise SystemExit(f"[frames] no frames sampled at every {args.every} "
                             f"in {frames_dir}")
        print(_emit(paths, args.emit))
        return

    frames_dir = args.frames_dir
    ext = args.ext or None
    if args.run_dir:
        layout = load_layout(args.run_dir)
        key = "masks_dir" if args.masks else "frames_dir"
        frames_dir = frames_dir or layout.get(key, "")
        if args.masks and not args.ext:
            ext = ".png"
    if not frames_dir:
        p.error("window needs --frames-dir (or --run-dir with a layout.json)")

    offsets = [int(x) for x in args.offsets.split(",") if x.strip()] or None
    paths = resolve_window(
        frames_dir, args.anchor, offsets=offsets,
        radius=args.radius, step=args.step, ext=ext,
        existing_only=not args.all)
    if not paths:
        raise SystemExit(f"[frames] no window frames resolved for anchor "
                         f"{args.anchor!r} in {frames_dir}")
    print(_emit(paths, args.emit))


if __name__ == "__main__":
    main()
