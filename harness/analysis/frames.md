# `frames` — order frame ids temporally, resolve a review window by offset

Two tiny helpers for the one thing every temporal tool kept re-deriving: frames are
named by their numeric index (`000040.jpg` → 40), so their **temporal** order is that
integer — not dict-insertion order, not lexical accident (bare `9` vs `40` vs `100`).
`frames.py` owns that convention once, as importable functions **and** a CLI.

## Three jobs

1. **`order_frames(names)`** — sort frame ids/paths into temporal order (by numeric
   index), de-duped, form preserved (paths stay paths, bare ids stay bare). Mixed
   zero-pad widths and full paths all sort correctly; a digit-less name falls back to
   a stable lexical key so the sort never throws. `analysis.temporal.sequence.diff_sequence` uses this
   for its default order, so per-step residuals follow the timeline even when the
   `pose.json` frames were authored out of order.

2. **`resolve_window(frames_dir, anchor, ...)`** — given an **anchor** frame and a
   set of **offsets** (explicit, or a `radius`+`step`), return the existing frame
   **paths** around it in temporal order. Offsets are in **frame-number units**,
   matching the naming. This is the neighborhood you stage when one frame is in
   question — e.g. the `--frames` list for a
   [`frame_sheet`](viz/frame_sheet.md) around a questioned pose.

3. **`sample_frames(frames_dir, step, ...)`** — pick **every Nth** frame across a
   dir, striding in **frame-number units** (`start, start+step, …`) rather than by
   position in the file list, so gaps in the numbering don't shift the sampling. This
   is the selector behind `run.sh --every N`: it expands to the `--frames` list the
   multi-frame path would otherwise take by hand. Optional `--start`/`--stop` bound
   the frame-number range (default: the earliest/latest frame present).

Both surfaces can read a run's **`layout.json`** directly (`--run-dir RUN`) instead
of hand-typed paths — the run manifest already declares the `frames_dir`,
`masks_dir`, and the ordered `frames` list (the same contract `measure_depth`
consumes). `order --run-dir RUN` orders the layout's frame list; `window --run-dir
RUN` pulls `frames_dir` from it (or `masks_dir` as `.png` with `--masks`).

## Run (artscript env, from `harness/`)

```bash
# order a handful of ids (repeatable / CSV / glob all accepted)
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.frames order \
    --frames 000040.jpg,000000.jpg,000080.jpg --emit csv
# -> 000000.jpg,000040.jpg,000080.jpg

# ...or order whatever the run's layout.json lists (no paths to type)
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.frames order \
    --run-dir RUN --emit csv

# resolve a 5-frame window around an anchor, on disk, in temporal order
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.frames window \
    --frames-dir RUN/frames --anchor 000040.jpg --radius 10 --step 5
# -> .../000030.jpg .../000035.jpg .../000040.jpg .../000045.jpg .../000050.jpg

# ...or let layout.json supply frames_dir (--masks resolves the mask window as .png)
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.frames window \
    --run-dir RUN --anchor 000040.jpg --radius 10 --step 5

# sample every 10th frame across a dir (stride by frame number), CSV for --frames
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.frames sample \
    --frames-dir RUN/frames --every 10 --emit names
# -> 000000.jpg,000010.jpg,000020.jpg,...   (--run-dir RUN works too)
```

`run.sh` wires this in directly: `./run.sh --capture DIR --every 10` samples the
capture's frames dir and feeds the result to `--frames` (mutually exclusive with
`--frames`).

## `sample` flags

| flag | meaning |
|---|---|
| `--frames-dir` | directory the frames live in (or supplied by `--run-dir`) |
| `--run-dir` | run dir whose `layout.json` supplies `frames_dir` (or `masks_dir` with `--masks`) |
| `--masks` | with `--run-dir`, sample `masks_dir` as `.png` instead of frames |
| `--every` | **stride in frame-number units** (required), e.g. `10` = every 10th frame |
| `--start` / `--stop` | inclusive frame-number bounds (default: earliest/latest present) |
| `--ext` | restrict to this extension (e.g. `.png` for masks) |
| `--all` | emit every strided number even if its file is absent (default: existing only) |
| `--emit` | `paths` (default), `csv`, `args`, or `names` (basenames CSV, for `run.sh --frames`) |

`--emit args` prints the resolved window as an exact `--frames …` argument list, so
it splices straight into any tool that takes one:

```bash
WIN=$(python -m analysis.frames window --frames-dir CAPTURE/frames \
        --anchor 000040.jpg --radius 10 --step 5 --emit args)
python -m analysis.viz.frame_sheet --frames-dir CAPTURE/frames $WIN \
    --out-dir RUN/sheets --basename around_000040
```

## `window` flags

| flag | meaning |
|---|---|
| `--frames-dir` | directory the frames live in (or supplied by `--run-dir`) |
| `--run-dir` | run dir whose `layout.json` supplies `frames_dir` (or `masks_dir` with `--masks`) |
| `--masks` | with `--run-dir`, resolve the window in `masks_dir` as `.png` instead of frames |
| `--anchor` | anchor frame, e.g. `000040.jpg`; its zero-pad **width** and **extension** are inferred from it |
| `--offsets` | explicit comma offsets in frame-number units, e.g. `=-10,-5,0,5,10`. Use `=` when it starts with a minus so argparse doesn't read it as a flag. Overrides `--radius`/`--step` |
| `--radius` / `--step` | symmetric ±window (default 10) and stride (default 5) when `--offsets` is absent |
| `--ext` | override the frame extension — e.g. `.png` to resolve the matching **mask** window from a `.jpg` anchor |
| `--all` | keep offsets whose file is absent (default: existing files only) |
| `--emit` | `paths` (one per line, default), `csv`, or `args` (a ready-to-splice `--frames …`) |

Frame 0 (the anchor) is always in the window. Offsets that resolve to a negative
frame number are skipped. `order`/`window` share `--emit`.

## Notes

- Pure stdlib; no bpy, so it runs in the `artscript` analysis env (and would in
  Blender's python too). It builds on `analysis.lib.io.frame_stem` for the stem
  convention, so the filename contract stays defined in one place.
