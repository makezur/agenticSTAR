#!/usr/bin/env bash
# video_to_frames.sh — dump a video into the frames/ layout tools/make_capture.py
# reads: <out>/frames/000000.jpg, 000001.jpg, ... (6-digit zero-based stems =
# temporal order). Prints the fps to pass as --fps.
#
#   tools/video_to_frames.sh clip.mov my_input            # every frame
#   tools/video_to_frames.sh clip.mov my_input --every 5  # keep every 5th frame,
#                                                          # stems keep the source index
#   tools/video_to_frames.sh clip.mov my_input --max-dim 720
#
# Then add <out>/mask_object/<stem>.png (+ optional mask_hand/) for the frames you
# keep — binary PNGs, same size as the frame — and run make_capture.py.
set -euo pipefail

usage() { sed -n '2,13p' "$0"; }
[[ $# -ge 2 ]] || { usage >&2; exit 2; }
SRC="$1"; OUT="$2"; shift 2
EVERY=1
MAX_DIM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --every) EVERY="${2:?}"; shift 2 ;;
    --max-dim) MAX_DIM="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "video_to_frames.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg is required" >&2; exit 1; }
[[ -f "$SRC" ]] || { echo "no such video: $SRC" >&2; exit 1; }
[[ "$EVERY" =~ ^[0-9]+$ && "$EVERY" -ge 1 ]] || { echo "--every must be a positive integer" >&2; exit 2; }

mkdir -p "$OUT/frames"
vf=""
if [[ -n "$MAX_DIM" ]]; then
  # Downscale so the longest edge is MAX_DIM, keeping even dimensions.
  vf="scale='if(gt(iw,ih),min($MAX_DIM,iw),-2)':'if(gt(iw,ih),-2,min($MAX_DIM,ih))'"
fi
# Every frame is written first so the stem IS the zero-based source frame index
# (a select filter + frame_pts would number by timestamp instead); --every N then
# drops the ones that are not multiples of N.
ffmpeg -hide_banner -loglevel error -i "$SRC" ${vf:+-vf "$vf"} -fps_mode passthrough \
  -start_number 0 -q:v 2 "$OUT/frames/%06d.jpg"
if [[ "$EVERY" -gt 1 ]]; then
  for f in "$OUT"/frames/*.jpg; do
    stem="$(basename "$f" .jpg)"
    (( 10#$stem % EVERY == 0 )) || rm -f "$f"
  done
fi
fps="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
       -of default=noprint_wrappers=1:nokey=1 "$SRC" | awk -F/ '{ if ($2) printf "%.4f", $1/$2; else print $1 }')"
n="$(ls "$OUT/frames" | wc -l)"
echo "wrote $n frames -> $OUT/frames (source fps $fps)"
echo "next: add $OUT/mask_object/<stem>.png [+ mask_hand/], then"
echo "  micromamba run -n pi3x python tools/make_capture.py --src $OUT --out captures/<name> --fps $fps"
