# Agentic STAR: Shape Tracking and Reconstruction from Monocular Videos

*Kirill Mazur, Nikita Karaev, Matthew Chang, Jitendra Malik, Nur Muhammad "Mahi" Shafiullah*  
Amazon FAR (Frontier AI and Robotics)

Project page: https://agenticstar.github.io/

## Overview

This is a harness for agentic shape tracking of articulated objects. A coding
agent (Codex or Claude Code) looks at the frames of a monocular video, writes a
Blender script that builds the object from primitives, poses it per frame in
front of the known cameras and declares its joints, then renders, compares to
the frames and iterates. The output is a GLB with named parts and a `pose.json`
with per-frame poses and joint states.

The two main components the agent works with are the pose search views
([sweep/apply](harness/views/sweeps/sweep.md)) and the
[temporal tool](harness/analysis/temporal/report.md). The task the agent reads
is [AGENT_TASK.md](AGENT_TASK.md); the `scene.py` contract is in
[conventions/](conventions/).

The repo was designed and tested with GPT-5.6-Sol and Claude Fable 5. The
results in the paper were obtained with the mechanism module turned on
(`--enable mechanism`; it is off by default). Newer models tend to do better
with it off. The GPT-5.6-Sol results were obtained with the `critic` branch,
which adds a second VLM as a critic; newer models should work fine without it.

## Installation

Linux x86_64 with an NVIDIA GPU. You need `micromamba`, `git`, `curl`, `tar`,
`xz`, `python3`, `tmux`, `bubblewrap`, `iproute2` and `ffmpeg` on the host.
Root (or `CAP_NET_ADMIN`) is needed for the sandbox's network namespace.

```bash
./install.sh                          # artscript env, Blender 4.2.5, pi3x env

export ANTHROPIC_API_KEY=sk-ant-...   # for Claude Code
tools/install_claude_bwrap.sh         # once; stores the key in a state dir outside the checkout

export OPENAI_API_KEY=sk-...          # for Codex
tools/install_codex_bwrap.sh
```

`claude` and/or `codex` must be on `PATH`.

## Running

One supervised run on the committed garden-shears example, on GPU 0:

```bash
tools/run_kf.sh --agent claude --timeout-hours 4 examples/garden_shears/capture:0
tools/claude-supervisor attach --run-dir runs/capture_kf_<stamp>    # watch it
```

Use `--agent codex` and `tools/agent-supervisor` for Codex. Add
`--enable mechanism` to reproduce the paper setting. Outputs land in
`runs/<name>/`: `scene.py`, `mesh/object.glb`, `mesh/pose.json`, and every
pass's renders and side-by-sides under `iterations/`. A run takes hours and
many tokens; `--timeout-hours` bounds it.

The agent runs inside a Bubblewrap sandbox with network access limited to the
model provider's API host. `SANDBOX_NET=open` disables that for debugging.
`./run.sh --help` lists the remaining options (GPU pinning, frame selection,
resume, `--prompt-only`).

## Input format

A capture is a directory with:

```
frames/000000.jpg, 000005.jpg, ...   video frames
mask_object/<stem>.png               binary object mask per frame, same size as the frame
mask_hand/<stem>.png                 optional; occluder mask, treated as don't-care
tracking/cameras.npz                 per-frame camera-to-world and intrinsics
tracking/keyframes.json              the frames that have a camera
```

Masks can come from any segmenter. Cameras come from
[Pi3X](https://github.com/yyfz/Pi3), which `tools/make_capture.py` runs
jointly over the keyframes:

```bash
tools/video_to_frames.sh clip.mov my_input --every 5
# add my_input/mask_object/<stem>.png for those frames
micromamba run -n pi3x python tools/make_capture.py --src my_input --out captures/my_object
tools/run_kf.sh --agent claude captures/my_object:0
```

`--depth` additionally keeps Pi3X's per-frame pointmaps in `depth/` for the
depth scorer. `examples/garden_shears/input/` is such an input directory and
`examples/garden_shears/capture/` is what the converter produced from it. The
schema, reader and validator live in `datasets/common/`.

We thank the authors of Pi3 for releasing their model and code.

## Citation

```bibtex
@misc{mazur2026agenticstar,
  title        = {Agentic STAR: Shape Tracking and Reconstruction from Monocular Videos},
  author       = {Mazur, Kirill and Karaev, Nikita and Chang, Matthew and Malik, Jitendra and Shafiullah, Nur Muhammad},
  year         = {2026},
  howpublished = {\url{https://agenticstar.github.io/}},
  note         = {Project page}
}
```

## License

MIT, see [LICENSE](LICENSE). `harness/views/sweeps/lib/_vendor/` carries two
optimizer routines adapted from SciPy under its BSD-3-Clause license
([SCIPY_LICENSE.txt](harness/views/sweeps/lib/_vendor/SCIPY_LICENSE.txt)).
