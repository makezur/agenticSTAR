# `critic.py` — the VLM realism + identity judge (independent second opinion)

**Answers:** "does this reconstruction read as a plausible real object (realism),
and is it recognisably the SAME object as the photo (identity) — and if not, what
in the shape / pose / joints should change?"

Where [`silhouette.py`](../silhouette.md) (silhouette IoU) is the **numeric gate** and
[`depth.py`](../depth.md) is a **strong-but-noisy geometric guide** beside it, and
your own `Read` of the standalone overlap/render/source
images is your **first-person** visual judge, `critic.py` adds a **separate VLM's**
verdict beside them. It looks at the real photo next to the render(s) — the matched
view **and** the turntable's novel orbit views — and returns **scores + ranked,
tagged fixes** that plug straight into the "shape vs pose vs joint" decision in
[AGENT_TASK.md](../../../../AGENT_TASK.md).

**IoU and the critic fail in opposite directions — that's why you keep both.** IoU
is **precise but local and blind**: it is the best, most precise signal *in the local
neighbourhood* (once you're close, nothing pins the exact pose/scale/joint better),
but it's depth-blind, sees one silhouette, and can't tell shape from pose or spot a
near-symmetric flip. The critic is the **opposite** — imprecise (can't measure
degrees) but **global/holistic**: it judges realism + identity across novel views and
tells you *which way* things are wrong, including exactly what IoU can't see. So they
are not two estimates of one number to average.

The **hard checks** are silhouette IoU and turntable coherence; depth is a
strong-but-noisy geometric guide (weigh it, don't gate on it). The critic is the
independent judge you weigh beside them — alongside IoU, one of your two reliable
reads of pose/identity, and often the one that resolves what a noisy depth reading
only hints at. **When the critic is not happy, trust its direction and investigate**
— it is complaining about the global picture IoU's local precision can't see, so a
green IoU never talks you out of it. Distinguish its two outputs: a low *score* is a
soft prompt (a strong hint to fix something, not a finalize-blocker by itself), but a
**high-severity tagged `shape`/`pose`/`joint_state`/`joint_definition` fix IS a
finalize gate** — `aggregate.py`
lists it under FINALIZE REVIEW REQUIRED and it must be reconciled or justified (see
[aggregate.md](../../rollup/aggregate.md) + AGENT_TASK.md Stopping criteria). "Not a blocker" applies
to the *score*, never to a high-severity fix. And it **degrades gracefully**: with no
usable credentials it prints a skip message and exits `0`, so it never breaks the
iteration loop.

**It is additive, not a replacement.** Keep using your own `Read` on the individual
`overlap_<frame>.png` / `match_<frame>.png` / source / `depth_residual_<frame>.png`
images and the `composite`/`timeline` panels every iteration — the critic is a second
pair of eyes beside your first-person read, not instead of it.

## Run (artscript env)
```bash
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.scorers.critic \
    --source <frame image> \
    --render PASS_DIR/match_<frame>.png \
    --overlap PASS_DIR/overlap_<frame>.png \
    --turntable 'PASS_DIR/turntable_*_az*_el*.png' \
    --bg-mode black \
    --mask MASKS_DIR/<stem>.png \
    --hand-mask HAND_MASKS_DIR/<stem>.png \
    --metrics PASS_DIR/metrics_<frame>.json \
    --depth PASS_DIR/depth_<frame>.json \
    --pose PASS_DIR/pose.json \
    --frame <frame> \
    --out PASS_DIR/critic_<frame>.json
    # add --crops (+ optional --crop-pad 0.12) to also send zoomed detail crops.
```
(PASS_DIR = the `views/<NNNN>/` dir `render.sh` printed.)
Run it only for an explicit coverage or suspect frame after `composite.py` has
written that pass's overlap image + metrics and the turntable/depth have rendered.
Routine iterations and candidate sweeps make no VLM calls. `--source`/`--render`
are required; `--overlap` and `--turntable` are optional but strongly recommended
(the overlap shows where silhouettes disagree; the turntable is the ONLY view that
exposes detached parts and novel-view identity).

### Grounding (optional — feed it the numbers the agent already computed)
`--metrics`, `--depth`, and (optionally) `--sweep` pass the hard
signals for **this** frame into the prompt so the critic's prose can **reconcile
vision with the numbers** instead of guessing in a vacuum — e.g. "depth says the
object is too close AND the whole silhouette is oversized in the overlap → pull it
further from the camera (a pose fix, not build())". These are the exact files
`composite.py`/`depth.py`/`sweep` already wrote this pass; each is
**optional** and an absent/unreadable file is skipped silently (the critic still
runs image-only). The report records which were used in `"grounded_on": [...]`.
The critic stays an **independent** judge: a high IoU never talks it out of a
detached part it can see on the turntable.

From the `shape_pass` pipeline grounding is **OFF by default** (it can confuse the
visual judge); opt in per run with `harness/utils/shape_pass.sh ... --critic-ground` to feed
each frame's `metrics`/`depth`. Direct CLI invocation is likewise image-only unless
you pass `--metrics`/`--depth`/`--sweep`.

## What it sends the VLM (both comparisons + the numbers)
1. the **SOURCE** photo, the **MATCH** render, and the **overlap** → matched-view
   identity, proportions, colour, silhouette agreement;
2. the **SOURCE** photo + the `turntable_<state>_az<AAA>_el<±EE>` orbit views (up
   to `--max-turntable`, default 16; the view set is 8 angles per articulation
   state, including a raised and a dropped pair) → novel-view identity +
   realism/coherence. The prompt tells the VLM the turntable frames are
   **arbitrary orbit views of the 3D object, NOT aligned to the photo** — judge
   whether they show the *same object* and look *plausible*, not whether they
   pixel-match.
3. a **DECLARED RIG CONTEXT** block from `--pose` — joint definitions plus this
   frame's states, so the critic can distinguish a wrong value from a missing or
   structurally wrong joint;
4. (when grounded) a compact **NUMERIC CONTEXT** block — this frame's IoU, depth
   verdict, and any sweep best (pose order and/or joint states) — with an instruction to reconcile the
   visual read with it (see § Grounding).

`--turntable` is repeatable and each value may be a glob; images are de-duped and
capped at `--max-turntable` (default 16, logged if it drops any).

**The cap is spent ACROSS articulation states, not down the listing.** The paths
sort grouped by state, so the budget is dealt round-robin: every state is
represented, and a capped run drops **angles**, never a whole configuration.

### Consistent backdrop (`--bg-mode`) — one background for every image
The match + turntable renders are **RGBA on transparent film** (`film_transparent`,
`rig/render.py`); a VLM composites that transparency over an uncontrolled backdrop, so
a **dark object can wash out**. `--bg-mode` picks ONE backdrop for the match render,
the turntable frames, **and** the masked source (below) so they always share the same
background at judge time — never a mismatch:
- **`black` (default)** — alpha-composite / fill all three onto **uniform solid black**,
  so dark parts stay visible and the comparison is controlled;
- **`alpha`** — leave the renders' transparent film as-is (masked source gets a matching
  transparent backdrop). Byte-identical to the old behaviour for the renders.
The backdrop colour + the alpha-over-solid blend live in `core/renderer_settings.py`
(`BACKDROP_BGR`, `present()`) — the ONE place shared with `rig/render.py`'s
`film_transparent` flag and the other tools that present transparent renders
(`composite.py`); the on-disk renders are never modified (their alpha is
still needed by the IoU/depth scorers).

### Masked source (`--mask` / `--hand-mask`) — an apples-to-apples comparand
`--source` is the **full** real photo (background + hand). With `--mask` (the object
silhouette PNG) the critic **also** sends a **MASKED SOURCE** — the photo with its
background (and, with `--hand-mask`, the hand/occluder) blanked to the **same backdrop**
as the renders — so the VLM can compare the object 1:1 against the MATCH render on an
identical background. It is **additive** (the full SOURCE is still sent). Absent /
unreadable `--mask` skips it silently (like `--overlap`). Uses `rasters.load_mask` +
`build_keep`.

### Detail crops (`--crops`, opt-in) — higher fidelity for shape
`--crops` (requires `--mask`) additionally sends **ZOOMED DETAIL** crops of the match
render + masked source, cropped to a padded box (`--crop-pad`, default 0.12) around the
object (union of the source-mask bbox and the render-silhouette bbox — reusing
`silhouette.tight_bbox`). These give the VLM more pixels for fine **shape** judgment and
are labelled **shape-only — NOT for global position/framing/pose** (a crop discards the
framing that pose reads from). Off by default and not run in the routine shape-pass loop.

## Credentials — API keys only
The critic makes its **own** API call, so it needs a key in its environment (it
cannot borrow the driving agent's session). Two backends:

- **`--provider anthropic`** (default model `claude-opus-5`) — `ANTHROPIC_API_KEY`.
  Inside a Claude Code run the launcher already exports the agent's key, so it
  works with nothing extra. Inside a Codex run it needs the second key
  `tools/install_codex_bwrap.sh` stores when `ANTHROPIC_API_KEY` is exported at
  setup (the launcher then passes it in and allows `api.anthropic.com`).
- **`--provider openai`** (default model `gpt-5.6-sol`) — `OPENAI_API_KEY`, or,
  when that is unset, the Codex API-key login in `$CODEX_HOME/auth.json`. Inside a
  Codex run it therefore works with nothing extra. Inside a Claude Code run it needs
  the second key `tools/install_claude_bwrap.sh` stores when `OPENAI_API_KEY` is
  exported at setup.
- **`--provider auto`** (the default) takes the first of the two with a usable key,
  Anthropic first. `run.sh --critic-provider` pins one per run in
  `RUN_DIR/run_config.json`; `--api-key-env` names a different env var.

Both SDKs (`anthropic`, `openai`) are in `environment.yml`. With no usable
SDK/key the tool prints `[critic] SKIPPED — …` and exits `0`.

## Output — `critic_<frame>.json` (what a VLM should say)
The **contract** every backend is held to (scores clamped to `[0,1]`, `tag`
coerced to one of `shape`/`pose`/`joint_state`/`joint_definition`,
discrepancies ranked biggest-first). Each
discrepancy is a **directional MOTION correction** — which view shows it, which way
it's off, and roughly how much — leaving the exact value to be resolved downstream:
```json
{
  "frame": "000080.jpg",
  "realism": 0.0-1.0,          // plausible real object across all views
  "identity": 0.0-1.0,         // same object, same pose + articulation, as the photo
  "summary": "one-line verdict",
  "discrepancies": [           // ranked, biggest first
    {
      "tag": "shape" | "pose" | "joint_state" | "joint_definition",
      "part": "door",                     // named part
      "severity": "high" | "med" | "low",
      "evidence_views": ["match", "overlap"],  // where it's visible (optional)
      "observation": "In the match view and the overlap (yellow extra volume down the left edge) the door is swung open much further than the photo — the photo shows it roughly half-open, the render is nearly flat against the side.",
      "adjustment": "Close the hinge substantially, back toward rest — it's open about twice as far as it should be; the exact angle is resolved downstream."
    }
  ],
  "realism_notes": "...", "identity_notes": "...",
  "provider": "anthropic", "model": "claude-opus-5",
  "status": "ok", "n_images": 11, "n_turntable": 8,  // n_images varies with masked-source / crops

  "grounded_on": ["metrics", "depth"],   // which numeric signals were fed in
  "gate": {"pass_realism": 0.7, "pass_identity": 0.7,
           "realism_pass": true, "identity_pass": false}
}
```
The **tag** is the whole point: it drops each fix into AGENT_TASK.md's central
decision — `shape` → edit `build()`, `pose` → edit that frame's `pose` (rotation /
translation / depth), `joint_state` → edit a declared joint's frame value, and
`joint_definition` → add or repair JOINTS, then rerender and repeat pose
refinement. Legacy `joint` records normalize to `joint_state`. The
`observation`/`adjustment` are deliberately **wordy and directional** so the
recipient knows *how the object has to move*; a `pose` example reads "the whole
object is rotated too far clockwise and sits too low — spin it back and lift it a
little" (the exact magnitude is resolved downstream by `sweep`, which the agent runs
— the critic states only the direction).

**`pose` also catches a locally-good-but-semantically-wrong pose.** `pose` explicitly
covers a **near-symmetric flip** — a left/right mirror, front/back or top/bottom flip,
or a ~180° yaw where the **silhouette matches** but a distinctive feature (handle,
hinge, logo, spout) faces the wrong way. This is precisely the
**near-symmetric mis-orientation** AGENT_TASK.md warns a passing IoU (or a clean depth
reading) does *not* clear: the outline is symmetric, so the numbers **tie** across the
flip and can't see it — the critic's eyes are the only signal that can. The critic
describes the **rough direction** to fix it as a *large, discrete* reorientation and
leaves *how* to re-pose and re-converge to the agent (a guardrail keeps it from crying
flip on a genuinely symmetric object where either orientation is defensible).

**`shape` also covers geometric fidelity.** Beyond wrong/missing/detached/
misproportioned parts, `shape` now flags an **under-modelled** part: the right part,
roughly placed, but built from too few primitives for the form the photo/turntable
shows (a flat box where the outline is clearly curved/tapered/faceted, one blob where
there are distinct lobes or steps). The critic **describes in words which part needs
higher fidelity and the true geometric form it should take** — it does *not* prescribe
primitives or Blender ops (no "add a cylinder / use convex_hull / bevel"); *how* to
reach that shape is the agent's call, per AGENT_TASK.md rule 4 ("use the simplest
sufficient primitive model, then increase fidelity when evidence demands it").
A **guardrail** keeps it from asking for
gratuitous detail — a simple object modelled simply that already matches the photo is
correct.

**Why not an ID-colour map?** A per-part flat-colour render already exists (the `ids`
view → `ids.png`), but it colours by **named part**, and primitives get boolean-unioned
into one part, so it can't reveal "only one primitive covers this whole area" — that
part-level coarseness is already visible in the `match`/`turntable` RGB the critic sees.
So the fidelity signal is prompt-only; feeding `ids.png` is a possible future addition
only if per-part attribution (not primitive count) proves useful.

**No fabricated PRECISE numbers.** A VLM can't measure degrees/translation/scale to
fine precision from 2D renders, so the critic states **direction + relative
magnitude** (it MAY name a coarse discrete landmark it reads reliably — a ~90°
quarter-turn, a ~180° flip — but never a fine value like "23°") and hands the
exact value to `sweep` (which searches pose orders and/or joint states against the
mask + depth). Treat its prose as the *which-way* (± a coarse landmark), and the
sweep as the *how-much* — **for `pose`/`joint_state` fixes only**. Read the `discrepancies` and act on
the highest-severity one. **Do not gate a fix on the numbers corroborating it:**
- a `shape` fix is exactly the case the numbers **can't** corroborate — IoU is
  silhouette-only and depth is noisy, so a too-blocky/under-modelled/misproportioned
  part can leave both looking fine. Trust the critic + the turntable and fix `build()`;
  a `sweep` cannot fix geometry, and a high IoU can sit on wrong geometry.
- a `pose` flip is near-symmetric, so the numbers **tie** across it and can't see it
  either — trust the critic's eyes.
Never defer a high-severity fix because "the IoU looks OK". The numbers say *how far
off*, the critic + turntable say *what kind of error* — you need both.

**Exit code (soft gate):** `0` iff `realism ≥ --pass-realism` AND
`identity ≥ --pass-identity` (both default `0.7`); `1` otherwise; `2` only on a hard
error (unreadable required image, or the model never returned parseable JSON — then
the JSON is written with `"status": "error"`). A skip (no creds) is exit `0`.

## Swap the VLM
The judgement is provider-independent: the rubric + JSON contract live in one
`SYSTEM_PROMPT` string (plus a `GROUNDING_PROMPT` appended only when numeric
context is supplied), and each backend is a small class exposing
`complete(system, images, user_text) -> str`. To add a provider, write one more
class (base64 the images the way that SDK expects) and one branch in
`make_backend()`; the prompt, parsing, and schema enforcement are shared. Existing
backends: `AnthropicBackend` (Anthropic Messages API) and `OpenAIBackend` (OpenAI
Responses API). Pick the model with `--model` (or `critic.model` in
`run_config.json`); it must accept images.

## Where it fits
The independent identity/realism opinion beside the numeric signals (the IoU gate
+ the depth guide). `harness/utils/shape_pass.sh` runs it
only with `--critic-frames` or `--critic-all`, after selected frames' numeric scoring,
and prints each fresh verdict inline. The output is stamped with the pass and
`scene.py` hash; older verdicts are stale historical context, not current evidence.
Fold fresh tagged fixes into the shape/pose/joint decision; never let a passing
critic override a failing IoU gate or a
detached part on the turntable, and never let a passing IoU override a high-severity
critic fix (`aggregate.py` gates those as FINALIZE REVIEW REQUIRED).
