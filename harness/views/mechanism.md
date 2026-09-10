# `mechanism` view — the BLIND A/B choice on the JOINTS *declaration*

**Answers:** "which of these two arcs is the real mechanism?"

**A module.** Everything on this page — the view, the sheets, the pick, both
gates — exists only in a run whose `RUN_DIR/modules.json` enables `mechanism`
(`core/modules.py`; `run.sh --enable/--disable mechanism`). In a run without it
the view is refused, `mechanism_calls check` passes saying so, and the run's
`AGENT_TASK.md` carries none of this.

Each joint is stepped across its declared `limit` **twice** — once about the
declared `axis`, once about its **negation** — in the **canonical** frame, from
**two picked viewpoints** each. The two arcs go on the page as **ARM A** and
**ARM B**, and the page never says which is which. **No source, no pose, nothing
scored.** Rendered every pass alongside the turntable; a rigid object has no
joint to sweep and the view says so and writes nothing.

```
PASS_DIR/mechanism_<joint>_<arm>_r<R>_s<NN>_<state>.png
PASS_DIR/mechanism_sheets/mechanism_sheet_<joint>_A.png   <- READ THESE TWO
PASS_DIR/mechanism_sheets/mechanism_sheet_<joint>_B.png
PASS_DIR/mechanism_sheets/mechanism_sheet_<joint>.png     <- both, same-state
PASS_DIR/mechanism_sheets/mechanism_sheet_manifest.json
```

**One page per arm, plus a stacked one.** Per-arm pages get the full width (on
the stacked layout every tile is half-size). Decide on the stacked page,
look closely on the per-arm ones.

**Why a forced choice and not a verdict.** An absolute question — *does this arc
look right?* — has a default answer, and it is yes. A choice between two rendered
arcs has no default, and the harness knows the answer, so the pick is **graded**.

## Why this is not the turntable

They are complementary halves of the shape gate, and neither substitutes for the
other:

| | turntable | mechanism |
|---|---|---|
| moves | the **camera** | the **joint** |
| holds | articulation | the camera |
| per | observed articulation state | declared joint |
| frame | the reference frame's base pose | canonical |
| source tile | yes (identity anchor) | **no** |
| answers | is the shape coherent from every angle | does the joint swing the right way |

The turntable renders every observed *state*, so it never shows a joint *moving*.
That is exactly the blind spot below.

## The bug it exists for

**Negating a revolute `axis` is the same as negating the state.** So any state that
is its own negation modulo the rotation is a **fixed point** of the sign flip — and
for the usual `limit: (0, 180)` hinge that is **both endpoints**.

A book cover authored to open *backwards through its own pages* renders
**pixel-identically** to a correct one at 0° and at 180°. Those are also the two
states an agent verifies hardest, because they are the legible extremes. So
silhouette IoU, the side-by-side, the critic and the turntable **all agree**, and
the error survives review. It shows only in the intermediate angles, which is
where an agent is least confident anyway — so the mismatch gets attributed to pose
and chased with sweeps that can never fix it.

Hence the A/B pair. Rendered as arcs the two are unmistakable:

```
axis (0,0,-1)   +0  +22  +45  +68  +90  +112  +135  +158  +180   <- opens like a book
axis (0,0, 1)   +0  +22  +45  +68  +90  +112  +135  +158  +180   <- dives through the pages
                 ^                                        ^
                 identical under either sign ─────────────┘
```

## How to read it

**Compare the two arms tile by tile at the same state.** Every tile prints the
joint's state in degrees. Put ARM A's `+22°` beside ARM B's `+22°` and ask which
one a real object does. One of them is right; there is no third option and no
"both look plausible".

- **The interior tiles carry the sign; the endpoints carry nothing.** The two
  arms render **identically** at `+0` and at the limit — they are the fixed
  points of the flip. An argument built from them cites the only two tiles that
  cannot distinguish the arms, which is why the gate refuses evidence that names
  only endpoints.
- **The small steps off rest are the most legible.** At the earliest few degrees
  one arm lifts the child away into free space and the other dips it straight
  into the body it should be opening away from. By `+90` a mirrored hinge has
  often swung somewhere that looks superficially plausible again.
- **Name landmarks, not impressions.** Which face is toward camera, what
  occludes what, which parts are visible that should be hidden. "The lid lifts
  away into free space" is true of one arm and unfalsifiable from a tile; "the
  rear-mounted SOS button is facing camera at `+22°`" is a fact about exactly one
  arm.

**The pick is recorded and GRADED.** After authoring or changing a joint,
`python -m analysis.mechanism_calls ask --run-dir RUN_DIR --pass-dir PASS_DIR`
emits `RUN_DIR/mechanism_calls.json` — one question per joint, pick blank — and
you write the arm label plus evidence naming at least **three interior states**
and what the child does at each. `... check --run-dir RUN_DIR` is a hard gate
(finalization, and `windows plan`):

- the **declared** arm, argued from the interior → PASS;
- the **mirrored** arm → BLOCKS: `WRONG JOINT IMPLEMENTATION`, naming the `axis`
  sign to write in `JOINTS`. Your pick stands; the rig is what changes;
- `unsure` → BLOCKS, no override (add `--mechanism-rows` and look again).

**The gate reads `mesh/pose.json`, which only a RENDER regenerates**, so editing
`scene.py` is half the fix. `check` notices that case and says `scene.py ALREADY
has axis … RE-RENDER` rather than repeating itself.

**One pick settles the sign.** Freshness hashes `type`/`origin`/`limit`/`child`
and the axis's *unsigned LINE*, not its sign — the two arms ARE that line's two
signs, so flipping it re-renders the identical pair. Writing the named sign makes
the existing pick correct; nothing is re-asked. Hashing the sign instead would
loop forever:

```
axis (1,0,0)  -> pick the real arc -> "wrong, write (-1,0,0)"
axis (-1,0,0) -> pick the real arc -> "wrong, write (1,0,0)"     ...forever
```

**Which arm is declared is not a convention.** It comes from the declaration
hash, so it is stable for a given rig (a re-render reproduces the same page) but
varies joint to joint and rig to rig — there is no "A means declared" to lean on
instead of looking.

**Nothing publishes the signed axis until the pick is graded** — not the header,
the CLI, or the JSON (both artifacts carry the declaration minus `axis` plus the
unsigned `axis_line`). Withholding it from the image while writing it into the
file beside it would be theatre. It appears after a wrong pick, as the fix.

What a mismatch means:

- **A reversed arc** — at intermediate values the group swings into the body
  instead of out into free space. Fix the **sign** of `axis` in `JOINTS`; a pose
  sweep cannot fix a rig error.
- **The wrong pivot line** — the group hinges about the wrong edge (a cover
  rotating about its far edge rather than the spine): `origin` is off the
  mechanism's axis, or `axis` names the wrong direction entirely.
- **Travel that stops short or overshoots** — the arc reaches an implausible
  configuration at the limit, or never reaches the observed one: `limit` is wrong.
- **Parts left behind** — something that should ride the joint stays put, or
  something that should stay put rides along: the `child` group is wrong.

The header is two terse lines (header height is height the tiles do not get);
the `axis` is withheld. Each row prints its viewpoint and angle off the swing
plane, and **NEAR EDGE-ON** defers to the other row. Both arms get the SAME
viewpoints — a per-arm camera would let a reader tell them apart without looking
at the mechanism.

## Tuning the cost

- `--mechanism-rows` (render, default 2) — viewpoints per joint. 1 for a quick
  cheap look; 3–4 when a joint stays ambiguous from two. Cost is
  joints × **arms** × rows × samples renders — the A/B pair doubles it, which is
  the price of the only check that catches a mirrored hinge.
- `--mechanism-samples` (render, default 7) — states per row, endpoints included.
  7 rather than 9 because tile size is page width / samples and this page lives or
  dies on tile legibility; 7 still leaves five interior states, well past the
  three a pick must cite.
- `--columns` (sheet builder; `--mechanism-columns` via shape_pass, default 9) —
  tiles per line before a row's strip wraps. Match it to a raised
  `--mechanism-samples` so a strip stays one line.

## How it is sampled and posed

- `--mechanism-samples` (default 7) states spanning the declared `limit`,
  **endpoints included**. An unlimited joint (genuinely continuous, e.g. a wheel)
  sweeps a full turn.
- **Every other joint held at rest** (0.0, clamped into its own limit), so one
  strip is one joint's doing. A collision needing two joints at once is out of
  scope here.
- **Canonical frame, no base pose.** The `axis` and `origin` under inspection are
  declared in canonical coordinates, so that is the frame the arc must be read in.
  Borrowing a frame's rotation would tilt the swing plane and put you back to
  inferring a sign from an oblique view.
- **Two rows per joint, viewpoints PICKED per joint, each fixed across its
  sweep.** A swing plane that contains the view direction projects both axis
  signs onto the same line — a single hard-coded camera has this view's own blind
  spot in miniature. So the viewpoints are chosen from the turntable's natural
  ring (level horizon, slightly above — a picked view can never look weird),
  scored by their angle **off the swing plane**, ties broken toward the classic
  az35/el20 so the camera only moves when that default is near-degenerate. The
  two rows sit **≥ 90° apart in azimuth**, so they cannot both be edge-on — and
  the second row resolves the one ambiguity a single silhouette keeps
  (toward-the-camera vs away). Each row's header prints its viewpoint and its
  off-plane angle; below 15° it says **NEAR EDGE-ON — read the other row**.
- Framed once per joint to the object's **full swept extent across BOTH arms**,
  so the group cannot walk out of shot or change scale between tiles — the same
  (center, radius) serves every arm and row, so the only thing differing between
  rows is the viewpoint. Framing per arm would render one strip slightly larger
  than the other and leak which arm is which through scale alone; the blinding is
  only as good as the framing.

## When to read it

**Every time you author or change a joint's `axis`, `origin`, `limit`, or
`child`** — and once more before finalizing. See
[AGENT_TASK.md](../../AGENT_TASK.md) (Signals, and the finalize checklist) and
[conventions/JOINTS.md](../../conventions/JOINTS.md) § the `axis` SIGN.

## Related

- [turntable.md](turntable.md) — the other half of the shape gate.
- [analysis/self_intersection.md](../analysis/self_intersection.md) — the
  numeric counterpart. It reports part-pair overlap for the **authored** frames;
  note a non-watertight joint child is **excluded**, so `unknown_pairs` is the
  line to read before believing a clean verdict.
- [analysis/viz/mechanism_sheet.py](../analysis/viz/mechanism_sheet.py) — the
  strip builder; shares its filename grammar via `core/mechanism_views.py`.
