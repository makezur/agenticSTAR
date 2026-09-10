"""args.py — CLI parsing for the render rig (everything after the '--' separator).

Flags are grouped by concern (views, per-view tuning, camera framing, multi-frame
/ Pi3X, engine, export). Each flag's help string is the ground-truth
documentation for that flag; the per-view docs in views/*.md explain when to
use a whole view.
"""

import argparse
import sys

from core import mechanism_views
from core import turntable_views


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(description="Blender render rig")
    p.add_argument("--scene", required=True, help="path to agent scene.py")
    p.add_argument("--out", required=True,
                   help="RENDERS ROOT dir. Each invocation writes its outputs "
                        "into a fresh numbered pass subdir (<out>/0001, /0002, "
                        "...); the wrapper PRINTS that pass dir at the end — read "
                        "this run's renders from it. A view not rendered this pass "
                        "is ABSENT from its dir (no stale leftovers under a "
                        "live-looking name).")
    p.add_argument("--pass-label", default="",
                   help="optional descriptive metadata stored in MANIFEST.json; "
                        "pass directories are always immutable numeric ids. The "
                        "reserved label 'final' is rejected.")
    p.add_argument("--intent", default="",
                   help="what this pass CHANGES and why, recorded to "
                        "<pass>/intent.json at allocation — before the render, "
                        "so an interrupted run still finds the declaration")
    p.add_argument("--rollback", default="",
                   help="what to KEEP if this pass is rejected, recorded "
                        "alongside --intent in <pass>/intent.json")
    p.add_argument("--views", default="match",
                   help="comma list: match,turntable,ids,sweep,apply,osweep,oapply,"
                        "oapply_all,crop,visibility,depth. osweep/oapply fit each "
                        "frame independently; oapply_all commits ONE shared rotation "
                        "to every frame")
    p.add_argument("--crop-margin", type=float, default=0.6,
                   help="'crop' view: overscan margin per side (fraction of the "
                        "match frame) used to see how much of the object falls "
                        "OUTSIDE the frame — raise if the object is clipped even "
                        "in the overscan")
    p.add_argument("--visibility-margin", type=float, default=0.1,
                   help="'visibility' view: overscan margin per side (fraction of "
                        "the match frame) for the per-part solo renders, so a part "
                        "spilling off-frame is still captured (and per-part clipping "
                        "measured). 0 = analyze only inside the true frame (skip the "
                        "overscan) when you only care about occlusion")
    # --- sweep view tuning ---------------------------------------------------
    # The 'sweep' view searches per-frame POSES for max silhouette IoU by rendering
    # the REAL geometry at each candidate pose (see views/sweeps/sweep.md).
    p.add_argument("--sweep-mask", default="",
                   help="'sweep' view: path to the OBJECT mask (png) candidate "
                        "silhouettes are scored against (IoU). REQUIRED to run the "
                        "sweep; usually the same mask you pass to silhouette.py")
    p.add_argument("--sweep-hand-mask", default="",
                   help="'sweep' view: optional binary HAND (occluder) mask; its "
                        "region is scored as don't-care and iou_visible (over "
                        "K=not-hand) ranks candidates, mirroring silhouette.py")
    p.add_argument("--sweep-hand-dilate", type=float, default=0.0,
                   help="'sweep' view: grow the hand ignore-region by this fraction "
                        "of the longer side before ignoring (0 = the default: use "
                        "the hand mask exactly, as the segmenter produced it — the default "
                        "everywhere). Dilating only ever REMOVES pixels from "
                        "scoring, and it eats agreement faster than it eats "
                        "error, so it can lower iou_visible)")
    p.add_argument("--sweep-quality", type=float, default=1.0,
                   help="'sweep' view: speed<->fidelity ratio, clamped to (0.05,1]. "
                        "Sets the "
                        "scoring render resolution = quality x match-res (floored at "
                        "~96px long side) AND scales the default grid density. Use a "
                        "low value (fast, coarse) when far from the fit; raise toward "
                        "1.0 as the pose converges. The winner is always re-rendered "
                        "at full match-res regardless")
    p.add_argument("--sweep-ranges", default="",
                   help="'sweep' view: per-DOF search bounds as camera-frame ORDER "
                        "increments around the base pose, ';'-separated "
                        "'dof:min,max,steps' for grid or 'dof:min,max' for DE. "
                        "dof in roll,yaw,pitch (DEGREES, "
                        "orbits about the object centre — roll=in-plane about the "
                        "optical axis, yaw=about camera-up, pitch=about "
                        "camera-right), "
                        "dpx,dpy (image-plane shift as a FRACTION of frame width/height), "
                        "tz (depth nudge as a "
                        "FRACTION of the object's depth: dt_z=tz*depth). A JOINT "
                        "name is an ABSOLUTE state (revolute: degrees). Omitted DOFs "
                        "are held. Empty = a coarse default box (see sweeps/sweep.md) whose "
                        "step counts scale with --sweep-quality")
    p.add_argument("--sweep-strategy", choices=("grid", "de"), default="de",
                   help="'sweep' view: grid uses the Cartesian lattice and optional "
                        "refinement; de uses bounded SciPy differential evolution "
                        "inside the basin selected by --sweep-pose-start and these "
                        "ranges")
    p.add_argument("--sweep-refine", type=int, default=1,
                   help="'sweep' view: number of coarse->fine passes after the "
                        "initial grid; each re-centers on the current best and shrinks "
                        "every range by --sweep-refine-shrink (same step count)")
    p.add_argument("--sweep-refine-shrink", type=float, default=0.5,
                   help="'sweep' view: per-pass range shrink factor for refinement "
                        "(0.5 = halve the search window each pass)")
    p.add_argument("--sweep-de-max-evals", type=int, default=3600,
                   help="'sweep --sweep-strategy de': total DE evaluation budget "
                        "split across the --sweep-topk independent restarts, before "
                        "optional local polishing")
    p.add_argument("--sweep-de-popsize", type=int, default=8,
                   help="'sweep --sweep-strategy de': SciPy population multiplier "
                        "(population is approximately this value times active DOFs)")
    p.add_argument("--sweep-de-seed", type=int, default=17,
                   help="'sweep --sweep-strategy de': deterministic random seed")
    p.add_argument("--sweep-de-mutation", default="0.5,1.0",
                   help="'sweep --sweep-strategy de': mutation F as one value, or "
                        "a min,max dithering interval (default 0.5,1.0)")
    p.add_argument("--sweep-de-recombination", type=float, default=0.8,
                   help="'sweep --sweep-strategy de': crossover probability CR "
                        "in [0,1] (default 0.8)")
    p.add_argument("--sweep-polish", choices=("none", "powell"), default="none",
                   help="'sweep --sweep-strategy de': optional bounded local solver "
                        "started from DE's best result (default: none)")
    p.add_argument("--sweep-polish-max-evals", type=int, default=200,
                   help="'sweep --sweep-strategy de --sweep-polish powell': local "
                        "solver evaluation budget")
    # --- unified space + grid search (see views/sweeps/sweep.md) -------------
    # The 'sweep' view searches a UNIFIED DOF vector: name pose ORDER DOFs
    # (roll/yaw/pitch/dpx/dpy/tz) and/or JOINT names in one --sweep-ranges
    # spec. The space is inferred from the named DOFs (override with --sweep-space).
    p.add_argument("--sweep-space", default="", choices=["", "pose", "joint", "both"],
                   help="'sweep' view: which DOF space to search — 'pose' (order "
                        "increments), 'joint' (articulation states), or 'both' "
                        "(coupled). Default '' = INFER from the DOFs named in "
                        "--sweep-ranges (pose names -> pose, joint names -> joint, a "
                        "mix -> both; empty ranges -> pose). Also picks the "
                        "empty-ranges default box")
    p.add_argument("--sweep-angle-preset", default="",
                   help="'sweep' view: per-axis size PRESET for the empty-"
                        "--sweep-ranges default box, ';'-separated 'axis:preset'. "
                        "axis in roll,yaw,pitch; preset in tiny (+/-5deg), "
                        "standard (+/-15deg, the default), large (+/-30deg), "
                        "huge (+/-45deg). A preset may be suffixed with a "
                        "DIRECTION to make the band ONE-SIDED (spending the whole "
                        "2*half budget on the side you saw): roll:PRESET_cw/_ccw, "
                        "yaw:PRESET_leftcloser/_rightcloser, "
                        "pitch:PRESET_topcloser/_bottomcloser (which EDGE swings "
                        "toward the camera; signs in conventions/DIRECTIONS.md). "
                        "PREFER a directional preset when you can see which way the "
                        "pose is off (you usually can) — only tiny is a genuine "
                        "sign-agnostic touch-up. Overrides "
                        "ONLY the named angle axes' default search band; unnamed "
                        "rotation axes start standard and dpx/dpy/tz retain "
                        "their local defaults. Explicit "
                        "--sweep-ranges numbers for an axis override its preset. "
                        "A full-swing search (unknown facing) is an explicit "
                        "range, e.g. 'yaw:-180,180,9' with low --sweep-quality "
                        "(coarse), then refine into the winning basin")
    p.add_argument("--sweep-joints", default="",
                   help="'sweep' view: comma-separated joint names to restrict the "
                        "DEFAULT (empty --sweep-ranges) joint/both sweep to; ignored "
                        "when --sweep-ranges names joints explicitly")
    p.add_argument("--sweep-candidates", default="",
                   help="'sweep' view: explicit list of pose candidates to score, as "
                        "inline JSON text or a path to a JSON file (bypasses the grid; "
                        "--sweep-ranges/--sweep-refine ignored). Schema in sweeps/sweep.md")
    p.add_argument("--sweep-pose-start", default="",
                   help="'sweep' view: the JSON placement the grid starts FROM (its "
                        "centre); defaults to the frame's current pose. Carries NO "
                        "scale — the object's shared SCALE is applied by the engine, "
                        "and a placement stating a different one is refused")
    p.add_argument("--sweep-start-shift", default="",
                   help="'sweep' view: a directional ORDER composed onto the pose "
                        "start BEFORE searching, so the grid hunts a window AROUND the "
                        "shifted pose (direction + search in one call). Same grammar "
                        "as --apply: ';'-separated 'dof:value' (ONE value each) where "
                        "a pose DOF (roll/yaw/pitch/dpx/dpy/tz) is a camera-frame "
                        "order increment and a JOINT name is an ABSOLUTE state. Unlike "
                        "--sweep-pose-start (placement only) this CARRIES joints. Pair "
                        "with --sweep-angle-preset / --sweep-ranges for the band, e.g. "
                        "'--sweep-start-shift yaw:90 --sweep-angle-preset yaw:standard' "
                        "searches +/-15deg around a 90deg turn. A pose shift moves the "
                        "window; a joint ALSO named in --sweep-ranges keeps its explicit "
                        "band (the shift is ignored for it). Mutually exclusive with "
                        "--sweep-candidates; composes after --sweep-pose-start")
    p.add_argument("--sweep-topk", type=int, default=0,
                   help="'sweep' view: for DE, run this many independent seeded "
                        "restarts, list one winner from each, and render every winner "
                        "(default 6). For grid/candidates, list this many entries in "
                        "plain combined-score order (default 10). 0 selects that "
                        "strategy default. `rank` names an entry everywhere")
    p.add_argument("--sweep-dump-topk", type=int, default=0,
                   help="sweep-family views: also re-render up to N candidates at "
                        "full match-res to <stem>_top_00.png .. "
                        "<stem>_top_<N-1>.png (00 = the numeric winner, the same "
                        "pose as <stem>_best.png). 0 (default) still panels the "
                        "winner alone. The N are the BEST CANDIDATE PER ORDERED "
                        "GRID CELL — the cell scale is the grid you asked for in "
                        "--sweep-ranges, so adjacent near-duplicates from one "
                        "optimum collapse to their best and refine hits inside one "
                        "cell count once. A cheap 'show me every side' contact "
                        "sheet for eyeballing which orientation matches the photo "
                        "when several poses tie on IoU — pair it with a wide-range "
                        "--sweep-ranges 'yaw:-180,180,12;pitch:-80,80,5'. You get "
                        "FEWER than N when fewer cells are occupied (the report's "
                        "panels.cells_occupied says how many); an explicit "
                        "--sweep-candidates set has no grid, so there it is plain "
                        "score order. This flag applies to grid/candidate searches; "
                        "DE instead renders its --sweep-topk restart winners")
    p.add_argument("--sweep-visuals", default="auto",
                   choices=("auto", "all", "none"),
                   help="sweep-family views: how many candidates become IMAGES. "
                        "'auto' (default) honours --sweep-dump-topk, and always "
                        "renders at least the winner, so an order never lands as a "
                        "bare score. For DE, auto/all render every restart winner. "
                        "For other searches, 'all' panels EVERY scored candidate — the "
                        "honest 'show me the whole grid' — checked against the "
                        "visual budget (100 panels) because the count comes from "
                        "the grid, not from you; add --waive-visual-budget to mean "
                        "it. 'none' scores only (no candidate images, no sheet); "
                        "<stem>_best.png is still written, it is the view's output")
    p.add_argument("--waive-visual-budget", action="store_true",
                   help="proceed past the visual budget (see --sweep-visuals): "
                        "yes, you really do want that many panels")
    p.add_argument("--sweep-depth-weight", type=float, default=None,
                   help="'sweep' view: weight of the observed-depth penalty in the "
                        "candidate ranking: combined = gate IoU - weight * "
                        "min(depth_canon, 1), where depth_canon is the candidate's "
                        "RAW conf-weighted depth error |render_Z - observed_Z| as a "
                        "fraction of the object's longest dimension (the shared "
                        "SCALE). IoU alone is depth-blind — a nearer+smaller pose "
                        "projects the same silhouette — so this breaks the tie "
                        "with real geometry. RAW error on purpose: a fitted "
                        "scale assumes the pose is right and hides pose "
                        "error. Default: the run's depth_config.json `weight` "
                        "(the depth backend's default), else 0.1. Needs "
                        "--tracking (the Z pass rides the same scoring "
                        "render, so it costs no extra raster); without it, or "
                        "with weight 0, ranking is pure IoU as before")
    p.add_argument("--sweep-grid-slice", default="",
                   help="'sweep' view: 'INDEX/TOTAL' — score only this shard of the "
                        "flattened coarse grid (stride slice), for parallel sharding "
                        "across TOTAL Blender processes; the caller merges the "
                        "per-shard sweep.json files. Refinement is disabled when "
                        "slicing (coarse grid only)")
    p.add_argument("--sweep-timeout", type=float, default=0.0,
                   help="'sweep' view: wall-clock budget in SECONDS for candidate "
                        "scoring (0 = no limit, the default). When the deadline "
                        "passes, the candidate loop and any remaining refine passes "
                        "stop and the BEST-SO-FAR is ranked, re-rendered at full "
                        "match-res, and written to sweep.json/sweep.txt (flagged "
                        "timed_out); if not even one candidate finished, a minimal "
                        "sweep.json is still written. Granularity is one candidate — "
                        "it does not interrupt a single in-flight render (use the "
                        "RENDER_TIMEOUT env var on render.sh for a hard GPU-wedge "
                        "backstop). Progress prints to stderr while scoring")
    # --- apply view (imperative directional order — see views/sweeps/apply.md) -------
    # The 'apply' view is the imperative sibling of 'sweep': it composes ONE
    # directional order onto the frame's current pose, RENDERS it, and SCORES that one
    # config against the mask (a sweep of a single candidate). Use it to CHECK a
    # specific tilt/turn/open; use 'sweep' to SEARCH for the best-fitting one. It
    # reuses sweep's scoring flags (--sweep-mask required, --sweep-depth-weight, etc.).
    p.add_argument("--apply", default="",
                   help="'apply' view: a directional ORDER to compose onto the "
                        "--frame frame's current pose, then RENDER + SCORE it against "
                        "--sweep-mask (a single-candidate sweep). ';'-separated "
                        "'dof:value' (ONE value each, not a range). dof is a "
                        "camera-frame pose ORDER INCREMENT — roll/yaw/pitch (DEGREES, "
                        "orbit about the object centre), "
                        "dpx/dpy (image-plane shift as a FRACTION of "
                        "frame width/height), tz (depth nudge as a FRACTION of the "
                        "object's depth) — OR a declared JOINT name set to an ABSOLUTE "
                        "state (revolute: degrees; clamped to its limit). Mix freely. "
                        "Reuses sweep's "
                        "scoring flags (--sweep-mask/-hand-mask/-depth-weight/-quality/"
                        "-base). Writes apply_best.png + apply.json/.txt (IoU/depth + a "
                        "paste-ready pose/joints block)")
    # --- osweep / oapply / oapply_all views (object-centric reorientation) ----
    # sweep/apply correct in the CAMERA frame (left-multiply one frame's pose). The
    # object-centric views do the OPPOSITE: a rotation in the object's own CANONICAL
    # frame, RIGHT-multiplied onto a pose (M' = M @ R_extra), then render + score
    # every frame against its mask. This expresses what a camera-frame yaw cannot —
    # "turn it about its own up-axis" is not a yaw except for an upright object.
    # Two SCOPES, and the difference is the whole point:
    #   osweep / oapply  PER FRAME: each frame gets its OWN rotation (searched /
    #                    named). Semantically N independent sweep/apply runs sharing
    #                    one Blender process and one candidate set.
    #   oapply_all       SHARED: ONE rotation for every frame — the fix for an object
    #                    BUILT mis-oriented, so it reads wrong in every frame and one
    #                    decision commits it. There is deliberately no searched
    #                    whole-run mode: ranking a shared rotation by MEAN IoU can
    #                    pick one that is mediocre everywhere.
    # Knobs are the canonical ROTATION VECTOR rx/ry/rz (degrees about object
    # +X/+Y/+Z; axis = direction, angle = norm — NOT the camera-frame yaw/pitch of
    # sweep). Scores per-frame masks from --masks-dir; reuses sweep's
    # --sweep-quality/-refine/-depth-weight/-timeout.
    p.add_argument("--osweep-ranges", default="",
                   help="'osweep' view: per-DOF search grid over the canonical "
                        "object rotation, ';'-separated 'dof:min,max,steps'. dof in "
                        "rx (about canonical +X right), ry (about +Y front), rz "
                        "(about +Z up), DEGREES — a ROTATION VECTOR in the object's "
                        "OWN frame (multi-axis = ONE rotation about the tilted axis; "
                        "NOT the camera-frame yaw/pitch of --sweep-ranges). Every "
                        "frame searches this SAME grid INDEPENDENTLY and keeps its "
                        "own argmax by its own gate IoU — N sweeps, nothing "
                        "aggregated. Omitted DOFs held at 0. Empty = a default box "
                        "(standard preset, +/-15deg, on all three axes) whose step "
                        "counts scale with --sweep-quality")
    p.add_argument("--osweep-angle-preset", default="",
                   help="'osweep' view: per-axis size PRESET for the empty-"
                        "--osweep-ranges default box, ';'-separated 'axis:preset'. "
                        "axis in rx,ry,rz; preset in tiny (+/-5deg), standard "
                        "(+/-15deg, the default), large (+/-30deg), huge "
                        "(+/-45deg). No directional suffix here: a canonical "
                        "rotation has no fixed on-screen direction (see "
                        "conventions/DIRECTIONS.md), so bands are symmetric. "
                        "Overrides "
                        "ONLY the named axes; explicit --osweep-ranges numbers "
                        "win (a full swing is e.g. 'rz:-180,180,9')")
    p.add_argument("--opreset", default="",
                   help="ALL object-centric views: an AXIS plus a RANGE OF ANGLES, "
                        "';'-separated 'axis:preset' — the one flag to reach for "
                        "instead of hand-computing ranges. axis in x,y,z (rx/ry/rz "
                        "accepted) = the object's OWN +X right / +Y front / +Z up. "
                        "preset is a TURN SET: flip (0,180), quarters "
                        "(0,90,180,270), octants (every 45deg), full "
                        "([-180,180], steps from --sweep-quality); or a SIZE band "
                        "tiny/standard/large/huge (+/-5/15/30/45deg); or an explicit "
                        "comma-separated angle LIST in degrees ('z:0,90,180,270'). "
                        "Numeric form is ALWAYS a list, never min,max,steps (that is "
                        "--osweep-ranges). Every turn set INCLUDES 0 so the no-op "
                        "baseline competes. Two axes = the Cartesian product "
                        "('z:quarters;x:tiny'). OR 'so3:N' (osweep ONLY — the "
                        "imperative verbs panel NAMED rotations and refuse a "
                        "blind draw; stands ALONE): identity "
                        "+ N Haar-uniform random rotations from a BAKED seed — "
                        "deterministic, and so3:2N extends so3:N without "
                        "reshuffling. The RE-LOCALISE move when tracking is lost "
                        "and no axis is suspected (if you know the axis, 'z:full' "
                        "is ~7 candidates instead of 64); pair with --sweep-refine "
                        "to descend from the coarse winner. 'so3:N@SEED' re-rolls "
                        "the draw when the default one judged wrong everywhere. "
                        "Assumes placement is "
                        "roughly right: IoU cannot rank rotations whose render "
                        "misses the mask entirely. No directional suffixes: a canonical "
                        "axis has no fixed on-screen direction (the same shared "
                        "rz:+30 moves a landmark by up to 169deg differently between "
                        "frames — see conventions/DIRECTIONS.md); signed NUMBERS are "
                        "fine. On the SEARCH view it becomes the candidate grid "
                        "(refine applies to band presets only — there is no basin "
                        "between 90deg and 180deg); on the imperative views it "
                        "becomes a labelled candidate SET, so each angle is rendered "
                        "and panelled per frame. Precedence: --osweep-ranges > "
                        "--opreset > --osweep-angle-preset")
    p.add_argument("--oapply", default="",
                   help="'oapply' and 'oapply_all' views: the canonical object "
                        "rotation(s) to right-multiply onto a frame's pose, then "
                        "RENDER + SCORE every frame against its --masks-dir mask (an "
                        "osweep of exactly the named candidates). '|' separates "
                        "candidates, ';' separates 'dof:value' terms inside one; dof "
                        "in rx/ry/rz (DEGREES about the object's OWN +X/+Y/+Z — a "
                        "rotation vector, NOT the camera-frame yaw/pitch of --apply). "
                        "Sugar: 'flip:x|y|z' is the 180deg flip about a canonical axis "
                        "(rx/ry/rz = 180); 'identity' the no-flip baseline; 'flips' the "
                        "DEFAULT SYMMETRY PANEL (identity|flip:x|flip:y|flip:z) — "
                        "the seam-flip check for near-symmetric objects. Use osweep "
                        "to SEARCH instead. WHICH VIEW: 'oapply' lets EACH FRAME pick "
                        "its own winner (frames disagree — a window seam); "
                        "'oapply_all' commits ONE rotation to every frame (the object "
                        "was built mis-oriented). A set of at most "
                        "core/visual_budget.MAX_SET_RENDERS (8) re-renders EVERY "
                        "candidate per frame (<view>_<label>_<frame>.png) and the .txt "
                        "gets a frames x candidates gate-IoU matrix; the winner is "
                        "always <view>_best_<frame>.png. Both write a "
                        "<view>_poses.json fragment for multiagent.windows merge/apply. Pair "
                        "with --frames to scope to a SUBSET (e.g. a window seam); in "
                        "shared 'oapply_all' the report then authors a paste block for "
                        "every scored frame (inheritance is off across a subset) and "
                        "WARNS if the reference frame is inside the subset (pasting it "
                        "would leak the rotation to out-of-subset frames) — per-frame "
                        "'oapply' always authors every frame, so neither applies")
    # --- camera framing (turntable orbit) + resolution -----------------------
    # The match/depth camera is always the fixed camera 0 (identity extrinsic);
    # per-frame intrinsics come from Pi3X (--tracking) or an iPhone-13 placeholder.
    # These flags only drive the turntable's orbit + the default resolution.
    p.add_argument("--azimuth", type=float, default=0.0,
                   help="deg — rotate the WHOLE turntable view set about the up "
                        "axis (0 = the first ring view is the camera-0 front view). "
                        "Was documented as an orbit knob but read by nothing: the "
                        "four angles were literals in views/turntable.py.")
    p.add_argument("--elevation", type=float, default=20.0,
                   help="deg above horizon — the elevation of the turntable's LEVEL "
                        "RING. The raised/dropped views of 'sphere8' have their own "
                        "absolute elevations (core/turntable_views.py), so this does "
                        "not push them past the pole.")
    p.add_argument("--turntable-views", default=turntable_views.DEFAULT_VIEW_SET,
                   choices=sorted(turntable_views.VIEW_SETS),
                   help="which turntable view set: 'sphere8' (default — a level "
                        "4-azimuth ring plus a raised and a dropped pair, so the top "
                        "and underside are actually seen) or 'ring4' (the level ring "
                        "only — half the renders per articulation state)")
    p.add_argument("--turntable-jitter", type=int, default=None,
                   help="SEED — perturb the turntable directions deterministically "
                        "(same seed, same set). Default: unset, so every pass looks "
                        "from the SAME directions and one iteration's renders are "
                        "comparable to the last's. Use this for a deliberate "
                        "look-from-somewhere-new pass; the angles land in the "
                        "filenames either way.")
    p.add_argument("--mechanism-samples", type=int,
                   default=mechanism_views.DEFAULT_SAMPLES,
                   help="states per joint PER ROW for the 'mechanism' view, "
                        "which sweeps ONE joint across its declared limit in the "
                        "canonical frame from two picked fixed viewpoints (no "
                        "source, nothing scored) so a wrong `axis` SIGN is "
                        "visible as an arc going the wrong way. Endpoints "
                        "included; the default reads as an arc in one strip.")
    p.add_argument("--mechanism-rows", type=int,
                   default=mechanism_views.DEFAULT_ROWS,
                   help="viewpoints (rows) per joint for the 'mechanism' view, "
                        "picked >=90 deg apart in azimuth so no swing plane can "
                        "be edge-on to all of them. Default "
                        f"{mechanism_views.DEFAULT_ROWS}: the second row "
                        "resolves toward-vs-away ambiguity. 1 = quick look; "
                        "3-4 when a joint stays ambiguous from two.")
    p.add_argument("--radius", default="auto", help="'auto' or float distance")
    p.add_argument("--fov", type=float, default=45.0, help="horizontal FOV deg")
    p.add_argument("--ortho", action="store_true", help="orthographic camera")
    p.add_argument("--res", default="800",
                   help="render resolution: 'N' (square NxN) or 'WxH'")
    p.add_argument("--match-res", default="",
                   help="path to an image (usually the source); render at its "
                        "exact pixel resolution. Overrides --res.")
    # --- explicit (measured) camera intrinsics -------------------------------
    # The camera is ALWAYS the fixed camera-0 (identity extrinsic); only the
    # intrinsics K are configurable. Priority: --intrinsics / --camera-json (this
    # block) > --tracking per-frame K > iPhone-13 placeholder. An explicit K applies
    # to ALL frames and is resolution-agnostic: its stated W,H is only the grid it
    # was measured on (the default canvas), and it is rescaled to whatever
    # resolution is actually rendered.
    p.add_argument("--intrinsics", default="",
                   help="explicit pinhole K 'fx,fy,cx,cy,W,H' in pixels (e.g. "
                        "measure_depth.py's intrinsics_for_render). Overrides "
                        "--tracking K for all frames.")
    p.add_argument("--camera-json", default="",
                   help="path to a camera JSON (see examples/camera_example.json); "
                        "its 'intrinsics' block is used. A non-identity 'pose' is "
                        "REJECTED — the harness uses a fixed camera-0 extrinsic and "
                        "moves the object, not the camera.")
    # --- multi-frame / per-frame tracking cameras -----------------------------
    p.add_argument("--tracking", "--pi3x", dest="tracking", default="",
                   help="capture tracking dir (cameras.npz + keyframes.json — "
                        "<capture>/tracking). Supplies the per-frame known cameras: "
                        "each frame's intrinsics K[k] and its extrinsic relative to "
                        "the reference frame (used to SEED that frame's object "
                        "pose). Absent -> auto-resolved from the run's layout.json "
                        "capture (see --no-auto-tracking); if that fails too, a "
                        "single frame with iPhone-13 placeholder intrinsics. "
                        "--pi3x is the old name, still accepted.")
    p.add_argument("--no-auto-tracking", "--no-auto-pi3x",
                   dest="no_auto_tracking", action="store_true",
                   help="do NOT auto-resolve --tracking from the run's layout.json "
                        "when no --tracking/--intrinsics is given. By default the "
                        "harness walks up from the output dir to RUN_DIR/layout.json "
                        "and uses its capture's tracking dir (so a manual sweep still "
                        "scores under the real camera, not the iPhone-13 "
                        "placeholder); pass this to force the placeholder fallback "
                        "instead. --no-auto-pi3x is the old name, still accepted.")
    p.add_argument("--frames", default="",
                   help="comma list of frame names to render (keys of scene.py's "
                        "FRAMES / basenames in Pi3X keyframes.json). Empty -> all "
                        "frames declared in FRAMES.")
    p.add_argument("--masks-dir", default="",
                   help="dir of per-frame OBJECT masks, one '<frame-stem>.png' per "
                        "frame (the harness-wide convention). REQUIRED by the "
                        "osweep/oapply/oapply_all views (they score every frame's "
                        "silhouette against its own mask); unused by other views.")
    p.add_argument("--hand-masks-dir", default="",
                   help="optional dir of per-frame HAND (occluder) masks "
                        "'<frame-stem>.png'; a frame's mask region is scored as "
                        "don't-care (iou_visible gates that frame) for the "
                        "osweep/oapply/oapply_all views. "
                        "Frames without a hand mask file score iou_raw.")
    p.add_argument("--ref-frame", default="",
                   help="which frame anchors the gauge (identity extrinsic). Empty "
                        "-> scene.py's REFERENCE_FRAME.")
    p.add_argument("--frame", default="",
                   help="which single frame the pose diagnostics (sweep/ids/"
                        "visibility/crop) run against. Empty -> the reference frame.")
    p.add_argument("--debug-project", default="",
                   help="'x,y,z' world point; print its projected pixel for the "
                        "match camera (sanity-check pose/intrinsics)")
    # resident worker mode (the render pool): after the one-time build() +
    # canonical capture + engine/camera/lighting setup, DON'T render the frame
    # list and exit — instead serve per-request renders on stdin forever (until
    # EOF/SIGTERM). The build() geometry stays resident; each request carries its
    # own pose/joints/camera. See harness/pool/serve.py + pool/README.md.
    p.add_argument("--serve", action="store_true",
                   help="resident render-worker mode: build once, then serve "
                        "line-delimited JSON render requests on stdin (used by "
                        "pool/manager.py). Ignores --frames.")
    # render engine. EEVEE_NEXT (rasterizer) is the DEFAULT: ~3-10x faster than
    # Cycles on these primitive scenes and it still supports the two things the
    # metrics rely on — a transparent film (alpha silhouette -> IoU) and emission
    # shaders (the ID pass). Use --engine CYCLES for a final path-traced pass
    # (true GI/soft shadows) when judging color fidelity or exporting.
    p.add_argument("--engine", default="BLENDER_EEVEE_NEXT",
                   choices=["CYCLES", "BLENDER_EEVEE_NEXT"])
    p.add_argument("--device", default="OPTIX", choices=["OPTIX", "CUDA", "CPU"],
                   help="Cycles compute device (ignored by EEVEE)")
    p.add_argument("--samples", type=int, default=64,
                   help="Cycles path samples / EEVEE TAA render samples")
    # export
    p.add_argument("--export", default="", help="GLB output path ('' = skip)")
    p.add_argument("--remesh", default="voxel:0.0",
                   help="'voxel:S' with S>0 to force watertight remesh before export")
    return p.parse_args(argv)
