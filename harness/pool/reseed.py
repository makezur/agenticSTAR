"""reseed.py — save a pose+joints pick, then spend it as the seed of the next hop.

Refining a frame is coordinate descent: sweep a few DOFs, PICK a candidate by eye,
sweep the next few from that pick. Spelling hop N+1 by hand means re-typing a
quaternion, a translation and every joint state, and the pick is rarely rank 0 (IoU is
a proxy; the silhouette-best candidate is routinely an edge-on trap), so an arbitrary
candidate must be addressable.

Every report writes the FULL absolute joint state per candidate, so a reseed is a read:
one normalized record {frame, pose, joints, provenance}, one reader per dialect on
disk, one writer per output shape. Pose and joints always travel together — carrying
the placement alone reverts the articulation while the pose lands, which looks
plausible and is wrong.

Every failure is LOUD (a silent fallback to the committed pose is the bug this
removes), and a selector that matches nothing lists what is present.

Contract and CLI: pool/reseed.md. Pure Python — no bpy, no cv2 — so the manager can
call it at claim time.
"""

import argparse
import json
import os
import sys

from core import state_json

# --------------------------------------------------------------------------- #
# the normalized record
# --------------------------------------------------------------------------- #
# {
#   "frame":   "000250.jpg" | None,   # None when the source cannot say
#   "pose":    {"quaternion": [w,x,y,z], "translation": [x,y,z], "scale": float?},
#   "joints":  {name: state, ...},    # every declared joint, absolute
#   "provenance": {"source": path, "selector": str|None, "dialect": str,
#                  "order_id": str|None, "rank": int|None, "score": float|None},
# }
#
# `provenance` rides into whatever the record is written to, so "why is this pose
# here" stays answerable three hops later.

POSE_KEYS = ("quaternion", "translation", "scale")


class ReseedError(Exception):
    """Any unresolvable reseed. Carries a message naming what was tried."""


def _read_json(path):
    if not os.path.isfile(path):
        raise ReseedError(f"reseed source {path!r} does not exist")
    try:
        with open(path) as f:
            return json.load(f)
    except ValueError as exc:
        raise ReseedError(f"reseed source {path!r} is not valid JSON: {exc}")


def split_address(address):
    """`<path>#<selector>` -> (path, selector|None).

    One syntax for "which record in that file", because every dialect needs it.
    Split on the LAST '#': a selector never contains one, while a path might.
    """
    text = str(address)
    if "#" not in text:
        return text, None
    path, _, selector = text.rpartition("#")
    if not path:                       # "#rank=0" with no path
        raise ReseedError(f"reseed address {address!r} has no path before the '#'")
    return path, (selector or None)


def _pose_block(raw, where):
    """Normalize a pose dict to {quaternion, translation, scale?}.

    Accepts the quaternion spelling every dialect on disk uses. `rotation_euler`
    is accepted too because an AUTHORED scene.py frame may use it, but it is NOT
    converted here (that needs mathutils, which is bpy-side) — it is refused with
    the reason, since a reseed that silently dropped a rotation would be the
    worst possible failure of this module."""
    if not isinstance(raw, dict):
        raise ReseedError(f"{where}: pose is {type(raw).__name__}, expected an object")
    if raw.get("quaternion") is None:
        if raw.get("rotation_euler") is not None:
            raise ReseedError(
                f"{where}: pose carries 'rotation_euler' but no 'quaternion'. "
                "Reseeding is quaternion-native (every report on disk writes one); "
                "converting Euler angles needs the rig's rotation order and is "
                "deliberately not done here. Re-render the frame to get a "
                "quaternion, or author one.")
        raise ReseedError(f"{where}: pose has no 'quaternion'")
    if raw.get("translation") is None:
        raise ReseedError(f"{where}: pose has no 'translation'")
    quat = [float(v) for v in raw["quaternion"]]
    trans = [float(v) for v in raw["translation"]]
    if len(quat) != 4:
        raise ReseedError(f"{where}: quaternion has {len(quat)} components, want 4")
    if len(trans) != 3:
        raise ReseedError(f"{where}: translation has {len(trans)} components, want 3")
    out = {"quaternion": quat, "translation": trans}
    if raw.get("scale") is not None:
        out["scale"] = float(raw["scale"])
    return out


def _joints_block(raw, where):
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ReseedError(f"{where}: joints is {type(raw).__name__}, expected an object")
    return {str(k): float(v) for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# dialect detection
# --------------------------------------------------------------------------- #
SWEEP_REPORT = "sweep_report"          # sweep.json / apply.json (single-frame)
OBJECT_REPORT = "object_report"        # osweep/oapply/oapply_all (one entry per frame)
POSE_JSON = "pose_json"                # mesh/pose.json (object_pose per frame)
FRAME_MAP = "frame_map"                # fragment / progress.json / poses.json
ORDER = "order"                        # a pool order body


def detect_dialect(doc, path=""):
    """Which dialect a loaded JSON document speaks.

    Keyed on STRUCTURE, not filename: a sweep report copied to another name must
    still be readable, and an agent writing progress.json in a spool dir must not
    be misread as a report."""
    if not isinstance(doc, dict):
        raise ReseedError(f"{path or 'document'} is not a JSON object")
    # The object verbs are detected by their OWN positive signal — a top-level
    # `frames` LIST — and checked FIRST. They have no `best`: the winner is per
    # frame, so there is no single pose to name, and neither
    # `shared_report.write_report` nor `write_per_frame_report` emits the key.
    # The camera-frame verbs carry `frame` SINGULAR and never a top-level `frames`
    # (`engine`'s meta is a closed dict literal), so the two shapes cannot collide.
    #
    # `analysis.viz.sweep_sides.is_object_report` asks a deliberately STRICTER
    # question — it also requires `ranked[].images`, because it answers "are there
    # per-frame pixels to panel", not "which reader parses this" — and it is not
    # imported here: `sweep_sides` pulls in cv2, which this module stays free of so
    # the manager can call it on the claim path.
    if "ranked" in doc and isinstance(doc.get("frames"), list):
        return OBJECT_REPORT
    if "ranked" in doc and "best" in doc:
        return SWEEP_REPORT
    frames = doc.get("frames")
    if isinstance(frames, dict):
        entry = next(iter(frames.values()), None)
        if isinstance(entry, dict) and "object_pose" in entry:
            return POSE_JSON
        return FRAME_MAP
    if "views" in doc or "sweep" in doc or "seed" in doc:
        return ORDER
    raise ReseedError(
        f"{path or 'document'}: unrecognized JSON shape (keys: "
        f"{sorted(doc)[:8]!r}). Reseed reads a sweep/apply report, an "
        "osweep/oapply report, mesh/pose.json, a multiagent.windows fragment or "
        "progress.json, or a pool order.")


# --------------------------------------------------------------------------- #
# readers — one per dialect
# --------------------------------------------------------------------------- #
def _sweep_entries(doc):
    """The addressable candidates of a sweep/apply report, in listing order.

    Addressed by the entry's own `rank` — the NUMERIC ranking, which is what
    `sweep.txt`'s `rank` column prints. The listing is GAPPY (topk cuts it, then
    panelled candidates from below the cut are appended keeping their true rank), so
    line N is routinely not rank N; resolving against the line number would hand back
    a different candidate."""
    ranked = doc.get("ranked")
    if not isinstance(ranked, list):
        return []
    return [r for r in ranked if isinstance(r, dict)]


def _sweep_describe(doc):
    """What a selector-miss should list: the candidates that DO exist."""
    entries = _sweep_entries(doc)
    ranks = [e.get("rank") for e in entries if e.get("rank") is not None]
    cids = [e.get("candidate_id") for e in entries if e.get("candidate_id")]
    bits = []
    if ranks:
        bits.append(f"rank= one of {ranks}")
    if cids:
        bits.append(f"candidate_id= one of {cids}")
    bits.append("'best' (the numeric winner)")
    return "; ".join(bits)


def read_sweep_report(doc, selector, path):
    """A sweep/apply report -> one record.

    A candidate's `joints` IS the full absolute state (`_rec_out` writes every declared
    joint, swept or held), so it is read, not assembled. `meta.held_joints` stays in the
    report for the human "Swept: … Held: …" line; nothing reconstructs a state from it.
    """
    frame = doc.get("frame")

    def record_of(entry, sel, rank=None):
        pose = entry.get("pose")
        if pose is None:
            # a joint-only sweep emits no `pose` block; the pose start is the
            # pose every candidate shared, so that IS this candidate's pose.
            pose = doc.get("pose_start")
            if pose is None:
                raise ReseedError(
                    f"{path}#{sel}: candidate has no 'pose' and the report has no "
                    "'pose_start' to fall back on, so its placement is unknown")
        return {
            "frame": frame,
            "pose": _pose_block(pose, f"{path}#{sel}"),
            "joints": _joints_block(entry.get("joints"), f"{path}#{sel} joints"),
            "provenance": {
                "source": path, "selector": sel, "dialect": SWEEP_REPORT,
                "order_id": doc.get("order_id"),
                "rank": rank,
                "score": entry.get("combined"),
            },
        }

    if selector in (None, "best"):
        best = doc.get("best")
        if best is None:
            raise ReseedError(
                f"{path}: 'best' is null — no candidate finished scoring (see the "
                "report's 'status'). There is nothing to reseed from; re-run the "
                "sweep with a larger timeout or a smaller grid.")
        return record_of(best, "best", rank=0)

    key, _, value = selector.partition("=")
    key, value = key.strip(), value.strip()
    entries = _sweep_entries(doc)
    if key == "rank":
        try:
            want = int(value)
        except ValueError:
            raise ReseedError(f"{path}#{selector}: rank must be an integer")
        for e in entries:
            if int(e.get("rank", -1)) == want:
                return record_of(e, selector, rank=want)
        raise ReseedError(
            f"{path}#{selector}: no ranked entry with rank {want}. Available: "
            f"{_sweep_describe(doc)}. NOTE the report's `topk` bounds `ranked`, so "
            "a candidate outside the top-k is not addressable — order a larger "
            "`topk` on the sweep whose candidates you may want to descend from.")
    if key == "candidate_id":
        for e in entries:
            if str(e.get("candidate_id")) == value:
                return record_of(e, selector, rank=e.get("rank"))
        raise ReseedError(
            f"{path}#{selector}: no candidate with candidate_id {value!r}. "
            f"Available: {_sweep_describe(doc)}")
    raise ReseedError(
        f"{path}#{selector}: unknown selector for a sweep report. Use "
        f"'#rank=N', '#candidate_id=X', or '#best'. Available: "
        f"{_sweep_describe(doc)}")


def read_object_report(doc, selector, path):
    """An osweep/oapply/oapply_all report -> one record, for ONE frame.

    Refused without a frame selector: these verbs score a canonical rotation
    across N frames, so the winner is PER FRAME and there is no single pose the
    report names. Reseeding from one without saying which frame would silently
    pick whichever frame happened to be first."""
    frames = doc.get("frames") or []
    names = [str(f.get("frame")) for f in frames if isinstance(f, dict)]
    if selector is None:
        raise ReseedError(
            f"{path}: this is a {doc.get('view', 'canonical-rotation')} report — its "
            "winner is PER FRAME, not one pose. Name the frame: "
            f"'{os.path.basename(path)}#<frame>'. Frames: {names}")
    for fb in frames:
        if not isinstance(fb, dict):
            continue
        if str(fb.get("frame")) == selector:
            pose = fb.get("pose")
            if pose is None:
                raise ReseedError(f"{path}#{selector}: frame block has no 'pose'")
            return {
                "frame": selector,
                "pose": _pose_block(pose, f"{path}#{selector}"),
                # the frame's committed state, which the writer records because
                # these verbs render it (they search rotation, not articulation).
                "joints": _joints_block(fb.get("joints"),
                                        f"{path}#{selector} joints"),
                "provenance": {"source": path, "selector": selector,
                               "dialect": OBJECT_REPORT,
                               "order_id": doc.get("order_id"),
                               "rank": None,
                               "score": (fb.get("combined")
                                         if fb.get("combined") is not None
                                         else fb.get("iou_visible"))},
            }
    raise ReseedError(f"{path}#{selector}: no such frame in the report. "
                      f"Frames: {names}")


def read_pose_json(doc, selector, path):
    """mesh/pose.json -> one frame's committed record (`object_pose` + `joints`)."""
    frames = doc.get("frames") or {}
    name = _resolve_frame_selector(frames, selector, path)
    entry = frames[name]
    pose = entry.get("object_pose")
    if pose is None:
        raise ReseedError(f"{path}#{name}: entry has no 'object_pose'")
    return {
        "frame": name,
        "pose": _pose_block(pose, f"{path}#{name}"),
        "joints": _joints_block(entry.get("joints"), f"{path}#{name} joints"),
        "provenance": {"source": path, "selector": selector, "dialect": POSE_JSON,
                       "order_id": None, "rank": None, "score": None},
    }


def read_frame_map(doc, selector, path):
    """A multiagent.windows fragment / progress.json / poses.json -> one frame's record.

    The refiner's own saved decision, so `provenance` picks up whatever chain the
    writer left behind (a reseed written by this module records where it came
    from, which is what makes a multi-hop descent auditable)."""
    frames = doc.get("frames") or {}
    name = _resolve_frame_selector(frames, selector, path)
    entry = frames[name]
    if not isinstance(entry, dict):
        raise ReseedError(f"{path}#{name}: entry is not an object")
    pose = entry.get("pose")
    if pose is None:
        raise ReseedError(
            f"{path}#{name}: entry has no 'pose'. A frame recorded with notes but "
            "no pose is an in-progress decision, not a seed.")
    prov = {"source": path, "selector": selector, "dialect": FRAME_MAP,
            "order_id": None, "rank": None, "score": None}
    if isinstance(entry.get("reseed"), dict):
        # a chain: keep the previous hop's origin so provenance is a trail
        prov["parent"] = entry["reseed"]
    return {
        "frame": name,
        "pose": _pose_block(pose, f"{path}#{name}"),
        "joints": _joints_block(entry.get("joints"), f"{path}#{name} joints"),
        "provenance": prov,
    }


def read_order(doc, selector, path):
    """A pool order body -> the pose+joints it carries.

    An order spells placement two ways (`pose` at the top level, or `sweep.pose_start`),
    so both resolve; `joints` is always top-level. This reader exists so a hop can
    descend from an order that was ALREADY written (re-running a hop with one DOF
    widened) without re-deriving where that order's numbers came from."""
    sweep = doc.get("sweep") if isinstance(doc.get("sweep"), dict) else {}
    pose = doc.get("pose") or sweep.get("pose_start")
    if isinstance(pose, str):
        raise ReseedError(
            f"{path}: this order's sweep.pose_start is a PATH ({pose!r}), not an inline "
            "placement — reseed from that file instead.")
    if pose is None:
        raise ReseedError(
            f"{path}: order carries neither a top-level 'pose' nor a "
            "'sweep.pose_start' placement, so there is no pose to reseed from.")
    return {
        "frame": doc.get("frame"),
        "pose": _pose_block(pose, path),
        "joints": _joints_block(doc.get("joints"), f"{path} joints"),
        "provenance": {"source": path, "selector": selector, "dialect": ORDER,
                       "order_id": doc.get("id"), "rank": None, "score": None},
    }


def _resolve_frame_selector(frames, selector, path):
    """Which frame key a selector names, with the single-record shorthand.

    A file holding exactly ONE frame needs no selector — the common case for a
    per-hop scratch record. Anything else refuses and lists the choices rather
    than picking the first, which would be a silent wrong-frame reseed."""
    if not isinstance(frames, dict) or not frames:
        raise ReseedError(f"{path}: no 'frames' to select from")
    if selector is None:
        if len(frames) == 1:
            return next(iter(frames))
        raise ReseedError(
            f"{path}: holds {len(frames)} frames, so a selector is required. "
            f"Use '{os.path.basename(path)}#<frame>'. Frames: {sorted(frames)}")
    if selector in frames:
        return selector
    # tolerate a stem for a .jpg key (agents write both)
    matches = [k for k in frames if os.path.splitext(k)[0] == selector]
    if len(matches) == 1:
        return matches[0]
    raise ReseedError(f"{path}#{selector}: no such frame. "
                      f"Frames: {sorted(frames)}")


_READERS = {
    SWEEP_REPORT: read_sweep_report,
    OBJECT_REPORT: read_object_report,
    POSE_JSON: read_pose_json,
    FRAME_MAP: read_frame_map,
    ORDER: read_order,
}


def read(address, frame=None, cross_frame=False):
    """`<path>#<selector>` -> the normalized record. The one entry point.

    `frame`, when given, is CHECKED against the record rather than used to
    resolve it: a pose from another frame is meaningless as a seed and is a
    plausible copy-paste slip, so it refuses instead of proceeding.

    `cross_frame=True` is the caller SAYING SO: the record's pose is carried to
    another frame verbatim. See `resolve_order_seed` for why that is a flag rather
    than the default (and why the refusal it waives is still the right default).
    A cross-frame record keeps its own `frame` in `provenance.source_frame`, so the
    ledger records which frame a hop actually descended from."""
    path, selector = split_address(address)
    doc = _read_json(path)
    dialect = detect_dialect(doc, path)
    rec = _READERS[dialect](doc, selector, path)
    if frame is not None and rec.get("frame") is not None:
        if str(rec["frame"]) != str(frame):
            if not cross_frame:
                raise ReseedError(
                    f"{address}: this record is for frame {rec['frame']!r} but the "
                    f"reseed targets {frame!r}. A pose from another frame is not a "
                    "seed — check the address. If you MEANT to carry a neighbour's "
                    'pose across, say so: add "seed_cross_frame": true to the order '
                    "(the pose lands verbatim, so the camera moved between the two "
                    "frames and the sweep must absorb that motion — give it enough "
                    "range).")
            rec["provenance"]["source_frame"] = str(rec["frame"])
            rec["provenance"]["cross_frame"] = True
            rec["frame"] = str(frame)
    return rec


# --------------------------------------------------------------------------- #
# writers — one per dialect
# --------------------------------------------------------------------------- #
_rounded_pose = state_json.round_pose      # conventions/state_json.md §5
_rounded_joints = state_json.round_joints


def as_frame_entry(rec, note=None):
    """The record as a fragment / progress.json / poses.json frame entry.

    `reseed` records the provenance INSIDE the entry, so the descent chain is
    readable from the file the agent already inspects — and `read_frame_map`
    picks it up as `parent`, making a chain of hops a chain in the data too."""
    entry = {"pose": _rounded_pose(rec["pose"])}
    if rec.get("joints"):
        entry["joints"] = _rounded_joints(rec["joints"])
    entry["reseed"] = dict(rec.get("provenance") or {})
    if note:
        entry["notes"] = note
    return entry


def as_order(rec, order_id, frame=None, views="sweep", sweep=None):
    """The record as a pool ORDER body that DESCENDS from it.

    Pose and joints travel together; the placement's KEY depends on `views` —
    `sweep.pose_start` for a pure `sweep`/`apply` order, top-level `pose` otherwise (see
    `_placement_key_for`), so `--views osweep` cannot emit a pose_start the run discards
    while the joints apply alone. `sweep` merges the caller's ranges/refine/topk
    either way.

    The emitted keys are a subset of serve.py's accepted top-level keys — the worker
    rejects an unknown one — which the test asserts."""
    body = {"id": str(order_id), "views": views}
    if frame or rec.get("frame"):
        body["frame"] = str(frame or rec["frame"])
    sweep_obj = dict(sweep or {})
    # a dict, not a JSON string: `_apply_sweep_overrides` json.dumps a non-str
    # `pose_start` itself, and the object form is what a human editing the order can
    # actually read and tweak. Same for the top-level `pose`, which serve.py reads
    # as a dict.
    if _placement_key_for(body) == "pose_start":
        sweep_obj["pose_start"] = _rounded_pose(rec["pose"])
    else:
        body["pose"] = _rounded_pose(rec["pose"])
    # `sweep` is emitted even when the placement went to `pose`: the caller's
    # ranges/topk still belong to the order, and an empty object is harmless.
    body["sweep"] = sweep_obj
    if rec.get("joints"):
        body["joints"] = _rounded_joints(rec["joints"])
    return body


def as_placement(rec):
    """The record's bare placement — what `--sweep-pose-start` takes on its own.

    Emitted for completeness (a human pasting into a shell), and deliberately
    NOT what `--as order` uses: a placement alone is the revert waiting to
    happen, since it carries no joints."""
    return _rounded_pose(rec["pose"])


def write_to(address, rec, note=None):
    """Merge the record into a frame-keyed JSON file at `<path>#<frame>`.

    Read-modify-write rather than replace: progress.json is the refiner's running
    record of every frame it has decided, so a reseed must add one frame's entry
    without touching the others. The file is created if absent (the first hop of a
    window writes it), and written temp-then-replace — the same atomicity
    pool/manager.py uses for results, so a concurrent reader never sees a
    half-written file."""
    path, selector = split_address(address)
    frame = selector or rec.get("frame")
    if not frame:
        raise ReseedError(
            f"{address}: no frame to write under. Give one as a selector "
            f"('{os.path.basename(path)}#<frame>') — the source record does not "
            "name a frame.")
    doc = {}
    if os.path.isfile(path):
        doc = _read_json(path)
        if not isinstance(doc, dict):
            raise ReseedError(f"{path} is not a JSON object")
    frames = doc.setdefault("frames", {})
    if not isinstance(frames, dict):
        raise ReseedError(f"{path}: 'frames' is not an object")
    existing = frames.get(str(frame)) if isinstance(frames.get(str(frame)),
                                                    dict) else {}
    entry = as_frame_entry(rec, note=note)
    # keep the refiner's own annotations (confidence/notes) unless we are
    # replacing them: a reseed is a pose update, not a verdict.
    merged = {k: v for k, v in existing.items()
              if k not in ("pose", "joints", "reseed")}
    merged.update(entry)
    if note is None and existing.get("notes"):
        merged["notes"] = existing["notes"]
    frames[str(frame)] = merged
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# the `seed` order key — a reseed spent at claim time
# --------------------------------------------------------------------------- #
def _placement_key_for(req):
    """WHICH key a seed's placement lands in: `sweep.pose_start` or top-level `pose`.

    Not a gate — every view can be seeded. Two channels, different reach:

      * `sweep.pose_start` moves WHERE THE SEARCH STARTS only — read by
        `views/sweeps/engine` alone, for `sweep`/`apply` (see SWEEP_POSE_START_VIEWS).
      * top-level `pose` moves the FRAME, applied by serve.py to the resolved
        FrameSpec before any view runs, so it reaches every view.

    A pure `sweep`/`apply` order renders the same pixels either way, and `pose_start`
    is the narrower key. Everything else takes `pose`, since `pose_start` is DISCARDED
    there while the joints still apply. MIXED views take `pose` too: `_resolve_pose_start`
    falls back to `ctx.placement`, so the sweep still centres on the seed while
    `match` renders it as well."""
    from core.sweep_family import SWEEP_POSE_START_VIEWS
    raw = req.get("views") or "match"
    names = [v.strip() for v in str(raw).split(",") if v.strip()]
    if names and all(v in SWEEP_POSE_START_VIEWS for v in names):
        return "pose_start"
    return "pose"


def resolve_order_seed(req):
    """Expand an order's `"seed": "<path>#<selector>"` into a placement + `joints`.

    THE HOP THIS EXISTS FOR — one address instead of ~10 hand-copied floats:

        {"id": "w04-000250-b2", "views": "sweep", "frame": "000250.jpg",
         "seed": "spool/renders/w04-000250-b1/sweep.json#rank=14",
         "sweep": {"ranges": "yaw:-6,6,7"}}

    Works on ANY view; `_placement_key_for` decides which key the placement takes.
    Same override contract as the manager's other claim-time fills: a key the order
    already carries has decided, so an explicit placement or `joints` beside a `seed`
    WINS — that is how an agent nudges one half without restating the other.

    UNLIKE those fills, a failure is NOT silent: a seed that failed to resolve would
    render the frame's committed pose — a plausible picture of the wrong configuration.
    A bad address raises, the order comes back ok:false.

    `"seed_cross_frame": true` AUTHORIZES a seed whose record belongs to a DIFFERENT
    frame, carrying the pose verbatim. Frame-scoping is still the default because the
    failure it catches is real, but "start the next frame from the last one I fitted" is a
    legitimate and common move — a 10-frame gap in a hand-held capture is a small
    screen-space step, and the neighbour's pose is a far better grid centre than the
    frame's own committed one. Verbatim means the inter-frame CAMERA motion is not
    compensated: the sweep has to absorb it, so give it range. The waiver is per
    order and is recorded (`provenance.source_frame` + a `cross_frame` ledger field),
    because "which frame did this pose come from" is the first question a bad seam
    raises.

    NO SCALE IS EVER FILLED. A record's pose carries `scale` only when its source
    dialect had one (a report does; a fragment/progress entry deliberately does not —
    conventions/state_json.md §4), and this function will not invent one: the
    engine's `_resolve_pose_start` stamps the frame's shared SCALE onto it. Filling a
    default here is exactly the bug that scored a SCALE-0.6 object at 1.0.

    Mutates `req` and returns {"pose_start"|"pose": ..., "joints": ...} naming what was
    filled. `seed` becomes `seed_from` so the claimed order records its origin."""
    address = req.get("seed")
    if not address:
        return {}
    # KEPT on the order, not consumed: the claimed order stays self-describing
    # ("this hop crossed frames on purpose"), and re-resolving is still idempotent
    # because the early return above fires once `seed` is gone.
    cross = req.get("seed_cross_frame", False)
    if isinstance(cross, str):
        cross = cross.strip().lower() in ("1", "true", "yes", "on")
    rec = read(address, frame=req.get("frame"), cross_frame=bool(cross))
    filled = {}
    # `seed` is meaningless without a search to seed, so an order with no `sweep`
    # object gets one.
    sweep = req.get("sweep")
    if not isinstance(sweep, dict):
        sweep = {}
        req["sweep"] = sweep
    # EITHER placement key already on the order counts as "already decided", whichever
    # one this order's views would have taken: they are two spellings of one decision,
    # so filling the other beside it would be two placements for one render.
    key = _placement_key_for(req)
    if "pose_start" not in sweep and "pose" not in req:
        if key == "pose_start":
            sweep["pose_start"] = as_placement(rec)
        else:
            req["pose"] = as_placement(rec)
        filled[key] = as_placement(rec)
    if "joints" not in req and rec.get("joints"):
        req["joints"] = _rounded_joints(rec["joints"])
        filled["joints"] = req["joints"]
    del req["seed"]
    req["seed_from"] = str(address)
    return filled


def order_seed_source_frame(req):
    """The frame a claimed order's seed came FROM, when it crossed frames — else None.

    Read back off the resolved order rather than returned from `resolve_order_seed`,
    whose return value means "which order keys were filled" and is logged as such.
    The address in `seed_from` names the source frame anyway (`…#000340.jpg`); this
    is the answer for the report-shaped dialects, whose selector is a rank."""
    if not req.get("seed_cross_frame"):
        return None
    address = str(req.get("seed_from") or "")
    if not address:
        return None
    try:
        path, selector = split_address(address)
        doc = _read_json(path)
        rec = _READERS[detect_dialect(doc, path)](doc, selector, path)
    except ReseedError:
        return None
    return rec.get("frame")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_sweep_settings(args):
    sweep = {}
    if args.sweep_ranges:
        sweep["ranges"] = args.sweep_ranges
    if args.sweep_space:
        sweep["space"] = args.sweep_space
    if args.refine is not None:
        sweep["refine"] = int(args.refine)
    if args.topk is not None:
        sweep["topk"] = int(args.topk)
    if args.dump_topk is not None:
        sweep["dump_topk"] = int(args.dump_topk)
    return sweep


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="pool.reseed",
        description="Save a pose+joints pick, or spend it as the next hop's seed.")
    p.add_argument("--from", dest="src", required=True,
                   metavar="PATH[#SELECTOR]",
                   help="source record: a sweep/apply report (#rank=N, "
                        "#candidate_id=X, #best), an osweep/oapply report "
                        "(#<frame>), mesh/pose.json (#<frame>), a fragment / "
                        "progress.json (#<frame>), or a pool order")
    p.add_argument("--to", dest="dst", metavar="PATH[#FRAME]",
                   help="merge the record into this frame-keyed JSON file")
    p.add_argument("--as", dest="emit", default=None,
                   choices=("order", "entry", "placement", "record"),
                   help="print the record in this shape instead of writing it")
    p.add_argument("--frame", help="assert the record is for this frame")
    p.add_argument("--id", dest="order_id", help="order id for --as order")
    p.add_argument("--views", default="sweep", help="order views (--as order)")
    p.add_argument("--sweep-ranges", help="order's sweep.ranges (--as order)")
    p.add_argument("--sweep-space", help="order's sweep.space (--as order)")
    p.add_argument("--refine", type=int, help="order's sweep.refine")
    p.add_argument("--topk", type=int, help="order's sweep.topk")
    p.add_argument("--dump-topk", type=int, help="order's sweep.dump_topk")
    p.add_argument("--note", help="notes to record with --to")
    a = p.parse_args(argv)

    if not a.dst and not a.emit:
        p.error("give --to (save it) or --as (spend it); doing neither is a no-op")

    try:
        rec = read(a.src, frame=a.frame)
        if a.dst:
            path = write_to(a.dst, rec, note=a.note)
            prov = rec["provenance"]
            print(f"reseed: {prov['source']}"
                  f"{'#' + prov['selector'] if prov.get('selector') else ''} "
                  f"-> {path} [{rec.get('frame')}] "
                  f"pose + {len(rec['joints'])} joint state(s)")
        if a.emit == "order":
            if not a.order_id:
                p.error("--as order needs --id (order ids are immutable within a "
                        "spool, so it cannot be generated for you)")
            print(json.dumps(as_order(rec, a.order_id, frame=a.frame,
                                      views=a.views,
                                      sweep=_parse_sweep_settings(a)),
                             indent=2))
        elif a.emit == "entry":
            print(json.dumps(as_frame_entry(rec, note=a.note), indent=2))
        elif a.emit == "placement":
            print(json.dumps(as_placement(rec), indent=2))
        elif a.emit == "record":
            print(json.dumps(rec, indent=2))
    except ReseedError as exc:
        print(f"reseed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
