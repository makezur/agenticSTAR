#!/usr/bin/env python3
"""freshness.py — the pose hash that stales a temporal verdict mechanically.

A temporal verdict ("this 47 deg step is the lid being thrown open, and you can
see the hinge line clear the rim in 000120") is a statement about TWO SPECIFIC
POSES. Move either of them and the sentence usually still READS fine while no
longer being about anything that is in the trajectory — which is how a stale
verdict outlives the evidence for it. Nothing in the words says they went stale,
so a verdict carries a hash of what it was written about, and the hash is
recomputed from the poses at validation time. Mismatch = stale = uncalled: that
step needs looking at again, by whoever owns it.

What goes into the hash, and why each part has to:
  * both frames' quaternion and translation — the obvious half;
  * both frames' JOINT STATES — an articulation change moves the object on
    screen exactly as a base-pose change does, so a re-hinged frame whose base
    pose never moved must stale its verdict too;
  * the run's SHARED SCALE — a rescale changes NEITHER frame's pose numbers and
    still changes every rendered silhouette and every canonical residual, so
    either the scale is in the hash or a rescale silently keeps every verdict in
    the run (conventions/state_json.md 9).

What deliberately stays OUT: the measured residual. The hash answers "are these
still the poses I judged", not "did the numbers change". Deriving it from a
measurement would make it depend on the measuring code, so a fix to the residual
maths would stale every verdict in the run without a pose having moved.

Pure stdlib on purpose — no numpy, no rig/, no analysis/: the process side
(multiagent) and any later consumer must both be able to stamp and check a
verdict without pulling in the measurement stack.
"""

import hashlib
import json

# Quantize at the WIRE precision (core.state_json.POSE_DP). That is what makes a
# hash survive a JSON round trip: a pose written to a fragment, read back by the
# merge, and re-hashed must give the same digest. Full float precision would not
# — the same pose would hash differently either side of serialization.
QUANT_DP = 6

# 12 hex chars = 48 bits. A refiner copies this into its fragment, so legibility
# is worth more here than the collision margin of a full sha1: the threat model
# is a verdict that quietly went stale, not an adversary forging one.
HASH_LEN = 12

# Bumped only if WHAT is hashed changes (which re-stales every verdict in flight
# — that is the honest outcome, and the number is how a reader sees why).
PAYLOAD_VERSION = 1


def _component(value):
    """One scalar at the serialized precision.

    `+ 0.0` folds -0.0 onto 0.0: json spells them differently ("-0.0" vs "0.0"),
    so without this a pose that merely round-tripped through a negative zero
    would hash as a different pose."""
    return round(float(value), QUANT_DP) + 0.0


def _canonical_quat(q):
    """A quaternion at wire precision with its SIGN canonicalized (first non-zero
    component positive).

    q and -q are the same rotation — the double cover. A writer that happened to
    negate all four components (a sweep result re-signed on its way through a
    seed, say) has not moved the object one degree, and staling a verdict for
    that would train everyone to re-stamp verdicts without re-looking, which is
    the one habit this whole mechanism exists to prevent."""
    out = [_component(v) for v in q]
    lead = next((v for v in out if abs(v) > 0.0), 0.0)
    if lead < 0.0:
        out = [(-v) + 0.0 for v in out]
    return out


def _placement(entry, what):
    """One frame entry -> the {q, t, j} the hash is taken over.

    Accepts either authority spelling (`object_pose` on a committed record,
    `pose` on a proposal — conventions/state_json.md 3) so the same function
    hashes a merged trajectory and an in-progress fragment overlay."""
    entry = entry or {}
    pose = entry.get("object_pose") or entry.get("pose") or {}
    q, t = pose.get("quaternion"), pose.get("translation")
    if not q or not t:
        raise ValueError(
            f"{what} carries no quaternion+translation, so there is nothing to "
            "stamp a temporal verdict against; resolve the pose first (a "
            "defaulted identity would hash as a real pose)")
    return {"q": _canonical_quat(q),
            "t": [_component(v) for v in t],
            "j": {str(name): _component(v)
                  for name, v in (entry.get("joints") or {}).items()}}


def pose_hash(entry_a, entry_b, scale=1.0):
    """The freshness stamp for the STEP from `entry_a` to `entry_b`.

    `entry_a` / `entry_b` — pose.json-shaped frame entries (`object_pose` or
    `pose`, with `joints` as a sibling), in temporal order.
    `scale` — the run's shared scale (the pose.json / merged document's
    top-level `scale`), not a per-frame echo.

    Returns a short hex digest. Two calls agree exactly when both frames' poses,
    both frames' joint states, and the shared scale are unchanged at wire
    precision; any other change is a different hash, i.e. a stale verdict.
    """
    payload = {"v": PAYLOAD_VERSION,
               "a": _placement(entry_a, "the step's FROM frame"),
               "b": _placement(entry_b, "the step's TO frame"),
               "scale": _component(1.0 if scale is None else scale)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:HASH_LEN]
