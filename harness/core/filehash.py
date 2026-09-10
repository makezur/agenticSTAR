"""filehash.py — the one whole-file digest used to fingerprint scene.py.

A whole-file sha1: ANY edit (FRAMES, comments, whitespace) changes it. That is
deliberate — it is the pool-recycle trigger and the frozen-scene proof for
multiagent.windows briefs and fragments, where "the file changed at all" is
exactly the question. (The finer FRAMES-excluding fingerprint the merge/apply
gates compare is multiagent.scene_source.ex_frames_sha1.)

Lives in bpy-free core/ so Blender's python (render_wrapper, pool/serve.py) and
the analysis env (multiagent.windows, pool/manager.py) share one
implementation.
"""

import hashlib


def bytes_sha1(data):
    """sha1 hex digest of bytes already in hand.

    For callers that must hash the SAME read they otherwise consume — the pass
    snapshot writes scene.py's bytes and fingerprints them together, so that a
    later `sha1(snapshot) == manifest.scene_sha1` check compares two independent
    reads and can actually fail. Re-opening the path there would make the two
    values agree by construction and the check vacuous.
    """
    return hashlib.sha1(data).hexdigest()


def file_sha1(path):
    """sha1 hex digest of the file's bytes. Raises OSError when unreadable —
    for callers where a missing scene.py is a hard error (plan validation,
    manifest writing)."""
    with open(path, "rb") as f:
        return bytes_sha1(f.read())


def file_sha1_or_none(path):
    """file_sha1, or None when the file is unreadable — for pollers (the pool
    recycle watch) where absence just means 'no change yet'."""
    try:
        return file_sha1(path)
    except OSError:
        return None
