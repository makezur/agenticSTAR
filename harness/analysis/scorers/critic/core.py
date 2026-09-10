"""core — every operation behind the critic CLI (wrapper.py drives these).

The two prompts (loaded from the sibling .txt files), the swappable VLM backends,
response parsing + schema enforcement, numeric grounding, and the image/turntable
helpers. No argparse and no orchestration live here — see wrapper.py. See
critic.md for the full rubric.
"""

import base64
import glob
import json
import os

import cv2
import numpy as np

from analysis.lib import panels
from analysis.lib import rasters
from analysis.scorers.silhouette import tight_bbox
from core import renderer_settings, turntable_views
# How a transparent render is presented on a backdrop (the flag, the backdrop
# colour, and the alpha-over-solid blend) lives in core.renderer_settings — the
# ONE place shared with rig/render.py and composite.py.

# ------------------------------------------------------------------------- #
# What the VLM is asked to do — the rubric + output contract (see critic.md).
# The two prompts live in sibling .txt files (system_prompt.txt appended with
# grounding_prompt.txt only when numeric context is supplied) so the judgement is
# provider-independent AND the text is plain, editable prose: swap the backend,
# keep the prompt; edit the prompt without touching code.
# ------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_prompt(name):
    """Read a prompt .txt file that ships beside this module (verbatim bytes)."""
    with open(os.path.join(_HERE, name), encoding="utf-8") as f:
        return f.read()


# The rubric + JSON contract sent to any backend.
SYSTEM_PROMPT = _load_prompt("system_prompt.txt")
# Appended to SYSTEM_PROMPT only when the caller passes numeric grounding (--metrics
# / --depth / --sweep). Keeps the critic an independent visual judge while letting its
# prose RECONCILE vision with the numeric signals the agent computed.
GROUNDING_PROMPT = _load_prompt("grounding_prompt.txt")


# ------------------------------------------------------------------------- #
# Image helpers
# ------------------------------------------------------------------------- #
def _media_type(path):
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/png")


def _load_image(path):
    """(media_type, base64_str) for a file on disk, or None if unreadable."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        print(f"[critic] WARNING: could not read image {path}: {e}")
        return None
    return _media_type(path), base64.standard_b64encode(data).decode("ascii")


def _encode_bgr(arr):
    """PNG media tuple from a cv2/numpy BGR or BGRA array, or None."""
    try:
        ok, buf = cv2.imencode(".png", arr)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[critic] WARNING: could not PNG-encode a BGR array: {e}")
        return None
    if not ok:
        print("[critic] WARNING: cv2.imencode returned failure for a BGR array")
        return None
    return "image/png", base64.standard_b64encode(buf.tobytes()).decode("ascii")


def _render_arr(path, bg_mode):
    """Present a render (match or one turntable frame) in the chosen backdrop mode.

    Returns (croppable_array_or_None, (media_type, base64)). The array is a full-size
    BGR (black mode) or BGRA (alpha mode) image so --crops can reuse the exact pixels;
    it is None only when we fell back to the raw file. In 'black' mode the transparent
    film is alpha-composited onto renderer_settings.BACKDROP_BGR (an already-opaque
    image is a no-op, alpha all 255); in 'alpha' mode the RGBA is kept, with fully
    transparent pixels (alpha == 0) also zeroed in BGR. On read failure we
    fall back to the raw file (current behaviour): (None, _load_image(path))."""
    try:
        bgr, alpha = rasters.load_rgba(path)
    except Exception as e:
        print(f"[critic] WARNING: could not decode {path}: {e}")
        return None, _load_image(path)
    # black -> composite onto the shared backdrop; alpha -> keep the film as-is.
    out = renderer_settings.present(bgr, alpha, bg_mode)
    return out, _encode_bgr(out)


def _mask_source_arr(source, mask, hand, bg_mode):
    """The real photo with background (and optionally the hand/occluder) removed, on
    the SAME backdrop as the renders. Returns (croppable_array, (mt, b64)) or None.

    'black' -> object pixels kept, else the shared backdrop (a BGR array).
    'alpha' -> object pixels kept on a truly transparent backdrop (a BGRA array;
    masked-out pixels are zeroed in both alpha AND BGR). Either way the returned
    array is directly croppable by --crops."""
    try:
        bgr = rasters.load_bgr(source)
        m = rasters.load_mask(mask)
        if m.shape[:2] != bgr.shape[:2]:
            m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        obj = m > 0
        if hand:
            keep = rasters.build_keep(rasters.load_mask(hand), bgr.shape)
            obj = obj & (keep > 0)
    except Exception as e:
        print(f"[critic] WARNING: could not build masked source from "
              f"{source}/{mask}: {e}")
        return None
    # Present on the SAME backdrop as the renders: reuse renderer_settings.present
    # with the object silhouette as the alpha (black -> composite onto the shared
    # backdrop, alpha -> keep object pixels on a transparent backdrop).
    out = renderer_settings.present(bgr, obj.astype(np.uint8) * 255, bg_mode)
    return out, _encode_bgr(out)


def _detail_crops(match_arr, msrc_arr, mask, render, pad):
    """Zoomed crops of the MATCH render + MASKED SOURCE around the object bbox.

    Returns a list of (label, (media_type, base64)) — up to two, cropped to the union
    of the source-mask bbox and the render-silhouette bbox (they're 1:1 aligned),
    padded and clamped. Any failure / empty bbox -> []. Reuses the arrays already
    built for the full images (same channel count), so pixels + box are identical."""
    try:
        boxes = [bb for bb in (tight_bbox(rasters.load_mask(mask)),
                               tight_bbox(rasters.render_silhouette(render)))
                 if bb is not None]
        if not boxes:
            return []
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        ref = match_arr if match_arr is not None else msrc_arr
        if ref is None:
            return []
        h, w = ref.shape[:2]
        p = int(round(pad * max(h, w)))
        x0, y0 = max(0, x0 - p), max(0, y0 - p)
        x1, y1 = min(w, x1 + p), min(h, y1 + p)
        if x1 <= x0 or y1 <= y0:
            return []
    except Exception as e:
        print(f"[critic] WARNING: could not compute detail-crop bbox: {e}")
        return []

    crops = []
    _CROP = ("ZOOMED DETAIL — {} (crop; fine SHAPE/geometry ONLY — do NOT use for "
             "global position/framing/pose)")
    for arr, what in ((match_arr, "MATCH render"), (msrc_arr, "MASKED SOURCE")):
        if arr is None:
            continue
        try:
            enc = _encode_bgr(arr[y0:y1, x0:x1])
            if enc:
                crops.append((_CROP.format(what), enc))
        except Exception as e:
            print(f"[critic] WARNING: could not crop {what}: {e}")
    return crops


def _side_by_side(msrc_arr, match_arr):
    """MASKED SOURCE beside the MATCH render as ONE height-matched comparison
    panel -> (media_type, base64), or None if either array is missing.

    Reuses the arrays already built for the full images, preserving BGRA in
    'alpha' mode, scaled to a common height and labeled like composite.py."""
    if msrc_arr is None or match_arr is None:
        return None

    try:
        h = max(msrc_arr.shape[0], match_arr.shape[0])
        strip = [
            panels.Panel(panels.scale_to_height(msrc_arr, h), "MASKED SOURCE"),
            panels.Panel(panels.scale_to_height(match_arr, h), "MATCH render"),
        ]
        return _encode_bgr(panels.hstack_panels(panels.label_row(strip)))
    except Exception as e:
        print(f"[critic] WARNING: could not build side-by-side panel: {e}")
        return None


# ------------------------------------------------------------------------- #
# Swappable VLM backends — each exposes .complete(system, images, user_text)->str.
# `images` is a list of (media_type, base64_str). Add a provider by writing one
# more class and one more branch in make_backend(); the prompt/schema are shared.
# See critic.md § "Swap the VLM". Both backends are API-key only.
# ------------------------------------------------------------------------- #
class BackendUnavailable(Exception):
    """No usable SDK / credentials — the run is SKIPPED, not failed."""


class AnthropicBackend:
    """Anthropic Messages API (api.anthropic.com), authenticated by API key."""

    def __init__(self, model, key):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable(
                "the 'anthropic' package is not installed "
                "(micromamba run -n artscript pip install anthropic)"
            ) from e
        from anthropic import Anthropic

        self.model = model
        self._client = Anthropic(api_key=key)

    def complete(self, system, images, user_text, max_tokens=2000):
        content = [{"type": "image",
                    "source": {"type": "base64", "media_type": mt, "data": b64}}
                   for (mt, b64) in images]
        content.append({"type": "text", "text": user_text})
        msg = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


class OpenAIBackend:
    """OpenAI Responses API (api.openai.com), authenticated by API key."""

    def __init__(self, model, key):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable(
                "the 'openai' package is not installed "
                "(micromamba run -n artscript pip install openai)"
            ) from e
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=key)

    def complete(self, system, images, user_text, max_tokens=2000):
        content = [{"type": "input_image", "image_url": f"data:{mt};base64,{b64}"}
                   for (mt, b64) in images]
        content.append({"type": "input_text", "text": user_text})
        resp = self._client.responses.create(
            model=self.model, max_output_tokens=max_tokens, instructions=system,
            input=[{"role": "user", "content": content}],
        )
        return resp.output_text or ""


_DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-5.6-sol",
}
_PROVIDERS = ("anthropic", "openai")
_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def _codex_login_key():
    """The OpenAI key of the Codex API-key login ($CODEX_HOME/auth.json), or "".

    Inside a Codex sandbox the key is only in that file, never in the environment,
    so the critic reads it there when OPENAI_API_KEY is unset."""
    home = os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    try:
        with open(os.path.join(home, "auth.json")) as f:
            return str(json.load(f).get("OPENAI_API_KEY") or "")
    except (OSError, ValueError):
        return ""


def resolve_api_key(provider, api_key_env=""):
    """(env_var_name, key) for a provider; key is "" when nothing usable is set."""
    env = api_key_env or _KEY_ENV[provider]
    key = os.environ.get(env, "")
    if not key and provider == "openai" and not api_key_env:
        key = _codex_login_key()
    return env, key


def resolve_provider(provider, api_key_env=""):
    """'auto' -> the first provider with a usable key (anthropic, then openai)."""
    if provider != "auto":
        return provider
    for cand in _PROVIDERS:
        if resolve_api_key(cand, api_key_env)[1]:
            return cand
    raise BackendUnavailable(
        "no VLM API key found (set ANTHROPIC_API_KEY or OPENAI_API_KEY, or pass "
        "--provider explicitly)")


def make_backend(provider, model, api_key_env=""):
    """(backend, model_id, provider) for a --provider value ('auto' resolves to
    the first provider with a usable key)."""
    provider = resolve_provider(provider, api_key_env)
    if provider not in _PROVIDERS:
        raise BackendUnavailable(f"unknown --provider {provider!r} "
                                 "(expected auto | anthropic | openai)")
    model = model or _DEFAULT_MODELS[provider]
    env, key = resolve_api_key(provider, api_key_env)
    if not key:
        hint = (" or a Codex API-key login in $CODEX_HOME/auth.json"
                if provider == "openai" else "")
        raise BackendUnavailable(
            f"${env} is not set{hint} (needed for --provider {provider})")
    if provider == "anthropic":
        return AnthropicBackend(model, key), model, provider
    return OpenAIBackend(model, key), model, provider


# ------------------------------------------------------------------------- #
# Response parsing + schema enforcement (the model's raw text -> validated dict)
# ------------------------------------------------------------------------- #
def _extract_json(text):
    """Pull the JSON object out of a model reply (tolerates ```json fences /
    stray prose). Returns a dict, or raises ValueError."""
    t = text.strip()
    if t.startswith("```"):
        # drop the opening fence line and any trailing fence
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model reply")
    return json.loads(t[start:end + 1])


def _clamp01(v):
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return None


def _normalize(report):
    """Coerce a parsed reply into the fixed contract (clamp scores, canonicalise
    tags/severities, guarantee list shape)."""
    out = {
        "realism": _clamp01(report.get("realism")),
        "identity": _clamp01(report.get("identity")),
        "summary": str(report.get("summary", "")).strip(),
        "realism_notes": str(report.get("realism_notes", "")).strip(),
        "identity_notes": str(report.get("identity_notes", "")).strip(),
        "discrepancies": [],
    }
    raw = report.get("discrepancies") or []
    if isinstance(raw, dict):
        raw = [raw]
    for d in raw:
        if not isinstance(d, dict):
            continue
        tag = str(d.get("tag", "shape")).strip().lower()
        if tag == "joint":  # backward-compatible alias from older prompts/files
            tag = "joint_state"
        if tag not in ("shape", "pose", "joint_state", "joint_definition"):
            tag = "shape"  # default so it still routes into the decision
        sev = str(d.get("severity", "med")).strip().lower()
        if sev not in ("high", "med", "low"):
            sev = "med"
        item = {
            "tag": tag,
            "part": str(d.get("part", "unknown")).strip() or "unknown",
            "severity": sev,
            "observation": str(d.get("observation", "")).strip(),
            "adjustment": str(d.get("adjustment", "")).strip(),
        }
        ev = d.get("evidence_views")
        if isinstance(ev, str):
            ev = [ev]
        if isinstance(ev, (list, tuple)):
            ev = [str(v).strip() for v in ev if str(v).strip()]
            if ev:
                item["evidence_views"] = ev
        out["discrepancies"].append(item)
    return out


# ------------------------------------------------------------------------- #
# Numeric grounding — read the hard signals the agent already computed and format
# a compact block the VLM can reconcile its visual read against (see GROUNDING_PROMPT).
# ------------------------------------------------------------------------- #
def _load_json(path):
    """Parse a JSON file, or None if absent/unreadable (grounding is optional)."""
    if not path:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"[critic] NOTE: could not read grounding file {path}: {e} (skipping it)")
        return None


def _fmt_num(v, nd=3):
    try:
        return f"{float(v):.{nd}g}"
    except (TypeError, ValueError):
        return None


def build_grounding(metrics_path, depth_path, sweep_path):
    """(text_block, sources) from the optional grounding files.

    text_block is a compact human-readable NUMERIC CONTEXT block appended to the
    user message (empty string if no grounding was readable); sources is the list
    of which grounding kinds were actually used (recorded in the output JSON). The
    unified `sweep.json` carries BOTH the pose order and the joint states of the
    best candidate, so one file grounds both the pose and the articulation."""
    lines = []
    sources = []

    m = _load_json(metrics_path)
    if m is not None:
        gate = m.get("gate_field") or "iou_raw"
        parts = []
        for k in ("iou_raw", "iou_visible"):
            fv = _fmt_num(m.get(k))
            if fv is not None:
                parts.append(f"{k}={fv}")
        if m.get("pass") is not None:
            parts.append(f"pass={m.get('pass')} (gate={gate})")
        if parts:
            lines.append("- silhouette IoU: " + ", ".join(parts)
                         + ". (IoU is DEPTH-BLIND — a part floating toward the "
                           "camera still overlaps in 2D.)")
            sources.append("metrics")

    d = _load_json(depth_path)
    if d is not None:
        parts = []
        for k in ("depth_mae_canon", "depth_bias_canon"):
            fv = _fmt_num(d.get(k))
            if fv is not None:
                parts.append(f"{k}={fv}")
        if parts:
            lines.append("- observed depth: " + ", ".join(parts)
                         + ". (a large depth_mae_canon *suggests* the object may "
                           "be at the wrong distance => a POSE/translation-Z hint "
                           "(depth_bias_canon: + = too far, - = too near), not "
                           "build(); depth is noisy, so weigh it, don't gate.)")
            sources.append("depth")

    s = _load_json(sweep_path)
    if isinstance(s, dict) and s.get("best"):
        best = s["best"]
        bits = []
        # the sweep's camera-aligned pose ORDER (roll/yaw/pitch/dpx/dpy/tz) —
        # the which-way the search moved, the same vocabulary as a pose fix above.
        order = best.get("order") or {}
        nonzero = {k: v for k, v in order.items() if abs(float(v)) > 1e-9}
        if nonzero:
            bits.append("order=" + ", ".join(f"{k}={v:+.4g}"
                                              for k, v in nonzero.items()))
        # the unified sweep also carries the best JOINT states when joints were swept.
        joints = best.get("joints") or {}
        if joints:
            bits.append("joints=" + ", ".join(f"{k}={float(v):+.4g}"
                                               for k, v in joints.items()))
        iou = _fmt_num(best.get("iou_raw"))
        if iou is not None:
            bits.append(f"iou_raw={iou}")
        if bits:
            lines.append("- sweep best (a candidate the SEARCH found, may not be "
                         "pasted yet): " + ", ".join(bits))
            sources.append("sweep")

    if not lines:
        return "", sources
    block = ("\n\nNUMERIC CONTEXT the agent already computed for this frame "
             "(reconcile your visual read with it; do NOT just restate it):\n"
             + "\n".join(lines))
    return block, sources


def build_rig_context(pose_path, frame):
    """Compact declared-joint inventory for distinguishing state vs rig defects."""
    pose = _load_json(pose_path)
    if not isinstance(pose, dict):
        return ""
    defs = pose.get("joint_defs") or []
    frame_entry = (pose.get("frames") or {}).get(frame) or {}
    states = frame_entry.get("joints") or {}
    if not defs:
        inventory = "- declared joints: NONE"
    else:
        rows = []
        for joint in defs:
            if not isinstance(joint, dict):
                continue
            name = str(joint.get("name", "?"))
            rows.append(
                f"{name} (type={joint.get('type', 'fixed')}, "
                f"child={joint.get('child')}, axis={joint.get('axis')}, "
                f"limit={joint.get('limit')}, state={states.get(name, 0.0)})")
        inventory = "- declared joints: " + "; ".join(rows)
    return (
        "\n\nDECLARED RIG CONTEXT for this render:\n"
        + inventory
        + "\nUse `joint_state` only if one of these declared joints can produce "
          "the required relative motion. Use `joint_definition` when the visible "
          "motion requires a missing joint or a declared joint has the wrong "
          "type/axis/origin/children/limit."
    )


# ------------------------------------------------------------------------- #
def _resolve_turntable(patterns, cap):
    """Expand the repeatable/glob --turntable args into a de-duped, capped list.

    THE CAP IS SPENT ACROSS STATES, not down the sorted listing. A turntable renders
    a whole view set per ARTICULATION STATE, so the paths arrive grouped by state;
    the budget is dealt round-robin — one view from each state, then a second from
    each, until the cap runs out. Every state is represented, and the views each
    state keeps are its own first ones (the sheet's reading order — level ring
    before the raised/dropped pairs). A capped run drops ANGLES, never a whole
    configuration. Logs what it dropped — no silent truncation.
    """
    paths = []
    for pat in patterns or []:
        hits = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        for h in hits:
            if h not in paths and os.path.isfile(h):
                paths.append(h)
    if len(paths) <= cap:
        return paths

    # group by state, preserving the order each first appeared in
    by_state = {}
    for path in paths:
        parsed = turntable_views.parse_image_name(os.path.basename(path))
        # a path that isn't a turntable name (an explicitly-passed file) is its own
        # group, so it competes for the budget rather than being dropped.
        key = parsed[0] if parsed else os.path.basename(path)
        by_state.setdefault(key, []).append(path)

    kept, groups = [], list(by_state.values())
    for index in range(max(len(g) for g in groups)):
        for group in groups:
            if index < len(group) and len(kept) < cap:
                kept.append(group[index])
    # back into listing order, so the images the VLM is shown stay grouped by state
    kept_set = set(kept)
    kept = [p for p in paths if p in kept_set]
    print(f"[critic] NOTE: {len(paths)} turntable images found across "
          f"{len(groups)} state(s); sending {len(kept)} — up to "
          f"{-(-cap // len(groups))} view(s) per state, so every state is "
          f"represented (raise --max-turntable to include more)")
    return kept
