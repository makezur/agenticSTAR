"""
critic.py — an independent VLM judge for REALISM + IDENTITY (the second opinion).

Runs in the 'artscript' micromamba env (NOT Blender's python), from harness/:
  micromamba run -n artscript python -m analysis.scorers.critic \
      --source <frame image> --render PASS_DIR/match_<frame>.png \
      --overlap PASS_DIR/overlap_<frame>.png \
      --turntable 'PASS_DIR/turntable_*_az*_el*.png' \
      --frame <frame> --out PASS_DIR/critic_<frame>.json
  # PASS_DIR = the views/<NNNN>/ dir render.sh printed (renders live in per-pass
  # buckets under views/ — there is no 'latest' symlink).

Where silhouette.py is the silhouette-IoU gate and depth.py is a strong-but-noisy
geometric guide, this tool is the *visual* second opinion — alongside IoU, one of
the two reliable reads of pose/identity: a VLM looks at the
real photo beside the render(s) and answers two questions the numbers can't —
  1) REALISM  — does the reconstruction read as a plausible real object at all?
  2) IDENTITY — is it recognizably the SAME object as the photo (matched view AND
                the turntable's novel orbit views)?
— then emits a ranked list of concrete, tagged (shape / pose / joint-state /
joint-definition) fixes that
route straight into scene.py the way AGENT_TASK.md's "shape vs pose vs joint"
decision expects.

It is an independent judge beside the IoU gate, the depth guide, and the turntable
coherence check, and it DEGRADES
GRACEFULLY: with no usable credentials it prints a clear message and exits 0
(skipped, like composite.py skips its depth panels without --tracking) so it never
breaks the iteration loop. See critic.md for the full rubric + the JSON contract
(what a VLM should say) + how to add a provider.
"""

import argparse
import json

from analysis.lib.io import write_json
from analysis.scorers.critic import core
from core import renderer_settings, turntable_views
# All operations (prompts, backends, parsing, grounding, image + turntable helpers)
# live in core; this file is only the CLI surface + orchestration. renderer_settings
# is still needed here for the --bg-mode choices + the shared backdrop phrasing.


def main():
    p = argparse.ArgumentParser(
        prog="critic",  # stable name regardless of `-m` entry point (else __main__.py)
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="the REAL reference photo")
    p.add_argument("--render", required=True,
                   help="the matched-view render (match_<frame>.png)")
    p.add_argument("--overlap", default="",
                   help="optional silhouette-overlap image (overlap_<frame>.png)")
    p.add_argument("--turntable", action="append", default=[],
                   help="turntable orbit render(s); repeatable, and each value may "
                        f"be a glob (e.g. '.../{turntable_views.IMAGE_GLOB}')")
    p.add_argument("--max-turntable", type=int, default=16,
                   help="cap on turntable images sent (default 16). Spent ACROSS "
                        "articulation states (round-robin), so a capped run drops "
                        "angles rather than hiding a whole configuration from the "
                        "judge.")
    # backdrop + extra image inputs (see critic.md § What it sends the VLM)
    p.add_argument("--bg-mode", default="black", choices=list(renderer_settings.BG_MODES),
                   help="ONE consistent backdrop for match/turntable/masked-source: "
                        "'alpha' keeps the renders' transparent film as-is; 'black' "
                        "(default) composites/fills all three onto uniform black so "
                        "dark objects stay visible and comparisons share a backdrop")
    p.add_argument("--mask", default="",
                   help="optional object mask PNG (silhouette of the object in the "
                        "photo). Enables the MASKED SOURCE image and (with --crops) "
                        "the detail crops; absent = skipped silently")
    p.add_argument("--hand-mask", default="",
                   help="optional hand/occluder mask PNG; when given with --mask the "
                        "hand region is also blanked from the masked source")
    p.add_argument("--crops", action="store_true",
                   help="also send ZOOMED DETAIL crops of the match render + masked "
                        "source around the object bbox (fine SHAPE only; requires "
                        "--mask). Off by default")
    p.add_argument("--crop-pad", type=float, default=0.12,
                   help="detail-crop padding as a fraction of the longer side "
                        "(default 0.12)")
    p.add_argument("--frame", default="", help="frame name, recorded in the report")
    p.add_argument("--out", required=True, help="output critic_<frame>.json path")
    # optional NUMERIC GROUNDING — the hard signals the agent already computed this
    # pass. All optional; a missing/unreadable file is skipped silently so the
    # critic still runs image-only (see critic.md § Grounding).
    p.add_argument("--metrics", default="",
                   help="optional metrics_<frame>.json (silhouette IoU) to ground "
                        "the critique in")
    p.add_argument("--depth", default="",
                   help="optional depth_<frame>.json (Pi3X depth verdict) to ground "
                        "the critique in")
    p.add_argument("--sweep", default="",
                   help="optional sweep.json (best pose order AND/OR joint states of "
                        "the search's winning candidate) for context")
    p.add_argument("--pose", default="",
                   help="optional pass pose.json; supplies declared joint "
                        "definitions and this frame's states so the critic can "
                        "distinguish joint_state from joint_definition")
    # backend selection (API keys only; see critic.md § Credentials)
    p.add_argument("--provider", default="auto",
                   choices=["auto", "anthropic", "openai"],
                   help="VLM backend: 'anthropic' (ANTHROPIC_API_KEY) or 'openai' "
                        "(OPENAI_API_KEY, else the Codex API-key login in "
                        "$CODEX_HOME/auth.json). 'auto' (default) takes the first "
                        "of the two with a usable key")
    p.add_argument("--model", default="",
                   help="model id override (default per provider: anthropic = "
                        "claude-opus-5, openai = gpt-5.6-sol)")
    p.add_argument("--api-key-env", default="",
                   help="env var holding the API key (default per provider)")
    p.add_argument("--max-tokens", type=int, default=6000)
    # soft gate thresholds
    p.add_argument("--pass-realism", type=float, default=0.7)
    p.add_argument("--pass-identity", type=float, default=0.7)
    args = p.parse_args()

    # ---- gather images ----
    # backdrop phrasing shared by every gray/black-aware label so match, turntable
    # and masked source describe the SAME backdrop the pixels actually show.
    bg = renderer_settings.backdrop_phrase(args.bg_mode)
    images = []
    labels = []
    src = core._load_image(args.source)
    if src is None:
        print("[critic] ERROR: --source image is required and must be readable")
        raise SystemExit(2)
    images.append(src); labels.append("SOURCE photo (the real reference, full frame)")

    match_arr, rnd = core._render_arr(args.render, args.bg_mode)
    if rnd is None:
        print("[critic] ERROR: --render image is required and must be readable")
        raise SystemExit(2)
    images.append(rnd)
    labels.append("MATCH render (reconstruction posed at the photo's viewpoint — "
                  f"compare 1:1 to the source) — shown on {bg} (not part of the "
                  "object; ignore it)")

    # MASKED SOURCE — the photo with background (and optionally the hand) blanked to
    # the SAME backdrop as the renders, for a clean side-by-side (additive; the full
    # SOURCE is still sent above). Absent/unreadable --mask skips it silently.
    msrc_arr = None
    if args.mask:
        msrc = core._mask_source_arr(args.source, args.mask, args.hand_mask, args.bg_mode)
        if msrc is not None and msrc[1] is not None:
            msrc_arr = msrc[0]
            images.append(msrc[1])
            labels.append("MASKED SOURCE (the real photo with background"
                          + (" and hand/occluder" if args.hand_mask else "")
                          + f" blanked to the SAME {bg} as the MATCH render, for a "
                          "clean 1:1 side-by-side identity/proportion/colour "
                          "comparison; the backdrop is not the object)")

    # SIDE-BY-SIDE — masked source beside the match render, height-matched on the
    # SAME backdrop, so the VLM reads identity/proportion/pose 1:1 without having
    # to align two separate full-frame images itself. Needs both arrays.
    sbs = core._side_by_side(msrc_arr, match_arr)
    if sbs is not None:
        images.append(sbs)
        labels.append("SIDE-BY-SIDE — MASKED SOURCE (left) beside the MATCH render "
                      f"(right), scaled to equal height on the SAME {bg}, for a "
                      "direct 1:1 identity / proportion / pose comparison")

    if args.overlap:
        ov = core._load_image(args.overlap)
        if ov is not None:
            images.append(ov)
            labels.append("SILHOUETTE overlap (green=agree, yellow=render only "
                          "(EXTRA volume), magenta=photo only (MISSING volume), "
                          "blue=our render behind the hand (present but NOT "
                          "scored — not an error), gray=hand/occluder don't-care "
                          "region — ignore it) — where the outlines disagree")

    # DETAIL CROPS (opt-in) — inserted BEFORE the turntable block so the turntable
    # range math below is untouched. Each crop carries its own per-image label.
    if args.crops:
        if not args.mask:
            print("[critic] NOTE: --crops needs --mask (for the object bbox); "
                  "skipping detail crops")
        else:
            for lab, enc in core._detail_crops(match_arr, msrc_arr, args.mask,
                                                args.render, args.crop_pad):
                images.append(enc)
                labels.append(lab)

    tt_paths = core._resolve_turntable(args.turntable, args.max_turntable)
    n_before_tt = len(images)
    for tp in tt_paths:
        _, img = core._render_arr(tp, args.bg_mode)
        if img is not None:
            images.append(img)
    n_tt = len(images) - n_before_tt
    if n_tt:
        labels.append(f"TURNTABLE orbit views ({n_tt} images) — ARBITRARY angles of "
                      "the reconstructed 3D object, NOT aligned to the photo. Judge "
                      "whether they depict the SAME object and look like a plausible "
                      "real object; do NOT expect them to pixel-match the photo.")

    # optional numeric grounding — the hard signals the agent already computed.
    grounding_block, grounded_on = core.build_grounding(
        args.metrics, args.depth, args.sweep)
    rig_context = core.build_rig_context(args.pose, args.frame)
    system_prompt = core.SYSTEM_PROMPT + (core.GROUNDING_PROMPT if grounding_block else "")

    # user text = an index of what each attached image is (+ the numeric context)
    user_text = ("Judge this reconstruction. The attached images, in order:\n"
                 + "\n".join(f"- Image {i+1}: {lab}" for i, lab in enumerate(labels))
                 + f"\n\nThere are {len(images)} images total"
                 + (f" (images {n_before_tt+1}-{len(images)} are turntable orbit views)."
                    if n_tt else ".")
                 + grounding_block
                 + rig_context
                 + "\nReturn the JSON verdict now.")
    if grounded_on:
        print(f"[critic] grounding on: {', '.join(grounded_on)}")

    # ---- call the backend (graceful skip if unavailable) ----
    provider = args.provider
    try:
        backend, model, provider = core.make_backend(args.provider, args.model,
                                                      args.api_key_env)
    except core.BackendUnavailable as e:
        print(f"[critic] SKIPPED — {e}. The critic is an optional signal; "
              "the IoU gate, depth guide, and turntable check are unaffected.")
        raise SystemExit(0)

    raw = None
    report = None
    last_err = None
    for attempt in (1, 2):  # one retry on unparseable output
        try:
            raw = backend.complete(system_prompt, images, user_text, args.max_tokens)
            report = core._normalize(core._extract_json(raw))
            break
        except core.BackendUnavailable as e:
            print(f"[critic] SKIPPED — {e}.")
            raise SystemExit(0)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            print(f"[critic] attempt {attempt}: could not parse a JSON verdict "
                  f"({e}); {'retrying' if attempt == 1 else 'giving up'}")
        except Exception as e:  # network / API error — do not crash the loop
            print(f"[critic] backend error ({type(e).__name__}: {e})")
            last_err = e
            break

    if report is None:
        err = {"frame": args.frame, "provider": provider, "model": model,
               "status": "error", "error": str(last_err),
               "raw_reply": (raw or "")[:2000]}
        write_json(args.out, err)
        print(f"[critic] wrote {args.out} (status=error)")
        raise SystemExit(2)

    realism_pass = report["realism"] is not None and report["realism"] >= args.pass_realism
    identity_pass = report["identity"] is not None and report["identity"] >= args.pass_identity
    report.update({
        "frame": args.frame, "provider": provider, "model": model,
        "status": "ok", "n_images": len(images), "n_turntable": n_tt,
        "grounded_on": grounded_on,
        "gate": {"pass_realism": args.pass_realism, "pass_identity": args.pass_identity,
                 "realism_pass": realism_pass, "identity_pass": identity_pass},
    })
    write_json(args.out, report)

    r = "—" if report["realism"] is None else f"{report['realism']:.2f}"
    i = "—" if report["identity"] is None else f"{report['identity']:.2f}"
    print(f"[critic] wrote {args.out}")
    print(f"[critic] realism={r} identity={i}  {report['summary']}")
    for d in report["discrepancies"]:
        print(f"  [{d['severity']:>4}] {d['tag']:>5} · {d['part']}: {d['adjustment']}")

    raise SystemExit(0 if (realism_pass and identity_pass) else 1)


if __name__ == "__main__":
    main()
