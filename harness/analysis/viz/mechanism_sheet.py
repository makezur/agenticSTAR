"""Build one mechanism sheet per JOINT from a render pass — the arc as strips.

One page per joint: each picked viewpoint (row) laid out as a STRIP in sweep
order (a sequence, unlike the turntable's unordered grid — direction of travel
is the thing being judged), labelled with its viewpoint and angle off the swing
plane. No source tile: mid-sweep states are configurations the capture never
showed, and the page answers "does the rig do what you declared", not a match.
The page header is deliberately two terse lines: header height is height the
tiles do not get, and on this page the tiles are the evidence. The signed `axis`
is WITHHELD (one arm is swept about it — printing it would answer the question).

    RUN_DIR/views/<NNNN>/mechanism_sheets/mechanism_sheet_<joint>.png
                                         mechanism_sheet_manifest.json

Shares its filename grammar with the renderer via `core.mechanism_views`, and
its chrome with every other page via `analysis.lib.panels`.
"""

import argparse
import glob
import os

import cv2

from analysis.lib import panels, rasters
from analysis.lib.io import read_json, write_json
from analysis.viz import frame_sheet
from core import mechanism_calls
from core import mechanism_views


DEFAULT_TILE_HEIGHT = 300
# A sweep's default is 7 states, and a 7-wide strip at 300px is ~2100px — wide, but
# ONE row, which is the property the page exists for. Wrapping is the fallback for
# a hand-ordered 20-state sweep, not the normal case.
DEFAULT_COLUMNS = 9

TILE_GAP = 10
TILE_SEPARATOR_BGR = (70, 70, 70)
PAGE_HEADER_HEIGHT = 56          # two lines: identity, then THE QUESTION
PAGE_BG = (30, 30, 30)


def find_tiles(pass_dir):
    """Every mechanism render in `pass_dir`, grouped by joint, then arm, then row.

    Returns `[(joint, [(arm, [(row, [(index, state, path)])])])]` with joints and
    arms in sorted order, rows in row order, and each row's tiles in SWEEP order
    — sorted by the sample index, not by state, so a sweep running from positive
    to negative still reads as one motion.

    A pre-arm pass parses with `arm=None` under a single `None` arm;
    `write_sheets` refuses to build a forced choice out of one arm.
    """
    pass_dir = os.path.abspath(pass_dir)
    found = {}
    for path in sorted(glob.glob(os.path.join(pass_dir,
                                              mechanism_views.IMAGE_GLOB))):
        parsed = mechanism_views.parse_image_name(os.path.basename(path))
        if parsed is None:                    # not one of ours; leave it alone
            continue
        joint, arm, row, index, state = parsed
        found.setdefault(joint, {}).setdefault(arm, {}).setdefault(
            row, []).append((index, state, path))
    return [(joint,
             [(arm, [(row, sorted(tiles, key=lambda t: t[0]))
                     for row, tiles in sorted(rows.items())])
              for arm, rows in sorted(arms.items(),
                                      key=lambda kv: (kv[0] is None,
                                                      str(kv[0])))])
            for joint, arms in sorted(found.items())]


def _tile(path, index, state, total, tile_height, bg_mode):
    """One tile as an UNLABELLED `panels.Panel`: the render, titled by the STATE
    it shows, captioned by its position in the sweep.

    The state leads the title because it is what a reader quotes back ("it is
    already inside the body by +50"); the position is the caption because it only
    orients you within the strip. Unlabelled because bar height is a property of
    the ROW — see `turntable_sheet._tile` for the failure that taught this."""
    render = rasters.load_render(path, bg_mode)
    image = panels.scale_to_height(render, int(tile_height))
    return panels.Panel(image, f"{state:+.0f}", sub=f"{index + 1}/{total}")


def _joint_defs(pass_dir):
    """{joint name: its definition} from the pass's pose.json, or {}.

    The DECLARATION is what the page is evidence about, so it is read from the
    same artifact the renderer posed from rather than re-derived. Absent or
    unreadable -> the header simply omits it; a missing pose.json must not cost
    the sheet."""
    for candidate in ("pose.json", os.path.join("..", "..", "mesh", "pose.json")):
        doc = read_json(os.path.join(pass_dir, candidate))
        if doc:
            defs = doc.get("joint_defs") or doc.get("JOINTS") or []
            named = {j.get("name"): j for j in defs if j.get("name")}
            if named:
                return named
    return {}


def _blind_definition(joint_def):
    """A joint definition with the signed `axis` REMOVED, for the manifest — the
    rest is shared by both arms, so it orients without answering. `template`
    re-derives the hash from this plus `axis_line`."""
    if not joint_def:
        return {}
    return {k: v for k, v in dict(joint_def).items() if k != "axis"}


def _declaration(joint_def):
    """The declaration as one line, MINUS the axis — one arm is swept about
    exactly that axis, so printing it answers the question. It appears only after
    a wrong pick, as the fix."""
    if not joint_def:
        return ""
    def vec(key):
        value = joint_def.get(key)
        if value is None:
            return "?"
        return "(" + ", ".join(f"{float(v):g}" for v in value) + ")"
    limit = joint_def.get("limit")
    limit_text = ("free" if limit is None
                  else f"[{float(limit[0]):g}, {float(limit[1]):g}]")
    return (f"{joint_def.get('type', '?')}   origin {vec('origin')}   "
            f"limit {limit_text}   axis WITHHELD (one arm is its mirror)")


ROW_HEADER_HEIGHT = 24
ARM_HEADER_HEIGHT = 30


def _arm_header(width, channels, arm, n_rows):
    """The band opening one arm's block: its blind label, nothing about the axis.
    Brighter/taller than a row header because the ARM is the unit of decision."""
    band = panels.solid(ARM_HEADER_HEIGHT, width, (44, 44, 52), channels)
    white = (235, 235, 235, 255) if channels == 4 else (235, 235, 235)
    dim = (170, 170, 170, 255) if channels == 4 else (170, 170, 170)
    label = f"ARM {arm}" if arm is not None else "ARM ?"
    cv2.putText(band, label, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.66,
                white, 2, cv2.LINE_AA)
    if n_rows > 1:            # with one row there is no "same arc" to explain
        cv2.putText(band, f"{n_rows} viewpoints, same arc", (118, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, dim, 1, cv2.LINE_AA)
    return band


def _row_header(width, channels, row, az_el, off_plane):
    """One line naming the row's VIEWPOINT and its angle off the swing plane.

    The angle is printed so a marginal view announces itself: below
    NEAR_EDGE_ON_DEG the arc is foreshortened enough that the reader should lean
    on the other row, and the header says so instead of letting a flattened arc
    under-sell a correct (or wrong) sign. Unknown viewpoint (no joint def to
    recompute from) -> just the row number."""
    band = panels.solid(ROW_HEADER_HEIGHT, width, (22, 22, 22), channels)
    if az_el is None:
        text, color = f"row {row}", (165, 165, 165)
    else:
        az, el = az_el
        text = f"az{az:.0f}/el{el:+.0f}   {off_plane:.0f}° off swing plane"
        color = (165, 165, 165)
        if off_plane is not None and off_plane < mechanism_views.NEAR_EDGE_ON_DEG:
            text += "   NEAR EDGE-ON — read the other row"
            color = (60, 170, 235)      # amber-ish in BGR: flag, not decoration
    if channels == 4:
        color = color + (255,)
    cv2.putText(band, text, (12, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                color, 1, cv2.LINE_AA)
    return band


def _page_header(width, channels, joint, joint_def, index, total, n_states,
                 arms, this_arm=None):
    """TWO terse lines: identity, then the question. Header height is height the
    tiles do not get, and here the tiles ARE the evidence."""
    header = panels.solid(PAGE_HEADER_HEIGHT, width, (15, 15, 15), channels)
    white = (225, 225, 225, 255) if channels == 4 else (225, 225, 225)
    dim = (165, 165, 165, 255) if channels == 4 else (165, 165, 165)
    flag = (80, 200, 245, 255) if channels == 4 else (80, 200, 245)
    labels = "/".join(str(a) for a in arms if a is not None) or "?"
    left = (f"MECHANISM  {joint}  —  ARM {this_arm}" if this_arm
            else f"MECHANISM  {joint}")
    right = f"{n_states} states x {len(arms)} arms"
    cv2.putText(header, left, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                white, 2, cv2.LINE_AA)
    right_width = cv2.getTextSize(
        right, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
    cv2.putText(header, right, (max(12, width - right_width - 12), 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, dim, 1, cv2.LINE_AA)
    limit = (joint_def or {}).get("limit")
    span = (f"  [{float(limit[0]):g}, {float(limit[1]):g}]"
            if limit is not None else "")
    question = (f"ONE of {labels} is the real mechanism (the other is its "
                f"MIRROR) — identical at the endpoints, so read the INTERIOR "
                f"tiles.{span}")
    cv2.putText(header, question, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                flag, 1, cv2.LINE_AA)
    line = (150, 150, 150, 255) if channels == 4 else (150, 150, 150)
    cv2.line(header, (0, PAGE_HEADER_HEIGHT - 2),
             (width, PAGE_HEADER_HEIGHT - 2), line, 2)
    return header


def _grid(tiles, row_counts):
    """`panels.Panel` tiles labelled and stacked into rows of `row_counts`.

    One `label_row` call for the whole page, so every row's bar is the same height
    and tiles line up across rows (`turntable_sheet._grid` documents the failure
    this prevents)."""
    labelled = panels.label_row(tiles)
    rows, start = [], 0
    for count in row_counts:
        rows.append(panels.hstack_panels(labelled[start:start + count],
                                        sep_width=TILE_GAP,
                                        sep_bgr=TILE_SEPARATOR_BGR))
        start += count
    return panels.vstack_rows(rows, sep_height=TILE_GAP,
                              sep_bgr=TILE_SEPARATOR_BGR)


def write_sheets(pass_dir, out_dir="", columns=DEFAULT_COLUMNS,
                 tile_height=DEFAULT_TILE_HEIGHT, bg_mode="black"):
    """Write one sheet per joint. Returns `(page_paths, manifest_path)`.

    Raises `ValueError` when the pass holds no mechanism renders — the caller
    asked for sheets of something that was not rendered, and silently writing an
    empty manifest would read as "the check is clean".
    """
    columns, tile_height = int(columns), int(tile_height)
    if columns < 1:
        raise ValueError("--columns must be at least 1")
    if tile_height < 1:
        raise ValueError("--tile-height must be positive")

    pass_dir = os.path.abspath(pass_dir)
    joints = find_tiles(pass_dir)
    if not joints:
        raise ValueError(
            f"no mechanism renders in {pass_dir} (looked for "
            f"{mechanism_views.IMAGE_GLOB}) — render the view first: "
            "harness/render.sh RUN_DIR/scene.py RUN_DIR/views "
            "--views mechanism")

    out_dir = os.path.abspath(
        out_dir or os.path.join(pass_dir, "mechanism_sheets"))
    os.makedirs(out_dir, exist_ok=True)
    defs = _joint_defs(pass_dir)

    page_paths, manifest_joints = [], []
    for index, (joint, arms) in enumerate(joints, start=1):
        joint_def = defs.get(joint)
        arm_labels = [arm for arm, _rows in arms]
        if len(arm_labels) < 2:
            # A one-armed page is not a forced choice. Refuse rather than emit a
            # page whose single strip a reader would dutifully approve.
            raise ValueError(
                f"joint {joint!r} has only {len(arm_labels)} swept arm "
                f"({arm_labels!r}) in {pass_dir} — the mechanism page is a BLIND "
                "A/B choice between the declared arc and its mirror, and one arm "
                "cannot pose it. Re-render the mechanism view with current code "
                "(harness/render.sh ... --views mechanism); tiles named without "
                "an arm field are from before the pair existed")
        # the same deterministic picker the renderer ran, re-run from the same
        # declaration — so each row header can name its viewpoint and angle off
        # the swing plane without a side-channel. The picker takes the DECLARED
        # axis, but its answer is arm-independent (negating an axis leaves the
        # swing plane, and so the legibility of every viewpoint, unchanged), so
        # re-running it here leaks nothing. No declaration in reach -> rows are
        # still laid out, just labelled by number alone.
        n_rows = max(len(rows) for _arm, rows in arms)
        picked = (mechanism_views.pick_views(joint_def.get("axis", (0, 0, 1)),
                                             joint_def.get("type", "revolute"),
                                             rows=n_rows)
                  if joint_def else [])
        # ONE PAGE PER ARM. Each arm gets the full page width, so a tile is as
        # large as the sheet can make it — on a stacked page the two arms shared
        # the width and every tile came out half-size, which is a bad trade for
        # the one view whose job is showing a subtle interior difference. The
        # combined page is still written (below) for same-state comparison.
        arm_pages, manifest_arms, n_states = {}, [], 0
        arm_bodies = {}
        for arm, rows in arms:
            manifest_rows = []
            arm_bands = []
            for row, tiles in rows:
                built = [_tile(path, i, state, len(tiles), tile_height, bg_mode)
                         for i, (_idx, state, path) in enumerate(tiles)]
                # ONE ROW per viewpoint while the sweep fits the budget — the
                # strip's whole value. Only a longer sweep wraps, and then into
                # balanced rows so the last is not a lone tile in a field of
                # black.
                row_counts = (frame_sheet.balance_pages(len(built), columns)
                              if len(built) > columns else [len(built)])
                strip = _grid(built, row_counts)
                az_el = picked[row] if row < len(picked) else None
                off_plane = (mechanism_views.swing_plane_angle_deg(
                                 az_el[0], az_el[1],
                                 joint_def.get("axis", (0, 0, 1)))
                             if az_el and joint_def else None)
                arm_bands.append(_row_header(strip.shape[1], strip.shape[2],
                                             row, az_el, off_plane))
                arm_bands.append(strip)
                n_states = max(n_states, len(tiles))
                manifest_rows.append({
                    "row": row,
                    "view": ({"azimuth": az_el[0], "elevation": az_el[1],
                              "off_swing_plane_deg": round(off_plane, 2)}
                             if az_el and off_plane is not None else None),
                    "columns": max(row_counts),
                    "states": [
                        {"state": round(state, 4), "index": idx,
                         "image": os.path.basename(path), "tile": i + 1}
                        for i, (idx, state, path) in enumerate(tiles)
                    ],
                })
            width = max(b.shape[1] for b in arm_bands)
            head = _arm_header(width, arm_bands[0].shape[2], arm, len(rows))
            arm_bodies[arm] = panels.vstack_rows([head] + arm_bands,
                                                sep_height=0)
            manifest_arms.append({"arm": arm, "rows": manifest_rows,
                                  "page": None})

        for arm, body in arm_bodies.items():
            header = _page_header(body.shape[1], body.shape[2], joint,
                                  joint_def, index, len(joints), n_states,
                                  arm_labels, this_arm=arm)
            page = panels.vstack_rows([header, body], sep_height=0)
            page_path = os.path.join(out_dir,
                                     f"mechanism_sheet_{joint}_{arm}.png")
            if not cv2.imwrite(page_path, page):
                raise OSError(f"failed to write {page_path}")
            page_paths.append(page_path)
            for rec in manifest_arms:
                if rec["arm"] == arm:
                    rec["page"] = page_path
            print(f"[mechanism-sheet] wrote {page_path}  "
                  f"(arm {arm}: {n_rows} row(s) x {n_states} states of joint "
                  f"{joint})")

        # …and the two arms stacked, so the same state can be compared in one
        # saccade. Same tiles, no extra renders.
        combined_body = panels.vstack_rows(
            [arm_bodies[a] for a, _r in arms], sep_height=0)
        combined_header = _page_header(
            combined_body.shape[1], combined_body.shape[2], joint, joint_def,
            index, len(joints), n_states, arm_labels)
        combined = panels.vstack_rows([combined_header, combined_body],
                                      sep_height=0)
        page_path = os.path.join(out_dir, f"mechanism_sheet_{joint}.png")
        if not cv2.imwrite(page_path, combined):
            raise OSError(f"failed to write {page_path}")
        page_paths.append(page_path)
        print(f"[mechanism-sheet] wrote {page_path}  "
              f"(both arms stacked, for same-state comparison)")
        manifest_joints.append({
            "joint": joint,
            "page": page_path,
            # The definition MINUS the signed axis — the manifest sits in the
            # same directory as the page whose header withholds it, and a reader
            # who can `grep axis` never has to look at a tile. `axis_line` (the
            # unsigned direction) is what the declaration hash is about and is
            # safe to publish; the SIGN is the answer, so it is not here.
            "definition": _blind_definition(joint_def),
            "axis_line": (mechanism_calls.unsigned_axis(joint_def.get("axis"))
                          if joint_def else None),
            "arms": arm_labels,
            "arms_detail": manifest_arms,
        })

    manifest = {
        "schema_version": 1,
        "pass_dir": pass_dir,
        "bg_mode": bg_mode,
        "tile_height": tile_height,
        "joint_count": len(manifest_joints),
        "image_glob": mechanism_views.IMAGE_GLOB,
        "pages": page_paths,
        "joints": manifest_joints,
    }
    manifest_path = os.path.join(out_dir, "mechanism_sheet_manifest.json")
    write_json(manifest_path, manifest)
    print(f"[mechanism-sheet] wrote {manifest_path}")
    return page_paths, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pass-dir", required=True,
                        help="the views/<NNNN>/ pass dir render.sh printed")
    parser.add_argument("--out-dir", default="",
                        help="where the pages go (default: "
                             "PASS_DIR/mechanism_sheets)")
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS,
                        help="tiles per row before the strip wraps (default "
                             f"{DEFAULT_COLUMNS} — a default sweep stays ONE row)")
    parser.add_argument("--tile-height", type=int, default=DEFAULT_TILE_HEIGHT,
                        help=f"render height per tile (default {DEFAULT_TILE_HEIGHT})")
    parser.add_argument("--bg-mode", default="black",
                        help="backdrop for the transparent renders "
                             "(rasters.load_render)")
    args = parser.parse_args()
    write_sheets(args.pass_dir, out_dir=args.out_dir, columns=args.columns,
                 tile_height=args.tile_height, bg_mode=args.bg_mode)


if __name__ == "__main__":
    main()
