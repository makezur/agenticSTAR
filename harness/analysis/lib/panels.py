"""panels — shared panel drawing for the visual judge strips.

Owns the pieces the composite / depth / sheet panels all share:
labels, height scaling, solid fills, hstacking + vstacking with separators, the
silhouette-overlap colours and their legend, the TURBO heatmap, the signed
diverging (near/far) heatmap + its legend, and the debug timeline archive. Pure
drawing — no scoring, no I/O beyond writing the timeline PNG — so both `scorers/`
and `viz/` depend on it without a cycle.

The overlap COLOURS live here, not in `viz.composite`, because every tool that
paints or captions that panel must agree pixel-for-pixel: the composite strip,
the candidate sheets the pool writes after each sweep, and the critic's prompt.
One definition, imported by all of them (`viz.composite` re-exports the old
names). `viz.rows` builds the actual rows out of these primitives.

THE ALIGNMENT INVARIANT. A header bar is STACKED ABOVE its panel, not overlaid on
it: `label` returns a taller image, and no render pixel is ever covered. Panels sit
side by side to be compared, so the same y must be the same row of the capture in
all of them — equal renders under equal bars gives that exactly. Nothing a bar
contains can cost you image either: caption, colour key, or neither, the render
below is untouched. The legend never wraps (it shrinks its wording, then its type,
and is dropped rather than clipped), but that is about keeping the key readable on
one line, not about protecting pixels.

Bar height is therefore a property of the ROW, not of the panel: `BAR_HEIGHT`, or
`CAPTIONED_BAR_HEIGHT` when anything in the row carries a caption and every panel
must reserve the line. Build a row with `label_row`, which takes `Panel(image,
title, sub)` values and works that out for you — a caller that labels panels
one-by-one has to remember the rule, and a row of uncaptioned panels should not pay
for a caption line nothing in it uses.

See `Panel`, `label_row`, `label`, `BAR_HEIGHT`, `legend_fits`.
"""

import json
import os

import cv2
import numpy as np


MIN_DOWNSAMPLE_SCALE = 0.5


def resize_to_max_dimension(image, max_dimension):
    """Resize a completed panel toward a longest-edge target.

    A visual artifact may be made smaller, but never below half its native
    linear resolution: a lower target is intentionally treated as 0.5x rather
    than producing an unusable, token-cheap panel. Returns the resized image and
    auditable source/target metadata.
    """
    if isinstance(max_dimension, bool) or not isinstance(max_dimension, int) \
            or max_dimension < 0:
        raise ValueError(
            f"max_dimension must be an integer >= 0, got {max_dimension!r}")
    height, width = image.shape[:2]
    source_size = [width, height]
    longest = max(source_size)
    if not max_dimension or longest <= max_dimension:
        return image, {
            "source_size": source_size,
            "size": source_size,
            "scale": 1.0,
        }
    scale = max(MIN_DOWNSAMPLE_SCALE, max_dimension / longest)
    target = (max(1, int(round(width * scale))),
              max(1, int(round(height * scale))))
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA), {
        "source_size": source_size,
        "size": list(target),
        "scale": scale,
    }


def preview_path(path):
    """The derived preview path for a canonical panel image."""
    stem, ext = os.path.splitext(path)
    return f"{stem}_preview{ext}"


def variants_path(path):
    """The metadata sidecar for a canonical panel image."""
    stem, _ = os.path.splitext(path)
    return f"{stem}.variants.json"


def write_image_variants(
        path, image, preview_max_dimension=0, write_manifest=False):
    """Write a native canonical panel plus an optional reduced preview.

    The canonical `path` is always the untouched completed panel. When the
    configured target reduces it, the derived `_preview` file is written beside
    it and returned as the default inspection path. This preserves the pixels
    needed by a later native-file inspection or focused crop.
    `write_manifest` adds the shared `.variants.json` sidecar used by standalone
    side-by-sides; candidate sheets carry the same records in their page manifest.
    """
    preview, raster = resize_to_max_dimension(image, preview_max_dimension)
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, image):
        raise OSError(f"could not write panel image: {path}")
    selected = path
    if raster["scale"] < 1.0:
        selected = preview_path(path)
        if not cv2.imwrite(selected, preview):
            raise OSError(f"could not write panel preview: {selected}")
    else:
        stale_preview = preview_path(path)
        if os.path.isfile(stale_preview):
            os.unlink(stale_preview)
    variants = {
        "original": path,
        "preview": selected,
        "original_size": raster["source_size"],
        "preview_size": raster["size"],
        # Compatibility aliases for the candidate-sheet raster vocabulary.
        "source_size": raster["source_size"],
        "size": raster["size"],
        "scale": raster["scale"],
    }
    if write_manifest:
        manifest = variants_path(path)
        with open(manifest, "w") as handle:
            json.dump(variants, handle, indent=2)
            handle.write("\n")
        variants["manifest"] = manifest
    return variants


def scale_to_height(img, h):
    scale = h / img.shape[0]
    w = max(1, int(round(img.shape[1] * scale)))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def solid(height, width, bgr, channels=3):
    """A filled rectangle: separators, header bars, placeholder tiles.

    `channels=4` appends an opaque alpha, so a solid drawn beside BGRA panels
    stacks without the caller widening it by hand."""
    color = (*bgr, 255) if channels == 4 else bgr
    return np.full((height, width, channels), color, dtype=np.uint8)


# ------------------------------------------------------------------------- #
# Silhouette-overlap colours (BGR). One color per class, tuned so the two SCORED
# errors carry the loudest colors (extra volume is the actionable one),
# agreement stays a quiet green you can read across, the hand is a dim neutral
# painted UNDER everything, and our render inside the hand region gets its own
# calm blue — so "our volume is still there, just not scored" reads directly
# instead of vanishing into the ignored gray.
#
# These live in `lib` because the composite strip, the candidate sheets, and the
# critic's prompt must all describe the SAME pixels. `viz.composite` re-exports
# them under their original names.
# ------------------------------------------------------------------------- #
OVL_AGREE = (85, 140, 82)      # mask & render agree                muted green
OVL_EXTRA = (0, 235, 255)      # render only: EXTRA volume, SCORED   hot yellow
OVL_MISSING = (230, 0, 220)    # mask only:   MISSING volume, SCORED magenta
OVL_WAIVED = (225, 150, 60)    # our render under the hand — WAIVED  blue
OVL_HAND = (62, 62, 62)        # ignored hand, nothing of ours there dark gray

# Caption fragments describing the coding above — ONE definition, so the panel
# label, the critic's prompt, and the docs can't drift from the pixels.
#
# "MAYBE OCCLUDED" rather than the longer "POSSIBLE HAND OCCLUSION": it says the
# same thing (this volume may be hidden behind the hand, so it is not scored) in
# 449px instead of 521px, which is what lets the full wording fit a 512px panel's
# single legend row at a readable font instead of falling back to `HAND`.
OVERLAP_LEGEND = [
    (OVL_AGREE, "AGREE"),
    (OVL_EXTRA, "EXTRA"),
    (OVL_MISSING, "MISSING"),
    (OVL_WAIVED, "MAYBE OCCLUDED"),
]
# The same key with the hand entry shortened further, for panels still too narrow
# for one row (a 4-column candidate row is ~290-430px). Same colours, same order —
# only the words are abbreviated, so it cannot disagree with the full legend.
OVERLAP_LEGEND_SHORT = [
    (OVL_AGREE, "AGREE"),
    (OVL_EXTRA, "EXTRA"),
    (OVL_MISSING, "MISSING"),
    (OVL_WAIVED, "HAND"),
]
OVERLAP_CAPTION = ("green=AGREE yellow=EXTRA magenta=MISSING "
                   "blue=MAYBE OCCLUDED (our render behind the hand, not scored) "
                   "gray=hand (ignored)")


def overlap_legend(keep=None, short=False):
    """The overlap key for a panel, minus entries it cannot show.

    Without a keep-region there is no hand to waive, so the WAIVED swatch is
    dropped rather than promising a colour the pixels never use. `short` picks
    the abbreviated wording for narrow panels."""
    entries = OVERLAP_LEGEND_SHORT if short else OVERLAP_LEGEND
    if keep is not None:
        return list(entries)
    return [e for e in entries if e[0] != OVL_WAIVED]


def overlay_panel(source_mask, render_mask, keep=None):
    """Overlay the two silhouettes AS-RENDERED (raw 1:1, no normalization) at the
    NATIVE source-mask resolution. The camera is fixed and the object is built at
    true scale in front of it, so this direct overlap is what `iou_raw` scores.

    green = agreement, yellow = render only (EXTRA volume), magenta = mask only
    (MISSING volume). When `keep` (the non-hand keep-region, 0/1) is given the
    ignored HAND region is painted dim gray FIRST, and our render inside it is
    then lit in blue: that volume is present and legitimate (the object really
    does continue behind the hand) but neither penalized nor rewarded, so it must
    not read as either an error or a hole. Only the render extends under the hand
    — SAM labels occluded object pixels as hand, so the object mask is
    visible-only and mask ∩ hand is empty at hand_dilate=0.

    Returns the native-resolution panel; the caller scales it (the composite
    strip does, the standalone --overlap-out image keeps it full-res).
    """
    H, W = source_mask.shape[:2]
    rm = render_mask
    if rm.shape[:2] != (H, W):
        rm = cv2.resize(rm, (W, H), interpolation=cv2.INTER_NEAREST)
    panel = np.zeros((H, W, 3), dtype=np.uint8)
    sm = source_mask > 0
    rmb = rm > 0
    if keep is None:
        panel[sm] = OVL_MISSING
        panel[rmb] = OVL_EXTRA
        panel[sm & rmb] = OVL_AGREE
        return panel
    k = keep if keep.shape[:2] == (H, W) else cv2.resize(
        keep, (W, H), interpolation=cv2.INTER_NEAREST)
    ignored = k == 0
    scored = ~ignored
    panel[ignored] = OVL_HAND                    # dim, underneath
    panel[rmb & ignored] = OVL_WAIVED            # ours, present but unscored
    panel[sm & scored] = OVL_MISSING             # the SCORED classes on top
    panel[rmb & scored] = OVL_EXTRA
    panel[sm & rmb & scored] = OVL_AGREE
    return panel


_TITLE_FONT = 0.6         # the panel name
_LEGEND_FONT = 0.6        # same size as the title — readable at a glance
_SUB_FONT = 0.55          # the caption, whether it rides inline or on its own line
_SWATCH = 14              # swatch side, px
_GAP = 14                 # px between legend entries

_TITLE_LINE = 26          # px: the title line, which the legend shares
_CAPTION_LINE = 20        # px: the caption line, reserved whether or not it is used

# Font scales a legend may be drawn at, largest first. A narrow panel gets a
# SMALLER key, never a second row: one line is what makes the key scannable beside
# the title, and a key you have to read in two passes is a caption. 0.42 is the
# floor — below that the words stop being readable at all and `legend_font` returns
# None, which drops the key (and restores the title in its place).
LEGEND_FONTS = (0.6, 0.5, 0.42)

# THE header bar height, STACKED ABOVE the image rather than drawn over it.
#
# It takes exactly TWO values, and which one is in play is a property of the ROW,
# never of the individual panel:
#
#   BAR_HEIGHT          title/legend line only — every caption here rides inline
#   CAPTIONED_BAR_HEIGHT   + a caption line, because one in this row cannot
#
# Panels are compared side by side, so the same y must mean the same row of the
# capture in every one of them: equal-height renders under equal-height bars. A bar
# that sized itself to its OWN contents would offset a captioned panel's render from
# its uncaptioned neighbour's, which is the bug this replaced. So within a row the
# height is still one constant — an uncaptioned panel beside a captioned one does
# reserve the empty line — but a row that needs no second line does not pay for one.
# Because the line is charged to EVERY column, a caption is measured first and rides
# on the title line when it fits (`sub_fits_title_line`): "rank 00" beside RENDER is
# free, the depth panels' range/MAE/bias lines are what actually buy the band.
# `label` picks per panel and `hstack_panels` bottom-pads any shortfall, so a caller
# that mixes them still gets aligned renders; callers who want the whole row to share
# one bar pass `caption_line=` (see `viz.rows`).
BAR_HEIGHT = _TITLE_LINE                        # 26
CAPTIONED_BAR_HEIGHT = _TITLE_LINE + _CAPTION_LINE   # 46


def _legend_start(title):
    """The x the first legend entry begins at: after the title, or at the left
    margin when there is no title. A titleless panel is how the overlap column
    fits its key on ONE row — the swatches name the panel by themselves."""
    if not title:
        return 6
    return 6 + _text_width(title, _TITLE_FONT) + _GAP


def _text_width(text, font):
    """Rendered width of `text` at `font`, in the one face every bar uses."""
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 1)[0][0]


def _entry_width(text, font=_LEGEND_FONT):
    """Swatch + gap + text + trailing gap for one legend entry."""
    return _SWATCH + 5 + _text_width(text, font) + _GAP


def legend_width(entries, title="", font=_LEGEND_FONT):
    """Total px ONE row of `entries` needs beside `title`, incl. the right margin.

    The single measurement the one-row guarantee rests on: a legend is drawable
    iff this is ≤ the panel width. No packing, no wrapping — those are the states
    we are ruling out."""
    return _legend_start(title) + sum(
        _entry_width(text, font) for _, text in entries) + 4


def legend_font(entries, width, title="", fonts=LEGEND_FONTS):
    """The largest font scale that draws `entries` on ONE row inside `width`.

    None when even the smallest scale overflows, which is the caller's signal to
    drop the key rather than clip or wrap it. Wrapping is deliberately not an
    option here: see `LEGEND_FONTS`."""
    if not entries:
        return None
    for font in fonts:
        if legend_width(entries, title, font) <= width:
            return font
    return None


def sub_fits_title_line(title, sub, width, legend=None):
    """Can `sub` ride on the TITLE line, after the title, inside `width`?

    A caption is usually two or three words naming which candidate a panel shows —
    far too little to be worth a whole 20px band under every column in the row, and
    the band is charged to ALL of them (see `label_row`). So it is measured first,
    exactly as the legend is: it goes inline when it fits and only claims its own
    line when it genuinely cannot, which is the case for the depth panels' range /
    MAE / bias lines.

    A panel carrying a legend never inlines: the key already owns the title line.
    """
    if not sub or legend:
        return False
    return (_legend_start(title) + _text_width(sub, _SUB_FONT) + 6) <= width


def bar_height(caption_line=False):
    """The height of the header bar `label` stacks above a panel.

    `caption_line` is the ROW's answer to "is anything here captioned?", not this
    panel's: pass the same value for every panel in a row and their renders line
    up. Nothing else — not the title, not the legend, not the panel size — affects
    it. See `BAR_HEIGHT`."""
    return CAPTIONED_BAR_HEIGHT if caption_line else BAR_HEIGHT


def legend_fits(entries, width, height=None, title="", sub="", fonts=LEGEND_FONTS):
    """Can this legend be drawn on ONE row, unclipped, in a `width`-wide bar?

    Purely a horizontal question, since the bar is stacked above the panel rather
    than drawn over it: nothing the bar holds can cost the render pixels, so the
    only failure is a key too wide to fit on one line even at the smallest font.
    `height`/`sub` are accepted and ignored.

    Returns the chosen font scale (truthy) or None, so a caller can both test and
    use the result.
    """
    del height, sub                      # unused: the bar is stacked
    if not entries:
        return None
    return legend_font(entries, width, title, fonts)


class Panel:
    """A picture plus the words that belong over it — NOT yet labeled.

    What a module that owns a colormap hands back (see `scorers.depth`): it knows
    what its panels are called, but only the code assembling the row can know
    whether a neighbouring column carries a caption, and that decides the bar
    height for the whole row. Keeping the two apart is what stops "reserve the
    caption line" from being a rule every producer has to remember.

    `legend` is the optional colour key (as `label` takes it); `legend_font_scale`
    a preferred scale, honoured only if it fits. `ramp_legend` is the CONTINUOUS
    key variant — `(strip_bgr, left_text, right_text)`, drawn on the title line
    like the swatch legend (the object-units DEPTH panel's NEAR→FAR ramp).
    """

    __slots__ = ("image", "title", "sub", "legend", "legend_font_scale",
                 "ramp_legend")

    def __init__(self, image, title="", sub="", legend=None,
                 legend_font_scale=None, ramp_legend=None):
        self.image = image
        self.title = title
        self.sub = sub
        self.legend = legend
        self.legend_font_scale = legend_font_scale
        self.ramp_legend = ramp_legend


def label_row(row):
    """Label `Panel`s as ONE row: same bar height for all of them.

    `row` is a list of `Panel`, or a dict of them (the depth panels are keyed);
    returns the same shape with each value a labeled image. This is the alignment
    invariant made automatic: label panels individually and you have to remember it
    yourself.

    The caption line is reserved iff some panel has a `sub` THAT CANNOT RIDE ON ITS
    TITLE LINE. A caption naming which candidate a panel shows is two or three words
    and fits beside the title; the band is charged to every column in the row, so
    buying one for those words costs 20px across the whole strip to say what fits in
    the gap already there. The depth panels' range/MAE/bias lines are the case that
    genuinely needs it. Measured, not assumed — the same rule the legend follows.
    """
    keys = list(row.keys()) if isinstance(row, dict) else None
    items = list(row.values()) if keys else list(row)
    caption_line = any(
        p.sub and not sub_fits_title_line(p.title, p.sub, p.image.shape[1],
                                          p.legend)
        for p in items)
    out = [label(p.image, p.title, sub=p.sub, legend=p.legend,
                 legend_font_scale=p.legend_font_scale,
                 caption_line=caption_line, ramp_legend=p.ramp_legend)
           for p in items]
    return dict(zip(keys, out)) if keys else out


def label(img, text, sub="", legend=None, legend_font_scale=None,
          caption_line=None, ramp_legend=None):
    """Stamp a header bar on `img`: title, optional legend, optional subtitle.

    Prefer `label_row` when labeling several panels that will sit side by side: it
    works out the shared bar height, which this cannot do from one panel alone.

    `legend` is a list of (color_bgr, text) pairs drawn as color swatch + name on
    the TITLE line, right after `text`. Pass it the SAME color constants the panel
    was painted with — that way the key can't drift from the pixels the way a
    hand-typed "green=agree" in the title can.

    `ramp_legend` is the CONTINUOUS key: `(strip_bgr, left_text, right_text)`,
    drawn on the title line after the title (and after any swatch legend) as
    left_text | the strip resized to the line | right_text. For colormapped
    panels (the object-units DEPTH column), where discrete swatches cannot say
    "everything in between is a distance too". Build the strip from the SAME
    colormap the pixels use. Dropped whole when the panel is too narrow —
    half a ramp is not a key.

    `text` may be EMPTY, in which case the legend starts at the left margin and
    names the panel by itself — the overlap column does this, since spending the
    width on the word "SILHOUETTE" is what forces the key to shrink.

    THE BAR IS STACKED ABOVE THE IMAGE, so the result is TALLER than `img` and no
    render pixel is ever covered. Its height takes one of two values and does not
    otherwise vary, so panels of equal height come out equal and their renders line
    up row-for-row — that is the whole point:

      `caption_line=True`  reserve the caption line (`CAPTIONED_BAR_HEIGHT`)
      `caption_line=False` title/legend line only (`BAR_HEIGHT`)
      `caption_line=None`  (default) True iff THIS panel has a `sub`

    Pass it explicitly — the same value for every panel — when labeling a row whose
    panels are captioned unevenly, so the uncaptioned ones reserve the line too
    instead of riding 20px higher than their neighbours. The default is right for a
    lone panel and for a row that is uniformly captioned or uniformly not; it is
    what keeps an uncaptioned row from paying for a line nothing in it uses.

    The legend is always ONE row (it shrinks to the `LEGEND_FONTS` floor and is
    DROPPED past that rather than clipped — `legend_fits` tells a caller in advance),
    because a key you read in two passes stops being scannable.
    """
    has_alpha = img.ndim == 3 and img.shape[2] == 4
    font = None
    if legend:
        # honour the caller's scale only if it actually fits; otherwise shrink. If
        # even the floor overflows, DROP the key: a clipped legend spills swatches
        # past the panel edge and reads as garbage, and half a key is not a key.
        fonts = ((legend_font_scale,) + LEGEND_FONTS if legend_font_scale
                 else LEGEND_FONTS)
        font = legend_font(legend, img.shape[1], text, fonts)
        if font is None:
            legend = None
    # One height for every panel in a row — the alignment guarantee — and ADDED to
    # the image rather than taken out of it. The row decides whether the caption
    # line exists; absent an explicit answer, this panel's own `sub` decides.
    # A short caption rides on the TITLE line rather than buying a band the whole
    # row pays for (see `sub_fits_title_line`); a long one still gets its own line.
    if caption_line is None:
        caption_line = bool(sub) and not sub_fits_title_line(
            text, sub, img.shape[1], legend)
    # ...and once a row HAS a caption line, every caption in it uses that line: a
    # mix of inline and below-the-title captions is harder to scan than either.
    inline_sub = "" if caption_line else sub
    bar_h = bar_height(caption_line)
    bar = solid(bar_h, img.shape[1], (0, 0, 0), 4 if has_alpha else 3)
    white = (255, 255, 255, 255) if has_alpha else (255, 255, 255)
    green = (120, 255, 120, 255) if has_alpha else (120, 255, 120)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, _TITLE_FONT,
                white, 1, cv2.LINE_AA)
    if inline_sub:
        cv2.putText(bar, inline_sub, (_legend_start(text), 19),
                    cv2.FONT_HERSHEY_SIMPLEX, _SUB_FONT, green, 1, cv2.LINE_AA)
    y, x = 3, _legend_start(text)           # the single, title-line legend row
    for color, name in (legend or []):
        sw = tuple(color[:3]) + ((255,) if has_alpha else ())
        cv2.rectangle(bar, (x, y + 2), (x + _SWATCH, y + 2 + _SWATCH), sw, -1)
        cv2.rectangle(bar, (x, y + 2), (x + _SWATCH, y + 2 + _SWATCH),
                      (90, 90, 90, 255) if has_alpha else (90, 90, 90), 1)
        cv2.putText(bar, name, (x + _SWATCH + 5, y + _SWATCH + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font, white, 1, cv2.LINE_AA)
        x += _entry_width(name, font)
    if ramp_legend:
        _draw_ramp_legend(bar, x, ramp_legend, white, has_alpha)
    if sub and not inline_sub:
        cv2.putText(bar, sub, (6, bar_h - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    _SUB_FONT, green, 1, cv2.LINE_AA)
    if has_alpha:
        # LINE_AA blends into the alpha channel too, so antialiased glyph edges
        # would leave the bar semi-transparent and the text would ghost against
        # whatever is composited behind it. The bar is chrome, not content: force
        # it fully opaque. (The image's own alpha is untouched — that is the point
        # of stacking rather than drawing over it.)
        bar[..., 3] = 255
    return np.vstack([bar, img])


def _draw_ramp_legend(bar, x, ramp_legend, white, has_alpha):
    """Draw `(strip, left_text, right_text)` on the bar's title line from `x`.

    left_text | strip | right_text, the strip resized to the line height and to
    whatever width remains (capped — a key is a key, not a mural). Text and gaps
    step down together on narrow portrait panels before the key is dropped.
    Skipped entirely when even a 60px strip cannot fit: half a ramp is not a key."""
    strip, left_text, right_text = ramp_legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    layout = None
    for fs, gap in ((_LEGEND_FONT, _GAP), (0.5, 10), (0.42, 5)):
        lt_w = _text_width(left_text, fs)
        rt_w = _text_width(right_text, fs)
        room = bar.shape[1] - x - lt_w - rt_w - 3 * gap
        if room >= 60:
            layout = fs, gap, lt_w, room
            break
    if layout is None:
        return
    fs, gap, lt_w, room = layout
    bw, bh, y0 = min(180, room), _SWATCH, 5
    cv2.putText(bar, left_text, (x, y0 + _SWATCH - 1), font, fs, white, 1,
                cv2.LINE_AA)
    x += lt_w + gap
    resized = cv2.resize(strip, (bw, bh), interpolation=cv2.INTER_AREA)
    if has_alpha:
        resized = np.dstack([resized,
                             np.full((bh, bw), 255, dtype=np.uint8)])
    bar[y0:y0 + bh, x:x + bw] = resized
    x += bw + gap
    cv2.putText(bar, right_text, (x, y0 + _SWATCH - 1), font, fs, white, 1,
                cv2.LINE_AA)


def _match_alpha(panels):
    """Widen every panel to BGRA if ANY of them has alpha; else leave as BGR.

    Returns (panels, has_alpha). Stacking mixed channel counts is a ValueError
    from numpy, and every stacker here has to make the same choice, so it is made
    once."""
    has_alpha = any(p.ndim == 3 and p.shape[2] == 4 for p in panels)
    if has_alpha:
        panels = [
            p if p.shape[2] == 4
            else np.dstack([p, np.full(p.shape[:2], 255, dtype=np.uint8)])
            for p in panels
        ]
    return panels, has_alpha


def hstack_panels(parts, sep_width=4, sep_bgr=(60, 60, 60)):
    """hstack labeled panels into one strip with gray separators.

    `parts` is a list of panels, or a dict whose values are panels (the depth
    panels are keyed). The strip height is the TALLEST panel: panels of unequal
    height are top-aligned and the shorter ones bottom-padded with opaque black
    instead of failing to stack. There is deliberately no `height` argument —
    panels arrive already labeled, and resizing one here would squash its header
    bar and break the row-for-row alignment `label` exists to guarantee. Size the
    renders instead, before labeling (`scale_to_height`).

    `sep_width`/`sep_bgr` default to the composite strip's thin dark rule; the
    candidate sheets pass a wider, lighter one so the column boundary reads
    against the overlap colours. Folds the former per-tool copies (composite,
    depth, sheets).
    """
    panels = list(parts.values()) if isinstance(parts, dict) else list(parts)
    panels, has_alpha = _match_alpha(panels)
    h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < h:
            pad = np.zeros((h - p.shape[0], p.shape[1], p.shape[2]), p.dtype)
            if has_alpha:
                pad[..., 3] = 255
            p = np.vstack([p, pad])
        padded.append(p)
    return _interleave(padded, solid(h, sep_width, sep_bgr,
                                     4 if has_alpha else 3), np.hstack)


def vstack_rows(rows, sep_height=12, sep_bgr=(110, 110, 110)):
    """vstack full-width rows into one page with horizontal separators.

    The sheet counterpart of `hstack_panels`: rows of unequal width are
    left-aligned and the narrower ones right-padded, so a page whose rows came
    from different-aspect sources stacks instead of raising."""
    rows, has_alpha = _match_alpha(list(rows))
    w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < w:
            pad = np.zeros((r.shape[0], w - r.shape[1], r.shape[2]), r.dtype)
            if has_alpha:
                pad[..., 3] = 255
            r = np.hstack([r, pad])
        padded.append(r)
    if len(padded) == 1:
        return padded[0]
    return _interleave(padded, solid(sep_height, w, sep_bgr,
                                     4 if has_alpha else 3), np.vstack)


def _interleave(parts, separator, stack):
    """`stack` the parts with `separator` between every adjacent pair."""
    return stack([
        item
        for index, part in enumerate(parts)
        for item in ((separator, part) if index else (part,))
    ])


def turbo_heatmap(norm, valid):
    """TURBO-colormap a PRE-NORMALIZED (0..1) field; black where not valid.

    `norm` is clipped to [0,1] then mapped through COLORMAP_TURBO; pixels where
    `valid` is False are blacked out. Callers own the normalization (a fixed
    divisor for the RGB residual, a [dmin,dmax] range for depth) — this folds the
    shared normalize→colormap→blackout kernel that composite and depth duplicated.
    """
    heat = cv2.applyColorMap((np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8),
                             cv2.COLORMAP_TURBO)
    heat[~valid] = (0, 0, 0)
    return heat


def pct_range(arrays, lo=2, hi=98):
    """One [dmin, dmax] over the pooled FINITE values of several arrays.

    Percentiles, not min/max, so a few outlier pixels don't wash out the scale.
    Pooling is the point: pass the two maps being compared (the depth scorer's
    render + observed pair) or a whole clip's frames (the depth sheet) and every
    picture drawn from the result shares one scale. `[0, 1]` when nothing is
    finite anywhere — an empty selection stays drawable, not fatal. `dmax` is
    nudged above `dmin` so callers can divide by the span unguarded."""
    pool = [np.asarray(a)[np.isfinite(a)].ravel() for a in arrays]
    pool = np.concatenate([p for p in pool if p.size] or [np.zeros(0)])
    if pool.size == 0:
        return 0.0, 1.0
    dmin = float(np.percentile(pool, lo))
    dmax = float(np.percentile(pool, hi))
    return dmin, max(dmax, dmin + 1e-6)


def depth_heat(depth, dmin, dmax, valid=None, divisor=1.0):
    """TURBO heatmap of a DEPTH map over [dmin, dmax] (after `/ divisor`).

    The depth-specific normalization the scorer and the object-units sheet both
    need, folded the way `turbo_heatmap` folded the layer beneath it: non-finite
    (no-geometry) pixels are parked at `dmax` so they can't drag the ramp, then
    blacked out via `valid`. `valid` defaults to "wherever depth is finite";
    pass one to ALSO exclude pixels outside a scored region, and it is ANDed
    with the finite mask either way. `divisor` rescales the unit — the object
    scale `s`, to draw in object units instead of scene units."""
    finite = np.isfinite(depth)
    parked = np.where(finite, np.asarray(depth, np.float32) / divisor, dmax)
    keep = finite if valid is None else (valid & finite)
    return turbo_heatmap((parked - dmin) / max(dmax - dmin, 1e-6), keep)


def turbo_ramp_strip(width=180, height=14):
    """A NEAR→FAR turbo strip for the continuous keys (`ramp_legend`, the frame
    sheet's page header).

    Lives here, beside the LUT and beside the two things that DRAW it, so a key
    is built from the same colormap as the pixels it explains and cannot drift
    from them."""
    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    return turbo_heatmap(np.repeat(ramp, height, axis=0),
                         np.ones((height, width), bool))


def signed_diverging_heat(signed, valid, span, gamma=0.6):
    """Diverging blue↔white↔red colormap of a SIGNED depth residual.

    `signed` = render_depth − observed depth (observed units, +Z forward/OpenCV),
    so the sign tells you which WAY to move, not just how much:
      * signed < 0 -> render is NEARER than observed (in FRONT of it) -> BLUE
      * signed ≈ 0 -> agreement                                      -> WHITE
      * signed > 0 -> render is FARTHER than observed (BEHIND it)     -> RED
    (render too near -> push it back: +translation-Z or larger SCALE; too far ->
    the opposite: −translation-Z or smaller SCALE.) Intensity toward the poles
    grows with |error|, `span` sets the
    saturation scale (|signed| ≥ span clamps to full blue/red — same span the two
    depth panels share). `gamma` < 1 lifts small biases so a uniform near/far tint
    is still legible at a glance (the MAE number stays the quantitative measure).
    White center reads clearly against the black (unscored) background. Returns
    BGR uint8; black where not valid."""
    t = np.clip(signed / max(span, 1e-6), -1.0, 1.0)          # (H,W) in [-1,1]
    a = (np.abs(t) ** gamma)[..., None]                       # 0 agree, 1 pole
    white = np.array([255.0, 255.0, 255.0], np.float32)
    blue = np.array([255.0, 0.0, 0.0], np.float32)            # BGR (in front)
    red = np.array([0.0, 0.0, 255.0], np.float32)             # BGR (behind)
    pole = np.where((t < 0)[..., None], blue, red)
    heat = (white * (1.0 - a) + pole * a).astype(np.uint8)
    heat[~valid] = (0, 0, 0)
    return heat


def stamp_signed_legend(panel):
    """Draw a small blue→white→red colour key at the bottom-left of `panel`.

    The panel label is width-limited (portrait sources render a ~288px-wide
    panel, so the colour convention in the title truncates); this always-legible
    swatch keeps the standalone residual image self-describing regardless of
    aspect ratio. Mutates and returns `panel` (BGR uint8). blue = render NEARER
    (in front), red = FARTHER (behind)."""
    H, W = panel.shape[:2]
    bw = min(150, W - 20)                       # bar width (fit narrow panels)
    if bw < 40:
        return panel                            # too small to bother
    bh, x0 = 12, 8
    y1 = H - 10
    y0 = y1 - bh
    ramp = np.linspace(-1.0, 1.0, bw, dtype=np.float32)[None, :]   # left→right
    bar = signed_diverging_heat(np.repeat(ramp, bh, axis=0),
                                np.ones((bh, bw), bool), span=1.0)
    # dark backing so the swatch + text read over any heat colour behind them
    cv2.rectangle(panel, (x0 - 4, y0 - 16), (x0 + bw + 4, y1 + 4), (0, 0, 0), -1)
    panel[y0:y0 + bh, x0:x0 + bw] = bar
    cv2.putText(panel, "NEAR", (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, "FAR", (x0 + bw - 34, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (200, 200, 255), 1, cv2.LINE_AA)
    return panel


def timeline_save(img, timeline_dir, frame, suffix, tag=""):
    """Archive `img` into a per-frame debug timeline: one PNG per step.

    Layout: <timeline_dir>/<frame>/<NNN>_<suffix>.png, where <NNN> is `tag` if
    given (e.g. 'iter03'), else a zero-padded auto-increment = the count of
    existing '*_<suffix>.png' in that per-frame subfolder. `suffix` keeps
    different panel kinds (composite/depth) from colliding in the same folder,
    so they can be scrubbed side by side. Shared by composite and depth.
    Returns the written path, or None when `timeline_dir`/`frame` is unset.
    """
    if not timeline_dir or not frame:
        return None
    sub = os.path.join(timeline_dir, frame)
    os.makedirs(sub, exist_ok=True)
    if tag:
        stem = str(tag)
    else:
        n = len([f for f in os.listdir(sub) if f.endswith(f"_{suffix}.png")])
        stem = f"{n:03d}"
    path = os.path.join(sub, f"{stem}_{suffix}.png")
    cv2.imwrite(path, img)
    print(f"[timeline] wrote {path}  ({img.shape[1]}x{img.shape[0]})")
    return path
