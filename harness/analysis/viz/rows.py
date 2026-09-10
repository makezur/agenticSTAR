"""rows.py — the ONE builder for a comparison row, and the column registry.

Every tool that shows "here is the source, here is what we rendered, here is
where they disagree" builds the same row out of the same columns:

    SOURCE | RENDER | SILHOUETTE [ | OUR DEPTH | DEPTH RESIDUAL ]

(MASKED SOURCE remains in the registry as an opt-in column.)

`composite.py` writes one such row per frame, `sweep_sides.py` one per rendered
candidate, and `candidate_sheet.py` stacks N of them into a page. Those three used
to each assemble their own columns, which is how the candidate sheets ended up
showing the tuned overlap colours with NO colour key while composite showed one.
The row is built here now, once, so a column tuned in one place is tuned
everywhere.

WHY A REGISTRY, not a fixed strip. Columns are declared in `COLUMNS` with the
inputs they need, so adding one (depth today, normals or a flow field tomorrow)
means one entry here rather than an edit in three tools. A caller asks for the
columns it wants; `build_row` supplies only those it has the inputs for and says
in `RowResult.missing` what it dropped, instead of failing a whole page over one
absent depth map.

Inputs arrive as ARRAYS, not paths (`RowInputs`), because the callers differ in
where pixels come from: the candidate sheet caches one decoded source per frame
across many rows, while composite loads from disk. `from_paths` covers the
load-from-disk case.
"""

import numpy as np

from analysis.lib import panels, rasters
from core import renderer_settings


class RowInputs:
    """The pixels a row can be built from — every field optional but `render`.

    `keep` is the non-hand keep-region (0/1) that scopes scoring; when it is None
    the row's overlap column drops the WAIVED swatch from its key, because
    without a hand region no pixel can be waived.
    """

    def __init__(self, render, source=None, source_mask=None, masked_source=None,
                 render_mask=None, keep=None, depth_panels=None):
        self.render = render
        self.source = source
        self.source_mask = source_mask
        self.masked_source = masked_source
        self.render_mask = render_mask
        self.keep = keep
        # {'our': Panel, 'residual': Panel, ...} from scorers.depth.depth_panels —
        # `panels.Panel` values: picture AND words, since the depth colormap, its
        # legend and its range/MAE lines are that module's business and duplicating
        # them here is what we are removing. Labeled by `build_row` like any other
        # column, so the bar is decided once for the whole row.
        self.depth_panels = depth_panels or {}


def from_paths(source_path, mask_path, render_path, hand_mask_path="",
               hand_dilate=0.0, bg_mode="black", depth_panels=None):
    """`RowInputs` loaded from disk — the composite / sweep_sides path.

    Folds the load → resize-mask → keep-region → `present` sequence that
    `composite.side_by_side_panel` and `candidate_sheet._SourceCache` each had
    their own copy of. `mask_path` may be empty for a row with no mask (composite
    without `--mask`), which simply yields no masked-source or overlap column.
    """
    source = rasters.load_bgr(source_path)
    render = rasters.load_render(render_path, bg_mode)
    if not mask_path:
        return RowInputs(render, source=source, depth_panels=depth_panels)

    source_mask = rasters.resize_mask_to(rasters.load_mask(mask_path), source.shape)
    keep = None
    visible = source_mask > 0
    if hand_mask_path:
        keep = rasters.build_keep(
            rasters.load_mask(hand_mask_path), source.shape, hand_dilate)
        visible &= keep > 0
    masked_source = renderer_settings.present(
        source, visible.astype(np.uint8) * 255, bg_mode)
    return RowInputs(
        render, source=source, source_mask=source_mask,
        masked_source=masked_source,
        render_mask=rasters.render_silhouette(render_path), keep=keep,
        depth_panels=depth_panels)


# --------------------------------------------------------------------------- #
# The column registry. `needs` names the RowInputs fields a column cannot be drawn
# without; `build(inputs, opts)` returns that column's UNLABELED picture at native
# resolution, which `build_row` scales and labels. Every column works this way,
# including the depth ones: a column that arrived pre-labeled would carry a bar
# sized without knowing what the rest of the row holds, which is exactly the
# decision that has to be made once per row (see `build_row`).
#
# `sub_key` names the opts entry holding this column's caption, and
# `panel_key`/`title_from_panel` let a column take its picture and title from a
# `panels.Panel` its producer built (the depth columns, whose colormaps and names
# live in `scorers.depth`).
# --------------------------------------------------------------------------- #
def _overlap(inputs, opts):
    return panels.overlay_panel(inputs.source_mask, inputs.render_mask,
                                keep=inputs.keep)


def _depth_panel(inputs, key):
    """The `panels.Panel` scorers.depth built for `key`, or None if absent."""
    return (inputs.depth_panels or {}).get(key)


def _depth_image(inputs, key):
    panel = _depth_panel(inputs, key)
    return None if panel is None else panel.image


COLUMNS = {
    "source": {
        "title": "SOURCE",
        "needs": ("source",),
        "build": lambda i, o: i.source,
    },
    "masked_source": {
        "title": "MASKED SOURCE",
        "needs": ("masked_source",),
        "build": lambda i, o: i.masked_source,
    },
    "render": {
        "title": "RENDER",
        "needs": ("render",),
        "build": lambda i, o: i.render,
        # the only column that takes a caption: which candidate it is. Identity
        # only — scores stay out of the pixels (see sweep_sides.row_sub).
        "sub_key": "render_sub",
    },
    "overlap": {
        "title": "SILHOUETTE",
        "needs": ("source_mask", "render_mask"),
        "build": _overlap,
        # drawn from the SAME constants the pixels were painted with, so the key
        # cannot drift from the panel — this is the whole reason the row is shared.
        "legend": True,
        # The swatches already say what this panel is, so the KEY REPLACES THE
        # TITLE: spending ~120px on the word "SILHOUETTE" is what forced the key
        # to shrink or wrap, and it must never wrap. The title comes back only
        # when no legend fits at all, so the panel is never unlabelled.
        "legend_replaces_title": True,
    },
    # The depth columns' pictures AND their words come from scorers.depth, which
    # owns the colormaps and the range/MAE/bias lines. They are labeled here like
    # every other column, so a depth panel beside a silhouette panel gets the same
    # bar as of right rather than by the two agreeing on a constant.
    "depth": {
        "needs": (),
        "panel_key": "our",
        "build": lambda i, o: _depth_image(i, "our"),
        "title_from_panel": True,
    },
    "depth_residual": {
        "needs": (),
        "panel_key": "residual",
        "build": lambda i, o: _depth_image(i, "residual"),
        "title_from_panel": True,
    },
}

# "silhouette" is what agents call the overlap panel (it IS the render silhouette
# against the mask). Accept both spellings for the same column rather than
# renaming a documented --columns value with tests and readers. "match" is the old
# name of the `render` column, kept for the same reason: it is the name of the VIEW
# that produces those pixels (`--views match`), so agents and existing docs/scripts
# type it, but on a panel beside SOURCE the word that reads is RENDER.
COLUMN_ALIASES = {"silhouette": "overlap", "match": "render"}
VALID_COLUMNS = frozenset(COLUMNS) | frozenset(COLUMN_ALIASES)

# The default comparison row. `composite`, `sweep_sides` and `candidate_sheet`
# all start here and subtract (`--no-overlap`) or add (depth) from it.
#
# WHY THIS ORDER. The three columns you read left-to-right are the ones that
# answer "does the render match the photo": SOURCE, the RENDER of it, and the
# SILHOUETTE that says where they disagree. MASKED SOURCE (the same photo,
# background blanked to the render's backdrop) is NOT in the default: a
# supporting panel that repeats pixels two columns already show costs a
# quarter of every row's width, so it is opt-in (`--columns ...,masked_source`
# — still in the registry, and still the whole point of composite's dedicated
# side-by-side strip). Column ORDER is only ever cosmetic here: every consumer
# names its columns, `build_row` draws them in the order asked, and the
# manifest records what was drawn.
DEFAULT_COLUMNS = ("source", "render", "overlap")

# Columns the image comparison tools can supply. ``depth`` is optional: old
# sweep reports and depth-disabled runs simply omit it, while panelled pose
# candidates can attach their own rendered-depth artifact without any observed
# Pi3X input. A SET of allowed names, so it does not imply display order.
IMAGE_COLUMNS = ("source", "render", "overlap", "masked_source", "depth")


def render_caption(display_rank, label=None):
    """The RENDER column's caption: WHICH candidate the panel shows. Nothing else.

    The one place this wording is built, because sharing the row builder only ever
    unified the PIXELS — `render_sub` is a free string, so each caller stayed free
    to compose its own caption, and `sweep_sides` used that freedom to stamp
    "combined=0.8066" onto the picture while `composite` documented the opposite
    ("the IoU is NOT stamped on the panel") and the sheets kept scores in the
    manifest. A shared builder that prints whatever it is handed is not a single
    source for the words.

    Scores stay out: a number burned into pixels cannot be sorted, diffed or
    recomputed, it goes stale the moment the metric changes, and it competes with
    the pixels the panel exists to show. Identity is what a panel can carry that
    the report cannot — the report has the numbers, and it has them exactly.

    `display_rank` is an int or "best"; `label` an optional group name (an
    object-report's gate).
    """
    head = "best" if display_rank == "best" else f"rank {int(display_rank):02d}"
    return f"{label}  {head}" if label else head


def canonical(name):
    """A column name with aliases resolved; raises on an unknown one."""
    if name not in VALID_COLUMNS:
        raise ValueError(
            f"unknown column: {name}; choose from {', '.join(sorted(COLUMNS))}")
    return COLUMN_ALIASES.get(name, name)


def parse_columns(value, drop=(), require=(), allowed=None):
    """A validated, de-aliased, duplicate-free column tuple from a CSV string.

    `drop` removes columns after parsing (the `--no-overlap` opt-out); `require`
    names columns the caller cannot render a row without; `allowed` narrows the
    registry to the columns THIS tool can actually supply inputs for, so asking a
    candidate sheet for `depth` is a clear error rather than a column that
    silently never appears."""
    names = [v.strip() for v in str(value).split(",") if v.strip()]
    vocabulary = (VALID_COLUMNS if allowed is None else
                  {c for c in VALID_COLUMNS if COLUMN_ALIASES.get(c, c) in allowed})
    unknown = [v for v in names if v not in vocabulary]
    if unknown:
        raise ValueError(
            f"unknown columns: {', '.join(unknown)}; choose from "
            f"{', '.join(sorted(allowed if allowed is not None else COLUMNS))}")
    names = [COLUMN_ALIASES.get(v, v) for v in names]
    names = [v for v in names if v not in {COLUMN_ALIASES.get(d, d) for d in drop}]
    for needed in require:
        if needed not in names:
            raise ValueError(f"these panels require the '{needed}' column")
    if len(names) != len(set(names)):
        raise ValueError("columns must not repeat")
    return tuple(names)


def available(inputs, columns):
    """(drawable, missing) split of `columns` against the inputs actually present.

    A column is drawable when every field in its `needs` is non-None, and — for the
    depth columns, whose pictures are supplied by `scorers.depth` rather than built
    from the raw inputs — when that picture was supplied. Lets one absent depth map
    cost one column, not the page."""
    drawable, missing = [], []
    for name in columns:
        spec = COLUMNS[canonical(name)]
        ok = all(getattr(inputs, field) is not None for field in spec["needs"])
        if ok and spec.get("panel_key") and spec["build"](inputs, {}) is None:
            ok = False
        (drawable if ok else missing).append(canonical(name))
    return drawable, missing


class RowResult:
    """A built row: the `image`, the `columns` actually drawn, what was `missing`,
    and the `legend` entries the overlap column ended up carrying.

    Callers that only want pixels use `.image`; the sheets record `columns` in
    their manifest so a page says what it showed. `legend` is None when no key was
    drawn (no overlap column, or a panel too small to carry one) and
    `legend_font_scale` is the type size it was drawn at — together they make the
    fit decision in `_fit_legend` observable, and the one-row invariant testable,
    instead of something you have to read off the pixels."""

    def __init__(self, image, columns, missing, legend=None,
                 legend_font_scale=None):
        self.image = image
        self.columns = list(columns)
        self.missing = list(missing)
        self.legend = legend
        self.legend_font_scale = legend_font_scale


def build_row(inputs, columns=DEFAULT_COLUMNS, height=512, render_sub="",
              sep_width=4, sep_bgr=(60, 60, 60), short_legend=None):
    """Build ONE comparison row and report which columns it drew.

    `columns` is the requested order; columns whose inputs are absent are skipped
    (see `available`) rather than raising, so a mask-less or depth-less row still
    produces the panels it can. `render_sub` is the RENDER caption — build it with
    `render_caption`, the one owner of that wording — and `sep_width`/`sep_bgr` the
    column rule.

    `height` is the height of the RENDERS, not of the returned row: each panel's
    header bar is stacked above its image, so the row comes out one bar taller —
    `panels.BAR_HEIGHT`, or `panels.CAPTIONED_BAR_HEIGHT` if some column's caption
    cannot ride on its title line, since then they all reserve the line (see
    `panels.label_row`). Sizing the renders (rather than the total) is what keeps the
    pictures the same size across tools whose captions differ.

    `short_legend` forces the abbreviated overlap wording; the default (None)
    MEASURES the panel and abbreviates (then shrinks the type) only as far as
    keeping the key on ONE row requires — see `_fit_legend`. That measurement is
    why a 4-column 360px-tall candidate row can carry the colour key at all.
    """
    drawable, missing = available(inputs, columns)
    if not drawable:
        raise ValueError(
            "no drawable columns: asked for "
            f"{', '.join(canonical(c) for c in columns) or '(none)'}")
    opts = {"render_sub": render_sub}
    strip, drawn_legend, drawn_font = [], None, None
    for name in drawable:
        spec = COLUMNS[name]
        image = panels.scale_to_height(spec["build"](inputs, opts), height)
        legend, font, ramp = None, None, None
        if spec.get("title_from_panel"):
            # picture and words both from the module that owns them
            # (scorers.depth / viz.depth_units) — INCLUDING any key it supplied
            # (the monocular DEPTH panel ships its NEAR→FAR ramp). `panels.label`
            # fits or drops them exactly as it does the overlap key.
            built = _depth_panel(inputs, spec["panel_key"])
            title, sub = built.title, built.sub
            legend, font = built.legend, built.legend_font_scale
            ramp = built.ramp_legend
        else:
            title, sub = spec["title"], opts.get(spec.get("sub_key") or "", "")
        if spec.get("legend"):
            # measured against the title this panel will ACTUALLY carry: dropping
            # the word frees the space that was forcing a smaller legend font.
            title = "" if spec.get("legend_replaces_title") else spec["title"]
            legend, font = _fit_legend(inputs.keep, image.shape[1], title,
                                       short_legend)
            drawn_legend, drawn_font = legend, font
            if legend is None:
                title = spec["title"]        # no key fits: name the panel instead
        strip.append(panels.Panel(image, title, sub, legend, font,
                                  ramp_legend=ramp))
    # `label_row` gives every panel here the SAME bar: the caption line exists iff
    # some column's caption cannot ride beside its title. So a captioned RENDER
    # cannot ride lower than its neighbours, an identity caption ("rank 00") costs
    # the row nothing, and a row with nothing to caption — every candidate sheet —
    # wastes no empty band under its titles.
    return RowResult(
        panels.hstack_panels(panels.label_row(strip),
                             sep_width=sep_width, sep_bgr=sep_bgr),
        drawable, missing, legend=drawn_legend, legend_font_scale=drawn_font)


def _fit_legend(keep, width, title, forced):
    """(entries, font_scale) for the most readable overlap key this panel can carry.

    A SINGLE ROW IS AN INVARIANT, not a preference — a key you have to read in two
    passes has stopped being a key and become a caption. Everything else gives way
    to that, in order:

      1. full wording ("MAYBE OCCLUDED") at the largest font that fits one row
      2. short wording ("HAND") at the largest font that fits one row
      3. no key — a panel too narrow for even the short form at the smallest type

    Words shrink before type does, and type shrinks before anything wraps: the
    COLOURS are what you actually read off the key, and they are identical in every
    case. `panels.label` cannot wrap even if asked, so this cannot regress into a
    two-row bar by some caller passing an unmeasured legend.

    (None, None) when nothing fits, which is the caller's cue to title the panel.
    """
    for short in ((True,) if forced else (False, True)):
        entries = panels.overlap_legend(keep, short=short)
        font = panels.legend_fits(entries, width, title=title)
        if font:
            return entries, font
    return None, None
