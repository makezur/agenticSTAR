"""Exact per-pixel z-buffer rasterizer for projected triangle meshes.

`rasterize` keeps a real depth buffer: every triangle's pixels are
depth-tested individually, like the z-buffer behind the harness's rendered
`depth.npy`, so mask ownership and rendered depth agree. This matters especially for closed,
interpenetrating hand/object meshes: sorting whole triangles by mean depth can
assign a contact-boundary pixel to a surface that is actually behind another.

Conventions:
  * `entities` = [(id, p2d (V,2), z (V,), faces (T,3)), ...]; `id` is a small
    positive uint8 SEG id (<= 255), 0 stays background. Later entities do NOT
    win ties — an exact depth tie goes to the SMALLER id, so the output never
    depends on the order the entities were passed in.
  * OpenCV pixel coordinates: the center of pixel (col, row) is the integer
    (col, row), and coverage is sampled there (as a GL-style z-buffer does,
    unlike cv2.fillConvexPoly on rounded vertices).
  * z is the positive planar camera depth; 1/z is interpolated (that is what is
    linear in screen space), so `depth` is perspective-correct rather than a
    barycentric blend of z.

Pure numpy — no cv2, GL, or torch. It is vectorized over all triangles at once
and chunked to bound peak memory rather than looping in Python.
"""

import numpy as np

# packed sort key layout: pixel index | float32 depth bits | seg id
_ID_BITS = 8
_DEPTH_BITS = 32
_PIX_SHIFT = np.uint64(_ID_BITS + _DEPTH_BITS)
_MAX_PIXELS = 1 << (64 - _ID_BITS - _DEPTH_BITS)


def valid_faces(p2d, z, faces, W, H):
    """Faces fully in front of the camera and not absurdly off-screen.

    The off-screen limit bounds the candidate-pixel budget for triangles that
    project to enormous screen areas; it is the same 4*max(W,H) box the
    painter's-algorithm rasterizers used, so the visible set is unchanged.
    """
    zf = z[faces]
    pf = p2d[faces]
    lim = 4 * max(W, H)
    ok = (zf > 1e-3).all(axis=1)
    ok &= (np.abs(pf) < lim).all(axis=(1, 2))
    return faces[ok]


def gather(entities, W, H):
    """entities -> (uv (T,3,2) f64, z (T,3) f64, ids (T,) uint8): the
    camera-facing triangles of every entity, concatenated in argument order.

    `rasterize`'s `tri` output indexes this list, so callers that need
    per-triangle attributes (shading normals, part labels) build them here.
    """
    uv_l, z_l, id_l = [], [], []
    for seg_id, p2d, z, faces in entities:
        if not 0 < int(seg_id) < (1 << _ID_BITS):
            raise ValueError(f"seg id {seg_id} outside 1..255 (0 = background)")
        p2d = np.ascontiguousarray(p2d, np.float64)
        z = np.ascontiguousarray(z, np.float64)
        f = valid_faces(p2d, z, np.asarray(faces), W, H)
        if not len(f):
            continue
        uv_l.append(p2d[f])
        z_l.append(z[f])
        id_l.append(np.full(len(f), seg_id, np.uint8))
    if not uv_l:
        return (np.zeros((0, 3, 2)), np.zeros((0, 3)), np.zeros(0, np.uint8))
    return (np.concatenate(uv_l), np.concatenate(z_l), np.concatenate(id_l))


def _bboxes(uv, W, H):
    """Integer pixel-center bboxes, clipped to the image: (x0, y0, w, h)."""
    lo = np.ceil(uv.min(axis=1))
    hi = np.floor(uv.max(axis=1))
    x0 = np.clip(lo[:, 0], 0, W).astype(np.int64)
    x1 = np.clip(hi[:, 0], -1, W - 1).astype(np.int64)
    y0 = np.clip(lo[:, 1], 0, H).astype(np.int64)
    y1 = np.clip(hi[:, 1], -1, H - 1).astype(np.int64)
    return (x0, y0, np.maximum(x1 - x0 + 1, 0), np.maximum(y1 - y0 + 1, 0))


def _samples(uv, z, ids, W, H):
    """Covered pixel-center samples of a triangle batch -> (keys, tri).

    keys = pixel << 40 | float32_depth_bits << 8 | seg_id, so ascending order
    groups by pixel and, within a pixel, puts the nearest surface (then the
    smallest id) first — IEEE-754 positive floats order the same as their bit
    patterns read as unsigned ints. `tri` is the row of `uv` each key came
    from, parallel to `keys`.
    """
    x0, y0, bw, bh = _bboxes(uv, W, H)
    n = bw * bh
    idx = np.flatnonzero(n > 0)
    empty = (np.zeros(0, np.uint64), np.zeros(0, np.int64))
    if not len(idx):
        return empty
    n_k = n[idx]

    # expand every bbox into its pixel centers without a python loop
    tri = np.repeat(idx, n_k)
    k = np.arange(int(n_k.sum()), dtype=np.int64) \
        - np.repeat(np.cumsum(n_k) - n_k, n_k)
    w = bw[idx].repeat(n_k)
    px = x0[tri] + k % w
    py = y0[tri] + k // w

    # per-TRIANGLE coefficients (T-sized, not sample-sized): with dx,dy taken
    # from vertex c,  l1 = a1*dx + b1*dy,  l2 = a2*dx + b2*dy,  and
    # 1/z = ic + l1*(ia - ic) + l2*(ib - ic).
    a, b, c = uv[:, 0], uv[:, 1], uv[:, 2]
    den = ((b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0])
           + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1]))
    good = np.abs(den) > 1e-12          # degenerate (zero-area) projection
    den = np.where(good, den, 1.0)
    a1, b1 = (b[:, 1] - c[:, 1]) / den, (c[:, 0] - b[:, 0]) / den
    a2, b2 = (c[:, 1] - a[:, 1]) / den, (a[:, 0] - c[:, 0]) / den
    iz = 1.0 / z
    dia, dib, ic = iz[:, 0] - iz[:, 2], iz[:, 1] - iz[:, 2], iz[:, 2]

    # float32 per sample: 1e-6 slack is ~1/1000 pixel, far above its precision
    dx = (px - c[tri, 0]).astype(np.float32)
    dy = (py - c[tri, 1]).astype(np.float32)
    l1 = a1[tri].astype(np.float32) * dx + b1[tri].astype(np.float32) * dy
    l2 = a2[tri].astype(np.float32) * dx + b2[tri].astype(np.float32) * dy
    eps = np.float32(-1e-6)
    inside = good[tri] & (l1 >= eps) & (l2 >= eps) \
        & (l1 + l2 <= np.float32(1) - eps)
    if not inside.any():
        return empty

    tri, l1, l2 = tri[inside], l1[inside], l2[inside]
    inv = (ic[tri].astype(np.float32) + l1 * dia[tri].astype(np.float32)
           + l2 * dib[tri].astype(np.float32))
    hit = inv > np.float32(0)           # sample behind the camera plane
    if not hit.any():
        return empty
    tri = tri[hit]
    depth = np.float32(1) / inv[hit]
    pix = (py[inside][hit] * W + px[inside][hit]).astype(np.uint64)
    keys = ((pix << _PIX_SHIFT)
            | (depth.view(np.uint32).astype(np.uint64) << np.uint64(_ID_BITS))
            | ids[tri].astype(np.uint64))
    return keys, tri


def _chunks(uv, W, H, chunk_px):
    """Triangle-index ranges whose candidate-pixel bboxes sum to <= chunk_px
    (each range holds at least one triangle, so one huge near-camera triangle
    cannot blow the working set)."""
    _, _, bw, bh = _bboxes(uv, W, H)
    span = (bw * bh).astype(np.float64)
    out, acc, start = [], 0.0, 0
    for i, s in enumerate(span):
        if i > start and acc + s > chunk_px:
            out.append((start, i))
            start, acc = i, 0.0
        acc += s
    out.append((start, len(span)))
    return out


def rasterize(entities, W, H, chunk_px=4_000_000):
    """Z-buffer `entities` into (seg, depth, tri).

      seg   (H,W) uint8   nearest entity id per pixel, 0 = background
      depth (H,W) float32 nearest camera depth, same units as the input z;
                          0.0 = background
      tri   (H,W) int32   winning triangle as a row of `gather(entities,W,H)`,
                          -1 = background

    `chunk_px` bounds the candidate-pixel working set per batch (~40 B each);
    the result does not depend on where the batch boundaries fall.
    """
    uv, z, ids = gather(entities, W, H)
    seg = np.zeros(H * W, np.uint8)
    depth = np.zeros(H * W, np.float32)
    tri_out = np.full(H * W, -1, np.int32)
    shaped = lambda: (seg.reshape(H, W), depth.reshape(H, W),
                      tri_out.reshape(H, W))
    if not len(uv):
        return shaped()
    if H * W > _MAX_PIXELS:
        raise ValueError(f"{W}x{H} exceeds the packed z-buffer key's "
                         f"{_MAX_PIXELS} pixels")

    batches = [_samples(uv[lo:hi], z[lo:hi], ids[lo:hi], W, H) + (lo,)
               for lo, hi in _chunks(uv, W, H, chunk_px)]
    keys = np.concatenate([k for k, _, _ in batches])
    tri = np.concatenate([t + lo for _, t, lo in batches])
    if not len(keys):
        return shaped()

    order = np.argsort(keys, kind="stable")   # by pixel, then depth, then id
    keys, tri = keys[order], tri[order]
    first = np.ones(len(keys), bool)          # nearest sample of each pixel run
    pix = keys >> _PIX_SHIFT
    first[1:] = pix[1:] != pix[:-1]

    won, p = keys[first], pix[first].astype(np.int64)
    seg[p] = (won & np.uint64((1 << _ID_BITS) - 1)).astype(np.uint8)
    depth[p] = ((won >> np.uint64(_ID_BITS)) & np.uint64(0xFFFFFFFF)) \
        .astype(np.uint32).view(np.float32)
    tri_out[p] = tri[first]
    return shaped()
