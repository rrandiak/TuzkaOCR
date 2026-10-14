from __future__ import annotations

import cv2
import numpy as np

from .postprocess import (Region, TextLine, _baseline_to_polygon_normal, _extract_lines,
                          _merge_line_fragments, _order_regions, _region_hull)
from .reading_order import reorder_page

TRIGGER = 4.0
CAND_MIN = 3.0
GLYPH_W = (0.1, 3.0)
GLYPH_FILL = 0.15
GLYPH_ALIGN = 0.25
GLYPH_GAP = 2.0
GLYPH_MAX = 0.25
GLYPH_MIN = 3
GROW = 1.4
COARSE_TARGET = 40.0
F_MIN, F_MAX = 2.5, 8.0
MIN_COARSE_W = 160
FRAG_REL = 0.4
FRAG_MED = 3.0
FRAG_MAX = 1.3
EDGE_MARGIN = 0.1
INK_FRAC = 0.02
INK_GAP = 0.35
COVER_MAX = 0.7
SMALL_MAX = 0.5
MERGE_DY = 0.3
MERGE_GAP = 1.5
PAD = 0.08
ASC_PAD = 0.25
STRETCH = 1.3
BASELINE_PTS = 9
END_PAD = 0.1


def _h(ln) -> float:
    return max(1.0, float(ln.heights[0] + ln.heights[1]))


def _geom(ln):
    xs = [p[0] for p in ln.baseline]
    ys = [p[1] for p in ln.baseline]
    cy = float(np.mean(ys))
    return float(min(xs)), float(max(xs)), cy


def _box(ln):
    x0, x1, cy = _geom(ln)
    return x0, cy - ln.heights[0], x1, cy + ln.heights[1]


def _poly(bl, asc, desc):
    return _baseline_to_polygon_normal(bl, asc, desc, min_asc=1.0, min_desc=1.0)


def _scale_line(ln, k):
    return TextLine(baseline=[(x * k, y * k) for x, y in ln.baseline],
                    polygon=[(x * k, y * k) for x, y in ln.polygon],
                    heights=(ln.heights[0] * k, ln.heights[1] * k))


def _merge_candidates(cands):
    cands = sorted(cands, key=lambda c: _geom(c)[0])
    out = []
    for c in cands:
        if out:
            p = out[-1]
            px0, px1, pcy = _geom(p)
            cx0, cx1, ccy = _geom(c)
            h = 0.5 * (_h(p) + _h(c))
            if abs(pcy - ccy) <= MERGE_DY * h and cx0 - px1 <= MERGE_GAP * h:
                bl = sorted(list(p.baseline) + list(c.baseline))
                asc = max(p.heights[0], c.heights[0])
                desc = max(p.heights[1], c.heights[1])
                out[-1] = TextLine(baseline=bl, polygon=_poly(bl, asc, desc), heights=(asc, desc))
                continue
        out.append(c)
    return out


def _inside(fine_lines, x0, y0, x1, y1):
    out = []
    for i, ln in enumerate(fine_lines):
        fx0, fx1, fcy = _geom(ln)
        if x0 <= 0.5 * (fx0 + fx1) <= x1 and y0 <= fcy <= y1:
            out.append(i)
    return out


def _ink_extend(gray, cy, asc, x0, x1, lo, hi, h):
    if gray is None:
        return x0, x1
    H, W = gray.shape[:2]
    r0, r1 = int(max(0, cy - 0.8 * asc)), int(min(H, cy))
    if r1 - r0 < 2:
        return x0, x1
    band = gray[r0:r1].astype(np.float32)
    thr = 0.5 * (float(np.percentile(band, 5)) + float(np.percentile(band, 95)))
    ink = (band < thr).mean(axis=0) > INK_FRAC
    gap = max(1, int(INK_GAP * h))
    lo, hi = max(0, int(lo)), min(W - 1, int(hi))

    def walk(x, step, stop):
        last, run, x = x, 0, x + step
        while (x >= stop if step < 0 else x <= stop):
            if ink[x]:
                last, run = x, 0
            else:
                run += 1
                if run > gap:
                    break
            x += step
        return last

    nx0 = walk(int(np.clip(x0, lo, hi)), -1, lo)
    nx1 = walk(int(np.clip(x1, lo, hi)), 1, hi)
    return float(min(x0, nx0)), float(max(x1, nx1))


def _glyphs(gray, fine_lines, med):
    if gray is None:
        return []
    g = np.asarray(gray)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cen = [(0.5 * (a + b), cy) for a, b, cy in (_geom(ln) for ln in fine_lines)]
    cand = []
    for x, y, w, h, area in (stats[i] for i in range(1, n)):
        if not (TRIGGER * med <= h <= GLYPH_MAX * g.shape[0]):
            continue
        if not (GLYPH_W[0] * h <= w <= GLYPH_W[1] * h) or area < GLYPH_FILL * w * h:
            continue
        if any(x <= cx <= x + w and y <= cy <= y + h for cx, cy in cen):
            continue
        cand.append((float(x), float(y), float(x + w), float(y + h)))

    def aligned(a, b):
        ha, hb = a[3] - a[1], b[3] - b[1]
        gap = max(a[0], b[0]) - min(a[2], b[2])
        return (abs(a[3] - b[3]) <= GLYPH_ALIGN * max(ha, hb) and 0.5 <= ha / hb <= 2.0
                and gap <= GLYPH_GAP * max(ha, hb))
    return [a for a in cand if any(a is not b and aligned(a, b) for b in cand)]


def _coarse_candidates(lines, med, fine=()):
    big = [ln for ln in fine if _h(ln) >= TRIGGER * med]

    def keep(c):
        if _h(c) >= TRIGGER * med:
            return True
        if _h(c) < CAND_MIN * med:
            return False
        x0, y0, x1, y1 = _box(c)
        return bool(_inside(big, x0, y0, x1, y1))
    return _merge_candidates([c for c in lines if keep(c)])


def _accept(cand, fine_lines, med, gray=None, glyphs=()):
    x0, y0, x1, y1 = _box(cand)
    hc = _h(cand)
    cy = _geom(cand)[2]
    inside = _inside(fine_lines, x0, y0, x1, y1)
    big_h = max(FRAG_REL * hc, FRAG_MED * med)
    big = [i for i in inside if _h(fine_lines[i]) >= big_h]
    if big:
        if float(np.median([_h(fine_lines[i]) for i in big])) > FRAG_MAX * hc:
            return None
        bx0 = min(_geom(fine_lines[i])[0] for i in big)
        bx1 = max(_geom(fine_lines[i])[1] for i in big)
    else:
        gl = [b for b in glyphs if x0 <= 0.5 * (b[0] + b[2]) <= x1 and y0 <= 0.5 * (b[1] + b[3]) <= y1
              and b[3] - b[1] >= FRAG_REL * hc]
        if len(gl) < GLYPH_MIN:
            return None
        bx0, bx1 = min(b[0] for b in gl), max(b[2] for b in gl)
    lo, hi = -1e18, 1e18
    for i, ln in enumerate(fine_lines):
        fx0, fx1, fcy = _geom(ln)
        if _h(ln) >= big_h or not (y0 <= fcy <= y1):
            continue
        if fx1 <= bx0:
            lo = max(lo, fx1)
        elif fx0 >= bx1:
            hi = min(hi, fx0)
    margin = EDGE_MARGIN * hc
    lo, hi = lo + margin, hi - margin
    x0, x1 = max(x0, lo, -1e17), min(x1, hi)
    x0, x1 = min(x0, bx0), max(x1, bx1)
    x0, x1 = _ink_extend(gray, cy, cand.heights[0], x0, x1, max(lo, 0), min(hi, 1e9), hc)
    w = max(1.0, x1 - x0)
    inside = _inside(fine_lines, x0, y0, x1, y1)
    big = [i for i in inside if _h(fine_lines[i]) >= big_h]
    small = [i for i in inside if i not in big]
    ivs = sorted((max(x0, _geom(fine_lines[i])[0]), min(x1, _geom(fine_lines[i])[1])) for i in big)
    cover, cur = 0.0, -1e18
    for a, b in ivs:
        a = max(a, cur)
        if b > a:
            cover += b - a
        cur = max(cur, b)
    if len(big) < 2 and cover / w >= COVER_MAX:
        return None
    if sum(_geom(fine_lines[i])[1] - _geom(fine_lines[i])[0] for i in small) > SMALL_MAX * w:
        return None
    return x0, x1, inside


def _select(cands, fine, med, gray, glyphs):
    out = []
    for c in sorted(cands, key=_h, reverse=True):
        res = _accept(c, fine, med, gray, glyphs)
        if res is None:
            continue
        x0, x1, inside = res
        _, y0, _, y1 = _box(c)
        if any(_overlap((x0, y0, x1, y1), (a0, _box(a)[1], a1, _box(a)[3])) > 0.5 for a, a0, a1, _ in out):
            continue
        out.append((c, x0, x1, inside))
    return out


def _overlap(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / max(small, 1e-6)


def _clip_baseline(bl, x0, x1, n=BASELINE_PTS):
    pts = sorted((float(x), float(y)) for x, y in bl)
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    out = [(x, float(np.interp(x, xs, ys))) for x in np.linspace(x0, x1, n)]
    return [(int(round(x)), int(round(y))) for x, y in out]


def _order(regions, maps, gray, reading_order):
    if reading_order == "columns":
        return reorder_page(regions, np.maximum(maps[:, :, 4], 0), gray)
    return _order_regions(regions)


def apply_large_text(detector, img_bgr, regions, maps, gray, img_scale, reading_order="basic"):
    fine = [ln for r in regions for ln in r.lines]
    if len(fine) < 3:
        return regions
    med = float(np.median([_h(ln) for ln in fine]))
    trig = [_h(ln) for ln in fine if _h(ln) >= TRIGGER * med]
    glyphs = _glyphs(gray, fine, med)
    if not trig and len(glyphs) >= GLYPH_MIN:
        trig = [(b[3] - b[1]) / GROW for b in glyphs]
    if not trig:
        return regions
    f = float(np.clip(GROW * np.median(trig) / COARSE_TARGET, F_MIN, F_MAX))
    wf = maps.shape[1]
    wc = int(round(wf / f))
    if wc < MIN_COARSE_W:
        wc = min(wf, MIN_COARSE_W)
    cmaps, cscale, _ = detector.get_maps_at_width(img_bgr, wc)
    k = cscale / img_scale
    clines = _extract_lines(cmaps)
    clines = _merge_line_fragments(clines, np.maximum(cmaps[:, :, 4], 0))
    cands = [_scale_line(ln, k) for ln in clines]
    cands = _coarse_candidates(cands, med, fine)
    drop, new_regions = set(), []
    for c, x0, x1, inside in _select(cands, fine, med, gray, glyphs):
        drop.update(id(fine[i]) for i in inside)
        asc, desc = c.heights[0] * (1 + ASC_PAD), c.heights[1] * (1 + PAD)
        bl = _clip_baseline(c.baseline, x0 - PAD * _h(c), x1 + PAD * _h(c))
        ln = TextLine(baseline=bl, polygon=[(int(x), int(y)) for x, y in _poly(bl, asc, desc)],
                      heights=(asc, desc), large=True)
        new_regions.append(Region(lines=[ln], polygon=_region_hull([ln])))
    if not new_regions:
        return regions
    kept = []
    for r in regions:
        lines = [ln for ln in r.lines if id(ln) not in drop]
        if not lines:
            continue
        if len(lines) != len(r.lines):
            r = Region(lines=lines, polygon=_region_hull(lines))
        kept.append(r)
    return _order(kept + new_regions, maps, gray, reading_order)


def stretch_crop(gray, M):
    w = max(1, int(round(gray.shape[1] * STRETCH)))
    k = w / gray.shape[1]
    gray = cv2.resize(gray, (w, gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    p = int(round(END_PAD * gray.shape[0]))
    gray = cv2.copyMakeBorder(gray, 0, 0, p, p, cv2.BORDER_CONSTANT, value=int(np.percentile(gray, 95)))
    shift = np.array([[1.0, 0.0, -p], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return gray, M @ np.diag([1.0 / k, 1.0, 1.0]) @ shift
