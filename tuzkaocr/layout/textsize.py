from __future__ import annotations

import cv2
import numpy as np

WORK_W = 1200
FINE_W_MIN = 1600
FINE_BELOW = 7.0
FINE_AREA = 1000 * 1500
MIN_H = 4

C_XH = 2.019
C_WORD = 1.309
C_WORD_P25 = 1.403
C_PITCH = 0.918
LO_PITCH = 0.501


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _resize_w(g: np.ndarray, w: int) -> tuple[np.ndarray, float]:
    H, W = g.shape[:2]
    if W <= w:
        return g, 1.0
    s = w / W
    src = g
    while src.shape[1] >= 2 * w:
        src = cv2.pyrDown(src)
    return cv2.resize(src, (w, max(1, int(round(H * s)))), interpolation=cv2.INTER_AREA), s


def _binarise(g: np.ndarray) -> np.ndarray:
    k = max(15, (min(g.shape) // 40) | 1)
    small = cv2.resize(g, None, fx=max(0.25, 1.0 / g.shape[1]),
                       fy=max(0.25, 1.0 / g.shape[0]), interpolation=cv2.INTER_AREA)
    ks = max(5, (k // 4) | 1)
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks)))
    bg = cv2.blur(bg, (ks, ks))
    bg = cv2.resize(bg, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    norm = cv2.divide(g, np.maximum(bg, 1), scale=255)
    t = min(cv2.threshold(norm, 0, 255, cv2.THRESH_OTSU)[0], 200)
    return (norm < t).astype(np.uint8)


def _log_mode(vals: np.ndarray, bw: float = 0.06) -> tuple[float, float]:
    lv = np.log(vals.astype(np.float64))
    step = bw / 4
    lo = lv.min() - 3 * bw
    n = int((lv.max() + 3 * bw - lo) / step) + 1
    idx = np.clip(((lv - lo) / step).astype(int), 0, n - 1)
    hist = np.bincount(idx, minlength=n).astype(np.float64)
    ker = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
    sm = np.convolve(hist, ker, mode="same")
    j = int(np.argmax(sm))
    return float(np.exp(lo + j * step)), float(sm[max(0, j - 4):j + 5].sum() / max(sm.sum(), 1e-9))


def _xheight(b: np.ndarray) -> tuple[float, float, int]:
    H, W = b.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    st = st[1:]
    w, h, a = st[:, 2], st[:, 3], st[:, 4]
    fill = a / np.maximum(w * h, 1)
    keep = ((h >= MIN_H) & (h <= H / 15) & (w <= W / 10) & (w <= 4 * h) & (w * 8 >= h)
            & (fill > 0.12) & (fill < 0.95))
    hk = h[keep]
    if hk.size < 20:
        return float("nan"), 0.0, int(hk.size)
    xh, frac = _log_mode(hk)
    return xh, frac, int(hk.size)


def _word_heights(b: np.ndarray, xh: float) -> tuple[float, float]:
    kw = max(3, int(round(xh * 0.9)))
    sm = cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))
    _, _, st, _ = cv2.connectedComponentsWithStats(sm, connectivity=8)
    w, h = st[1:, 2], st[1:, 3]
    hk = h[(h >= xh * 0.9) & (h <= xh * 5) & (w >= 2.0 * h)]
    if hk.size < 5:
        return float("nan"), float("nan")
    return float(np.median(hk)), float(np.percentile(hk, 25))


def _pitch(b: np.ndarray, strips: int = 6) -> float:
    H, W = b.shape
    lo, hi = 5, max(8, min(H // 6, W // 8))
    if H <= lo:
        return float("nan")
    acc = np.zeros(hi + 1)
    for i in range(strips):
        prof = b[:, W * i // strips:W * (i + 1) // strips].sum(axis=1).astype(np.float64)
        prof -= cv2.blur(prof.reshape(-1, 1), (1, int(hi * 2) | 1)).ravel()
        e = float((prof ** 2).sum())
        if e <= 0:
            continue
        f = np.fft.rfft(prof, n=2 * H)
        acc += np.fft.irfft(f * np.conj(f))[:hi + 1] / e
    seg = acc[lo:hi + 1]
    if seg.size < 3 or not np.any(seg > 0):
        return float("nan")
    jmax = int(np.argmax(seg))
    peaks = [k for k in range(1, seg.size - 1)
             if seg[k] >= seg[k - 1] and seg[k] >= seg[k + 1] and seg[k] >= 0.6 * seg[jmax]]
    j = peaks[0] if peaks else jmax
    off = 0.0
    if 0 < j < seg.size - 1:
        d = seg[j - 1] - 2 * seg[j] + seg[j + 1]
        if d != 0:
            off = 0.5 * (seg[j - 1] - seg[j + 1]) / d
    return float(lo + j + off)


def features(img: np.ndarray) -> dict:
    g = _to_gray(img)
    W0 = g.shape[1]
    gs, s = _resize_w(g, WORK_W)
    b = _binarise(gs)
    xh, frac, ncc = _xheight(b)
    if np.isfinite(xh) and xh < FINE_BELOW and W0 > gs.shape[1] * 1.2:
        s = 1.0
        while W0 * s / 2 >= FINE_W_MIN:
            s /= 2
        H0 = g.shape[0]
        side = int(np.sqrt(FINE_AREA) / s)
        cw, ch = min(W0, side), min(H0, int(FINE_AREA / s / s / min(W0, side)))
        x0, y0 = (W0 - cw) // 2, (H0 - ch) // 2
        gs = g[y0:y0 + ch, x0:x0 + cw]
        while gs.shape[1] > cw * s * 1.5:
            gs = cv2.pyrDown(gs)
        s = gs.shape[1] / cw
        b = _binarise(gs)
        xh, frac, ncc = _xheight(b)
    word, word_p25 = _word_heights(b, xh) if np.isfinite(xh) else (float("nan"),) * 2
    pitch = _pitch(b)
    return {"scale": s, "n_cc": ncc, "xh_frac": frac, "xh_orig": xh / s, "word_orig": word / s,
            "word_p25_orig": word_p25 / s, "pitch_orig": pitch / s}


def combine(f: dict) -> tuple[float, float]:
    cues = np.array([f["xh_orig"] * C_XH, f["word_orig"] * C_WORD, f["word_p25_orig"] * C_WORD_P25])
    cues = cues[np.isfinite(cues) & (cues > 0)]
    p = f["pitch_orig"]
    if cues.size == 0:
        return (float(p * C_PITCH), 0.1) if np.isfinite(p) else (float("nan"), 0.0)
    est = float(np.median(cues))
    if np.isfinite(p):
        est = max(est, LO_PITCH * p)
    spread = float(np.log(cues.max() / cues.min())) if cues.size > 1 else 0.7
    conf = float(np.exp(-spread / 0.3)) * (cues.size / 3)
    return est, conf


def estimate_line_height(img: np.ndarray) -> tuple[float, float]:
    return combine(features(img))
