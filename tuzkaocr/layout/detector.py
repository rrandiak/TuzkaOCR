from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .large_text import apply_large_text
from .postprocess import maps_to_regions, Region
from .textsize import estimate_line_height

DOWNSAMPLE         = 3
MAX_ORIGINAL_WIDTH = 1500
MAX_SIDE           = 1536
WIDE_ASPECT        = 1.2     
GUTTER_RATIO       = 0.25    
MIN_SCALED_WIDTH   = 300
MAX_SCALED_PIXELS  = 6.0e6
MIN_BASELINE_PIXELS = 50
TEXT_HEIGHT_GAIN   = 1.07
TEXT_HEIGHT_MIN_CONF = 0.3
TEXT_HEIGHT_MIN_WIDTH_FRAC = 0.006
TEXT_HEIGHT_RECHECK = 5.0 / 3.0
                            

def _has_central_gutter(img_bgr: np.ndarray) -> bool:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    if W < 10:
        return False
    ink = (g < 128).mean(axis=0)
    k = max(3, W // 200)
    ink = np.convolve(ink, np.ones(k) / k, mode="same")
    med = float(np.median(ink[int(0.05 * W):int(0.95 * W)])) + 1e-6
    c0, c1 = int(0.42 * W), int(0.58 * W)
    return float(ink[c0:c1].min()) < GUTTER_RATIO * med


def _median_line_height(maps: np.ndarray) -> float | None:
    base = maps[..., 2] > 0.5
    if int(base.sum()) <= MIN_BASELINE_PIXELS:
        return None
    return float(np.median((maps[..., 0] + maps[..., 1])[base]))


def _scaled_width(w: int, h: int, line_height: float, target: float) -> int:
    width = w * target / line_height
    width = min(width, (MAX_SCALED_PIXELS * w / h) ** 0.5)
    width = max(min(width, w), min(MIN_SCALED_WIDTH, w))
    return max(1, int(round(width)))


def _sigmoid_inplace(x: np.ndarray) -> None:
    np.negative(x, out=x)
    np.exp(x, out=x)
    x += 1.0
    np.reciprocal(x, out=x)


class LayoutDetector:
    def __init__(self, model_path: str | Path, device: str = "cpu", threads: int = 4,
                 cpu_mem_arena: bool = True, column_split: bool = False,
                 reading_order: str = "basic", large_text: bool = False):
        self.column_split = column_split
        self.reading_order = reading_order
        self.large_text = large_text
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = max(1, threads // 2)
        opts.enable_cpu_mem_arena = cpu_mem_arena

        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=providers
        )
        self._input_name = self.session.get_inputs()[0].name
        self._needs_sigmoid = self._probe_sigmoid()
        print(f"[layout] model={model_path} sigmoid_in_code={self._needs_sigmoid}", flush=True)

    def _probe_sigmoid(self) -> bool:
        dummy = np.zeros((1, 3, 64, 64), dtype=np.float32)
        out = self.session.run(None, {self._input_name: dummy})[0]
        ch = out[0, 2:5]
        return bool(ch.min() < -0.001 or ch.max() > 1.001)

    def get_maps(self, img_bgr: np.ndarray, downsample: int | None = None) -> tuple[np.ndarray, float]:
        orig_h, orig_w = img_bgr.shape[:2]
        h, w = orig_h, orig_w

        if w > MAX_ORIGINAL_WIDTH:
            pre_scale = MAX_ORIGINAL_WIDTH / w
            img_bgr = cv2.resize(img_bgr,
                                 (MAX_ORIGINAL_WIDTH, max(1, int(h * pre_scale))),
                                 interpolation=cv2.INTER_AREA)
            h, w = img_bgr.shape[:2]

        ds     = downsample or DOWNSAMPLE
        img_ds = cv2.resize(img_bgr, (max(1, w // ds), max(1, h // ds)), interpolation=cv2.INTER_AREA)
        hd, wd = img_ds.shape[:2]

        if max(hd, wd) > MAX_SIDE:
            scale  = MAX_SIDE / max(hd, wd)
            img_ds = cv2.resize(img_ds,
                                (max(1, int(wd * scale)), max(1, int(hd * scale))),
                                interpolation=cv2.INTER_AREA)
            hd, wd = img_ds.shape[:2]

        img_scale = orig_h / hd

        ph = (32 - hd % 32) % 32
        pw = (32 - wd % 32) % 32
        img_pad = (np.pad(img_ds, ((0, ph), (0, pw), (0, 0)), mode="edge")
                   if ph or pw else img_ds)

        tensor = (img_pad.astype(np.float32).transpose(2, 0, 1)[None]) / 255.0

        out = self.session.run(None, {self._input_name: tensor})[0]
        if self._needs_sigmoid:
            _sigmoid_inplace(out[:, 2:5])

        maps = out[0, :, :hd, :wd].transpose(1, 2, 0)
        gray = cv2.cvtColor(img_ds, cv2.COLOR_BGR2GRAY)
        return np.ascontiguousarray(maps, dtype=np.float32), img_scale, gray

    def get_maps_at_width(self, img_bgr: np.ndarray, width: int) -> tuple[np.ndarray, float, np.ndarray]:
        orig_h, orig_w = img_bgr.shape[:2]
        height = max(1, int(round(orig_h * width / orig_w)))
        img = (img_bgr if (width, height) == (orig_w, orig_h)
               else cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA))
        ph = (32 - height % 32) % 32
        pw = (32 - width % 32) % 32
        img_pad = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge") if ph or pw else img
        tensor = img_pad.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.session.run(None, {self._input_name: tensor})[0]
        if self._needs_sigmoid:
            _sigmoid_inplace(out[:, 2:5])
        maps = np.ascontiguousarray(out[0, :, :height, :width].transpose(1, 2, 0), dtype=np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return maps, orig_h / height, gray

    def text_line_height(self, img_bgr: np.ndarray) -> float | None:
        est, conf = estimate_line_height(img_bgr)
        min_height = TEXT_HEIGHT_MIN_WIDTH_FRAC * img_bgr.shape[1]
        if np.isfinite(est) and est >= min_height and conf >= TEXT_HEIGHT_MIN_CONF:
            return est * TEXT_HEIGHT_GAIN
        maps, img_scale, _ = self.get_maps(img_bgr, 1)
        line_height = _median_line_height(maps)
        return None if line_height is None else line_height * img_scale

    def detect(self, img_bgr: np.ndarray, downsample: int | None = None,
               line_height: float | None = None) -> tuple[list[Region], float]:
        orig_h, orig_w = img_bgr.shape[:2]
        if orig_w > orig_h * WIDE_ASPECT and _has_central_gutter(img_bgr):
            return self._detect_split(img_bgr, downsample)

        maps = None
        if line_height and downsample is None:
            page_line_height = self.text_line_height(img_bgr)
            if page_line_height:
                width = _scaled_width(orig_w, orig_h, page_line_height, line_height)
                maps, img_scale, gray = self.get_maps_at_width(img_bgr, width)
                seen = _median_line_height(maps)
                if seen is not None and seen > TEXT_HEIGHT_RECHECK * line_height:
                    width = _scaled_width(orig_w, orig_h, seen * img_scale, line_height)
                    maps, img_scale, gray = self.get_maps_at_width(img_bgr, width)
        if maps is None:
            maps, img_scale, gray = self.get_maps(img_bgr, downsample)
        regions = self._regions(maps, gray)
        if self.large_text and downsample is None:
            regions = apply_large_text(self, img_bgr, regions, maps, gray, img_scale, self.reading_order)
        return regions, img_scale

    def _regions(self, maps: np.ndarray, gray: np.ndarray) -> list[Region]:
        return maps_to_regions(maps, gray, column_split=self.column_split,
                               reading_order=self.reading_order)

    def _detect_split(self, img_bgr: np.ndarray, downsample: int | None = None) -> tuple[list[Region], float]:
        h, w = img_bgr.shape[:2]
        mid = w // 2
        ds = downsample or DOWNSAMPLE

        left_maps, _, left_gray  = self.get_maps(img_bgr[:, :mid], downsample)
        L_wd = left_maps.shape[1]
        left_regions = self._regions(left_maps, left_gray)

        right_maps, _, right_gray = self.get_maps(img_bgr[:, mid:], downsample)
        R_wd = right_maps.shape[1]
        right_regions = self._regions(right_maps, right_gray)

        pre_scale = MAX_ORIGINAL_WIDTH / w if w > MAX_ORIGINAL_WIDTH else 1.0
        F_w = int(w * pre_scale) if w > MAX_ORIGINAL_WIDTH else w
        F_h = max(1, int(h * pre_scale))
        F_wd = max(1, F_w // ds)
        F_hd = max(1, F_h // ds)
        if max(F_hd, F_wd) > MAX_SIDE:
            s = MAX_SIDE / max(F_hd, F_wd)
            F_hd = max(1, int(F_hd * s))
            F_wd = max(1, int(F_wd * s))

        img_scale = h / F_hd

        sx_L = (F_wd / 2) / L_wd;  sy_L = F_hd / left_maps.shape[0]
        sx_R = (F_wd / 2) / R_wd;  sy_R = F_hd / right_maps.shape[0]
        x_off_R = F_wd / 2

        def _transform(region, sx, sy, x_off=0.0):
            region.polygon = [(x * sx + x_off, y * sy) for x, y in region.polygon]
            for line in region.lines:
                line.baseline = [(x * sx + x_off, y * sy) for x, y in line.baseline]
                line.polygon  = [(x * sx + x_off, y * sy) for x, y in line.polygon]
                line.heights  = (line.heights[0] * sy, line.heights[1] * sy)

        for r in left_regions:
            _transform(r, sx_L, sy_L)
        for r in right_regions:
            _transform(r, sx_R, sy_R, x_off_R)

        return left_regions + right_regions, img_scale
