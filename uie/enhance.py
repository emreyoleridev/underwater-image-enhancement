"""Enhancement methods. All functions take and return uint8 BGR images (OpenCV convention)."""
from __future__ import annotations

import cv2
import numpy as np


# --- Histogram equalization ------------------------------------------------------------

def he_rgb(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization applied independently to each color channel.
    Stretches every channel to a flat histogram, which also removes the color cast,
    but tends to over-amplify noise and produce unnatural colors."""
    return cv2.merge([cv2.equalizeHist(c) for c in cv2.split(img)])


def he_luma(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization on the luminance (Y) channel only.
    Improves contrast while keeping chroma, so the color cast is preserved."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


# --- CLAHE -----------------------------------------------------------------------------

def clahe(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on the L channel of CIELAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    lab[..., 0] = op.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# --- White balance ---------------------------------------------------------------------

def gray_world(img: np.ndarray) -> np.ndarray:
    """Gray-World white balance: scale each channel so that its mean equals the global mean."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    f *= means.mean() / np.maximum(means, 1e-6)
    return np.clip(f, 0, 255).astype(np.uint8)


def compensate_red(img: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Red-channel compensation (Ancuti et al., 2018).
    Water absorbs red light first, so the red channel is boosted with information from the
    better-preserved green channel:  Ir' = Ir + alpha * (mean(G) - mean(R)) * (1 - Ir) * Ig."""
    f = img.astype(np.float32) / 255.0
    b, g, r = cv2.split(f)
    r = r + alpha * (g.mean() - r.mean()) * (1.0 - r) * g
    return (np.clip(cv2.merge([b, g, r]), 0, 1) * 255).astype(np.uint8)


def underwater_wb(img: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Red-channel compensation followed by Gray-World white balance."""
    return gray_world(compensate_red(img, alpha))


# --- Helpers ---------------------------------------------------------------------------

def stretch(img: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Per-channel percentile contrast stretch."""
    out = np.empty_like(img)
    for i in range(3):
        lo, hi = np.percentile(img[..., i], (low, high))
        out[..., i] = np.clip((img[..., i].astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255)
    return out


def gamma(img: np.ndarray, g: float = 1.0) -> np.ndarray:
    if g == 1.0:
        return img
    lut = (np.linspace(0, 1, 256) ** (1.0 / g) * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def saturation(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    if factor == 1.0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# --- Combined pipeline -----------------------------------------------------------------

def pipeline(
    img: np.ndarray,
    red_alpha: float = 1.0,
    white_balance: bool = True,
    clip_limit: float = 2.0,
    tile: int = 8,
    stretch_pct: float = 0.5,
    gamma_value: float = 1.0,
    sat_factor: float = 1.0,
) -> np.ndarray:
    """Proposed pipeline: color correction first, then local contrast enhancement.

    1. Red-channel compensation   (restore the attenuated red channel)
    2. Gray-World white balance    (remove the blue/green cast)
    3. Percentile contrast stretch (use the full dynamic range)
    4. CLAHE on L channel          (local contrast without touching color)
    5. Optional gamma / saturation fine-tuning
    """
    out = img
    if red_alpha > 0:
        out = compensate_red(out, red_alpha)
    if white_balance:
        out = gray_world(out)
    if stretch_pct > 0:
        out = stretch(out, stretch_pct, 100 - stretch_pct)
    if clip_limit > 0:
        out = clahe(out, clip_limit, tile)
    out = gamma(out, gamma_value)
    out = saturation(out, sat_factor)
    return out


METHODS = {
    "HE (RGB)": he_rgb,
    "HE (Luminance)": he_luma,
    "CLAHE": clahe,
    "Gray-World WB": gray_world,
    "Underwater WB": underwater_wb,
    "Pipeline (WB + CLAHE)": pipeline,
}


def enhance(img: np.ndarray, method: str) -> np.ndarray:
    return METHODS[method](img)
