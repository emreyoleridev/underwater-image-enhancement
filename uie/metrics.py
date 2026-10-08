"""Image quality metrics.

Full-reference (need a ground-truth image):  PSNR, SSIM.
No-reference (underwater-specific):         UCIQE (Yang & Sowmya, 2015), UIQM (Panetta et al., 2016).
All functions take uint8 BGR images.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(img: np.ndarray, ref: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, img, data_range=255))


def ssim(img: np.ndarray, ref: np.ndarray) -> float:
    return float(structural_similarity(ref, img, channel_axis=2, data_range=255))


# --- UCIQE -----------------------------------------------------------------------------

def uciqe(img: np.ndarray) -> float:
    """UCIQE = 0.4680*sigma_c + 0.2745*con_l + 0.2576*mu_s, computed in CIELAB (values in [0, 1])."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64) / 255.0
    L, a, b = lab[..., 0], lab[..., 1] - 0.5, lab[..., 2] - 0.5
    chroma = np.sqrt(a**2 + b**2)
    sigma_c = chroma.std()
    lo, hi = np.percentile(L, (1, 99))
    con_l = hi - lo
    sat = np.divide(chroma, L, out=np.zeros_like(chroma), where=L > 0)
    mu_s = np.clip(sat, 0, 1).mean()
    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)


# --- UIQM ------------------------------------------------------------------------------

def _trimmed_stats(x: np.ndarray, alpha: float = 0.1) -> tuple[float, float]:
    x = np.sort(x.ravel())
    k = int(alpha * x.size)
    t = x[k : x.size - k]
    return float(t.mean()), float(t.var())


def _uicm(rgb: np.ndarray) -> float:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mu_rg, var_rg = _trimmed_stats(r - g)
    mu_yb, var_yb = _trimmed_stats((r + g) / 2 - b)
    return -0.0268 * np.hypot(mu_rg, mu_yb) + 0.1586 * np.sqrt(var_rg + var_yb)


def _blocks(ch: np.ndarray, size: int) -> np.ndarray:
    h, w = (ch.shape[0] // size) * size, (ch.shape[1] // size) * size
    return ch[:h, :w].reshape(h // size, size, w // size, size).swapaxes(1, 2).reshape(-1, size * size)


def _eme(ch: np.ndarray, size: int = 8) -> float:
    blk = _blocks(ch, size)
    mx, mn = blk.max(1), blk.min(1)
    valid = (mn > 0) & (mx > 0)
    if not valid.any():
        return 0.0
    return float(2.0 / blk.shape[0] * np.sum(np.log(mx[valid] / mn[valid])))


def _uism(rgb: np.ndarray) -> float:
    weights = (0.299, 0.587, 0.114)
    total = 0.0
    for i, w in enumerate(weights):
        ch = rgb[..., i]
        gx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.hypot(gx, gy)
        edge = edge / max(edge.max(), 1e-12) * 255.0
        total += w * _eme(edge * ch / 255.0)
    return total


def _uiconm(rgb: np.ndarray, size: int = 8) -> float:
    gray = rgb.mean(axis=2)
    blk = _blocks(gray, size)
    mx, mn = blk.max(1), blk.min(1)
    top, bot = mx - mn, mx + mn
    valid = (top > 0) & (bot > 0)
    if not valid.any():
        return 0.0
    ratio = top[valid] / bot[valid]
    return float(abs(np.sum(ratio * np.log(ratio))) / blk.shape[0])


def uiqm(img: np.ndarray) -> float:
    """UIQM = 0.0282*UICM + 0.2953*UISM + 3.5753*UIConM (colorfulness, sharpness, contrast)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    return float(0.0282 * _uicm(rgb) + 0.2953 * _uism(rgb) + 3.5753 * _uiconm(rgb))
