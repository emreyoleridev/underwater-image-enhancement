"""Streamlit app: interactive underwater image enhancement.

Run:  streamlit run app.py
"""
import io
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from scripts.download_uieb import download
from uie import METHODS, enhance, pipeline, psnr, ssim, uciqe, uiqm

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MAX_SIDE = 1024

st.set_page_config(page_title="Underwater Image Enhancement", page_icon="🌊", layout="wide")


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    s = MAX_SIDE / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


@st.cache_resource(show_spinner="Downloading UIEB sample images…")
def ensure_samples(n: int = 12) -> list[str]:
    """Use the local dataset if present, otherwise fetch a few UIEB pairs (e.g. on Streamlit Cloud)."""
    existing = sorted(p.name for p in (DATA / "raw").glob("*.png")) if (DATA / "raw").exists() else []
    if existing:
        return existing
    try:
        return download(n=n, out=DATA)
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def metrics(img_bytes: bytes, ref_bytes: bytes | None) -> dict:
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    out = {"UCIQE": uciqe(img), "UIQM": uiqm(img)}
    if ref_bytes is not None:
        ref = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)
        out |= {"PSNR": psnr(img, ref), "SSIM": ssim(img, ref)}
    return out


def histogram(img: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(5, 2.2))
    for i, (name, col) in enumerate([("Blue", "#2a78d6"), ("Green", "#008300"), ("Red", "#e34948")]):
        h = cv2.calcHist([img], [i], None, [256], [0, 256]).ravel()
        ax.plot(h / h.sum(), color=col, linewidth=1.5, label=name)
    ax.set_xlim(0, 255); ax.set_yticks([]); ax.set_title(title, loc="left", fontsize=10)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


# --- Sidebar: input -----------------------------------------------------------------------
st.sidebar.title("🌊 Underwater Enhancement")
source = st.sidebar.radio("Image source", ["Upload your own", "UIEB sample"], horizontal=True)

raw, ref = None, None
if source == "Upload your own":
    up = st.sidebar.file_uploader("Underwater image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    up_ref = st.sidebar.file_uploader("Optional ground-truth (enables PSNR / SSIM)", type=["png", "jpg", "jpeg"])
    if up:
        raw = decode(up.getvalue())
    if up_ref and raw is not None:
        ref = cv2.resize(decode(up_ref.getvalue()), raw.shape[1::-1], interpolation=cv2.INTER_AREA)
else:
    samples = ensure_samples()
    if not samples:
        st.sidebar.warning("Couldn't load UIEB samples. Upload your own image instead.")
    else:
        name = st.sidebar.selectbox("Sample", samples)
        raw = decode((DATA / "raw" / name).read_bytes())
        ref = cv2.resize(decode((DATA / "reference" / name).read_bytes()), raw.shape[1::-1],
                         interpolation=cv2.INTER_AREA)

# --- Sidebar: method ----------------------------------------------------------------------
st.sidebar.divider()
method = st.sidebar.selectbox("Method", list(METHODS), index=len(METHODS) - 1)
params = {}
if method == "Pipeline (WB + CLAHE)":
    st.sidebar.caption("Pipeline parameters")
    params = dict(
        red_alpha=st.sidebar.slider("Red-channel compensation α", 0.0, 2.0, 1.0, 0.1),
        white_balance=st.sidebar.checkbox("Gray-World white balance", True),
        stretch_pct=st.sidebar.slider("Contrast stretch (clip %)", 0.0, 5.0, 0.5, 0.1),
        clip_limit=st.sidebar.slider("CLAHE clip limit", 0.0, 6.0, 2.0, 0.1),
        tile=st.sidebar.slider("CLAHE tile grid", 2, 16, 8),
        gamma_value=st.sidebar.slider("Gamma", 0.5, 2.0, 1.0, 0.05),
        sat_factor=st.sidebar.slider("Saturation", 0.5, 2.0, 1.0, 0.05),
    )
elif method == "CLAHE":
    params = dict(clip_limit=st.sidebar.slider("Clip limit", 0.5, 6.0, 2.0, 0.1),
                  tile=st.sidebar.slider("Tile grid", 2, 16, 8))
elif method == "Underwater WB":
    params = dict(alpha=st.sidebar.slider("Red-channel compensation α", 0.0, 2.0, 1.0, 0.1))

# --- Main ---------------------------------------------------------------------------------
st.title("Underwater Image Enhancement")
st.caption("Classical color correction & contrast enhancement with OpenCV — CLAHE, white balance, "
           "histogram equalization — evaluated with PSNR, SSIM, UCIQE and UIQM.")

if raw is None:
    st.info("⬅️ Upload an underwater photo or pick a UIEB sample from the sidebar to get started.")
    st.stop()

out = METHODS[method](raw, **params)
ref_png = to_png(ref) if ref is not None else None

tab_single, tab_all, tab_about = st.tabs(["Before / After", "Compare all methods", "How it works"])

with tab_single:
    cols = st.columns(3 if ref is not None else 2)
    cols[0].image(rgb(raw), caption="Original", width="stretch")
    cols[1].image(rgb(out), caption=method, width="stretch")
    if ref is not None:
        cols[2].image(rgb(ref), caption="Reference (ground truth)", width="stretch")

    m_raw, m_out = metrics(to_png(raw), ref_png), metrics(to_png(out), ref_png)
    mcols = st.columns(len(m_out))
    for c, k in zip(mcols, m_out):
        fmt = "{:.2f}" if k == "PSNR" else "{:.3f}"
        c.metric(k, fmt.format(m_out[k]), fmt.format(m_out[k] - m_raw[k]) + " vs original")

    h1, h2 = st.columns(2)
    h1.pyplot(histogram(raw, "Original — channel histograms"))
    h2.pyplot(histogram(out, f"{method} — channel histograms"))

    st.download_button("⬇️ Download enhanced image", to_png(out), "enhanced.png", "image/png")

with tab_all:
    st.caption("Every method with default parameters on the current image.")
    results = {"Original": raw, **{m: enhance(raw, m) for m in METHODS}}
    if ref is not None:
        results["Reference"] = ref
    grid = st.columns(4)
    for i, (name, img) in enumerate(results.items()):
        grid[i % 4].image(rgb(img), caption=name, width="stretch")
    table = pd.DataFrame({n: metrics(to_png(img), None if n == "Reference" else ref_png)
                          for n, img in results.items()}).T
    st.dataframe(table.style.format("{:.3f}", na_rep="—").highlight_max(
        axis=0, subset=pd.IndexSlice[[i for i in table.index if i not in ("Original", "Reference")], :],
        color="#2a78d633"), width="stretch")
    st.caption("Highlighted = best enhancement method for that metric. All metrics: higher is better.")

with tab_about:
    st.markdown("""
**Why underwater images look bad.** Water absorbs red light within a few metres, and suspended
particles scatter light. The result is a blue/green color cast and hazy, low-contrast images.

| Method | What it does |
|---|---|
| **HE (RGB)** | Global histogram equalization on each RGB channel. Removes the cast but often produces garish colors and noise. |
| **HE (Luminance)** | Equalizes only the Y channel; boosts contrast but keeps the cast. |
| **CLAHE** | Adaptive, clip-limited equalization of the L channel in CIELAB. Local contrast without noise blow-up; doesn't fix color. |
| **Gray-World WB** | Scales channels so each has the same mean. Over-reddens when red is nearly absent. |
| **Underwater WB** | Compensates the red channel from green first (Ancuti et al., 2018), then Gray-World. |
| **Pipeline** | Red compensation → Gray-World → percentile stretch → CLAHE → optional gamma/saturation. |

**Metrics.** PSNR/SSIM compare against a reference image (only when one is available).
UCIQE and UIQM are no-reference underwater quality measures of colorfulness, sharpness and contrast.
""")
