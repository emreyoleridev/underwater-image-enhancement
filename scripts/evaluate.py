"""Run every enhancement method on the dataset, compute metrics and produce report figures."""
import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uie import METHODS, enhance, psnr, ssim, uciqe, uiqm  # noqa: E402

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GRAY, ORANGE = "#2a78d6", "#a3a29c", "#eb6834"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 150,
})

METRICS = {"PSNR": "PSNR (dB) ↑", "SSIM": "SSIM ↑", "UCIQE": "UCIQE ↑", "UIQM": "UIQM ↑"}


def evaluate(data: Path, max_side: int) -> pd.DataFrame:
    rows = []
    files = sorted((data / "raw").glob("*.png"))
    for i, f in enumerate(files, 1):
        raw = cv2.imread(str(f))
        ref = cv2.imread(str(data / "reference" / f.name))
        scale = max_side / max(raw.shape[:2])
        if scale < 1:  # keep evaluation fast on large images
            raw = cv2.resize(raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            ref = cv2.resize(ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for name in ["Raw input", *METHODS, "Reference"]:
            t = time.perf_counter()
            out = raw if name == "Raw input" else ref if name == "Reference" else enhance(raw, name)
            ms = (time.perf_counter() - t) * 1000
            rows.append({
                "image": f.name, "method": name, "time_ms": ms,
                "PSNR": psnr(out, ref) if name != "Reference" else np.nan,
                "SSIM": ssim(out, ref) if name != "Reference" else np.nan,
                "UCIQE": uciqe(out), "UIQM": uiqm(out),
            })
        if i % 20 == 0:
            print(f"{i}/{len(files)} images", flush=True)
    print()
    return pd.DataFrame(rows)


def order(df):
    return ["Raw input", *METHODS, "Reference"]


def plot_bars(summary: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    names = list(summary.index)
    for ax, (m, label) in zip(axes, METRICS.items()):
        s = summary[(m, "mean")]
        e = summary[(m, "std")].fillna(0)
        best = s.drop(["Raw input", "Reference"]).idxmax()
        colors = [ORANGE if n == best else GRAY if n in ("Raw input", "Reference") else BLUE for n in names]
        y = np.arange(len(s))[::-1]
        ax.barh(y, s.fillna(0).values, xerr=e.values, color=colors, height=0.62,
                error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 2})
        for yi, v in zip(y, s.values):
            txt = "n/a (is GT)" if np.isnan(v) else f"{v:.3f}" if v < 10 else f"{v:.2f}"
            ax.text(0.01 if np.isnan(v) else v * 0.03, yi, txt, va="center", ha="left", fontsize=8,
                    color=INK_2 if np.isnan(v) else "white", fontweight="bold",
                    transform=ax.get_yaxis_transform() if np.isnan(v) else ax.transData)
        ax.set_yticks(y, names if ax is axes[0] else [""] * len(names))
        ax.set_title(label, loc="left")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Mean metric per method on 100 UIEB images (error bar = 1 std; orange = best method)",
                 x=0.01, ha="left", color=INK_2, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "metrics_bar.png", bbox_inches="tight")
    plt.close(fig)


def plot_box(df: pd.DataFrame, out: Path):
    names = [n for n in order(df) if n != "Reference"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, m in zip(axes, ["PSNR", "UIQM"]):
        data = [df[df.method == n][m].dropna().values for n in names]
        bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.55,
                        medianprops={"color": INK, "linewidth": 1.5},
                        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": GRAY, "markeredgecolor": "none"})
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(GRAY if n == "Raw input" else BLUE)
            patch.set_alpha(0.75)
            patch.set_edgecolor("white")
        ax.set_yticks(range(1, len(names) + 1), names if ax is axes[0] else [""] * len(names))
        ax.invert_yaxis()
        ax.set_title(f"{METRICS[m]} distribution", loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "metrics_box.png", bbox_inches="tight")
    plt.close(fig)


def plot_grid(data: Path, names: list[str], out: Path, max_side: int):
    cols = ["Raw input", *METHODS, "Reference"]
    fig, axes = plt.subplots(len(names), len(cols), figsize=(2.3 * len(cols), 1.85 * len(names)))
    for r, n in enumerate(names):
        raw = cv2.imread(str(data / "raw" / n))
        ref = cv2.imread(str(data / "reference" / n))
        s = min(1, 400 / max(raw.shape[:2]))
        raw = cv2.resize(raw, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ref = cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        for c, m in enumerate(cols):
            img = raw if m == "Raw input" else ref if m == "Reference" else enhance(raw, m)
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(m, fontsize=9)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "before_after_grid.jpg", bbox_inches="tight", dpi=110, pil_kwargs={"quality": 88})
    plt.close(fig)


def plot_histograms(data: Path, name: str, out: Path):
    raw = cv2.imread(str(data / "raw" / name))
    imgs = {"Raw input": raw, "CLAHE": enhance(raw, "CLAHE"),
            "Pipeline (WB + CLAHE)": enhance(raw, "Pipeline (WB + CLAHE)"),
            "Reference": cv2.imread(str(data / "reference" / name))}
    chan = [("Blue", "#2a78d6"), ("Green", "#008300"), ("Red", "#e34948")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), gridspec_kw={"height_ratios": [1.3, 1]})
    for c, (title, img) in enumerate(imgs.items()):
        axes[0, c].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, c].set_title(title, fontsize=10)
        axes[0, c].axis("off")
        ax = axes[1, c]
        for i, (cn, col) in enumerate(chan):
            h = cv2.calcHist([img], [i], None, [256], [0, 256]).ravel()
            ax.plot(h / h.sum(), color=col, linewidth=1.6, label=cn)
        ax.set_xlim(0, 255)
        ax.set_yticks([])
        if c == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Per-channel color histograms: the raw red channel is crushed near zero; the pipeline restores a balanced distribution",
                 x=0.01, ha="left", color=INK_2, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "histograms.jpg", bbox_inches="tight", dpi=120, pil_kwargs={"quality": 88})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--max-side", type=int, default=640)
    args = ap.parse_args()
    (args.out / "figures").mkdir(parents=True, exist_ok=True)

    df = evaluate(args.data, args.max_side)
    df.to_csv(args.out / "per_image_metrics.csv", index=False)

    summary = df.groupby("method")[[*METRICS, "time_ms"]].agg(["mean", "std"]).reindex(order(df))
    summary.round(4).to_csv(args.out / "summary.csv")
    means = summary.xs("mean", axis=1, level=1).round(3)
    print(means.to_string())
    (args.out / "summary.md").write_text(means.to_markdown())

    # share of images on which each method beats the raw input
    raw = df[df.method == "Raw input"].set_index("image")
    win = {m: {k: (df[df.method == m].set_index("image")[k] > raw[k]).mean() * 100 for k in METRICS}
           for m in METHODS}
    pd.DataFrame(win).T.round(1).to_csv(args.out / "win_rate_vs_raw.csv")
    print(pd.DataFrame(win).T.round(1).to_string())

    figs = args.out / "figures"
    plot_bars(summary, figs)
    plot_box(df, figs)
    # pick examples: the 5 images with the lowest raw PSNR (most degraded) spread across the set
    worst = raw["PSNR"].sort_values().index.tolist()
    picks = [worst[i] for i in (0, 10, 25, 45, 70)]
    plot_grid(args.data, picks, figs, args.max_side)
    plot_histograms(args.data, worst[5], figs)
    print("Figures written to", figs)


if __name__ == "__main__":
    main()
