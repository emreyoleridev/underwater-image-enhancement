"""Extra experiments for the report (ablation + parameter sensitivity) and report figures."""
import sys
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
from uie.enhance import clahe, compensate_red, gray_world, pipeline, stretch  # noqa: E402

DATA, RES = ROOT / "data", ROOT / "results"
FIG = RES / "figures_report"
MAX_SIDE = 640

TR = {
    "Raw input": "Raw input", "HE (RGB)": "HE (RGB)", "HE (Luminance)": "HE (Luminance)",
    "CLAHE": "CLAHE", "Gray-World WB": "Gray-World WB", "Underwater WB": "Underwater WB",
    "Pipeline (WB + CLAHE)": "Proposed pipeline", "Reference": "Reference",
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GRAY, ORANGE = "#2a78d6", "#a3a29c", "#eb6834"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200,
})


def load_pairs():
    for f in sorted((DATA / "raw").glob("*.png")):
        raw, ref = cv2.imread(str(f)), cv2.imread(str(DATA / "reference" / f.name))
        s = MAX_SIDE / max(raw.shape[:2])
        if s < 1:
            raw = cv2.resize(raw, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            ref = cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        yield f.name, raw, ref


def score(out, ref):
    return {"PSNR": psnr(out, ref), "SSIM": ssim(out, ref), "UCIQE": uciqe(out), "UIQM": uiqm(out)}


# --- Experiments ---------------------------------------------------------------------------

ABLATION = {
    "Full pipeline": {},
    "w/o red compensation": {"red_alpha": 0.0},
    "w/o white balance": {"white_balance": False},
    "w/o contrast stretch": {"stretch_pct": 0.0},
    "w/o CLAHE": {"clip_limit": 0.0},
}
CLIPS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]


def run_experiments():
    abl, sens = [], []
    for i, (name, raw, ref) in enumerate(load_pairs(), 1):
        for k, kw in ABLATION.items():
            abl.append({"image": name, "config": k, **score(pipeline(raw, **kw), ref)})
        for c in CLIPS:
            sens.append({"image": name, "param": "clip_limit", "value": c, **score(pipeline(raw, clip_limit=c), ref)})
        for a in ALPHAS:
            sens.append({"image": name, "param": "red_alpha", "value": a, **score(pipeline(raw, red_alpha=a), ref)})
        if i % 20 == 0:
            print(f"{i} images", flush=True)
    abl, sens = pd.DataFrame(abl), pd.DataFrame(sens)
    abl.to_csv(RES / "ablation_per_image.csv", index=False)
    sens.to_csv(RES / "sensitivity_per_image.csv", index=False)
    a = abl.groupby("config")[["PSNR", "SSIM", "UCIQE", "UIQM"]].mean().reindex(ABLATION).round(4)
    a.to_csv(RES / "ablation.csv")
    s = sens.groupby(["param", "value"])[["PSNR", "SSIM", "UCIQE", "UIQM"]].mean().round(4)
    s.to_csv(RES / "sensitivity.csv")
    print(a.to_string(), "\n", s.to_string())
    return s


# --- Figures -------------------------------------------------------------------------------

def fig_sensitivity(s):
    params = [("clip_limit", "CLAHE clip limit"), ("red_alpha", "Red compensation coefficient α")]
    metrics = [("PSNR", "PSNR (dB)", BLUE), ("UIQM", "UIQM", ORANGE)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), sharex="col")
    for c, (param, xlabel) in enumerate(params):
        d = s.loc[param]
        for r, (m, ylabel, col) in enumerate(metrics):
            ax = axes[r, c]
            ax.plot(d.index, d[m], color=col, marker="o", markersize=6, linewidth=2)
            default = 2.0 if param == "clip_limit" else 1.0
            ax.axvline(default, color=INK_2, linewidth=0.8, linestyle="--")
            ax.set_ylabel(ylabel)
            if r == 0:
                ax.set_title(xlabel, loc="left", fontsize=11)
            if r == 1:
                ax.set_xlabel(xlabel + "  (dashed line = default)")
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def fig_clip_tradeoff(s):
    d = s.loc["clip_limit"]
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 3.6), sharex=True)
    for ax, (m, label, col) in zip(axes, [("PSNR", "PSNR (dB)", BLUE), ("UIQM", "UIQM", ORANGE)]):
        ax.plot(d.index, d[m], color=col, marker="o", markersize=5, linewidth=2)
        ax.axvline(2.0, color=INK_2, linewidth=0.8, linestyle="--")
        ax.set_ylabel(label, fontsize=10)
        ax.tick_params(labelsize=9)
    axes[0].text(2.1, d["PSNR"].max(), "default", fontsize=8.5, color=INK_2, va="top")
    axes[1].set_xlabel("CLAHE clip limit", fontsize=10)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(FIG / "clip_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def fig_bars():
    df = pd.read_csv(RES / "per_image_metrics.csv")
    order = list(TR)
    g = df.groupby("method")[["PSNR", "SSIM", "UCIQE", "UIQM"]].agg(["mean", "std"]).reindex(order)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    names = [TR[n] for n in order]
    for ax, m in zip(axes.ravel(), ["PSNR", "SSIM", "UCIQE", "UIQM"]):
        s, e = g[(m, "mean")], g[(m, "std")].fillna(0)
        best = s.drop(["Raw input", "Reference"]).idxmax()
        colors = [ORANGE if n == best else GRAY if n in ("Raw input", "Reference") else BLUE for n in order]
        y = np.arange(len(s))[::-1]
        ax.barh(y, s.fillna(0).values, xerr=e.values, color=colors, height=0.62,
                error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 2})
        for yi, v in zip(y, s.values):
            if np.isnan(v):
                ax.text(0.01, yi, "n/a (ground truth)", va="center", fontsize=8.5, color=INK_2,
                        transform=ax.get_yaxis_transform())
            else:
                ax.text(v * 0.03, yi, f"{v:.3f}" if v < 10 else f"{v:.2f}", va="center", fontsize=8.5,
                        color="white", fontweight="bold")
        ax.set_yticks(y, names)
        ax.set_title(f"{m} (higher is better)" if m != "PSNR" else "PSNR, dB (higher is better)", loc="left", fontsize=11)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "metric_means.png", bbox_inches="tight")
    plt.close(fig)


def fig_box():
    df = pd.read_csv(RES / "per_image_metrics.csv")
    order = [n for n in TR if n != "Reference"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, m in zip(axes, ["PSNR", "UIQM"]):
        data = [df[df.method == n][m].values for n in order]
        bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.55,
                        medianprops={"color": INK, "linewidth": 1.5},
                        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": GRAY, "markeredgecolor": "none"})
        for patch, n in zip(bp["boxes"], order):
            patch.set_facecolor(GRAY if n == "Raw input" else BLUE)
            patch.set_alpha(0.75)
            patch.set_edgecolor("white")
        ax.set_yticks(range(1, len(order) + 1), [TR[n] for n in order] if ax is axes[0] else [""] * len(order))
        ax.invert_yaxis()
        ax.set_title(f"{m} distribution" + (" (dB)" if m == "PSNR" else ""), loc="left", fontsize=11)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "metric_distributions.png", bbox_inches="tight")
    plt.close(fig)


def _thumb(name):
    raw, ref = cv2.imread(str(DATA / "raw" / name)), cv2.imread(str(DATA / "reference" / name))
    s = min(1, 400 / max(raw.shape[:2]))
    return (cv2.resize(raw, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
            cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))


def _show(ax, img, title=None):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)


def fig_grid(picks):
    cols = list(TR)
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.1 * len(cols), 1.65 * len(picks)))
    for r, n in enumerate(picks):
        raw, ref = _thumb(n)
        for c, m in enumerate(cols):
            img = raw if m == "Raw input" else ref if m == "Reference" else enhance(raw, m)
            _show(axes[r, c], img, TR[m] if r == 0 else None)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "visual_comparison.jpg", bbox_inches="tight", dpi=160, pil_kwargs={"quality": 90})
    plt.close(fig)


def fig_stages(name):
    raw, ref = _thumb(name)
    s1 = compensate_red(raw)
    s2 = gray_world(s1)
    s3 = stretch(s2, 0.5, 99.5)
    s4 = clahe(s3)
    stages = [(raw, "(a) Raw input"), (s1, "(b) Red compensation"), (s2, "(c) Gray-World WB"),
              (s3, "(d) Contrast stretch"), (s4, "(e) CLAHE (output)"), (ref, "(f) Reference")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, (img, t) in zip(axes.ravel(), stages):
        _show(ax, img, t)
    fig.tight_layout(pad=0.4, h_pad=1.6)
    fig.savefig(FIG / "pipeline_stages.jpg", bbox_inches="tight", dpi=160, pil_kwargs={"quality": 90})
    plt.close(fig)


def fig_hist(name):
    raw, ref = _thumb(name)
    imgs = [("Raw input", raw), ("CLAHE", clahe(raw)), ("Proposed pipeline", pipeline(raw)), ("Reference", ref)]
    chan = [("Blue", "#2a78d6"), ("Green", "#008300"), ("Red", "#e34948")]
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.6), gridspec_kw={"height_ratios": [1.3, 1]})
    for c, (t, img) in enumerate(imgs):
        _show(axes[0, c], img, t)
        ax = axes[1, c]
        for i, (cn, col) in enumerate(chan):
            h = cv2.calcHist([img], [i], None, [256], [0, 256]).ravel()
            ax.plot(h / h.sum(), color=col, linewidth=1.5, label=cn)
        ax.set_xlim(0, 255); ax.set_yticks([]); ax.set_xlabel("Pixel value")
        if c == 0:
            ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "histograms.jpg", bbox_inches="tight", dpi=160, pil_kwargs={"quality": 90})
    plt.close(fig)


def run_wilcoxon():
    """Paired two-sided Wilcoxon signed-rank tests: proposed pipeline vs. other methods."""
    from scipy.stats import wilcoxon

    df = pd.read_csv(RES / "per_image_metrics.csv")
    by = lambda m: df[df.method == m].set_index("image")
    a = by("Pipeline (WB + CLAHE)")
    rows = []
    for other in ["Raw input", "CLAHE", "Underwater WB", "HE (RGB)"]:
        b = by(other).reindex(a.index)
        for m in ["PSNR", "SSIM", "UCIQE", "UIQM"]:
            rows.append({"A": "Pipeline (WB + CLAHE)", "B": other, "metric": m,
                         "median_diff": float(np.median(a[m] - b[m])), "p": wilcoxon(a[m], b[m]).pvalue})
    pd.DataFrame(rows).to_csv(RES / "wilcoxon.csv", index=False)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    run_wilcoxon()
    if "--figures-only" in sys.argv:
        s = pd.read_csv(RES / "sensitivity.csv", index_col=[0, 1])
    else:
        s = run_experiments()
    fig_sensitivity(s)
    fig_clip_tradeoff(s)
    fig_bars()
    fig_box()
    raw = pd.read_csv(RES / "per_image_metrics.csv").query("method == 'Raw input'").set_index("image")
    worst = raw["PSNR"].sort_values().index.tolist()
    fig_grid([worst[i] for i in (0, 10, 25, 45, 70)])
    fig_stages(worst[10])
    fig_hist(worst[5])
    print("Figures ->", FIG)


if __name__ == "__main__":
    main()
