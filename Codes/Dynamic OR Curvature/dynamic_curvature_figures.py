#!/usr/bin/env python3
"""Create the ECDF and appendix distribution figures for the 19-tau analysis."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
INPUT = HERE / "output" / "curvatures_19tau.csv.gz"
FIG_DIR = HERE / "figures"
STAGES = ["E14-16", "E16-18", "L1", "L2", "L3"]
APPENDIX_STAGES = ["L1", "L2", "L3"]
APPENDIX_COLORS = {"L1": "C0", "L2": "C1", "L3": "C2"}
COLORS = {"E14-16": "#0072B2", "E16-18": "#E69F00", "L1": "#009E73", "L2": "#D55E00", "L3": "#CC79A7"}
STYLES = {"E14-16": "-", "E16-18": "--", "L1": "-.", "L2": ":", "L3": "-"}


def values(df, stage, tau):
    mask = (df["stage"] == stage) & np.isclose(df["tau"].to_numpy(float), tau, rtol=1e-10, atol=1e-12)
    return df.loc[mask, "curvature"].to_numpy(float)


def ecdf(x):
    x = np.sort(x)
    return x, np.arange(1, len(x) + 1) / len(x)


def kernel_sd(kde):
    return float(np.sqrt(np.asarray(kde.covariance).squeeze()))


def make_ecdf(df):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.45), sharex=True, sharey=True)
    for ax, tau, panel in zip(axes, [1.0, 10.0], ["(a)", "(b)"]):
        for stage in STAGES:
            x, y = ecdf(values(df, stage, tau))
            ax.step(x, y, where="post", lw=1.9, color=COLORS[stage], ls=STYLES[stage], label=stage)
        ax.set_xlim(0.25, 0.95); ax.set_ylim(0, 1.01)
        ax.set_xlabel("Distance-scaled dynamic OR curvature"); ax.set_title(rf"{panel} $\tau={tau:g}$")
        ax.grid(alpha=0.18, linewidth=0.7); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Empirical cumulative probability")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "ecdf_tau1_tau10_19tau_xlim_025_095.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "ecdf_tau1_tau10_19tau_xlim_025_095.png", dpi=240, bbox_inches="tight"); plt.close(fig)


def make_hist_kde(df, tau):
    arrays = {stage: values(df, stage, tau) for stage in APPENDIX_STAGES}
    pooled = np.concatenate(list(arrays.values())); lo, hi = pooled.min(), pooled.max()
    edges = np.linspace(lo, hi, 31); grid = np.linspace(lo, hi, 1200)
    h = 0.5 * kernel_sd(gaussian_kde(pooled, bw_method="scott"))
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for stage in APPENDIX_STAGES:
        ax.hist(arrays[stage], bins=edges, density=True, histtype="stepfilled", alpha=0.10, edgecolor=APPENDIX_COLORS[stage], facecolor=APPENDIX_COLORS[stage], linewidth=0.8)
    for stage in APPENDIX_STAGES:
        x = arrays[stage]; kde = gaussian_kde(x, bw_method=h / np.std(x, ddof=1))
        ax.plot(grid, kde(grid), color=APPENDIX_COLORS[stage], lw=2.2, label=stage)
    ax.set_title(f"Raw curvature distributions at tau = {tau:g}"); ax.set_xlabel("Distance-scaled dynamic OR curvature"); ax.set_ylabel("Density")
    ax.set_xlim(lo, hi); ax.legend(frameon=False, title="Stage"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); pdf = FIG_DIR / f"raw_overlay_tau_{tau:g}.pdf"; png = FIG_DIR / f"raw_overlay_tau_{tau:g}.png"
    fig.savefig(pdf, bbox_inches="tight"); fig.savefig(png, dpi=400, bbox_inches="tight"); plt.close(fig)


def main():
    df = pd.read_csv(INPUT)
    if not {"stage", "tau", "curvature"}.issubset(df.columns) or not np.isfinite(df.curvature.to_numpy(float)).all():
        raise ValueError("Invalid curvature input")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_ecdf(df); make_hist_kde(df, 0.1); make_hist_kde(df, 1.0)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
