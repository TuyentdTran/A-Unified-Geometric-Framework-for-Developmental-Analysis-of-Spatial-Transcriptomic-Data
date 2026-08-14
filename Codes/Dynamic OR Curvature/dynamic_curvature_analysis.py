#!/usr/bin/env python3
"""Compute the distribution comparisons reported for the 19-tau analysis."""

from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

HERE = Path(__file__).resolve().parent
INPUT = HERE / "output" / "curvatures_19tau.csv.gz"
OUTPUT_DIR = HERE / "output" / "analysis"
STAGES = ["E14-16", "E16-18", "L1", "L2", "L3"]
ADJACENT = list(zip(STAGES[:-1], STAGES[1:]))


def values(df, stage, tau):
    mask = (df["stage"] == stage) & np.isclose(df["tau"].to_numpy(float), tau, rtol=1e-10, atol=1e-12)
    return df.loc[mask, "curvature"].to_numpy(float)


def log_tau_average(x, y):
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap(y, x=x) / (x[-1] - x[0]))


def main():
    df = pd.read_csv(INPUT)
    required = {"stage", "tau", "curvature"}
    if not required.issubset(df.columns) or not np.isfinite(df["curvature"].to_numpy(float)).all():
        raise ValueError("Invalid curvature input")
    taus = np.sort(df["tau"].unique().astype(float))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for stage in STAGES:
        for tau in taus:
            x = values(df, stage, tau)
            q1, median, q3 = np.quantile(x, [0.25, 0.5, 0.75])
            summaries.append((stage, tau, len(x), np.mean(x), median, q1, q3, q3 - q1))
    pd.DataFrame(summaries, columns=["stage", "tau", "n", "mean", "median", "q1", "q3", "iqr"]).to_csv(OUTPUT_DIR / "stage_summaries.csv", index=False)

    pairwise = []
    for a, b in combinations(STAGES, 2):
        for tau in taus:
            x, y = values(df, a, tau), values(df, b, tau)
            med_x, med_y = np.median(x), np.median(y)
            iqr_x = np.quantile(x, 0.75) - np.quantile(x, 0.25)
            iqr_y = np.quantile(y, 0.75) - np.quantile(y, 0.25)
            raw = wasserstein_distance(x, y)
            scaled = wasserstein_distance((x - med_x) / iqr_x, (y - med_y) / iqr_y)
            pairwise.append((a, b, tau, np.log10(tau), raw, scaled))
    pairwise = pd.DataFrame(pairwise, columns=["stage_a", "stage_b", "tau", "log10_tau", "raw_w1", "centered_scaled_w1"])
    pairwise.to_csv(OUTPUT_DIR / "pairwise_w1_by_tau.csv", index=False)

    adjacent = pd.concat([pairwise[(pairwise.stage_a == a) & (pairwise.stage_b == b)].assign(transition=f"{a} to {b}") for a, b in ADJACENT], ignore_index=True)
    adjacent.to_csv(OUTPUT_DIR / "adjacent_w1_by_tau.csv", index=False)
    rows = []
    for a, b in ADJACENT:
        g = adjacent[(adjacent.stage_a == a) & (adjacent.stage_b == b)]
        raw_avg = log_tau_average(g.log10_tau.to_numpy(float), g.raw_w1.to_numpy(float))
        scaled_avg = log_tau_average(g.log10_tau.to_numpy(float), g.centered_scaled_w1.to_numpy(float))
        raw_wins = sum(np.isclose(g.loc[np.isclose(g.tau, tau), "raw_w1"].iloc[0], adjacent[np.isclose(adjacent.tau, tau)].raw_w1.max(), rtol=1e-12, atol=1e-15) for tau in taus)
        scaled_wins = sum(np.isclose(g.loc[np.isclose(g.tau, tau), "centered_scaled_w1"].iloc[0], adjacent[np.isclose(adjacent.tau, tau)].centered_scaled_w1.max(), rtol=1e-12, atol=1e-15) for tau in taus)
        rows.append((f"{a} to {b}", raw_avg, raw_wins, scaled_avg, scaled_wins))
    table3 = pd.DataFrame(rows, columns=["transition", "raw_w1_avg", "raw_largest_tau_count", "centered_scaled_w1_avg", "centered_scaled_largest_tau_count"])
    table3.to_csv(OUTPUT_DIR / "table3_adjacent_summary.csv", index=False)
    print(table3.to_string(index=False))


if __name__ == "__main__":
    main()
