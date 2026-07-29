"""
Classifies real graphs to geodesic time points using OR curvature distributions,
with SHARED (geodesic-referenced) standardization applied before every pairwise
comparison.

Standardization convention
--------------------------
Before each pairwise discrepancy is computed, both distributions are
standardized using the mean and standard deviation of the geodesic
distribution at that time point:

    a_s = (a - mu_geo) / sigma_geo     [real graph, shifted into geodesic space]
    b_s = (b - mu_geo) / sigma_geo     [geodesic, which becomes zero-mean, unit-variance]

This ensures that the geodesic distribution is centred at zero with unit variance, 
while preserving the relative displacement of the real graph distribution from the
geodesic reference. Concretely: if a real graph's curvature is shifted
0.3 standard deviations above the geodesic at t=0.25 but only 0.05
above the geodesic at t=0.75, that signal is retained and contributes
to the assignment.

Scalar methods
------------------------------------
Under shared standardization, the scalar methods measure how many geodesic
standard deviations the real graph's summary statistic lies from the
geodesic's summary statistic. Specifically:

    scalar-mean   : |mean(real) - mean(geo)| / sigma_geo
    scalar-median : |median(real) - median(geo)| / sigma_geo

Three choices of delta
----------------------
  - 'wasserstein' : 1-D Wasserstein distance (no standardization applied).
  - 'kl'          : Symmetrized KL divergence on geodesic-standardized KDEs.
  - 'scalar'      : normalized absolute difference of mean or median, using
                    geodesic statistics for rescaling.
"""

import numpy as np
from scipy.stats import wasserstein_distance, kendalltau, spearmanr, gaussian_kde
import matplotlib.pyplot as plt
import warnings


def _standardize_shared(a, b):
    """
    Standardize both arrays using the mean and std of b (the geodesic).

    Returns (a_s, b_s) where b_s is zero-mean, unit-variance, and a_s is
    shifted and scaled by the same constants. The originals are not modified.
    """
    mu  = b.mean()
    sig = max(b.std(), 1e-8)
    return (a - mu) / sig, (b - mu) / sig


def _wasserstein_delta(a, b):
    """1-D Wasserstein distance between two sample arrays."""
    return wasserstein_distance(a, b)


def _kl_delta(a, b, n_grid=512, bandwidth=None):
    """
    Symmetrized KL divergence KL(p||q) + KL(q||p) between KDE estimates.

    Parameters
    ----------
    a, b      : 1-D sample arrays (assumed already standardized by the caller)
    n_grid    : number of grid points for numerical integration
    bandwidth : KDE bandwidth; None uses Scott's rule
    """
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    margin = 0.1 * max(hi - lo, 1e-8)
    grid = np.linspace(lo - margin, hi + margin, n_grid)

    p = gaussian_kde(a, bw_method=bandwidth)(grid)
    q = gaussian_kde(b, bw_method=bandwidth)(grid)

    dx = grid[1] - grid[0]
    p /= p.sum() * dx
    q /= q.sum() * dx

    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    return (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))) * dx


def _scalar_delta_shared(real_vals, geo_vals, stat="mean"):
    """
    normalized absolute difference between a summary statistic of the real
    graph and the same statistic of the geodesic, measured in units of the
    geodesic's standard deviation.

    Parameters
    ----------
    real_vals : 1-D array, raw curvature samples for the real graph
    geo_vals  : 1-D array, raw curvature samples for the geodesic
    stat      : {'mean', 'median'}
    """
    sigma_geo = max(geo_vals.std(), 1e-8)
    if stat == "mean":
        return abs(np.mean(real_vals) - np.mean(geo_vals)) / sigma_geo
    elif stat == "median":
        return abs(np.median(real_vals) - np.median(geo_vals)) / sigma_geo
    else:
        raise ValueError(f"stat must be 'mean' or 'median', got '{stat}'.")


def _extract_curvature_values(curvature_evolution, N, t):
    """
    Return a 1-D array of finite upper-triangle curvature values for (N, t),
    mirroring the convention used in analyze_or_curvature().
    """
    mat = curvature_evolution[N][t]
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    vals = mat[mask]
    return vals[np.isfinite(vals)]


def classify_real_graphs(
    curvature_evolution_geo,
    curvature_evolution_real,
    t_values,
    N_geo=None,
    N_real=None,
    method="wasserstein",
    scalar_stat="mean",
    kl_bandwidth=None,
    verbose=True,
):
    """
    For each real graph, find the geodesic time point t* whose curvature
    distribution is closest under the chosen discrepancy (with shared
    geodesic-referenced standardization), then evaluate temporal ordering
    via rank correlation.

    Parameters
    ----------
    curvature_evolution_geo  : dict {N: {t: curvature_matrix}}
        Output of analyze_or_curvature() for the GW geodesic.
    curvature_evolution_real : dict {N: {t: curvature_matrix}}
        Output of analyze_or_curvature() for the real graphs.
    t_values    : list of float
        Shared time points (e.g. [0.0, 0.25, 0.5, 0.75, 1.0]).
    N_geo       : int or None
        Key to use in curvature_evolution_geo. Defaults to the first key.
    N_real      : int or None
        Key to use in curvature_evolution_real. Defaults to the first key.
    method      : {'wasserstein', 'kl', 'scalar'}
    scalar_stat : {'mean', 'median'}
        Summary statistic for method='scalar'. Ignored otherwise.
    kl_bandwidth : float or None
        KDE bandwidth for method='kl'. None uses Scott's rule.
    verbose     : bool

    Returns
    -------
    dict with keys:
        'true_t', 'predicted_t', 'distance_matrix',
        'kendall_tau', 'kendall_pvalue', 'spearman_rho', 'spearman_pvalue'
    """

    if N_geo is None:
        N_geo = list(curvature_evolution_geo.keys())[0]
    if N_real is None:
        N_real = list(curvature_evolution_real.keys())[0]

    if method not in ("wasserstein", "kl", "scalar"):
        raise ValueError(f"method must be 'wasserstein', 'kl', or 'scalar', got '{method}'.")

    if method == "scalar":
        d = _scalar_delta_shared(real_vals, geo_vals[t_geo], stat=scalar_stat)
    elif method == "wasserstein":
        d = _wasserstein_delta(real_vals, geo_vals[t_geo])
    else:  # "kl"
        # Standardize using geodesic statistics before KL, to focus on shape
        a_s, b_s = _standardize_shared(real_vals, geo_vals[t_geo])
        d = _kl_delta(a_s, b_s, bandwidth=kl_bandwidth)

    geo_vals = {
        t: _extract_curvature_values(curvature_evolution_geo, N_geo, t)
        for t in t_values
    }
    for t, vals in geo_vals.items():
        if vals.size == 0:
            warnings.warn(f"Geodesic curvature distribution at t={t} is empty.")

    n = len(t_values)
    distance_matrix = np.full((n, n), np.nan)
    predicted_t = []

    if verbose:
        print(f"\n{'='*62}")
        print(f"  Curvature-Based Temporal Classification")
        print(f"  Standardization     : shared (geodesic-referenced)")
        print(f"  Discrepancy measure : {method_label}")
        print(f"  Geodesic N          : {N_geo}")
        print(f"  t values            : {t_values}")
        print(f"{'='*62}")

    for i, t_real in enumerate(t_values):
        real_vals = _extract_curvature_values(curvature_evolution_real, N_real, t_real)

        if real_vals.size == 0:
            warnings.warn(f"Real graph curvature at t={t_real} is empty.")
            predicted_t.append(np.nan)
            continue

        distances = []
        for j, t_geo in enumerate(t_values):
            if geo_vals[t_geo].size == 0:
                distances.append(np.inf)
                continue

            if method == "scalar":
                d = _scalar_delta_shared(real_vals, geo_vals[t_geo], stat=scalar_stat)
            else:
                # Standardize both arrays using the geodesic's statistics
                a_s, b_s = _standardize_shared(real_vals, geo_vals[t_geo])
                if method == "wasserstein":
                    d = _wasserstein_delta(a_s, b_s)
                else:
                    d = _kl_delta(a_s, b_s, bandwidth=kl_bandwidth)

            distance_matrix[i, j] = d
            distances.append(d)

        best_idx = int(np.argmin(distances))
        t_star = t_values[best_idx]
        predicted_t.append(t_star)

        if verbose:
            dist_str = "  ".join(
                f"t={t_values[j]:.2f}: {distances[j]:.4f}" for j in range(n)
            )
            print(f"\n  Real graph t={t_real:.2f}")
            print(f"    Distances  -> {dist_str}")
            print(f"    Predicted  -> t* = {t_star:.2f}  "
                  f"{'[CORRECT]' if t_star == t_real else '[WRONG]'}")

    # ------------------------------------------------------------------
    # Rank correlation
    # ------------------------------------------------------------------
    true_ranks = np.arange(n, dtype=float)
    pred_ranks = np.array([
        t_values.index(pt) if (pt in t_values and not np.isnan(pt)) else np.nan
        for pt in predicted_t
    ], dtype=float)

    valid = np.isfinite(pred_ranks)
    n_valid = valid.sum()

    if n_valid < 2 or np.all(pred_ranks[valid] == pred_ranks[valid][0]):
        tau, tau_p, rho, rho_p = np.nan, np.nan, np.nan, np.nan
        if verbose:
            print(
                "\n  [WARNING] All predictions collapsed to the same time point. "
                "Rank correlation is undefined."
            )
    else:
        tau, tau_p = kendalltau(true_ranks[valid], pred_ranks[valid])
        rho, rho_p = spearmanr(true_ranks[valid], pred_ranks[valid])

    fmt = lambda v: f"{v:.4f}" if np.isfinite(v) else "nan"
    if verbose:
        print(f"\n{'='*62}")
        print(f"  Rank Correlation Results")
        print(f"{'='*62}")
        print(f"  True t values      : {[f'{t:.2f}' for t in t_values]}")
        print(f"  Predicted t values : {[f'{p:.2f}' if not np.isnan(p) else 'NaN' for p in predicted_t]}")
        print(f"\n  Kendall's tau      : {fmt(tau)}  (p = {fmt(tau_p)})")
        print(f"  Spearman's rho     : {fmt(rho)}  (p = {fmt(rho_p)})")
        print(f"{'='*62}\n")

    return {
        "true_t": t_values,
        "predicted_t": predicted_t,
        "distance_matrix": distance_matrix,
        "kendall_tau": tau,
        "kendall_pvalue": tau_p,
        "spearman_rho": rho,
        "spearman_pvalue": rho_p,
    }


def plot_classification_results(results, t_values, method="wasserstein", figsize=(14, 5)):
    """
    Two-panel figure: discrepancy heatmap (left) and true vs. predicted
    scatter annotated with rank correlations (right).

    Parameters
    ----------
    results  : dict returned by classify_real_graphs()
    t_values : list of float
    method   : str — used only for the heatmap colorbar label
    figsize  : tuple
    """
    dist_mat    = results["distance_matrix"]
    predicted_t = results["predicted_t"]
    tau, tau_p  = results["kendall_tau"],  results["kendall_pvalue"]
    rho, rho_p  = results["spearman_rho"], results["spearman_pvalue"]

    label_map = {
        "wasserstein":   "Wasserstein (shared std)",
        "kl":            "Sym. KL divergence (shared std)",
        "scalar_mean":   "Scalar mean (shared std)",
        "scalar_median": "Scalar median (shared std)",
    }
    method_label = label_map.get(method, method)
    t_labels = [f"{t:.2f}" for t in t_values]
    n = len(t_values)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Heatmap
    im = ax.imshow(dist_mat, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label=method_label)
    ax.set_xticks(range(n)); ax.set_xticklabels(t_labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(t_labels)
    ax.set_xlabel("Geodesic t", fontsize=11)
    ax.set_ylabel("Real graph t", fontsize=11)
    ax.set_title(f"Pairwise Discrepancy\n({method_label})", fontsize=12)
    for i in range(n):
        row = dist_mat[i]
        if np.any(np.isfinite(row)):
            ax.plot(int(np.nanargmin(row)), i, marker="*", color="white",
                    markersize=14, markeredgecolor="black", markeredgewidth=0.6)

    # Scatter
    true_idx = np.arange(n)
    pred_idx = np.array([
        t_values.index(pt) if (not np.isnan(pt) and pt in t_values) else np.nan
        for pt in predicted_t
    ], dtype=float)
    ax2.scatter(true_idx, pred_idx, s=80, zorder=3, color="steelblue", edgecolors="k")
    for i in range(n):
        if np.isfinite(pred_idx[i]):
            ax2.annotate(f"t={t_values[i]:.2f}", (true_idx[i], pred_idx[i]),
                         textcoords="offset points", xytext=(6, 4), fontsize=8.5)
    ax2.plot([0, n-1], [0, n-1], "--", color="gray", linewidth=1, label="Perfect prediction")
    ax2.set_xticks(true_idx);     ax2.set_xticklabels(t_labels)
    ax2.set_yticks(np.arange(n)); ax2.set_yticklabels(t_labels)
    ax2.set_xlabel("True time point", fontsize=11)
    ax2.set_ylabel("Predicted time point", fontsize=11)
    ax2.set_title("True vs. Predicted Time Step\n(shared standardization)", fontsize=12)
    ax2.legend(fontsize=9)

    fmt = lambda v: f"{v:.3f}" if np.isfinite(v) else "nan"
    ax2.text(0.04, 0.96,
             f"Kendall $\\tau$ = {fmt(tau)}  (p = {fmt(tau_p)})\n"
             f"Spearman $\\rho$ = {fmt(rho)}  (p = {fmt(rho_p)})",
             transform=ax2.transAxes, fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    plt.tight_layout()
    plt.show()


def compare_methods(
    curvature_evolution_geo,
    curvature_evolution_real,
    t_values,
    N_geo=None,
    N_real=None,
    kl_bandwidth=None,
):
    """
    Run all four classification variants under shared (geodesic-referenced)
    standardization and print a side-by-side summary.

    Returns
    -------
    results_w       : Wasserstein (geodesic-standardized)
    results_kl      : Sym. KL divergence (geodesic-standardized)
    results_sc_mean : Scalar mean (geodesic-normalized)
    results_sc_med  : Scalar median (geodesic-normalized)
    """
    shared = dict(
        curvature_evolution_geo=curvature_evolution_geo,
        curvature_evolution_real=curvature_evolution_real,
        t_values=t_values, N_geo=N_geo, N_real=N_real,
    )

    print("\n>>> Running Wasserstein (shared std)...")
    results_w = classify_real_graphs(**shared, method="wasserstein")

    print("\n>>> Running KL divergence (shared std)...")
    results_kl = classify_real_graphs(**shared, method="kl", kl_bandwidth=kl_bandwidth)

    print("\n>>> Running scalar mean (shared std)...")
    results_sc_mean = classify_real_graphs(**shared, method="scalar", scalar_stat="mean")

    print("\n>>> Running scalar median (shared std)...")
    results_sc_med = classify_real_graphs(**shared, method="scalar", scalar_stat="median")

    fmt = lambda v: f"{v:.4f}" if np.isfinite(v) else "   nan"
    print("\n" + "="*76)
    print("  Summary — shared (geodesic-referenced) standardization")
    print("="*76)
    print(f"  {'Metric':<22} {'Wasserstein':>12} {'KL':>12} {'Scalar mean':>12} {'Scalar med':>12}")
    print(f"  {'-'*72}")
    for label, key in [
        ("Kendall tau",      "kendall_tau"),
        ("Kendall p-value",  "kendall_pvalue"),
        ("Spearman rho",     "spearman_rho"),
        ("Spearman p-value", "spearman_pvalue"),
    ]:
        print(f"  {label:<22}"
              f" {fmt(results_w[key]):>12}"
              f" {fmt(results_kl[key]):>12}"
              f" {fmt(results_sc_mean[key]):>12}"
              f" {fmt(results_sc_med[key]):>12}")
    print("="*76)

    return results_w, results_kl, results_sc_mean, results_sc_med
