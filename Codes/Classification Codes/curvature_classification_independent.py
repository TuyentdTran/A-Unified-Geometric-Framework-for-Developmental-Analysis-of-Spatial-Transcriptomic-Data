"""
Classifies real graphs to geodesic time points using OR curvature distributions,
with INDEPENDENT standardization applied before every pairwise comparison.

Standardization convention
--------------------------
Before each pairwise discrepancy is computed, both distributions are
standardized independently: each is shifted by its own mean and rescaled by
its own standard deviation, producing zero-mean, unit-variance samples.

Scalar methods
------------------------------------
Independent standardization forces the mean of every distribution to exactly
zero, so |mean(a_std) - mean(b_std)| = 0 for all pairs, making scalar-mean
uninformative. For scalar-median the result is near-zero but not exactly zero
(the median of a standardized distribution is not generally zero unless the
distribution is symmetric). For this reason both scalar methods operate on the
unstandardized distributions.

Three choices of delta
----------------------
  - 'wasserstein' : 1-D Wasserstein distance (no standardization applied).
  - 'kl'          : Symmetrized KL divergence on independently standardized KDEs.
  - 'scalar'      : Absolute difference of mean or median on RAW samples
                    (standardization not applied; see note above).
"""

import numpy as np
from scipy.stats import wasserstein_distance, kendalltau, spearmanr, gaussian_kde
import matplotlib.pyplot as plt
import warnings


def _standardize_independent(a, b):
    """
    Standardize each array independently to zero mean and unit variance.

    Returns the standardized copies (a_s, b_s). The originals are not
    modified. Arrays with near-zero variance are left unscaled to avoid
    division by zero.
    """
    a_s = (a - a.mean()) / max(a.std(), 1e-8)
    b_s = (b - b.mean()) / max(b.std(), 1e-8)
    return a_s, b_s


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


def _scalar_delta(a, b, stat="mean"):
    """
    Absolute difference between a scalar summary statistic of two RAW arrays.

    Parameters
    ----------
    a, b : 1-D arrays of curvature samples (raw, unstandardized)
    stat : {'mean', 'median'}
    """
    if stat == "mean":
        return abs(np.mean(a) - np.mean(b))
    elif stat == "median":
        return abs(np.median(a) - np.median(b))
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
    distribution is closest under the chosen discrepancy (with independent
    standardization applied for distributional methods), then evaluate
    temporal ordering via rank correlation.

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
        Discrepancy measure. Wasserstein and KL apply independent
        standardization; scalar uses raw values (see module docstring).
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

    use_standardization = (method == "kl")

    if method == "wasserstein":
        delta_fn = _wasserstein_delta
        method_label = "Wasserstein (raw, no std)"
    elif method == "kl":
        delta_fn = lambda a, b: _kl_delta(a, b, bandwidth=kl_bandwidth)
        method_label = "Sym. KL divergence (independent std)"
    else:
        delta_fn = lambda a, b: _scalar_delta(a, b, stat=scalar_stat)
        method_label = f"Scalar ({scalar_stat}) [raw, no std]"

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
        print(f"  Standardization     : independent")
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

            if use_standardization:
                a, b = _standardize_independent(real_vals, geo_vals[t_geo])
            else:
                a, b = real_vals, geo_vals[t_geo]

            d = delta_fn(a, b)
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
    dist_mat   = results["distance_matrix"]
    predicted_t = results["predicted_t"]
    tau, tau_p = results["kendall_tau"],  results["kendall_pvalue"]
    rho, rho_p = results["spearman_rho"], results["spearman_pvalue"]

    label_map = {
        "wasserstein": "Wasserstein (indep. std)",
        "kl":          "Sym. KL divergence (indep. std)",
        "scalar_mean": "Scalar mean (raw)",
        "scalar_median": "Scalar median (raw)",
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
    ax2.set_xticks(true_idx);    ax2.set_xticklabels(t_labels)
    ax2.set_yticks(np.arange(n)); ax2.set_yticklabels(t_labels)
    ax2.set_xlabel("True time point", fontsize=11)
    ax2.set_ylabel("Predicted time point", fontsize=11)
    ax2.set_title("True vs. Predicted Time Step\n(independent standardization)", fontsize=12)
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
    Run all four classification variants under independent standardization
    and print a side-by-side summary.

    Returns
    -------
    results_w       : Wasserstein (independently standardized)
    results_kl      : Sym. KL divergence (independently standardized)
    results_sc_mean : Scalar mean (raw)
    results_sc_med  : Scalar median (raw)
    """
    shared = dict(
        curvature_evolution_geo=curvature_evolution_geo,
        curvature_evolution_real=curvature_evolution_real,
        t_values=t_values, N_geo=N_geo, N_real=N_real,
    )

    print("\n>>> Running Wasserstein (independent std)...")
    results_w = classify_real_graphs(**shared, method="wasserstein")

    print("\n>>> Running KL divergence (independent std)...")
    results_kl = classify_real_graphs(**shared, method="kl", kl_bandwidth=kl_bandwidth)

    print("\n>>> Running scalar mean (raw)...")
    results_sc_mean = classify_real_graphs(**shared, method="scalar", scalar_stat="mean")

    print("\n>>> Running scalar median (raw)...")
    results_sc_med = classify_real_graphs(**shared, method="scalar", scalar_stat="median")

    fmt = lambda v: f"{v:.4f}" if np.isfinite(v) else "   nan"
    print("\n" + "="*76)
    print("  Summary — independent standardization")
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
