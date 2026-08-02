"""
Classifies each bootstrap slice against the five real reference curvature
distributions using the same three discrepancy measures already established 
for the geodesic classification method:

    delta_scalar  -- absolute difference of mean or median curvature
    delta_W       -- 1-Wasserstein distance (raw curvature values)
    delta_KL      -- symmetrized KL divergence (standardized KDEs)

Classification rule: nearest-neighbour — each slice is assigned to the stage
whose reference curvature distribution minimizes delta.

Evaluation: classification accuracy and a confusion matrix.  Rank correlation 
is also reported since the stages have a natural temporal ordering.
"""

import numpy as np
import pandas as pd
import warnings
from scipy.stats import kendalltau, spearmanr

from curvature_classification_shared import (
    _extract_curvature_values,
    _wasserstein_delta,
    _kl_delta,
    _scalar_delta_shared,
    _standardize_shared,
)

# Developmental stage ordering (used for ordinal rank correlation)
STAGE_ORDER = ['E14_16', 'E16_18', 'L1', 'L2', 'L3']

# Map stage name to its t-value in the original pipeline
STAGE_T = {
    'E14_16': 0.0,
    'E16_18': 0.0345,
    'L1':     0.1552,
    'L2':     0.5862,
    'L3':     1.0,
}


def extract_reference_distributions(curvature_evolution_real, stage_t_map=STAGE_T):
    """
    Pull one curvature vector per stage from curvature_evolution_real
    (the {N: {t: matrix}} object from the original notebook pipeline).

    Parameters
    ----------
    curvature_evolution_real : dict  {N: {t: curvature_matrix}}
    stage_t_map              : dict  {stage_name: t_value}

    Returns
    -------
    ref_distributions : dict  {stage_name: 1-D np.ndarray of finite curvatures}
    """
    N = list(curvature_evolution_real.keys())[0]
    ref_distributions = {}

    for stage, t in stage_t_map.items():
        vals = _extract_curvature_values(curvature_evolution_real, N, t)
        if vals.size == 0:
            warnings.warn(
                f"Reference curvature distribution for stage '{stage}' "
                f"(t={t}) is empty — this stage will never be predicted."
            )
        ref_distributions[stage] = vals

    return ref_distributions


def classify_slices_against_real(
    curvature_by_slice,
    curvature_evolution_real,
    method='wasserstein',
    scalar_stat='mean',
    kl_bandwidth=None,
    stage_order=STAGE_ORDER,
    stage_t_map=STAGE_T,
    verbose=True,
):
    """
    Classify each bootstrap slice against the five real reference curvature
    distributions using nearest-neighbour assignment under the chosen
    discrepancy measure.

    Parameters
    ----------
    curvature_by_slice : dict  {(stage, bootstrap_idx): curvature_matrix}
        Output of compute_slice_curvatures() from multi_slice_classification.py.
        The stage key is the *true* stage label and is used for evaluation.
    curvature_evolution_real : dict  {N: {t: curvature_matrix}}
        Original five-graph curvature object from the notebook.
    method      : {'wasserstein', 'kl', 'scalar'}
    scalar_stat : {'mean', 'median'}  (used when method='scalar')
    kl_bandwidth : float or None  (KDE bandwidth for KL; None = Scott's rule)
    stage_order : list of str  -- defines the temporal ordering for rank corr.
    stage_t_map : dict  {stage: t_value}
    verbose     : bool

    Returns
    -------
    results_df : pd.DataFrame  with one row per slice, columns:
                     true_stage, bootstrap_idx, predicted_stage, correct,
                     true_rank, predicted_rank, min_discrepancy,
                     + one column per stage with its discrepancy value
    summary    : dict with accuracy, kendall_tau, spearman_rho, and their p-values
    """
    if method not in ('wasserstein', 'kl', 'scalar'):
        raise ValueError(f"method must be 'wasserstein', 'kl', or 'scalar', got '{method}'.")

    # Build reference distributions once
    ref_dists = extract_reference_distributions(curvature_evolution_real, stage_t_map)
    ref_stages = [s for s in stage_order if s in ref_dists and ref_dists[s].size > 0]

    method_label = {
        'wasserstein': 'Wasserstein (raw)',
        'kl':          f'Sym. KL divergence (standardized, bw={kl_bandwidth or "Scott"})',
        'scalar':      f'Scalar {scalar_stat}',
    }[method]

    if verbose:
        print(f"\n{'='*66}")
        print(f"  Slice-to-Stage Classification")
        print(f"  Discrepancy : {method_label}")
        print(f"  References  : {ref_stages}")
        print(f"  Slices      : {len(curvature_by_slice)} total")
        print(f"{'='*66}")

    rows = []

    for (true_stage, bootstrap_idx), curv_mat in curvature_by_slice.items():
        # Extract finite upper-triangle curvature values for this slice
        mask = np.triu(np.ones_like(curv_mat, dtype=bool), k=1)
        slice_vals = curv_mat[mask]
        slice_vals = slice_vals[np.isfinite(slice_vals)]

        if slice_vals.size == 0:
            warnings.warn(
                f"Empty curvature array for {true_stage}_bootstrap_{bootstrap_idx} "
                f"— skipping."
            )
            continue

        # Compute discrepancy against each reference stage
        discrepancies = {}
        for ref_stage in ref_stages:
            ref_vals = ref_dists[ref_stage]

            if method == 'scalar':
                # Scalar: reference distribution plays the role of 'geo_vals'
                d = _scalar_delta_shared(slice_vals, ref_vals, stat=scalar_stat)

            elif method == 'wasserstein':
                # Raw values — no standardization, so location and scal
                d = _wasserstein_delta(slice_vals, ref_vals)

            else:  # 'kl'
                # Standardize using the reference distribution's statistics
                a_s, b_s = _standardize_shared(slice_vals, ref_vals)
                d = _kl_delta(a_s, b_s, bandwidth=kl_bandwidth)

            discrepancies[ref_stage] = d

        # Nearest-neighbour assignment
        predicted_stage = min(discrepancies, key=discrepancies.get)
        correct = (predicted_stage == true_stage)

        true_rank = stage_order.index(true_stage) if true_stage in stage_order else np.nan
        pred_rank = stage_order.index(predicted_stage) if predicted_stage in stage_order else np.nan

        row = dict(
            true_stage=true_stage,
            bootstrap_idx=bootstrap_idx,
            predicted_stage=predicted_stage,
            correct=correct,
            true_rank=true_rank,
            predicted_rank=pred_rank,
            min_discrepancy=discrepancies[predicted_stage],
        )
        for s in ref_stages:
            row[f'd_{s}'] = discrepancies[s]

        rows.append(row)

        if verbose:
            status = 'CORRECT' if correct else f'WRONG -> {predicted_stage}'
            print(f"  {true_stage}_bootstrap_{bootstrap_idx:02d}  [{status}]  "
                  f"min_d={discrepancies[predicted_stage]:.4f}")

    results_df = pd.DataFrame(rows)
    results_df['true_stage'] = pd.Categorical(
        results_df['true_stage'], categories=stage_order, ordered=True
    )
    results_df['predicted_stage'] = pd.Categorical(
        results_df['predicted_stage'], categories=stage_order, ordered=True
    )
    results_df = results_df.sort_values(['true_stage', 'bootstrap_idx']).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    accuracy = results_df['correct'].mean()

    valid = results_df[['true_rank', 'predicted_rank']].dropna()
    if len(valid) < 2 or valid['predicted_rank'].nunique() < 2:
        tau, tau_p, rho, rho_p = np.nan, np.nan, np.nan, np.nan
    else:
        tau, tau_p = kendalltau(valid['true_rank'], valid['predicted_rank'])
        rho, rho_p = spearmanr(valid['true_rank'], valid['predicted_rank'])

    summary = dict(
        method=method_label,
        n_slices=len(results_df),
        accuracy=accuracy,
        n_correct=int(results_df['correct'].sum()),
        kendall_tau=tau,
        kendall_pvalue=tau_p,
        spearman_rho=rho,
        spearman_pvalue=rho_p,
    )

    fmt = lambda v: f'{v:.4f}' if np.isfinite(v) else 'nan'
    if verbose:
        print(f"\n{'='*66}")
        print(f"  Results — {method_label}")
        print(f"  Accuracy       : {summary['n_correct']} / {summary['n_slices']} "
              f"= {accuracy:.4f}")
        print(f"  Kendall tau    : {fmt(tau)}  (p = {fmt(tau_p)})")
        print(f"  Spearman rho   : {fmt(rho)}  (p = {fmt(rho_p)})")
        print(f"{'='*66}\n")

    return results_df, summary


def confusion_matrix_df(results_df, stage_order=STAGE_ORDER):
    """
    Return a DataFrame confusion matrix with true stages as rows and
    predicted stages as columns.  Cell (i, j) is the number of slices
    from true stage i predicted as stage j.
    """
    cm = pd.DataFrame(0, index=stage_order, columns=stage_order)
    for _, row in results_df.iterrows():
        ts = row['true_stage']
        ps = row['predicted_stage']
        if ts in stage_order and ps in stage_order:
            cm.loc[ts, ps] += 1
    cm.index.name = 'True \\ Predicted'
    return cm


def compare_methods_slice_classification(
    curvature_by_slice,
    curvature_evolution_real,
    kl_bandwidth=None,
    stage_order=STAGE_ORDER,
    stage_t_map=STAGE_T,
):
    """
    Run all four classification variants (Wasserstein, KL, scalar mean,
    scalar median) and print a side-by-side accuracy/correlation summary,
    plus a confusion matrix for each.

    Returns
    -------
    dict keyed by method string, values are (results_df, summary) tuples.
    """
    variants = [
        ('wasserstein', 'mean',   'Wasserstein'),
        ('kl',          'mean',   'KL divergence'),
        ('scalar',      'mean',   'Scalar mean'),
        ('scalar',      'median', 'Scalar median'),
    ]

    all_results = {}
    summaries = []

    for method, scalar_stat, label in variants:
        print(f"\n>>> {label} ...")
        df, summary = classify_slices_against_real(
            curvature_by_slice=curvature_by_slice,
            curvature_evolution_real=curvature_evolution_real,
            method=method,
            scalar_stat=scalar_stat,
            kl_bandwidth=kl_bandwidth,
            stage_order=stage_order,
            stage_t_map=stage_t_map,
            verbose=False,
        )
        summary['label'] = label
        all_results[label] = (df, summary)
        summaries.append(summary)

    # Summary table
    fmt = lambda v: f'{v:.4f}' if isinstance(v, float) and np.isfinite(v) else str(v)
    print(f"\n{'='*76}")
    print(f"  Slice-to-Stage Classification — Method Comparison")
    print(f"{'='*76}")
    header = f"  {'Method':<20} {'Accuracy':>10} {'N correct':>10} " \
             f"{'Kendall tau':>13} {'Spearman rho':>13}"
    print(header)
    print(f"  {'-'*72}")
    for s in summaries:
        print(f"  {s['label']:<20} {fmt(s['accuracy']):>10} "
              f"{s['n_correct']:>10} "
              f"{fmt(s['kendall_tau']):>13} "
              f"{fmt(s['spearman_rho']):>13}")
    print(f"{'='*76}\n")

    # Confusion matrices
    for label, (df, _) in all_results.items():
        print(f"  Confusion matrix — {label}")
        cm = confusion_matrix_df(df, stage_order)
        print(cm.to_string())
        print()

    return all_results