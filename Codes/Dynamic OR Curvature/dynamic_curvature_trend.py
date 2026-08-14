#!/usr/bin/env python3
"""Compute 19-scale dynamic OR curvature on the five stage networks."""

from pathlib import Path
import importlib
import os
import sys
import tempfile
import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "Data" / "full_correlations"
OUTPUT_DIR = HERE / "output"
TAUS = np.logspace(-2, 1, 19)
THRESHOLD = 0.2
STAGES = {
    "E14_16": ("E14-16", "corr_E14_16.csv"),
    "E16_18": ("E16-18", "corr_E16_18.csv"),
    "L1": ("L1", "corr_L1.csv"),
    "L2": ("L2", "corr_L2.csv"),
    "L3": ("L3", "corr_L3.csv")}
EXPECTED = {
    "E14_16": (171, 2499, 48), "E16_18": (171, 2533, 43), "L1": (171, 2677, 50),
    "L2": (171, 1675, 54), "L3": (171, 2840, 44)}


def load_compute_curvatures():
    temp = tempfile.TemporaryDirectory()
    package = Path(temp.name) / "geometric_clustering"
    package.mkdir()
    (package / "__init__.py").write_text(f"__path__.append({str(ROOT / 'geometric_clustering')!r})\n")
    sys.path.insert(0, temp.name)
    os.environ["PYTHONPATH"] = temp.name + os.pathsep + os.environ.get("PYTHONPATH", "")
    for name in ["geometric_clustering.curvature", "geometric_clustering.io", "geometric_clustering"]:
        sys.modules.pop(name, None)
    return temp, importlib.import_module("geometric_clustering.curvature").compute_curvatures


def build_graph(path):
    data = pd.read_csv(path).iloc[:, 1:].to_numpy(dtype=float, copy=True)
    if data.shape != (171, 171) or not np.isfinite(data).all():
        raise ValueError(f"Invalid correlation matrix: {path}")
    np.fill_diagonal(data, 0.5)
    return nx.from_numpy_array(np.where(np.abs(data * 2 - 1) > THRESHOLD, data, 0.0))


def compute_stage(graph, checkpoint_file, compute_curvatures):
    edges = [tuple(sorted(edge)) for edge in graph.edges]
    kappa = np.zeros((len(TAUS), graph.number_of_edges()))
    for vertices in nx.connected_components(graph):
        subgraph = graph.subgraph(vertices)
        if subgraph.number_of_nodes() <= 1:
            continue
        copied = nx.convert_node_labels_to_integers(subgraph, first_label=0, ordering="sorted")
        curv = compute_curvatures(copied, TAUS, n_workers=1, use_spectral_gap=True, measure_cutoff=1e-6, sinkhorn_regularisation=0, weighted_curvature=True, filename=str(checkpoint_file))
        sub_edges = [tuple(sorted(edge)) for edge in subgraph.edges]
        indices = [i for i, edge in enumerate(edges) if edge in sub_edges]
        kappa[:, indices] = curv
    if not np.isfinite(kappa).all():
        raise ValueError("Non-finite curvature values produced")
    return kappa, edges


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    package_temp, compute_curvatures = load_compute_curvatures()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "curvature.pkl"
            for order, (key, (label, filename)) in enumerate(STAGES.items(), start=1):
                graph = build_graph(DATA_DIR / filename)
                fingerprint = (graph.number_of_nodes(), graph.number_of_edges(), nx.number_connected_components(graph))
                if fingerprint != EXPECTED[key]:
                    raise ValueError(f"Unexpected graph for {key}: {fingerprint}; expected {EXPECTED[key]}")
                kappa, edges = compute_stage(graph, checkpoint, compute_curvatures)
                for ti, tau in enumerate(TAUS):
                    for edge_index, (u, v) in enumerate(edges):
                        rows.append((label, order, tau, np.log10(tau), edge_index, u, v, kappa[ti, edge_index]))
                print(f"{label}: {graph.number_of_edges()} edges, {len(TAUS)} tau values")
    finally:
        package_temp.cleanup()
    out = pd.DataFrame(rows, columns=["stage", "stage_order", "tau", "log10_tau", "edge_index", "node_u", "node_v", "curvature"])
    path = OUTPUT_DIR / "curvatures_19tau.csv.gz"
    out.to_csv(path, index=False, compression="gzip", float_format="%.18e")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
