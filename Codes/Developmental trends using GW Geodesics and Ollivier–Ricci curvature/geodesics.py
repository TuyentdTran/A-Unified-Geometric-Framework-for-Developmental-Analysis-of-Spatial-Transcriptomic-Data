import ot
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random, sys
import networkx as nx
from scipy import sparse
import os
import sys

from scipy.stats import false_discovery_control

from geometric_clustering.curvature import compute_curvatures, compute_OR_curvature


def compute_GW_geodesic(D1, D2, t_values, N, inf_treshold=1e9, initialize=True):
    """Compute the Gromov-Wasserstein geodesic between two distance matrices D1 and D2 using
    Gabriel Peyré, Marco Cuturi, and Justin Solomon works

    Consider two metric measure spaces (X, d_0, mu_0) and (Y, d_1, mu_1).
    In this application, X and Y are graphs,
    d_0 and d_1 are the l2 distance,
    mu_0 and mu_1 are assumed uniform distribution (for now)

    The geodesic distance matrix (or (t, 1 - t) weighted barycenter) is computed
    using the ot.gromov.entropic_gromov_barycenters function, based on the above
    paper. This avoids expanding into the product space of Sturm's work
    """

    geodesics = []

    # handle inf values in distance matrices

    D1 = np.nan_to_num(D1, posinf=inf_treshold)
    D2 = np.nan_to_num(D2, posinf=inf_treshold)

    for t in t_values:
        if len(D1) == len(D2) and len(D1) == N:
            if initialize:
                init = t * D2 + (1 - t) * D1
            else:
                init = None
            geodesics.append(
                ot.gromov.gromov_barycenters(
                    N,
                    [D1, D2],
                    lambdas=[1 - t, t],
                    maxiter=10000,
                    warmstartT=True,
                    init_C=init,
                    verbose=True,
                    symmetric=True,
                    stop_criterion="barycenter",
                    tol=1e-9,
                    armijo=True,
                )
            )
        else:
            geodesics.append(
                ot.gromov.gromov_barycenters(
                    N,
                    [D1, D2],
                    lambdas=[1 - t, t],
                    warmstarT=False,
                    maxiter=10000,
                    symmetric=True,
                )
            )

    return geodesics


def convert_shortest_path_matrix_to_graph_V2(
    A, lower_threshold=0.4, upper_threshold=1.5
):
    """
    Converts a continuous shortest path matrix to an undirected graph assuming that the graph is not
    weighted.

    Input: A shortest path matrix
    Output: Networkx graph
    """
    adjacency_matrix = sparse.csr_matrix(
        np.where(np.logical_and(A < upper_threshold, A > lower_threshold), 1, 0)
    )  # could be built better
    return nx.Graph(adjacency_matrix, nodetype=int)


def convert_shortest_path_matrix_to_graph(A):
    """
    Converts a shortest path matrix to an undirected graph assuming that the graph is not
    weighted. This can be constructed as the 1 values in the matrix correspond to the adjacency matrix.

    Input: A shortest path matrix
    Output: Networkx graph
    """
    adjacency_matrix = sparse.csr_matrix(np.where(A == 1, 1, 0))
    return nx.Graph(adjacency_matrix, nodetype=int)


def threshold_shortest_path_matrix(D, threshold_down=0.5):
    """
    Thresholds a shortest path matrix that contains non-integer values (i.e from the geodesics).
    We are assuming non-weighted graphs throughout, so need to convert these continuous distance matrices
    back to integers.

    Note that in doing this thresholding. We technically are creating matrices that might no longer
    be actual shortest path matrices. As they may not represent an actual graph anymore (values may be incompatible).
    However all that actually matters for our purpose is the adjacency matrix (i.e 1 values) hiding inside the shortest
    path matrix, which does correspond to a real graph.

    Input:
        D: matrix D representing some continuous values
        threshold_down: [i,i+threshold_down] -> i , (i+threshold_down, i+1] -> i+1 ,for integers i
    Output:
        A: D thresholded to the ints
    """

    return np.ceil(D - threshold_down)


def compute_GW_Geodesics_weighted(
    g1,
    g2,
    t_values,
    threshold_down=0.5,
    threshold_up=1.5,
    inf_treshold=1e9,
    weight_inf_treshold=0,
    delete_high_weights=False,
):
    """
    Computes a geodesic for weighted graphs g1 and g2 by spltting the process into 2 geodesic computations
    """

    # First compute the unweighted geodesics to get the graph structures
    D1 = nx.floyd_warshall_numpy(g1, weight=None)
    D2 = nx.floyd_warshall_numpy(g2, weight=None)

    geodesics = compute_GW_geodesic(
        D1,
        D2,
        t_values,
        nx.number_of_nodes(g1),
        inf_treshold=inf_treshold,
    )
    geodesic_graphs = [
        convert_shortest_path_matrix_to_graph_V2(D, threshold_down, threshold_up)
        for D in geodesics
    ]
    # Now compute the weighted geodesic matrices
    D1 = nx.floyd_warshall_numpy(g1)
    D2 = nx.floyd_warshall_numpy(g2)
    geodesics_weighted = compute_GW_geodesic(
        D1,
        D2,
        t_values,
        nx.number_of_nodes(g1),
        inf_treshold=weight_inf_treshold,
    )
    # add the weights to the already computed graph structure
    for g, D in zip(geodesic_graphs, geodesics_weighted):
        for u, v in g.edges:
            g.edges[u, v]["weight"] = D[u, v]
            if delete_high_weights and abs(D[u, v]) >= inf_treshold / 10.0:
                g.remove_edge(u, v)

    # error estimate
    errors = [
        np.linalg.norm(weighted_geodesic - nx.floyd_warshall_numpy(inducedgraph))
        for weighted_geodesic, inducedgraph in zip(geodesics_weighted, geodesic_graphs)
    ]

    return geodesic_graphs, errors


def compute_GW_Geodesics_weighted_alternative(
    g1, g2, t_values, threshold_down=0.1, threshold_up=1e3, inf_treshold=1e9
):
    """
    Computes a geodesic for weighted graphs g1 and g2 by spltting by assuming every weight is direct a connection
    """

    # Just compute
    D1 = nx.floyd_warshall_numpy(
        g1,
    )
    D2 = nx.floyd_warshall_numpy(g2)

    geodesics = compute_GW_geodesic(
        D1, D2, t_values, nx.number_of_nodes(g1), inf_treshold=inf_treshold
    )
    geodesic_graphs = [
        convert_shortest_path_matrix_to_graph_V2(D, threshold_down, threshold_up)
        for D in geodesics
    ]
    # add the weights to the already computed graph structure,
    for g, D in zip(geodesic_graphs, geodesics):
        for u, v in g.edges:
            g.edges[u, v]["weight"] = D[u, v]

    return geodesic_graphs


def plot_graphs(geodesics_graphs, geodesic_time):
    figs = []
    pos = nx.spring_layout(geodesics_graphs[0])
    cmap = plt.cm.Blues

    for graph, t in zip(geodesics_graphs, geodesic_time):
        fig, axes = plt.subplots(1, 1)

        if nx.is_weighted(graph):
            weights = nx.get_edge_attributes(graph, "weight").values()
            nx.draw(
                graph,
                pos,
                edge_color=weights,
                width=2.0,
                edge_cmap=cmap,
                edge_vmin=min(weights),
                edge_vmax=max(weights),
            )
        else:
            nx.draw(graph, pos, ax=axes)
        axes.set_title(f"t = {t:.2f}")
        figs.append(fig)
    return figs


def plot_graphs_and_curvatures(geodesics_graphs, geodesic_time, dynamic_ricii_times):
    figs = []
    kappas = []
    pos = nx.spring_layout(geodesics_graphs[0])

    for graph, t in zip(geodesics_graphs, geodesic_time):
        fig, axes = plt.subplots(1, 2)
        nx.draw(graph, pos, ax=axes[0])
        axes[0].set_title(f"t = {t:.2f}")
        kappa = compute_curvatures(graph, dynamic_ricii_times, n_workers=4)
        kappas.append(kappa)
        axes[1].plot(np.log10(dynamic_ricii_times), kappa, c="C0", lw=0.2)
        figs.append(fig)
    return figs, kappas


def plot_graphs_and_curvature_weighted(
    geodesics_graphs,
    geodesic_time,
    dynamic_ricii_times,
    edge_vmin=0,
    edge_vmax=1,
    pos=None,
):
    figs = []
    kappas = []
    if pos is None:
        pos = nx.spring_layout(geodesics_graphs[0])
    cmap = plt.cm.Blues

    for graph, t in zip(geodesics_graphs, geodesic_time):
        fig, axes = plt.subplots(1, 2)
        _, weights = zip(*nx.get_edge_attributes(graph, "weight").items())
        edges = [
            tuple(sorted(edge)) for edge in graph.edges
        ]  # sorting required for consistency to make subgraph operations easier
        nx.draw(
            graph,
            pos,
            ax=axes[0],
            edge_color=weights,
            edge_cmap=cmap,
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax,
        )
        axes[0].set_title(f"t = {t:.2f} Number of Edges: {len(edges)}")

        # compute the curvatures of each disconnected component separately.
        connected_vertices = nx.connected_components(graph)
        S = [graph.subgraph(c) for c in connected_vertices]

        kappa = np.zeros((len(dynamic_ricii_times), graph.number_of_edges()))
        for subgraph in S:
            if subgraph.order() > 1:
                # Their curvature code doesn't handle subgraphs (well any non-continuous indexing of vertices)
                # so copy it
                subgraph_copy = nx.convert_node_labels_to_integers(
                    subgraph, first_label=0, ordering="sorted"
                )
                kappa_sub = compute_curvatures(
                    subgraph_copy,
                    dynamic_ricii_times,
                    n_workers=4,
                    weighted_curvature=True,
                )
                subgraph_edges = [tuple(sorted(edge)) for edge in subgraph.edges]
                # map edges from subgraph to edges in full graph
                # TODO could be slightly improved as subgraph edges is ordered (i.e only need to search from full_edge onwards)
                subgraph_indices = [
                    i
                    for i, full_edge in enumerate(edges)
                    if full_edge in subgraph_edges
                ]

                kappa[:, subgraph_indices] = kappa_sub

        kappas.append(kappa)
        axes[1].plot(np.log10(dynamic_ricii_times), kappa, c="C0", lw=0.2)
        figs.append(fig)
    return figs, kappas


def compute_curvature_dynamic(geodesics_graphs, dynamic_ricii_times):
    kappas = []

    for graph in geodesics_graphs:
        edges = [
            tuple(sorted(edge)) for edge in graph.edges
        ]  # sorting required for consistency to make subgraph operations easier
        # compute the curvatures of each disconnected component separately.
        connected_vertices = nx.connected_components(graph)
        S = [graph.subgraph(c) for c in connected_vertices]

        kappa = np.zeros((len(dynamic_ricii_times), graph.number_of_edges()))
        for subgraph in S:
            if subgraph.order() > 1:
                # Their curvature code doesn't handle subgraphs (well any non-continuous indexing of vertices)
                # so copy it
                subgraph_copy = nx.convert_node_labels_to_integers(
                    subgraph, first_label=0, ordering="sorted"
                )
                kappa_sub = compute_curvatures(
                    subgraph_copy,
                    dynamic_ricii_times,
                    n_workers=4,
                    weighted_curvature=True,
                )
                subgraph_edges = [tuple(sorted(edge)) for edge in subgraph.edges]
                # map edges from subgraph to edges in full graph
                # TODO could be slightly improved as subgraph edges is ordered (i.e only need to search from full_edge onwards)
                subgraph_indices = [
                    i
                    for i, full_edge in enumerate(edges)
                    if full_edge in subgraph_edges
                ]

                kappa[:, subgraph_indices] = kappa_sub

        kappas.append(kappa)
    return kappas


def compute_curvature_OR(
    geodesics_graphs,
):
    kappas = []
    for graph in geodesics_graphs:
        edges = [
            tuple(sorted(edge)) for edge in graph.edges
        ]  # sorting required for consistency to make subgraph operations easier

        # compute the curvatures of each disconnected component separately.
        connected_vertices = nx.connected_components(graph)
        S = [graph.subgraph(c) for c in connected_vertices]

        kappa = np.zeros(graph.number_of_edges())
        for subgraph in S:
            if subgraph.order() > 1:
                # Their curvature code doesn't handle subgraphs (well any non-continuous indexing of vertices)
                # so copy it
                subgraph_copy = nx.convert_node_labels_to_integers(
                    subgraph, first_label=0, ordering="sorted"
                )
                kappa_sub = compute_OR_curvature(
                    subgraph_copy,
                    weighted_curvature=True,
                )
                subgraph_edges = [tuple(sorted(edge)) for edge in subgraph.edges]
                # map edges from subgraph to edges in full graph
                # TODO could be slightly improved as subgraph edges is ordered (i.e only need to search from full_edge onwards)
                subgraph_indices = [
                    i
                    for i, full_edge in enumerate(edges)
                    if full_edge in subgraph_edges
                ]
                kappa[subgraph_indices] = kappa_sub
        kappas.append(kappa)
    return kappas


def plot_avg_curvature(geodesic_time, kappa_values):
    avg = np.zeros(len(geodesic_time))
    for i, kappa in enumerate(kappa_values):
        avg[i] = np.average(kappa)

    plt.figure()
    plt.title("Average Ricci Curvature")
    plt.plot(geodesic_time, avg)


def plot_dynamic_curvature(geodesic_time, dynamic_ricii_times, kappa_values):
    figs = []
    for geotime, kappa in zip(geodesic_time, kappa_values):
        fig = plt.figure()
        plt.title(f"Dynamic Ricci Curvature for Geodesic {geotime}")
        plt.plot(np.log10(dynamic_ricii_times), kappa, c="C0", lw=0.2)
        figs.append(fig)
    return figs


def plot_dynamic_averages(geodesic_time, dynamic_ricii_times, kappa_values):
    figs = []

    avg = np.zeros(len(dynamic_ricii_times), len(geodesic_time))
    for i, kappa in enumerate(kappa_values):
        avg[:, i] = np.average(kappa)

    fig = plt.figure()
    plt.title("Average Dynamic Ricci Curvatures")
    plt.plot(geodesic_time, avg)
