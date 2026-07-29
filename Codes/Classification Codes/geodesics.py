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
from curvature import compute_curvatures

def compute_GW_geodesic(D1, D2, t_values, N):
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

  for t in t_values:
      if len(D1) == len(D2) and len(D1) == N :
          geodesics.append(ot.gromov.gromov_barycenters(N,
                                                         [D1,D2],
                                                             lambdas= [1-t,t],                 
                                                             maxiter=10000,
                                                                 warmstartT = True,
                                                             init_C= t*D2 + (1-t)*D1,
                                                             ))
      else:
           geodesics.append(ot.gromov.gromov_barycenters(N,
                                                             [D1,D2],
                                                             lambdas= [1-t,t],
                                                         warmstarT = True,
                                                             maxiter=10000,
                                                            ))

  return geodesics

def convert_shortest_path_matrix_to_graph_V2(A, lower_threshold = 0.4, upper_threshold = 1.5) :
    """
    Converts a continuous shortest path matrix to an undirected graph assuming that the graph is not 
    weighted. 

    Input: A shortest path matrix 
    Output: Networkx graph
    """
    adjacency_matrix = sparse.csr_matrix(np.where(np.logical_and(A < upper_threshold , A > lower_threshold),1,0)) #could be built better
    return nx.Graph(adjacency_matrix, nodetype=int)
    
def convert_shortest_path_matrix_to_graph(A) :
    """
    Converts a shortest path matrix to an undirected graph assuming that the graph is not 
    weighted. This can be constructed as the 1 values in the matrix correspond to the adjacency matrix. 

    Input: A shortest path matrix 
    Output: Networkx graph
    """
    adjacency_matrix = sparse.csr_matrix(np.where(A == 1, 1, 0))
    return nx.Graph(adjacency_matrix, nodetype=int)

def threshold_shortest_path_matrix(D,threshold_down = 0.5) : 
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

    return np.ceil(D-threshold_down)

def plot_graphs_and_curvatures(geodesics_graphs,geodesic_time,dynamic_ricii_times,threshold = 0.5) :
    figs = []
    
    for (graph, t) in (zip(geodesics_graphs, geodesic_time)):
        fig, axes = plt.subplots(1,2, figsize=(15, 6))
        nx.draw(graph,ax=axes[0])
        axes[0].set_title(f"t = {t:.2f}")
        kappas = compute_curvatures(graph, dynamic_ricii_times, n_workers=10)
        axes[1].plot(np.log10(dynamic_ricii_times), kappas, c="C0", lw=0.2)
        figs.append(fig)
    return figs

