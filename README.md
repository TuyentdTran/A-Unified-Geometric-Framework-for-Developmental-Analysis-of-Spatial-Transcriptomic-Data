# Optimal-Transport-Based-Curvature

## Codes 

### Tuyen 

Here the file GW Geodesic Ricci - Fruit Fly Ex3.ipynb contains the code used to generate the example in section Geodesics and Curvature. Here we implement the algorithm described in From Geodesics to Graphs in the file geodesics.py along with the code for computing the geodesics. In summary this notebook does the following: 
  1. Uses fruit fly high value gene data to construct gene networks for the fruit fly stages E14-16, E16-18, Larvae 1, Larvae 2, and Larvae 3. 
  2. Geodesics between the stages are computed.
  3. The algorithm in From Geodesics to Graphs is applied to construct gene networks for the geodesics.
  4. The Ollivier Ricci curvature and dynamic Ricci curvatures for the geodesic gene networks are computed and plotted. 

### Rank Ordering and Classification

The file Drosophila Classification - Final Version.ipynb contains the code used to implement the rank ordering and classification algorithms from Sections 3.2.1 and 3.2.2 and the results from Section 4.2. The notebook does the following:

  1. Create graphs corresponding to the gene expression networks of each developmental stage (E14-16, E16-18, L1, L2, and L3). In this setting, E14-16 and L3 serve as the initial and terminal graphs between which geodesics are computed, and all five graphs serve as the prototype graphs in the classification experiments.
  2. After computing geodesics between E14-16 and L3, we compute the Ollivier-Ricci curvature distributions for all data: the five prototype graphs, the 40 Gromov-Wasserstein geodesics, and the 50 gene expression networks corresponding to spatial slices at each temporal stage.
  3. Rank ordering is then performed on the 50 spatial slices by comparing their curvature distributions with those of the 40 Gromov-Wasserstein geodesics and plot the results.
  4. Classification is then performed on the 50 spatial slices by comparing their curvature distributions with those of the five prototype graphs for each developmental stage and plot the results.