# A Unified Geometric Framework for Developmental Analysis of Spatial Transcriptomic Data

This repository accompanies the paper *A Unified Geometric Framework for Developmental Analysis of Spatial Transcriptomic Data*

## Codes 

### Preprocessing 

 - Wisdm_corr_matrix_revised.ipynb implements the preprocessing procedure described in section 4.1.
 - wisdm_coot.ipynb contains the code to generate the results for section 4.5
 - Wisdm_corr_matrix_slice_ids.ipynb implements the preprocessing procedure described in appendix B.2

### Developmental trends using GW Geodesics and Ollivier–Ricci curvature 

Here the file Drosophilia GW Geodesics and Ricci curvatures.ipynb contains the code used to generate the example in section 4.3. The figures 6,7 and 13 and tables 2 and 4 are generated from this files.  The file geodesics.py implements the algorithm described in section 3.1.1.

In summary this notebook does the following: 
  1. Uses fruit fly high value gene data to construct gene networks for the fruit fly stages E14-16, E16-18, Larvae 1, Larvae 2, and Larvae 3. 
  2. Geodesics between the stages are computed.
  3. The algorithm in section 3.1.1 is applied to construct gene networks for the geodesics.
  5. The Ollivier Ricci curvature for the gene networks are computed and a stitched version is formed.
  6. The error computation in section 4.3 are computed.

### Rank Ordering and Classification

The file Drosophila Classification - Final Version.ipynb contains the code used to implement the rank ordering and classification algorithms from Sections 3.2.1 and 3.2.2 and the results from Section 4.2. The notebook does the following:

  1. Create graphs corresponding to the gene expression networks of each developmental stage (E14-16, E16-18, L1, L2, and L3). In this setting, E14-16 and L3 serve as the initial and terminal graphs between which geodesics are computed, and all five graphs serve as the prototype graphs in the classification experiments.
  2. After computing geodesics between E14-16 and L3, we compute the Ollivier-Ricci curvature distributions for all data: the five prototype graphs, the 40 Gromov-Wasserstein geodesics, and the 50 gene expression networks corresponding to spatial slices at each temporal stage.
  3. Rank ordering is then performed on the 50 spatial slices by comparing their curvature distributions with those of the 40 Gromov-Wasserstein geodesics and plot the results.
  4. Classification is then performed on the 50 spatial slices by comparing their curvature distributions with those of the five prototype graphs for each developmental stage and plot the results.

### Dynamic OR Curvature

This folder contains the analysis used for the dynamic Ollivier--Ricci curvature results in Sections 3.4 and 4.4. The scripts do the following:

  1. Compute distance-scaled dynamic OR curvature at 19 log-spaced diffusion scales from 0.01 to 10.
  2. Compute stage summaries, pairwise Wasserstein distances, log-scale averages, and the consecutive-transition counts.
  3. Create the ECDF figures and the appendix histogram/KDE figures. 

