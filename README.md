# RLMA-DR

This is the PyTorch implementation for paper "Relation-Level Diffusion with multi-level Attention on Heterogeneous Graphs for Drug Repositioning".
## Environment:
The codes of RLMA-DR are implemented and tested under the following development environment:
-  Python 3.8.19
-  cudatoolkit 11.5
-  pytorch 1.10.0
-  dgl 0.9.1
-  networkx 3.1
-  numpy 1.24.3
-  scikit-learn 1.3.0

## Datasets
We verify the effectiveness of our proposed method on three commonly-used benchmarks, i.e., <i>B-dataset, C-dataset, </i>and <i>F-dataset</i>.
| Dataset |  Drug |  Disease |  Protein |  Drug-Disease | Drug-Protein |  Disease-Protein|
|:-------:|:--------:|:--------:|:--------:|:-------:| :-------:| :-------:|
|B-dataset   | $269$ | $598$| $1021$ | $18416$ | $3110$ | $5898$ |
|C-dataset   | $663$ | $409$| $993$ | $2532$ | $3672$ | $10691$ |
|F-dataset   | $592$ | $313$| $2741$ | $1933$ | $3152$ | $47470$ |

