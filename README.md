# paper_MICLE

This is the PyTorch implementation for paper "Multi-view Contrastive Learning for Drug Repositioning on Heterogeneous Biological Networks".

## Introduction
This paper presents a novel Multi-view Contrastive Learning method for identifying underlying DDAs on biological networks (abbreviated as MICLE). MICLE consists of four crucial components, i.e., node representation learning, interview CL, intra-view CL and DDA predictor. The primary innovations lie in the effective characterization of graph heterogeneity and the design of two complementary CL objectives. To the best of our knowledge, it is the first time that graph heterogeneity is sufficiently characterized in the GCL paradigm devised for DDA prediction without resort to stochastic perturbation augmentation.

<img src='MICLE_figure.png'>

## Environment:
The codes of MICLE are implemented and tested under the following development environment:
-  Python 3.8.19
-  cudatoolkit 11.5
-  pytorch 1.10.0
-  dgl 0.9.1
-  networkx 3.1
-  numpy 1.24.3
-  scikit-learn 1.3.0

## Datasets
We verify the effectiveness of our proposed method on three commonly-used benchmarks, i.e., <i>B-dataset, C-dataset, </i>and <i>F-dataset</i>.
| Dataset |  Drug |  Disease |  Protein |  Drug-Disease | Drug-Protein |  Disease-Protein | Sparsity |
|:-------:|:--------:|:--------:|:--------:|:-------:| :-------:| :-------:| :-------:|
|B-dataset   | $269$ | $598$| $1021$ | $18416$ | $3110$ | $5898$ | $11.45\%$|
|C-dataset   | $663$ | $409$| $993$ | $2532$ | $3672$ | $10691$ | $0.93\%$|
|F-dataset   | $592$ | $313$| $2741$ | $1933$ | $3152$ | $47470$ | $1.04\%$|

These datasets can be downloaded from [google drive](https://drive.google.com/drive/folders/1w9orlSgM_HlwGwaVWPLYgRqbjdQc7RCv). Herein, we elaborate on the corresponding data files.
- <i>DrugFingerprint.csv</i>: The drug fingerprint similarities between each drug pairs.
- <i>DrugGIP.csv</i>: The drug Gaussian interaction profile (GIP) kernel similarities between each drug pairs.
- <i>DiseasePS.csv</i>: The disease phenotype similarities between each disease pairs.
- <i>DiseaseGIP.csv</i>: The disease GIP kernel similarities between each disease pairs.
- <i> DrugDiseaseAssociationNumber.csv </i>: The known drug-disease associations.
- <i> DrugProteinAssociationNumber.csv </i>: The known drug-protein associations.
- <i> ProteinDiseaseAssociationNumber.csv </i>: The known disease-protein associations.
- <i> Drug_mol2vec.csv </i>: The 300-dimensional mol2vec embeddings of drugs, obtained from paper [Mol2vec: unsupervised machine learning approach with chemical intuition. Journal of chemical information and modeling, 2018, 58(1): 27-35](https://pubmed.ncbi.nlm.nih.gov/29268609/).
- <i> DiseaseFeature.csv </i>: The 64-dimensional MeSH embeddings of diseases, obtained from paper [MeSHHeading2vec: a new method for representing MeSH headings as vectors based on graph embedding algorithm. Briefings in bioinformatics, 2021, 22(2): 2085-2095](https://academic.oup.com/bib/article/22/2/2085/5813844).
- <i> Protein_ESM.csv </i>: The 320-dimensional ESM-2 embeddings of proteins, obtained from paper [Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 2023, 379(6637): 1123-1130](https://www.science.org/doi/abs/10.1126/science.ade2574).

Above datasets utilized in this paper are consistent with the previous work [AMDGT: Attention aware multi-modal fusion using a dual graph transformer for drug–disease associations prediction. Knowledge-Based Systems, 2024, 284: 111329](https://github.com/JK-Liu7/AMDGT).


## Code Files:
The introduction of each <code> py </code> file is as follows:
- <i>contrastive_learning.py</i>: The implementation of inter-view and intra-view contrastive learning.
- <i>data_preprocessing.py</i>: The implementation of data preprocessing.
- <i>graph_transformer_layer.py</i>: The implementation of graph transformer layer.
- <i>graph_transformer.py</i>: The implementation of basic graph transformer.
- <i>metric.py</i>: The implementation of evaluation metrics.
- <i>model.py</i>: The implementation of entire MICLE model.
- <i>main.py</i>: The implementation of model training.
- <i>parse_args.py</i>: The parameter settings.


## How to Run the Code:
Please firstly download the datasets and unzip the downloaded files. Next, create the <code>Datasets/</code> folder and move the unzipped datasets into this folder. The command to train MICLE on the B-dataset, C-dataset or F-dataset is as follows.

<ul>
<li>B-dataset<pre><code>python main.py --dataset = B-dataset</code></pre>
</li>
<li>C-dataset<pre><code>python main.py --dataset = C-dataset</code></pre>
</li>
<li>F-dataset<pre><code>python main.py --dataset = F-dataset</code></pre>
</li>
</ul>
</body></html>
