# stVGP
A variational spatiotemporal Gaussian process framework designed to integrate multi-modal, multi-slice spatial transcriptomics (ST) data for coherent 3D tissue reconstruction.


![Fig 1.jpg](https://s2.loli.net/2025/08/07/dFUaJNTtykIsrCW.jpg)

# Installation
The stVGP package is developed based on the Python libraries Scanpy, PyTorch and PyG (PyTorch Geometric) framework, and can be run on GPU (recommend) or CPU. Before installing stVGP, please ensure that Scanpy, PyTorch, and PyG (PyTorch Geometric) are already installed. These dependencies are required for stVGP to function properly, but they are not automatically installed during the installation process to allow greater flexibility.

## Install stVGP
    pip install stVGP
The use of the mclust algorithm requires the rpy2 package (Python) and the mclust package (R). See https://pypi.org/project/rpy2/ and https://cran.r-project.org/web/packages/mclust/index.html for detail.

## Computing environment
Python environment：  
Python==3.8.19  
anndata==0.12.6  
numpy==1.24.3  
pandas==2.0.3  
Pillow==12.0.0  
scanpy==1.11.5  
scikit_learn==1.3.0  
scipy==1.10.1  
torch==1.13.1+cu117  
torch_geometric==2.7.0  
torchvision==0.14.1+cu117  
tqdm==4.65.0  

R environment：  
R==4.2.3  
Seurat==4.3.0  
sp==1.6-0  
spdep==1.3-3  
progress==1.2.2  

## Quick-start tutorial
Here, we provide guidance on using the stVGP sample data to help you quickly get started with our method. Here we provide two datasets for testing: the human dorsolateral prefrontal cortex (DLPFC) dataset (can be download at https://figshare.com/authors/Zedong_Wang/20593784) and the human developing heart dataset (can be download at https://figshare.com/authors/Zedong_Wang/20593784).

## DLPFC dataset  
import packages
```python
import stVGP as stvg
import scanpy as sc
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
```
# Read data
```python
# Read data
# This dataset can be downloaded directly from Figshare (https://figshare.com/authors/Zedong_Wang/20593784). 
# The raw data and citations can be found in the data description.
slices_use = [151673,151674,151675,151676]
adata_list_raw = []
for slice_id in slices_use:
    adata_i = sc.read("C:/Users/wzd/Desktop/setting/project/DLPFC/{}.h5ad".format(slice_id))
    adata_list_raw.append(adata_i)
```
# Data Preprocessing
```python
adata_list = stvg.st_preprocess(adata_list_raw,min_cells = 100)
```

```python
# Analysing spatial regionality genes
# Analysed with the help of R
# If a multi-slice spatial feature gene analysis is not performed, the first analysis is used by default
adata_st_list = stvg.select_gene(adata_list,
                                 save_data = True,
                                 savepath = 'C:/Users/wzd/Desktop/setting/project/DLPFC/select_gene/')
# Next, the genes will be analyzed using seurat and the spatial morans analysis will be performed on the filtered genes. 
# The details of the analysis are in gene_select.R
```

```python
After the previous code step completes, a file named "select_gene_9".txt will appear under the "savepath" specified in the select_gene function. 
In this step, we need to input this file into the stVGP R module for further analysis. 
Detailed code can be found in the Spatial genetic analysis DLPFC.R file. 
Only need to modify the input and output paths to complete the analysis of spatially differential genes.

Please note that the results selected here are highly version-dependent. 
Therefore, if you encounter execution issues at this step due to version incompatibility, 
you can proceed directly with data analysis by referencing our execution results (recommended). 
Alternatively, you may review our R execution environment and make necessary adjustments. 
At this step, you will obtain a new file in the path you specified, with the default name gene_morans_9.txt.
```
Read genetic information
```python
# Read genetic information
gene_morans_result = np.genfromtxt('C:/Users/wzd/Desktop/setting/project/DLPFC/select_gene/gene_morans_9.txt',
                                   dtype=np.str0,
                                   skip_header=1,
                                   delimiter ='\t')

sorted_indices = np.argsort(gene_morans_result[:,-1])[::-1]
# Select Genes
top_morans_indices = sorted_indices[:10]
select_gene_final = gene_morans_result[top_morans_indices,0]
```

```python
# Alignment using spatial information
# Gene expression parallel alignment
adata_st_list = stvg.gene_rigid_mapping_alignment(gene_input = select_gene_final,stadata_input = adata_st_list,align_model = "single_template_alignment",)
adata_st_list = stvg.STN_rigid_alignment(stadata_input = adata_st_list, select_gene_final = select_gene_final)
```

```python
# Save the result after alignment
# for adata in adata_st_list:
#     silce_id = str(int(adata.obs['slice_id'][0]))
#     adata.write("C:/Users/wzd/Desktop/setting/project/DLPFC/{}_slice.h5ad".format(silce_id))
```

```python
# Domain and batch effect
# Re-enter data
data_path = 'C:/Users/wzd/Desktop/setting/project/DLPFC/'
slice_idx = [151673, 151674, 151675, 151676]
adata_DLPFC_list = []
for slice_name in slice_idx:
    file_path = data_path + str(slice_name) + '_slice.h5ad'
    adata = sc.read(file_path)
    adata = adata[~adata.obs['layer'].isna()]
    adata_DLPFC_list.append(adata)
```

```python
# Processing relationships between data for the stVGP model
# 
slice_matrix,adj_matrix = stvg.adata_preprocess_adjnet(input_adata = adata_DLPFC_list,align_model = 'single_template_alignment')
```
```python
```
```python
```
```python
```
```python
```
```python
```
```python
```
```python
```
## Tutorials
Five step-by-step tutorials are included in the Tutorial folder

## Data and Preprocessing Workflow
The raw data files can be located by referring to Data_available.txt, which provides a list of all available datasets and their paths. All data files used in the tutorials can be generated directly through the provided code.


