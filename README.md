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

## Tutorials
Five step-by-step tutorials are included in the Tutorial folder

## Data and Preprocessing Workflow
The raw data files can be located by referring to Data_available.txt, which provides a list of all available datasets and their paths. All data files used in the tutorials can be generated directly through the provided code.


