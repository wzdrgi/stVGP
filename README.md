# stVGP
A variational spatiotemporal Gaussian process framework designed to integrate multi-modal, multi-slice spatial transcriptomics (ST) data for coherent 3D tissue reconstruction
<a href=" " target="_blank">< img src="https://s2.loli.net/2025/07/22/Y36aTcKHVmROlrW.jpg" /></a >

# Installation
The stVGP package is developed based on the Python libraries Scanpy, PyTorch and PyG (PyTorch Geometric) framework, and can be run on GPU (recommend) or CPU. Before installing stVGP, please ensure that Scanpy, PyTorch, and PyG (PyTorch Geometric) are already installed. These dependencies are required for stVGP to function properly, but they are not automatically installed during the installation process to allow greater flexibility.

## Install stVGP
    pip install stVGP
The use of the mclust algorithm requires the rpy2 package (Python) and the mclust package (R). See https://pypi.org/project/rpy2/ and https://cran.r-project.org/web/packages/mclust/index.html for detail.

## Tutorials
Five step-by-step tutorials are included in the Tutorial folder

## Data and Preprocessing Workflow
The raw data files can be located by referring to Data_available.txt, which provides a list of all available datasets and their paths. All data files used in the tutorials can be generated directly through the provided code.


