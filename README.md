# stVGP
A variational spatiotemporal Gaussian process framework designed to integrate multi-modal, multi-slice spatial transcriptomics (ST) data for coherent 3D tissue reconstruction.


![Fig 1.jpg](https://s2.loli.net/2025/08/07/dFUaJNTtykIsrCW.jpg)

# Installation
The stVGP package is developed based on the Python libraries Scanpy, PyTorch and PyG (PyTorch Geometric) framework, and can be run on GPU (recommend) or CPU. Before installing stVGP, please ensure that Scanpy, PyTorch, and PyG (PyTorch Geometric) are already installed. These dependencies are required for stVGP to function properly, but they are not automatically installed during the installation process to allow greater flexibility. For specific versions, refer to the Computing environment module or consult Python_requires.txt and R_requires.txt.

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

## Gene selection
Due to the strong version dependency of the R module in stVGP during gene selection, users encountering version-related issues preventing code execution may directly reference our results for subsequent alignment analysis. Alignment results are available for download at https://figshare.com/authors/Zedong_Wang/20593784 website.

## Quick-start tutorial
Here, we provide guidance on using the stVGP sample data to help you quickly get started with our method. Here we provide two datasets for testing: the human dorsolateral prefrontal cortex (DLPFC) dataset (can be download at https://figshare.com/authors/Zedong_Wang/20593784) and the human developing heart dataset (can be download at https://figshare.com/authors/Zedong_Wang/20593784).

Here, we first demonstrate the stVGP domain identification and gene expression prediction workflow on the DLPFC dataset. Subsequently, we demonstrate the stVGP batch-clearing workflow on the human cardiac dataset. Since only batch-clearing needs to be shown for the cardiac dataset, we skip the slice alignment process. Note that if users skip the alignment process, they should manually set `use_batch` to True in the `train_stVGP` function. All datasets can be download at https://figshare.com/authors/Zedong_Wang/20593784. Please replace the path for saving and reading data during the process with the location where you store your data.

DLPFC

import packages
```python
import stVGP as stvg
import scanpy as sc
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
```
Read data
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
Data preprocessing
```python
adata_list = stvg.st_preprocess(adata_list_raw,min_cells = 100)
```
Spatially specific gene selection
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
(can be download at https://figshare.com/authors/Zedong_Wang/20593784).
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
Align multiple slices
```python
# Alignment using spatial information
# Gene expression parallel alignment
adata_st_list = stvg.gene_rigid_mapping_alignment(gene_input = select_gene_final,stadata_input = adata_st_list,align_model = "single_template_alignment",)
adata_st_list = stvg.STN_rigid_alignment(stadata_input = adata_st_list, select_gene_final = select_gene_final)
```
Save data
```python
# Save the result after alignment
# for adata in adata_st_list:
#     silce_id = str(int(adata.obs['slice_id'][0]))
#     adata.write("C:/Users/wzd/Desktop/setting/project/DLPFC/{}_slice.h5ad".format(silce_id))
```
Domain and batch effect
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
Processing relationships
```python
# Processing relationships between data for the stVGP model
# Here, the selection of align_model must be consistent with the choice made in the gene_rigid_mapping_alignment function.
slice_matrix,adj_matrix = stvg.adata_preprocess_adjnet(input_adata = adata_DLPFC_list,align_model = 'single_template_alignment')
```
Run stVGP
```python
# Model training
recon_x, embedding, model_params,logvar = stvg.train_stVGP(
        ST_need_reconstruction_matrix = slice_matrix,
        all_spatial_net = adj_matrix,
        lr = 0.001,
        weight_decay = 0.0001,
        training_epoch = 1200,
        num_heads = 1,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        save_model = False,
        save_model_path = 'path',
        hidden_embedding = [512,24],
        random_seed = 112,
        optimize_method = 'adam',
        whether_gradient_clipping = False,
        gradient_clipping = 5.0,
        all_gat = True)
```
Save data
```python
# Save embedded layer data
# np.savetxt('C:/Users/wzd/Desktop/setting/project/DLPFC/embedding.txt',embedding,fmt='%s')
```
Cluster and compare
```python
# Domain result
from mclustpy import mclustpy
from sklearn.metrics import adjusted_rand_score

embedding = np.loadtxt('C:/Users/wzd/Desktop/setting/project/DLPFC/embedding.txt')

data_dim = np.cumsum(np.array([adata_DLPFC_list[0].X.shape[0],
                               adata_DLPFC_list[1].X.shape[0],
                               adata_DLPFC_list[2].X.shape[0],
                               adata_DLPFC_list[3].X.shape[0]]))
data_dim = np.insert(data_dim,0,0)

true_labels = np.vstack((np.array(adata_DLPFC_list[0].obs['layer']).reshape(-1,1),
                        np.array(adata_DLPFC_list[1].obs['layer']).reshape(-1,1),
                        np.array(adata_DLPFC_list[2].obs['layer']).reshape(-1,1),
                        np.array(adata_DLPFC_list[3].obs['layer']).reshape(-1,1))).ravel()
cluster_num = len(np.unique(true_labels))

res = mclustpy(embedding, G=cluster_num, modelNames='EEE', random_seed=1)
pre_labels = np.array(res['classification'])
ari = adjusted_rand_score(true_labels, pre_labels)
```
Save data and results
```python
# Storing Model Results
AP_z_slice_0 = [10.] * adata_DLPFC_list[0].X.shape[0]
AP_z_slice_1 = [20.] * adata_DLPFC_list[1].X.shape[0]
AP_z_slice_2 = [320.] * adata_DLPFC_list[2].X.shape[0]
AP_z_slice_3 = [330.] * adata_DLPFC_list[3].X.shape[0]

AP_z = np.vstack((np.array(AP_z_slice_0).reshape(-1,1),
                 np.array(AP_z_slice_1).reshape(-1,1),
                 np.array(AP_z_slice_2).reshape(-1,1),
                 np.array(AP_z_slice_3).reshape(-1,1)))

spatial_spots = np.vstack((np.array(adata_DLPFC_list[0].obsm['align_spatial']),
                          np.array(adata_DLPFC_list[1].obsm['align_spatial']),
                          np.array(adata_DLPFC_list[2].obsm['align_spatial']),
                          np.array(adata_DLPFC_list[3].obsm['align_spatial'])))

spatial_spots = np.hstack((spatial_spots,AP_z))

# np.savetxt('C:/Users/wzd/Desktop/setting/project/DLPFC/alignment_spatial.txt',spatial_spots,fmt='%s')
# np.savetxt('C:/Users/wzd/Desktop/setting/project/DLPFC/cluster.txt',pre_labels,fmt='%s')
# torch.save(model_params, "C:/Users/wzd/Desktop/setting/project/DLPFC/model.pth")
```
Prediction all slices
```python
# Mask-based prediction of gene expression
slice_idx = [151673, 151674, 151675, 151676]

spatial = np.loadtxt('C:/Users/wzd/Desktop/setting/project/DLPFC/alignment_spatial.txt')
embedding = np.loadtxt("C:/Users/wzd/Desktop/setting/project/DLPFC/embedding.txt")
domain = np.loadtxt("C:/Users/wzd/Desktop/setting/project/DLPFC/cluster.txt")

for slice_index,slice_name in enumerate(slice_idx):

    Train_spatial_up = spatial[:data_dim[slice_index]]
    Train_spatial_down = spatial[data_dim[slice_index+1]:]

    Embedding_up = embedding[:data_dim[slice_index]]
    Embedding_down = embedding[data_dim[slice_index+1]:]

    Pred_spatial = spatial[data_dim[slice_index]:data_dim[slice_index+1]]
    Pred_embedding_true = embedding[data_dim[slice_index]:data_dim[slice_index+1]]

    Train_spatial = np.vstack((Train_spatial_up,Train_spatial_down))
    Train_embedding = np.vstack((Embedding_up,Embedding_down))

    pred_embedding = stvg.get_3D_prediction(
                train_coordinates = Train_spatial,
                embedding = Train_embedding,
                spatial_pred = Pred_spatial,
                noise = False,
                noise_value = 0.00001,
                constant_value = 1.0,
                Rbf_value = 1024)
    
    # np.savetxt('C:/Users/wzd/Desktop/setting/project/DLPFC/{}_prediction.txt'.format(slice_idx[slice_index]),pred_embedding,fmt='%s')
```
Read our prediction data
```python
# Splicing prediction embedding for multi-slice gene reconstruction
prediction_embedding_151673 = np.loadtxt('C:/Users/wzd/Desktop/setting/project/DLPFC/151673_prediction.txt')
prediction_embedding_151674 = np.loadtxt('C:/Users/wzd/Desktop/setting/project/DLPFC/151674_prediction.txt')
prediction_embedding_151675 = np.loadtxt('C:/Users/wzd/Desktop/setting/project/DLPFC/151675_prediction.txt')
prediction_embedding_151676 = np.loadtxt('C:/Users/wzd/Desktop/setting/project/DLPFC/151676_prediction.txt')
model_checkpoint = torch.load("C:/Users/wzd/Desktop/setting/project/DLPFC/model.pth")

prediction_embedding = np.vstack((prediction_embedding_151673,prediction_embedding_151674,prediction_embedding_151675,prediction_embedding_151676))
prediction_embedding = torch.tensor(prediction_embedding,dtype=torch.float32)
```
Prepare gene prediction
```python
# Perform relational reconstruction for data reconstruction, utilizing gene expression data for approximation processing.
prediction_embedding = torch.tensor(prediction_embedding,dtype=torch.float32)
slice_matrix = torch.tensor(slice_matrix,dtype=torch.float32)
edge_list = []
edge_list.append(adj_matrix.row.tolist())
edge_list.append(adj_matrix.col.tolist())
adj_tensor = torch.LongTensor(edge_list)
```
Complete gene expression prediction
```python
# Complete gene expression prediction
Prediction_gene_expression = stvg.gene_prediction(
    slice_matrix = slice_matrix,
    prediction_embedding = prediction_embedding,           
    adj_matrix = adj_tensor,                        
    checkpoint = model_checkpoint,                         
    model_layer = [slice_matrix.shape[1],512,24,1],                        
    all_gat = True,                            
    logvar = None,                             
    device = torch.device('cuda:0')                           
)
```
Done!


Human heart dataset
```python
# Import packages and read raw data
import stVGP as stvg
import scanpy as sc
import anndata as ad
import numpy as np
import warnings
import torch
warnings.filterwarnings("ignore")

# Read data
# This dataset can be downloaded directly from Figshare (https://figshare.com/authors/Zedong_Wang/20593784). 
# The raw data and citations can be found in the data description.

# Here, we will perform the stVGP elimination batch workflow.
# Read data from different developmental stages and batches

# To visually demonstrate stVGP's ability to eliminate batch effects across different developmental stages, 
# we have omitted the cross-period alignment step here.

slices_use_45 = ["Human_heart_1","Human_heart_2","Human_heart_3",
                 "Human_heart_4"]

slices_use_65 = ["Human_heart_1","Human_heart_2","Human_heart_3",
                 "Human_heart_4","Human_heart_5","Human_heart_6",
                 "Human_heart_7","Human_heart_8","Human_heart_9"]

slices_use_9 = ["Human_heart_1","Human_heart_2","Human_heart_3",
                "Human_heart_4","Human_heart_5","Human_heart_6",]

adata_list_45 = []
adata_list_65 = []
adata_list_9 = []

for slice_id in slices_use_45:
    adata_i = sc.read("C:/Users/wzd/Desktop/setting/project/Human heart/4.5-5PCW/{}.h5ad".format(slice_id))
    adata_list_45.append(adata_i)

for slice_id in slices_use_65:
    adata_i = sc.read("C:/Users/wzd/Desktop/setting/project/Human heart/6.5PCW/{}.h5ad".format(slice_id))
    adata_list_65.append(adata_i)

for slice_id in slices_use_9:
    adata_i = sc.read("C:/Users/wzd/Desktop/setting/project/Human heart/9PCW/{}.h5ad".format(slice_id))
    adata_list_9.append(adata_i)
```
Data Preprocessing
```python
# Data Preprocessing
# Batch Data Preprocessing
adata_list = adata_list_45 + adata_list_65 + adata_list_9
for i, adata in enumerate(adata_list):
    adata.obs_names = str(i+1) + 'x' + adata.obs_names.astype(str)
ad_concat = ad.concat(adata_list)
adata_list = stvg.Batch_preprocess(adata_list,clear = True)
adata_re_concat = ad.concat(adata_list)
```
Prepare data
```python
# Extract whole-slide expression data
ST_need_reconstruction_matrix = adata_re_concat.X.toarray()
all_spatial_net = None
use_batch = True
```
Run stVGP
```python
# Please note that the output of stVGP may vary slightly depending on the specific image and batch settings selected.
recon_x, embedding, model_params, inference_outputs, generative_outputs = stvg.train_stVGP(
                                            ST_need_reconstruction_matrix = ST_need_reconstruction_matrix,
                                            all_spatial_net = all_spatial_net,
                                            use_batch = use_batch,
                                            batch_key = 'slice_id',
                                            adata_infor = adata_re_concat,
                                            use_image = False,
                                            adata_infor_image = None,
                                            GP_set = False,
                                            GP_spatial_infor = None,
                                            lr = 1e-3,
                                            weight_decay = 1e-4,
                                            training_epoch = 1000,
                                            num_heads = 1,
                                            device= torch.device('cuda' if torch.cuda.is_available() else 'cpu') ,
                                            save_model = False,
                                            save_model_path = 'path',
                                            hidden_embedding = [512,32],
                                            random_seed = 42,
                                            optimize_method = 'adam',
                                            whether_gradient_clipping = False,
                                            gradient_clipping = 5.0,
                                            all_gat = False,
                                            )
```
Save data
```python
# Storing hidden layer data and reslicing(if needed)
adata_re_concat.obsm['embedding'] = embedding
unique_slices = adata_re_concat.obs['slice_id'].unique()
adata_list = [
        adata_re_concat[adata_re_concat.obs['slice_id'] == s_id].copy() 
        for s_id in unique_slices
    ]
```
Batch correction
```python
# Batch correction and Result Visualization
sc.pp.neighbors(adata_re_concat, use_rep='embedding')
sc.tl.umap(adata_re_concat)
sc.pl.umap(adata_re_concat, color='slice_id', 
           title='Batch (Slice) Info',
           wspace=0.4)
```
Done!

For more data analysis and details, please refer to the stVGP tutorial (https://github.com/wzdrgi/stVGP/tree/main/Tutorial).

## Tutorials
For all datasets step-by-step tutorials are included in the Tutorial folder (https://github.com/wzdrgi/stVGP/tree/main/Tutorial). The tutorial folder contains all modules and reproduces some results and images from the stVGP paper.

## Data and Preprocessing Workflow
The raw data files can be located by referring to Data_available.txt, which provides a list of all available datasets and their paths. All data files used in the tutorials can be generated directly through the provided code.

## Computing Environment
Can refer to requires.txt and R_requires.txt for environment configuration. We conducted all data analysis on the Win10.

## Documentation
1. Spatial gene selection module: The function "select_gene" completes the rasterization of the space and the labeling of spatial tags.    
```
    Key Parameter:  
    Parameter "input_adata_list": A list of multi-slice spatial transcriptomes arranged in sequential order.
    Parameter "ref_adata_num": Index of adata to be analyzed spatially genetically.
    Parameter "spot_make": Number of x and y subspace divisions. After completion, the entire space will be partitioned into spot_make² number of subspaces.  
    Parameter "save_data": Whether or not to save.
    Parameter "key_words": keywords for spatial coordinates of the transcriptomics datasets. stVGP will search for data.obsm[key_words] 
    Parameter "savepath": Save location. When save_data was Ture, "savepath" can not be None. 
```
2. Rigid Alignment and STN Alignment: The function "gene_rigid_alignment" performs rigid alignment on all slices. It requires input of multi-slice information, selected gene information, and the alignment mode. The function "STN_rigid_alignment" performs non-rigid alignment on all slices and integrates rigid alignment. This mode must be run after multi-slice rigid alignment. It requires input of multi-slice information, selected gene information, and the alignment mode.  
```
    "gene_rigid_alignment" Key Parameter:  
    Parameter "stadata_input": A list of multi-slice spatial transcriptomes arranged in sequential order.  
    Parameter "gene_input": Selected list of spatial genes.  
    Parameter "ini_spatial","add_spatial": The original spatial coordinate keywords stored in the .obsm within the adata (anndata) and the new keywords storing the aligned coordinates after alignment slicing.  
    Parameter "align_model": The selected alignment model will either align all slices with a single slice or sequentially align all slices. Select 'single_template_alignment' or 'sequential_alignment'  
    Parameter "gene_input_list": When "align_model" is set to "sequential_alignment", "gene_input_list" is required: a two-dimensional list containing the gene selection results for each slice.  
    Parameter "ref_label": When aligning all slices with a template slice, the index of the diaphragm slice.  
    Parameter "align_method": Choosing between the ICP algorithm and the optimization algorithm.
    Parameter "icp_iterations","maxiter": Maximum iteration count for ICP algorithm or optimization algorithm.
    ###
    "STN_rigid_alignment" Key Parameter:
    Parameter "stadata_input": A list of multi-slice spatial transcriptomes arranged in sequential order.
    Parameter "select_gene_final": Selected list of spatial genes.
    Parameter "ref_label": When aligning all slices with a template slice, the index of the diaphragm slice.
    Parameter "ini_spatial": The original spatial coordinate keywords stored in the .obsm within the adata (anndata).  
    Parameter "STN_alignment_key": The rigid alignment spatial coordinate keywords stored in the .obsm within the adata (anndata).
    Parameter "add_spatial": The new keywords storing the aligned coordinates after alignment slicing.
    Parameter "gene_input_list": When "align_model" is set to "sequential_alignment", "gene_input_list" is required: a two-dimensional list containing the gene selection results for each slice.
    Parameter "align_model": The selected alignment model will either align all slices with a single slice or sequentially align all slices. Select 'single_template_alignment' or 'sequential_alignment'.
    Parameter "alignment_epoch": Total training rounds for alignment.
    Parameter "device": The device used for computation defaults to CUDA. When CUDA is unavailable, it falls back to the CPU.  
    Parameter "quantiles": Truncation range of the alignment process. 
    Parameter "attention": Enable attention mechanism fusion. Default to false.  
```
3. stVGP Preprocessing Module. stVGP provides extensive data preprocessing methods to facilitate adaptation to different data types. The primary functions are "st_preprocess", "adata_preprocess_adjnet", and "spatial_reconstruction". "st_preprocess" is primarily designed for raw data processing and is responsible for data cleansing. "adata_preprocess_adjnet" is responsible for converting data into the format suitable for training stVGP models. "spatial_reconstruction" primarily performs further processing on the data.   
```
    "st_preprocess" Key Parameter:  
    Parameter "input_adata_list": A list of multi-slice spatial transcriptomes arranged in sequential order.
    Parameter "n_hvg_group": Number of highly variable genes for reference anndata.
    Parameter "flavor": Methods for selecting highly variable genes.
    Parameter "min_genes": Minimum number of genes expressed required for a cell to pass filtering.
    Parameter "min_cells": Minimum number of cells expressed required for a gene to pass filtering.
    ###
    "adata_preprocess_adjnet" Key Parameter:  
    Parameter "input_adata": A list of AnnData objects (spatial transcriptomics slices), usually arranged in sequential order.
    Parameter "align_model": The alignment strategy to use. Options are 'single_template_alignment' (align all to one reference) or 'sequential_alignment' (align adjacent slices).
    Parameter "ref_label": The index of the reference slice in the input list (only used when align_model is 'single_template_alignment').
    Parameter "spatial_label": The key in `.obsm` that stores the aligned spatial coordinates to be used for network construction.
    Parameter "add_net_keywords_self": The key name used to store or retrieve the intra-slice adjacency matrix in `.obsm`.
    Parameter "n_neighbors": The number of nearest neighbors to select when constructing the spatial adjacency matrix.
    Parameter "no_cross": If True, the function will only construct intra-slice networks and exclude cross-slice (inter-slice) connections.
    ###
    "spatial_reconstruction" Key Parameter:  
    Parameter "adata": The input AnnData object containing spatial transcriptomics data.
    Parameter "alpha": The smoothing coefficient. It controls the weight of the neighbor-averaged expression in the final reconstruction (X_rec = alpha * smoothed_X + X).
    Parameter "n_neighbors": The number of spatial neighbors used to construct the nearest neighbor graph.
    Parameter "n_pcs": The number of principal components used to calculate cosine similarity weights between neighbors.
    Parameter "use_highly_variable": Whether to use highly variable genes when computing PCA.
    Parameter "normalize_total": Whether to normalize the data (normalize total counts per cell) before processing.
    Parameter "copy": If True, returns a new AnnData object; if False, modifies the input AnnData in place.
```
4. Domain and batch correction:The function “train_stVGP” requires input of merged slice gene expression data, spatial information, and selection of modes such as whether to eliminate batch effects or perform cross-modal fusion. Additionally, different selection modes yield distinct return details that must be examined within the code (https://github.com/wzdrgi/stVGP/blob/main/stVGP.py).
```
    Key Parameter:  
    Parameter "ST_need_reconstruction_matrix": The expression data after full-slice stitching, with dimensions spot * features.
    Parameter "all_spatial_net": Spatial adjacency across slices. "ST_need_reconstruction_matrix" and "all_spatial_net" can be directly obtained through stVGP's adata_preprocess_adjnet.
    Parameter "use_batch": If set to True, stVGP employs an encoding method for batch correction; otherwise, it uses the index mapping strategy for batch correction.
    Parameter "adata_infor": Information for all slices, represented as a list of all slices. If "use_batch" is true, the adata_infor should not be None.
    Parameter "batc_key": The storage keyword for batch information of all slices is stored in `adata.obsm[batc_key]`.
    Parameter "use_image": If set to True, stVGP performs cross-modal analysis using image modalities.
    Parameter "adata_infor_image": Storage of image information, representing a list of all slice adata. When "use_image" is set to True, "adata_infor_image" cannot be set to `None`.
    Parameter "GP_set": Switch for enabling Gaussian processes in hidden layers. 
    Parameter "GP_spatial_infor": After enabling the Gaussian process, all spatial coordinates of points required by the Gaussian process must be provided. When GP_set is set to True, GP_spatial_infor cannot be empty.
    Parameter "lr": stVGP learning rate, default value 0.001.
    Parameter "weight_decay": Weight decay (L2 penalty) coefficient, default value 1e-4.
    Parameter "training_epoch": The number of training iterations for stVGP, defaulting to 1500.
    Parameter "num_heads": The number of heads in the multi-head attention mechanism, defaulting to 1.
    Parameter "device": The device used for computation defaults to CUDA. When CUDA is unavailable, it falls back to the CPU..
    Parameter "save_model": Whether to save model parameters. Default is False. If set to True, manually configure the save_model_path..
    Parameter "save_model_path": Model storage path. When `save_model` is set to True, do not leave this field empty.
    Parameter "hidden_embedding": The hidden layer parameters for stVGP are specified as a list containing two numbers: the first number represents the hidden layer dimension, and the second number represents the embedding dimension.
    Parameter "random_seed": The random seed used for computation.
    Parameter "optimize_method": The optimizer selected by "optimize_method", defaulting to Adam.
    Parameter "whether_gradient_clipping": Enable gradient clipping. If set to Yes, configure "gradient_clipping".
    Parameter "gradient_clipping": Gradient clipping value, default is 5.0.
    Parameter "all_gat": all_gat, whether to replace the model's linear layer with GATConv.  
```
5. Gene prediction:The functions “get_3D_prediction” and “gene_prediction” predict the latent layer information and final gene expression of virtual slices, respectively. get_3D_prediction requires input of trained latent layer information and virtual spatial coordinates, while gene_prediction requires input of predicted latent layer information and initial slice expression information. They return the latent layer and gene expression of virtual slices, respectively.  
```
    Key Parameter:
    Parameter "train_coordinates": Spatial coordinates of the hidden layer used for training. 
    Parameter "embedding": Hidden layer information for training. The rows in embedding and train_coordinates should correspond to each other. The number of spots used for training.
    Parameter "spatial_pred": Spatial coordinate points requiring stVGP for prediction. 
    Parameter "noise": Whether to introduce white noise. 
    noise_value "noise": Noise level.
    constant_value "noise": Constant washout for Gaussian processes.
    Rbf_value "noise": Smoothness of Gaussian processes.   
```
