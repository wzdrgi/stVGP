import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch.nn.functional as F
import torch.nn as nn

import math
import torch
import sklearn.neighbors
import random
import os

from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from torch_geometric.nn.conv import GATConv
from torch.autograd import Variable
from torch_geometric.data import Data
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,RBF
from tqdm import tqdm
from scipy.sparse import isspmatrix

from numpy.typing import NDArray
from anndata import AnnData

def st_preprocess(input_adata_list: list, # list of spatial transcriptomics datasets
                n_hvg_group: int = 5000,  # number of highly variable genes for reference anndata
                flavor: str = "seurat",   # methods for selecting highly variable genes
                min_genes = 1,            # minimum number of genes expressed in a cell
                min_cells=1               # minimum number of cells expressed in a gene
                ):            
    
    adata_list_copy = input_adata_list.copy()
    print("Finding highly variable genes...")
    adata_common_list = []

    for i in range(len(adata_list_copy)):
        adata_st_new = adata_list_copy[i]
        adata_st_new.var_names_make_unique()

        # Remove mt
        adata_st_new = adata_st_new[:,(np.array(~adata_st_new.var.index.str.startswith("mt-")) & np.array(~adata_st_new.var.index.str.startswith("MT-")))]
        # Remove cells and genes with 0 counts
        sc.pp.filter_cells(adata_st_new, min_genes = min_genes)
        sc.pp.filter_genes(adata_st_new, min_cells = min_cells)
        sc.pp.normalize_total(adata_st_new, inplace=True, target_sum=1e4)
        sc.pp.log1p(adata_st_new)
        # Select hvgs
        if flavor in ["seurat","seurat_v3"]:
            if flavor == "seurat":
                sc.pp.highly_variable_genes(adata_st_new, flavor = flavor, n_top_genes = n_hvg_group)
            if flavor == "seurat_v3":
                sc.pp.highly_variable_genes(adata_st_new, flavor = flavor, n_top_genes = n_hvg_group)
        else:
            raise ValueError(f"Invalid flavor '{flavor}'. Please choose seurat or seurat_v3.") 
        
        adata_subset = adata_st_new[:, adata_st_new.var['highly_variable']]
        adata_common_list.append(adata_subset)
    
    common_genes = np.array(adata_common_list[0].var_names)
    for adata in adata_common_list[1:]:
        common_genes = np.intersect1d(common_genes,np.array(adata.var_names))
    # Subset all AnnData objects to include only common genes
    out_adata_st = [adata[:, list(common_genes)] for adata in adata_common_list]
    return out_adata_st

def select_gene(input_adata_list : list,        # list of spatial transcriptomics datasets
                ref_adata_num : int = 0,        # index of adata to be analyzed spatially genetically
                spot_make : int = 3,            # number of subspace divisions
                save_data : bool = False,       # whether or not to save   
                key_words : str = 'spatial',    # keywords for spatial coordinates of the transcriptomics datasets 
                savepath: str = ''              # save location
                ):
    
    adata_list_copy = input_adata_list.copy()
    # Used to store regional indicators
    adata_list_copy[ref_adata_num].obsm['marker_cluster'] = np.zeros((adata_list_copy[ref_adata_num].X.shape[0],1))

    # Perform data segmentation
    X_max, X_min = max(adata_list_copy[ref_adata_num].obsm[key_words][:,0]),min(adata_list_copy[ref_adata_num].obsm[key_words][:,0])
    Y_max, Y_min = max(adata_list_copy[ref_adata_num].obsm[key_words][:,1]),min(adata_list_copy[ref_adata_num].obsm[key_words][:,1])
    X_intervals = (X_max - X_min) / spot_make
    Y_intervals = (Y_max - Y_min) / spot_make
    X_indices = np.floor((adata_list_copy[ref_adata_num].obsm[key_words][:, 0] - X_min) / X_intervals)
    Y_indices = np.floor((adata_list_copy[ref_adata_num].obsm[key_words][:, 1] - Y_min) / Y_intervals)
    # Filter index subscripts
    mask_x = X_indices >= spot_make
    X_indices[mask_x] -= 1
    mask_y = Y_indices >= spot_make
    Y_indices[mask_y] -= 1

    # Redistricting
    adata_list_copy[ref_adata_num].obsm['marker_cluster'] = (X_indices * spot_make) + (Y_indices + 1)
    adata_list_copy[ref_adata_num].obsm['marker_cluster'] = np.array(adata_list_copy[ref_adata_num].obsm['marker_cluster']).reshape(-1,1)

    if isspmatrix(adata_list_copy[ref_adata_num].X):
        save_st = adata_list_copy[ref_adata_num].X.toarray()
    else:
        save_st = adata_list_copy[ref_adata_num].X
    
    save_st = np.hstack((np.array(adata_list_copy[ref_adata_num].obs_names).reshape(-1,1),
                         np.array(save_st),
                         adata_list_copy[ref_adata_num].obsm['marker_cluster'].reshape(-1,1),
                         adata_list_copy[ref_adata_num].obsm[key_words]))
    
    gene_names = list(adata_list_copy[ref_adata_num].var_names)
    gene_names.insert(0,'')
    gene_names = gene_names + ["marker_cluster","x","y"]

    save_st = save_st.tolist()
    save_st.insert(0,gene_names)
    save_st = np.array(save_st).T

    if save_data:
        np.savetxt(savepath + 'select_gene_{}.txt'.format(spot_make * spot_make),save_st,fmt='%s')
    return adata_list_copy

def get_slice_barycenter(input_adata : AnnData,   # spatial transcriptomics data
                         spatial_type : str  # keywords for spatial coordinates of the transcriptomics data 
                         ):

    ref_x_sum = np.sum(input_adata.obsm[spatial_type][:,0])
    ref_y_sum = np.sum(input_adata.obsm[spatial_type][:,1])
    barycenter_X = ref_x_sum / len(input_adata.obsm[spatial_type])
    barycenter_Y =  ref_y_sum / len(input_adata.obsm[spatial_type])
    return np.array([barycenter_X,barycenter_Y])

def barycenter_translation(ref_barycenter,align_barycenter):
    return ref_barycenter - align_barycenter

def gene_rotation(
        point_ref_cloud : NDArray,                      # coordinates of the reference point
        point_align_cloud : NDArray,                    # coordinates of the points to be aligned
        maxiter : int = 300,                            # maximum number of iterations
        ini_angle : float = 0,                          # initial solution for rotation angle
        if_all_angle : bool = False,                    # whether to test multiple rotation angles or not
        angle_params : list = [-60,-40,-20,0,20,40,60]  # angles to be tested
):
    point_ref_cloud = point_ref_cloud.copy()
    point_align_cloud = point_align_cloud.copy()

    def rotation_loss(R):
        sum_loss = 0
        theta = np.radians(R)
        for i in range(len(point_align_cloud)):
            point_align_cloud_x_transformation = math.cos(theta) * point_align_cloud[i][0] + math.sin(theta) *  point_align_cloud[i][1]
            point_align_cloud_y_transformation = math.cos(theta) * point_align_cloud[i][1] - math.sin(theta) *  point_align_cloud[i][0]
            if point_align_cloud_x_transformation == 0.0:
                point_align_cloud_x_transformation = point_align_cloud_x_transformation + 0.0001
            if point_ref_cloud[i][0] == 0.0:
                point_ref_cloud[i][0] = point_ref_cloud[i][0] + 0.0001
            error_point = (point_align_cloud_y_transformation/point_align_cloud_x_transformation - point_ref_cloud[i][1]/point_ref_cloud[i][0]) ** 2
            sum_loss = sum_loss + error_point
        return sum_loss
    
    if if_all_angle:
        angle_list = []
        for angle in angle_params:
            def rotation_loss(R):
                sum_loss = 0
                theta = np.radians(R)
                for i in range(len(point_align_cloud)):
                    point_align_cloud_x_transformation = math.cos(theta) * point_align_cloud[i][0] + math.sin(theta) *  point_align_cloud[i][1]
                    point_align_cloud_y_transformation = math.cos(theta) * point_align_cloud[i][1] - math.sin(theta) *  point_align_cloud[i][0]
                    if point_align_cloud_x_transformation == 0.0:
                        point_align_cloud_x_transformation = point_align_cloud_x_transformation + 0.0001
                    if point_ref_cloud[i][0] == 0.0:
                        point_ref_cloud[i][0] = point_ref_cloud[i][0] + 0.0001
                    error_point = (point_align_cloud_y_transformation/point_align_cloud_x_transformation - point_ref_cloud[i][1]/point_ref_cloud[i][0]) ** 2
                    sum_loss = sum_loss + error_point
                return sum_loss
            # restrictive condition
            Angle_limitation = []
            Angle_limitation.append({'type': 'ineq', 'fun': lambda W: -W + 180})
            Angle_limitation.append({'type': 'ineq', 'fun': lambda W: W + 180})
            R = np.ones(1) * angle
            rotation = minimize(rotation_loss, R, method='SLSQP' , constraints=Angle_limitation,options={'maxiter': maxiter,'disp': False})
            alpha_R  = np.radians(rotation.x)
            angle_list.append(alpha_R)
        return angle_list
    
    else:
        Angle_limitation = []
        Angle_limitation.append({'type': 'ineq', 'fun': lambda W: -W + 180})
        Angle_limitation.append({'type': 'ineq', 'fun': lambda W: W + 180})
        R = np.ones(1) * ini_angle
        rotation = minimize(rotation_loss, R, method='SLSQP' , constraints=Angle_limitation,options={'maxiter': maxiter,'disp': False})
        alpha_R = np.radians(rotation.x)
        return alpha_R

def mapping_alignment(
    gene_input : list,                                  # space-specific gene
    ref_adata : AnnData,                                # reference space transcriptomics data
    ali_adata : AnnData,                                # spatial transcriptomics data to be aligned
    ini_spatial : str = 'spatial',                      # unaligned coordinate keywords
    add_spatial : str = 'align_spatial',                # adding keywords after alignment
    if_all_angle: bool = False,                         # whether to test multiple rotation angles or not
    ini_angle :  float = 0.0,                           # initial solution for rotation angle
    angle_params : list = [-60,-40,-20,0,20,40,60]      # angles to be tested
    ):

    ref_adata = ref_adata.copy()
    ali_adata = ali_adata.copy()

    ref_list = []
    for i in range(len(gene_input)):
        if isspmatrix(ref_adata[:,gene_input[i]].X):
            gene_expre = np.array(ref_adata[:,gene_input[i]].X.todense(),dtype=np.float32)
        else:
            gene_expre = np.array(ref_adata[:,gene_input[i]].X,dtype=np.float32)
        spatial_coordinate_x = np.array(ref_adata.obsm[add_spatial][:,0].reshape(-1,1),dtype=np.float32)    
        spatial_coordiante_y = np.array(ref_adata.obsm[add_spatial][:,1].reshape(-1,1),dtype=np.float32) 
        spatial_coordinate_x = spatial_coordinate_x * gene_expre
        spatial_coordiante_y = spatial_coordiante_y * gene_expre
        ref_list.append([np.sum(spatial_coordinate_x)/np.sum(gene_expre),np.sum(spatial_coordiante_y)/np.sum(gene_expre)])
    ref_barycenter = get_slice_barycenter(ref_adata,add_spatial)
    ref_barycenter_copy = get_slice_barycenter(ref_adata,add_spatial)
    ref_list.append(ref_barycenter)
    
    point_ref_cloud = np.array(ref_list,dtype=np.float32)
    point_align_cloud = []

    for i in range(len(gene_input)):
        if isspmatrix(ali_adata[:,gene_input[i]].X):
            gene_expre = np.array(ali_adata[:,gene_input[i]].X.todense(),dtype=np.float32)
        else:
            gene_expre = np.array(ali_adata[:,gene_input[i]].X,dtype=np.float32)
        spatial_coordinate_x = np.array(ali_adata.obsm[ini_spatial][:,0].reshape(-1,1),dtype=np.float32)    
        spatial_coordiante_y = np.array(ali_adata.obsm[ini_spatial][:,1].reshape(-1,1),dtype=np.float32)  
        spatial_coordinate_x = spatial_coordinate_x * gene_expre
        spatial_coordiante_y = spatial_coordiante_y * gene_expre
        point_align_cloud.append([np.sum(spatial_coordinate_x)/np.sum(gene_expre),np.sum(spatial_coordiante_y)/np.sum(gene_expre)])
    
    aling_barycenter = get_slice_barycenter(ali_adata,ini_spatial)
    aling_barycenter_copy = get_slice_barycenter(ali_adata,ini_spatial)
    point_align_cloud.append(aling_barycenter)  
    point_align_cloud = np.array(point_align_cloud,dtype=np.float32)

    T = barycenter_translation(
        ref_barycenter = ref_barycenter,
        align_barycenter = aling_barycenter
    )

    point_align_cloud = point_align_cloud + np.array(T)

    point_ref_cloud = point_ref_cloud[:-1,] - point_ref_cloud[-1]
    point_align_cloud = point_align_cloud[:-1,] - point_align_cloud[-1]

    if if_all_angle:
        R_LIST = gene_rotation(
            point_ref_cloud = point_ref_cloud,
            point_align_cloud = point_align_cloud,
            maxiter = 300,
            if_all_angle = if_all_angle,
            angle_params = angle_params
        )
        for angle_index in range(len(R_LIST)):
            R = R_LIST[angle_index]
            add_spatial_all_angle = ''
            add_spatial_all_angle = add_spatial + str(angle_params[angle_index])
            trans_spatial_spots = []
            point_align_cloud_transformation = ali_adata.obsm[ini_spatial] + T.reshape(1,-1) - ref_barycenter_copy.reshape(1,-1)
            point_align_cloud_x_transformation = math.cos(R) * point_align_cloud_transformation[:,0] + math.sin(R) *  point_align_cloud_transformation[:,1] + ref_barycenter_copy[0]
            point_align_cloud_y_transformation = math.cos(R) * point_align_cloud_transformation[:,1] - math.sin(R) *  point_align_cloud_transformation[:,0] + ref_barycenter_copy[1]
            trans_spatial_spots.append(point_align_cloud_x_transformation)
            trans_spatial_spots.append(point_align_cloud_y_transformation)
            trans_spatial_spots = np.array(trans_spatial_spots,dtype=np.float32)
            trans_spatial_spots = trans_spatial_spots.T
            ali_adata.obsm[add_spatial_all_angle] = trans_spatial_spots
        return ali_adata
    
    else:
        R = gene_rotation(
            point_ref_cloud = point_ref_cloud,
            point_align_cloud = point_align_cloud,
            ini_angle = ini_angle,
            maxiter = 300)
        
    trans_spatial_spots = []
    point_align_cloud_transformation = ali_adata.obsm[ini_spatial] + T.reshape(1,-1) - ref_barycenter_copy.reshape(1,-1)
    point_align_cloud_x_transformation = math.cos(R) * point_align_cloud_transformation[:,0] + math.sin(R) *  point_align_cloud_transformation[:,1] + ref_barycenter_copy[0]
    point_align_cloud_y_transformation = math.cos(R) * point_align_cloud_transformation[:,1] - math.sin(R) *  point_align_cloud_transformation[:,0] + ref_barycenter_copy[1]
    trans_spatial_spots.append(point_align_cloud_x_transformation)
    trans_spatial_spots.append(point_align_cloud_y_transformation)
    trans_spatial_spots = np.array(trans_spatial_spots,dtype=np.float32)
    trans_spatial_spots = trans_spatial_spots.T
    ali_adata.obsm[add_spatial] = trans_spatial_spots
    return ali_adata

def mapping_alignment_sequential_alignment(
    gene_input : list,                                  # space-specific gene
    ref_adata : AnnData,                                # reference space transcriptomics data
    ali_adata : AnnData,                                # spatial transcriptomics data to be aligned
    ini_spatial : str = 'spatial',                      # unaligned coordinate keywords
    add_spatial : str = 'align_spatial',                # adding keywords after alignment
    if_all_angle: bool = False,                         # whether to test multiple rotation angles or not
    ini_angle :  float = 0.0,                           # initial solution for rotation angle
    angle_params : list = [-60,-40,-20,0,20,40,60]      # angles to be tested
):
    if if_all_angle :
        for angle in angle_params:
            key_words = add_spatial + str(angle)
            ref_list = []
            for i in range(len(gene_input)):
                if isspmatrix(ref_adata[:,gene_input[i]].X):
                    gene_expre = np.array(ref_adata[:,gene_input[i]].X.todense(),dtype=np.float32)
                else:
                    gene_expre = np.array(ref_adata[:,gene_input[i]].X,dtype=np.float32)

                spatial_coordinate_x = np.array(ref_adata.obsm[key_words][:,0].reshape(-1,1),dtype=np.float32)    
                spatial_coordiante_y = np.array(ref_adata.obsm[key_words][:,1].reshape(-1,1),dtype=np.float32) 

                spatial_coordinate_x = spatial_coordinate_x * gene_expre
                spatial_coordiante_y = spatial_coordiante_y * gene_expre

                ref_list.append([np.sum(spatial_coordinate_x)/np.sum(gene_expre),np.sum(spatial_coordiante_y)/np.sum(gene_expre)])

            ref_barycenter = get_slice_barycenter(ref_adata,key_words)
            ref_barycenter_copy = get_slice_barycenter(ref_adata,key_words)
            ref_list.append(ref_barycenter)
    
            point_ref_cloud = np.array(ref_list,dtype=np.float32)
            point_align_cloud = []
            for i in range(len(gene_input)):
                if isspmatrix(ali_adata[:,gene_input[i]].X):
                    gene_expre = np.array(ali_adata[:,gene_input[i]].X.todense(),dtype=np.float32)
                else:
                    gene_expre = np.array(ali_adata[:,gene_input[i]].X,dtype=np.float32)
                spatial_coordinate_x = np.array(ali_adata.obsm[ini_spatial][:,0].reshape(-1,1),dtype=np.float32)    
                spatial_coordiante_y = np.array(ali_adata.obsm[ini_spatial][:,1].reshape(-1,1),dtype=np.float32)  
                spatial_coordinate_x = spatial_coordinate_x * gene_expre
                spatial_coordiante_y = spatial_coordiante_y * gene_expre
                point_align_cloud.append([np.sum(spatial_coordinate_x)/np.sum(gene_expre),np.sum(spatial_coordiante_y)/np.sum(gene_expre)])
            
        
            aling_barycenter = get_slice_barycenter(ali_adata,ini_spatial)
            aling_barycenter_copy = get_slice_barycenter(ali_adata,ini_spatial)
            point_align_cloud.append(aling_barycenter)   

            point_align_cloud = np.array(point_align_cloud,dtype=np.float32)

            T = barycenter_translation(
                ref_barycenter = ref_barycenter,
                align_barycenter = aling_barycenter
            )

            point_align_cloud = point_align_cloud + np.array(T)

            point_ref_cloud = point_ref_cloud[:-1,] - point_ref_cloud[-1]
            point_align_cloud = point_align_cloud[:-1,] - point_align_cloud[-1]

            R = gene_rotation(
            point_ref_cloud = point_ref_cloud,
            point_align_cloud = point_align_cloud,
            ini_angle = angle,
            maxiter = 300)

            trans_spatial_spots = []
            point_align_cloud_transformation = ali_adata.obsm[ini_spatial] + T.reshape(1,-1) - ref_barycenter_copy.reshape(1,-1)
            point_align_cloud_x_transformation = math.cos(R) * point_align_cloud_transformation[:,0] + math.sin(R) *  point_align_cloud_transformation[:,1] + ref_barycenter_copy[0]
            point_align_cloud_y_transformation = math.cos(R) * point_align_cloud_transformation[:,1] - math.sin(R) *  point_align_cloud_transformation[:,0] + ref_barycenter_copy[1]
            trans_spatial_spots.append(point_align_cloud_x_transformation)
            trans_spatial_spots.append(point_align_cloud_y_transformation)
            trans_spatial_spots = np.array(trans_spatial_spots,dtype=np.float32)
            trans_spatial_spots = trans_spatial_spots.T
            
            ali_adata.obsm[key_words] = trans_spatial_spots
        return ali_adata

    else:
        ref_list = []
        for i in range(len(gene_input)):
            if isspmatrix(ref_adata[:,gene_input[i]].X):
                gene_expre = np.array(ref_adata[:,gene_input[i]].X.todense(),dtype=np.float32)
            else:
                gene_expre = np.array(ref_adata[:,gene_input[i]].X,dtype=np.float32)
            spatial_coordinate_x = np.array(ref_adata.obsm[add_spatial][:,0].reshape(-1,1),dtype=np.float32)    
            spatial_coordiante_y = np.array(ref_adata.obsm[add_spatial][:,1].reshape(-1,1),dtype=np.float32)  
            spatial_coordinate_x = spatial_coordinate_x * gene_expre
            spatial_coordiante_y = spatial_coordiante_y * gene_expre
            ref_list.append([np.sum(spatial_coordinate_x)/np.sum(gene_expre),np.sum(spatial_coordiante_y)/np.sum(gene_expre)])

        ref_barycenter = get_slice_barycenter(ref_adata,add_spatial)
        ref_barycenter_copy = get_slice_barycenter(ref_adata,add_spatial)
        ref_list.append(ref_barycenter)
        
        point_ref_cloud = np.array(ref_list,dtype=np.float32)
        point_align_cloud = []
        for i in range(len(gene_input)):
            if isspmatrix(ali_adata[:,gene_input[i]].X):
                gene_expre = np.array(ali_adata[:,gene_input[i]].X.todense(),dtype=np.float32)
            else:
                gene_expre = np.array(ali_adata[:,gene_input[i]].X,dtype=np.float32)
            spatial_coordinate_x = np.array(ali_adata.obsm[ini_spatial][:,0].reshape(-1,1),dtype=np.float32)    
            spatial_coordiante_y = np.array(ali_adata.obsm[ini_spatial][:,1].reshape(-1,1),dtype=np.float32)  
            spatial_coordinate_x = spatial_coordinate_x * gene_expre
            spatial_coordiante_y = spatial_coordiante_y * gene_expre
            point_align_cloud.append([np.sum(spatial_coordinate_x)/np.sum(gene_expre),np.sum(spatial_coordiante_y)/np.sum(gene_expre)])
        

        aling_barycenter = get_slice_barycenter(ali_adata,ini_spatial)
        aling_barycenter_copy = get_slice_barycenter(ali_adata,ini_spatial)
        point_align_cloud.append(aling_barycenter)   
        
        point_align_cloud = np.array(point_align_cloud,dtype=np.float32)

        T = barycenter_translation(
            ref_barycenter = ref_barycenter,
            align_barycenter = aling_barycenter
        )

        point_align_cloud = point_align_cloud + np.array(T)

        point_ref_cloud = point_ref_cloud[:-1,] - point_ref_cloud[-1]
        point_align_cloud = point_align_cloud[:-1,] - point_align_cloud[-1]

        R = gene_rotation(
            point_ref_cloud = point_ref_cloud,
            point_align_cloud = point_align_cloud,
            ini_angle = ini_angle,
            maxiter = 300)
        
        trans_spatial_spots = []
        point_align_cloud_transformation = ali_adata.obsm[ini_spatial] + T.reshape(1,-1) - ref_barycenter_copy.reshape(1,-1)
        point_align_cloud_x_transformation = math.cos(R) * point_align_cloud_transformation[:,0] + math.sin(R) *  point_align_cloud_transformation[:,1] + ref_barycenter_copy[0]
        point_align_cloud_y_transformation = math.cos(R) * point_align_cloud_transformation[:,1] - math.sin(R) *  point_align_cloud_transformation[:,0] + ref_barycenter_copy[1]
        trans_spatial_spots.append(point_align_cloud_x_transformation)
        trans_spatial_spots.append(point_align_cloud_y_transformation)
        trans_spatial_spots = np.array(trans_spatial_spots,dtype=np.float32)
        trans_spatial_spots = trans_spatial_spots.T
        ali_adata.obsm[add_spatial] = trans_spatial_spots

        return ali_adata
    
def gene_rigid_mapping_alignment(
    gene_input : list,                                  # space-specific gene
    stadata_input : list,                               # list of spatial transcriptomics datasets
    ini_spatial  : str = 'spatial',                     # unaligned coordinate keywords    
    add_spatial : str = 'align_spatial',                # adding keywords after alignment
    align_model : str = "single_template_alignment",    # patterns of alignment, single template alignment or sequential alignment
    gene_input_list : list = None,                      # space-specific gene list, if input then must be equal in length to stadata_input,single template alignment does not use this parameter
    angle_input_list : list = None,                     # initial angle of rotation for each slice, if input, needs to be equal to stadata_input
    ref_label : int = 0,                                # single template alignment parameter, which template to select
    if_all_angle : bool = False,                        # whether to test multiple rotation angles or not
    ini_angle : float = 0.0,                            # the initial rotation angle of the alignment, shared by all slices when angle_input_list is not provided.
    angle_params : list = [-60,-40,-20,0,20,40,60]      # angles to be tested
):
    if align_model.lower() not in ['single_template_alignment','sequential_alignment']:
        raise ValueError(f"Invalid flavor '{align_model}'. Please choose 'single_template_alignment' or 'sequential_alignment'.") 
    if align_model.lower() == 'single_template_alignment':
        if angle_input_list != None:
            if len(angle_input_list) != len(stadata_input) :
                raise ValueError(f"Invalid flavor angle_input_list. Please make sure the length is the same as the stadata_input.") 
            angle_input_list = angle_input_list.copy()
        if angle_input_list == None:
            angle_input_list = [ini_angle] * len(stadata_input)
        if if_all_angle:
            for angle in angle_params:
                angle_add_spatial = ''
                angle_add_spatial = add_spatial + str(angle)
                stadata_input[ref_label].obsm[angle_add_spatial] = stadata_input[ref_label].obsm[ini_spatial]
                stadata_input[ref_label].obsm[add_spatial] = stadata_input[ref_label].obsm[ini_spatial]
        else:
            stadata_input[ref_label].obsm[add_spatial] = stadata_input[ref_label].obsm[ini_spatial]
        for j in range(len(stadata_input)):
            if j == ref_label:
                continue
            else:
                stadata_input[j] = mapping_alignment(gene_input = gene_input,ref_adata = stadata_input[ref_label],
                        ali_adata = stadata_input[j],ini_spatial = ini_spatial,add_spatial = add_spatial,
                        if_all_angle = if_all_angle,ini_angle = angle_input_list[j],
                        angle_params = angle_params)
        return stadata_input
    
    if align_model.lower() == 'sequential_alignment':
        if angle_input_list != None:
            if len(angle_input_list) != len(stadata_input) :
                raise ValueError(f"Invalid flavor angle_input_list. Please make sure the length is the same as the stadata_input.") 
            angle_input_list = angle_input_list.copy()
        if angle_input_list == None:
            angle_input_list = [ini_angle] * len(stadata_input)
        if gene_input_list == None:
            for j in range(len(stadata_input)):
                if j == 0:
                    if if_all_angle:
                        for angle in angle_params:
                            angle_add_spatial = ''
                            angle_add_spatial = add_spatial + str(angle)
                            stadata_input[j].obsm[angle_add_spatial] = stadata_input[j].obsm[ini_spatial]
                    else:
                        stadata_input[j].obsm[add_spatial] = stadata_input[j].obsm[ini_spatial]
                        continue
                else:
                    stadata_input[j] = mapping_alignment_sequential_alignment(
                            gene_input = gene_input,ref_adata = stadata_input[j-1],
                            ali_adata = stadata_input[j],ini_spatial = ini_spatial,
                            add_spatial = add_spatial,if_all_angle = if_all_angle,
                            ini_angle = angle_input_list[j],angle_params = angle_params)  
            return stadata_input
        if gene_input_list != None:
            if len(gene_input_list) != len(stadata_input):
                raise ValueError("Invalid flavor. Please make sure that gene_input_list is the same length as stadata_input.") 
            for j in range(len(stadata_input)):
                if j == 0:
                    if if_all_angle:
                        for angle in angle_params:
                            angle_add_spatial = ''
                            angle_add_spatial = add_spatial + str(angle)
                            stadata_input[j].obsm[angle_add_spatial] = stadata_input[j].obsm[ini_spatial]
                    else:
                        stadata_input[j].obsm[add_spatial] = stadata_input[j].obsm[ini_spatial]
                        continue
                else:
                    stadata_input[j] = mapping_alignment_sequential_alignment(
                            gene_input = gene_input_list[j],
                            ref_adata = stadata_input[j-1],
                            ali_adata = stadata_input[j],
                            ini_spatial = ini_spatial,
                            add_spatial = add_spatial,
                            if_all_angle = if_all_angle,
                            ini_angle = angle_input_list[j],
                            angle_params = angle_params)
            return stadata_input
        
# Credit to https://github.com/ClayFlannigan/icp
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def alignment(
        point_ref_cloud : NDArray,          # coordinates of the reference point
        point_align_cloud : NDArray,        # coordinates of the points to be aligned
        maxiter : int = 300,                # maximum number of iterations
):
    def fun(X):
        sum_error = 0
        X[2] = (X[2]/180) * math.pi
        for i in range(len(point_align_cloud)):
            point_align_cloud_x_transformation = math.cos(X[2]) * point_align_cloud[i][0] + math.sin(X[2]) *  point_align_cloud[i][1] - math.cos(X[2]) * X[0] - math.sin(X[2]) * X[1] - point_ref_cloud[i][0]
            point_align_cloud_y_transformation = math.cos(X[2]) * point_align_cloud[i][1] - math.sin(X[2]) *  point_align_cloud[i][0] - math.cos(X[2]) * X[1] + math.sin(X[2]) * X[0] - point_ref_cloud[i][1]
            error_point = point_align_cloud_x_transformation ** 2 + point_align_cloud_y_transformation **2 
            sum_error = sum_error + error_point
        return sum_error

    Angle_limitation = []
    Angle_limitation.append({'type': 'ineq', 'fun': lambda W: -W[2] + 180})
    Angle_limitation.append({'type': 'ineq', 'fun': lambda W: W[2] + 180})

    X_ini = np.zeros(3)
    X_trans = minimize(fun, X_ini, method='SLSQP' , constraints=Angle_limitation, options={'maxiter':maxiter,'disp': False})

    T = [X_trans.x[0], X_trans.x[1]]
    alpha_R = (X_trans.x[2]/180) * math.pi
    R = [[math.cos(alpha_R),math.sin(alpha_R)],
         [-math.sin(alpha_R),math.cos(alpha_R)]]
    return T,R

def gene_rigid_alignment(
        gene_input : list,                                  # space-specific gene
        stadata_input : list,                               # list of spatial transcriptomics datasets
        ini_spatial : str = 'spatial',                      # unaligned coordinate keywords   
        add_spatial : str = 'align_spatial',                # adding keywords after alignment
        align_model : str = "single_template_alignment",    # patterns of alignment, single template alignment or sequential alignment
        gene_input_list : list = None,                      # space-specific gene list, if input then must be equal in length to stadata_input,single template alignment does not use this parameter
        ref_label : int = 0,                                # single template alignment parameter, which template to select
        align_method : str = 'optimize',                    # optimization methods used in the alignment process
        icp_iterations : int = 20,                          # maximum number of iterations for icp algorithm
        maxiter : int = 300,                                # maximum number of iterations of the optimization algorithm
):
    if align_model.lower() not in ['single_template_alignment','sequential_alignment']:
        raise ValueError(f"Invalid flavor '{align_model}'. Please choose 'single_template_alignment' or 'sequential_alignment'.") 
    if align_model.lower() == 'single_template_alignment':
        stadata_input[ref_label].obsm[add_spatial] = stadata_input[ref_label].obsm[ini_spatial]
        gene_input = gene_input.copy()
        stadata_input = stadata_input.copy()
        ref_list = []
        for i in range(len(gene_input)):
            if isspmatrix(stadata_input[ref_label][:,gene_input[i]].X):
                gene_expre = np.array(stadata_input[ref_label][:,gene_input[i]].X.todense(),dtype=np.float32)
            else:
                gene_expre = np.array(stadata_input[ref_label][:,gene_input[i]].X,dtype=np.float32)
            spatial_coordinate_x = np.array(stadata_input[ref_label].obsm[add_spatial][:,0].reshape(-1,1),dtype=np.float32)    
            spatial_coordiante_y = np.array(stadata_input[ref_label].obsm[add_spatial][:,1].reshape(-1,1),dtype=np.float32)  
            spatial_coordinate_x = spatial_coordinate_x * gene_expre
            spatial_coordiante_y = spatial_coordiante_y * gene_expre
            ref_list.append([np.sum(spatial_coordinate_x)/np.sum(gene_expre),np.sum(spatial_coordiante_y)/np.sum(gene_expre)])
        point_ref_cloud = np.array(ref_list,dtype=np.float32)

        for j in range(len(stadata_input)):
            if j == ref_label:
                continue
            else:
                point_align_cloud = []
                for i in range(len(gene_input)):
                    if isspmatrix(stadata_input[j][:,gene_input[i]].X):
                        gene_expre = np.array(stadata_input[j][:,gene_input[i]].X.todense(),dtype=np.float32)
                    else:
                        gene_expre = np.array(stadata_input[j][:,gene_input[i]].X,dtype=np.float32)
                    spatial_coordinate_x = np.array(stadata_input[j].obsm[ini_spatial][:,0].reshape(-1,1),dtype=np.float32)    
                    spatial_coordiante_y = np.array(stadata_input[j].obsm[ini_spatial][:,1].reshape(-1,1),dtype=np.float32)  
                    spatial_coordinate_x = spatial_coordinate_x * gene_expre
                    spatial_coordiante_y = spatial_coordiante_y * gene_expre
                    point_align_cloud.append([np.sum(spatial_coordinate_x)/np.sum(gene_expre),np.sum(spatial_coordiante_y)/np.sum(gene_expre)])
                point_align_cloud = np.array(point_align_cloud,dtype=np.float32)
            if align_method == 'optimize':
                T,R = alignment(
                    point_ref_cloud = point_ref_cloud,
                    point_align_cloud = point_align_cloud,
                    maxiter = maxiter)
                align_spatial_spots = stadata_input[j].obsm[ini_spatial]
                trans_spatial_spots = []
                point_align_cloud_x_transformation = R[0][0] * align_spatial_spots[:,0] + R[0][1] *  align_spatial_spots[:,1] - R[0][0] * T[0] - R[0][1] * T[1] 
                point_align_cloud_y_transformation = R[0][0] * align_spatial_spots[:,1] - R[0][1] *  align_spatial_spots[:,0] - R[0][0] * T[1] + R[0][1] * T[0]
                trans_spatial_spots.append(point_align_cloud_x_transformation)
                trans_spatial_spots.append(point_align_cloud_y_transformation)
                trans_spatial_spots = np.array(trans_spatial_spots,dtype=np.float32)
                trans_spatial_spots = trans_spatial_spots.T
                stadata_input[j].obsm[add_spatial] = trans_spatial_spots
            if align_method == 'icp':
                for icp_it in range(icp_iterations):
                    SUM_R_T,R,T = best_fit_transform(
                        A = point_ref_cloud,
                        B = point_align_cloud)
                    R[0][1] = - R[0][1]
                    R[1][0] = - R[1][0]
                    if icp_it == 0:
                        align_spatial_spots = stadata_input[j].obsm[ini_spatial]
                    else:
                        align_spatial_spots = stadata_input[j].obsm[add_spatial]
                    trans_spatial_spots = []
                    point_align_cloud_x_transformation = R[0][0] * align_spatial_spots[:,0] + R[0][1] *  align_spatial_spots[:,1] - R[0][0] * T[0] - R[0][1] * T[1] 
                    point_align_cloud_y_transformation = R[0][0] * align_spatial_spots[:,1] - R[0][1] *  align_spatial_spots[:,0] - R[0][0] * T[1] + R[0][1] * T[0]
                    trans_spatial_spots.append(point_align_cloud_x_transformation)
                    trans_spatial_spots.append(point_align_cloud_y_transformation)
                    trans_spatial_spots = np.array(trans_spatial_spots,dtype=np.float32)
                    trans_spatial_spots = trans_spatial_spots.T
                    stadata_input[j].obsm[add_spatial] = trans_spatial_spots
                    point_align_cloud_x = R[0][0] * point_align_cloud[:,0] + R[0][1] *  point_align_cloud[:,1] - R[0][0] * T[0] - R[0][1] * T[1] 
                    point_align_cloud_y = R[0][0] * point_align_cloud[:,1] - R[0][1] *  point_align_cloud[:,0] - R[0][0] * T[1] + R[0][1] * T[0]
                    point_align_cloud = []
                    point_align_cloud.append(point_align_cloud_x)
                    point_align_cloud.append(point_align_cloud_y)
                    point_align_cloud = np.array(point_align_cloud,dtype=np.float32)
                    point_align_cloud = point_align_cloud.T
        return stadata_input
    if align_model.lower() == 'sequential_alignment':
        if gene_input_list == None:
            gene_input = gene_input.copy()
            stadata_input = stadata_input.copy()

            for index in range(len(stadata_input)):
                if index == 0:
                    stadata_input[index].obsm[add_spatial] = stadata_input[index].obsm[ini_spatial]
                else:
                    ref_list = []
                    for i in range(len(gene_input)):
                        if isspmatrix(stadata_input[index-1][:,gene_input[i]].X):
                            gene_expre = np.array(stadata_input[index-1][:,gene_input[i]].X.todense(),dtype=np.float32)
                        else:
                            gene_expre = np.array(stadata_input[index-1][:,gene_input[i]].X,dtype=np.float32)
                        spatial_coordinate_x = np.array(stadata_input[index-1].obsm[add_spatial][:,0].reshape(-1,1),dtype=np.float32)    
                        spatial_coordiante_y = np.array(stadata_input[index-1].obsm[add_spatial][:,1].reshape(-1,1),dtype=np.float32)  
                        spatial_coordinate_x = spatial_coordinate_x * gene_expre
                        spatial_coordiante_y = spatial_coordiante_y * gene_expre
                        ref_list.append([np.sum(spatial_coordinate_x)/np.sum(gene_expre),np.sum(spatial_coordiante_y)/np.sum(gene_expre)])

                    point_ref_cloud = np.array(ref_list,dtype=np.float32)
                
                    point_align_cloud = []

                    for i in range(len(gene_input)):
                        if isspmatrix(stadata_input[index][:,gene_input[i]].X):
                            gene_expre = np.array(stadata_input[index][:,gene_input[i]].X.todense(),dtype=np.float32)
                        else:
                            gene_expre = np.array(stadata_input[index][:,gene_input[i]].X,dtype=np.float32)
                        spatial_coordinate_x = np.array(stadata_input[index].obsm[ini_spatial][:,0].reshape(-1,1),dtype=np.float32)    
                        spatial_coordiante_y = np.array(stadata_input[index].obsm[ini_spatial][:,1].reshape(-1,1),dtype=np.float32)  
                        spatial_coordinate_x = spatial_coordinate_x * gene_expre
                        spatial_coordiante_y = spatial_coordiante_y * gene_expre
                        point_align_cloud.append([np.sum(spatial_coordinate_x)/np.sum(gene_expre),np.sum(spatial_coordiante_y)/np.sum(gene_expre)])
                    
                    point_align_cloud = np.array(point_align_cloud,dtype=np.float32)

                    if align_method == 'optimize':

                        T,R = alignment(
                            point_ref_cloud = point_ref_cloud,
                            point_align_cloud = point_align_cloud,
                            maxiter = maxiter)
                        
                        align_spatial_spots = stadata_input[index].obsm[ini_spatial]
                        
                        trans_spatial_spots = []
                        point_align_cloud_x_transformation = R[0][0] * align_spatial_spots[:,0] + R[0][1] *  align_spatial_spots[:,1] - R[0][0] * T[0] - R[0][1] * T[1] 
                        point_align_cloud_y_transformation = R[0][0] * align_spatial_spots[:,1] - R[0][1] *  align_spatial_spots[:,0] - R[0][0] * T[1] + R[0][1] * T[0]
                        trans_spatial_spots.append(point_align_cloud_x_transformation)
                        trans_spatial_spots.append(point_align_cloud_y_transformation)
                        trans_spatial_spots = np.array(trans_spatial_spots,dtype=np.float32)
                        trans_spatial_spots = trans_spatial_spots.T
                        stadata_input[index].obsm[add_spatial] = trans_spatial_spots
                    
                    if align_method == 'icp':

                        for icp_it in range(icp_iterations):
                            SUM_R_T,R,T = best_fit_transform(
                                A = point_ref_cloud,
                                B = point_align_cloud)
                            R[0][1] = - R[0][1]
                            R[1][0] = - R[1][0]
                            
                            if icp_it == 0:
                                align_spatial_spots = stadata_input[index].obsm[ini_spatial]
                            else:
                                align_spatial_spots = stadata_input[index].obsm[add_spatial]
                            
                            trans_spatial_spots = []
                            point_align_cloud_x_transformation = R[0][0] * align_spatial_spots[:,0] + R[0][1] *  align_spatial_spots[:,1] - R[0][0] * T[0] - R[0][1] * T[1] 
                            point_align_cloud_y_transformation = R[0][0] * align_spatial_spots[:,1] - R[0][1] *  align_spatial_spots[:,0] - R[0][0] * T[1] + R[0][1] * T[0]
                            trans_spatial_spots.append(point_align_cloud_x_transformation)
                            trans_spatial_spots.append(point_align_cloud_y_transformation)
                            trans_spatial_spots = np.array(trans_spatial_spots,dtype=np.float32)
                            trans_spatial_spots = trans_spatial_spots.T
                            stadata_input[index].obsm[add_spatial] = trans_spatial_spots

                            point_align_cloud_x = R[0][0] * point_align_cloud[:,0] + R[0][1] *  point_align_cloud[:,1] - R[0][0] * T[0] - R[0][1] * T[1] 
                            point_align_cloud_y = R[0][0] * point_align_cloud[:,1] - R[0][1] *  point_align_cloud[:,0] - R[0][0] * T[1] + R[0][1] * T[0]
                            point_align_cloud = []
                            point_align_cloud.append(point_align_cloud_x)
                            point_align_cloud.append(point_align_cloud_y)
                            point_align_cloud = np.array(point_align_cloud,dtype=np.float32)
                            point_align_cloud = point_align_cloud.T
                
            return stadata_input
        
        if gene_input_list is not None:
            if len(gene_input_list) != len(stadata_input):
                raise ValueError("Invalid flavor. Please make sure that gene_input_list is the same length as stadata_input.") 
            stadata_input = stadata_input.copy()
            for index in range(len(stadata_input)):
                gene_input = gene_input_list[index].copy()
                if index == 0:
                    stadata_input[index].obsm[add_spatial] = stadata_input[index].obsm[ini_spatial]
                else:
                    ref_list = []
                    for i in range(len(gene_input)):
                        if isspmatrix(stadata_input[index-1][:,gene_input[i]].X):
                            gene_expre = np.array(stadata_input[index-1][:,gene_input[i]].X.todense(),dtype=np.float32)
                        else:
                            gene_expre = np.array(stadata_input[index-1][:,gene_input[i]].X,dtype=np.float32)
                        spatial_coordinate_x = np.array(stadata_input[index-1].obsm[add_spatial][:,0].reshape(-1,1),dtype=np.float32)    
                        spatial_coordiante_y = np.array(stadata_input[index-1].obsm[add_spatial][:,1].reshape(-1,1),dtype=np.float32)  
                        spatial_coordinate_x = spatial_coordinate_x * gene_expre
                        spatial_coordiante_y = spatial_coordiante_y * gene_expre
                        ref_list.append([np.sum(spatial_coordinate_x)/np.sum(gene_expre),np.sum(spatial_coordiante_y)/np.sum(gene_expre)])

                    point_ref_cloud = np.array(ref_list,dtype=np.float32)
                
                    point_align_cloud = []

                    for i in range(len(gene_input)):
                        if isspmatrix(stadata_input[index][:,gene_input[i]].X):
                            gene_expre = np.array(stadata_input[index][:,gene_input[i]].X.todense(),dtype=np.float32)
                        else:
                            gene_expre = np.array(stadata_input[index][:,gene_input[i]].X,dtype=np.float32)
                        spatial_coordinate_x = np.array(stadata_input[index].obsm[ini_spatial][:,0].reshape(-1,1),dtype=np.float32)    
                        spatial_coordiante_y = np.array(stadata_input[index].obsm[ini_spatial][:,1].reshape(-1,1),dtype=np.float32)  
                        spatial_coordinate_x = spatial_coordinate_x * gene_expre
                        spatial_coordiante_y = spatial_coordiante_y * gene_expre
                        point_align_cloud.append([np.sum(spatial_coordinate_x)/np.sum(gene_expre),np.sum(spatial_coordiante_y)/np.sum(gene_expre)])
                    
                    point_align_cloud = np.array(point_align_cloud,dtype=np.float32)

                    if align_method == 'optimize':

                        T,R = alignment(
                            point_ref_cloud = point_ref_cloud,
                            point_align_cloud = point_align_cloud,
                            maxiter = maxiter)
                        
                        align_spatial_spots = stadata_input[index].obsm[ini_spatial]
                        
                        trans_spatial_spots = []
                        point_align_cloud_x_transformation = R[0][0] * align_spatial_spots[:,0] + R[0][1] *  align_spatial_spots[:,1] - R[0][0] * T[0] - R[0][1] * T[1] 
                        point_align_cloud_y_transformation = R[0][0] * align_spatial_spots[:,1] - R[0][1] *  align_spatial_spots[:,0] - R[0][0] * T[1] + R[0][1] * T[0]
                        trans_spatial_spots.append(point_align_cloud_x_transformation)
                        trans_spatial_spots.append(point_align_cloud_y_transformation)
                        trans_spatial_spots = np.array(trans_spatial_spots,dtype=np.float32)
                        trans_spatial_spots = trans_spatial_spots.T
                        stadata_input[index].obsm[add_spatial] = trans_spatial_spots
                    
                    if align_method == 'icp':
                        for icp_it in range(icp_iterations):
                            SUM_R_T,R,T = best_fit_transform(
                                A = point_ref_cloud,
                                B = point_align_cloud)
                            R[0][1] = - R[0][1]
                            R[1][0] = - R[1][0]
                            
                            if icp_it == 0:
                                align_spatial_spots = stadata_input[index].obsm[ini_spatial]
                            else:
                                align_spatial_spots = stadata_input[index].obsm[add_spatial]
                                    
                            trans_spatial_spots = []
                            point_align_cloud_x_transformation = R[0][0] * align_spatial_spots[:,0] + R[0][1] *  align_spatial_spots[:,1] - R[0][0] * T[0] - R[0][1] * T[1] 
                            point_align_cloud_y_transformation = R[0][0] * align_spatial_spots[:,1] - R[0][1] *  align_spatial_spots[:,0] - R[0][0] * T[1] + R[0][1] * T[0]
                            trans_spatial_spots.append(point_align_cloud_x_transformation)
                            trans_spatial_spots.append(point_align_cloud_y_transformation)
                            trans_spatial_spots = np.array(trans_spatial_spots,dtype=np.float32)
                            trans_spatial_spots = trans_spatial_spots.T
                            stadata_input[index].obsm[add_spatial] = trans_spatial_spots

                            point_align_cloud_x = R[0][0] * point_align_cloud[:,0] + R[0][1] *  point_align_cloud[:,1] - R[0][0] * T[0] - R[0][1] * T[1] 
                            point_align_cloud_y = R[0][0] * point_align_cloud[:,1] - R[0][1] *  point_align_cloud[:,0] - R[0][0] * T[1] + R[0][1] * T[0]
                            point_align_cloud = []
                            point_align_cloud.append(point_align_cloud_x)
                            point_align_cloud.append(point_align_cloud_y)
                            point_align_cloud = np.array(point_align_cloud,dtype=np.float32)
                            point_align_cloud = point_align_cloud.T
                
            return stadata_input

def seek_corresponding_spot(
        ref_adata : AnnData,                                    # alignment templates for spatial transcriptomic data
        align_adata : AnnData,                                  # spatial transcriptomic data after alignment according to the alignment templatse
        spatial_label : str = 'align_spatial',                  # aligned spatial coordinate keywords
        add_corrseponding_words : str = 'batch_effect_mapping'  # keywords added after pairing
):
    ref_adata = ref_adata
    align_adata = align_adata
    align_adata_spot = np.array(align_adata.obsm[spatial_label],dtype=np.float32)
    ref_adata_spot = np.array(ref_adata.obsm[spatial_label],dtype=np.float32)
    distance_matrix = cdist(align_adata_spot,ref_adata_spot,metric='euclidean')
    max_indices_per_row = np.argmin(distance_matrix, axis=1)    
    max_indices_per_row = np.column_stack((np.arange(len(max_indices_per_row)),max_indices_per_row))
    align_adata.obsm[add_corrseponding_words] = max_indices_per_row[:,1]
    return align_adata

def adata_preprocess_dim(
        input_adata:list,                                       # a series of aligned sliced data
        ref_label:int = 0,                                      # alignment template for single template alignment model
        spatial_label:str = 'align_spatial',                    # keywords for aligned coordinates
        add_corrseponding_words:str = 'batch_effect_mapping'    # keywords added after pairing
        ):
    
    input_adata = input_adata.copy()
    for i in range(len(input_adata)):
        if i == ref_label:
            input_adata[i].obsm[add_corrseponding_words] = np.arange(0,input_adata[i].X.shape[0],1)
        else:
            input_adata[i] = seek_corresponding_spot(ref_adata=input_adata[ref_label],
                                                     align_adata=input_adata[i],
                                                     spatial_label = spatial_label,
                                                     add_corrseponding_words = add_corrseponding_words)
    return input_adata

def connect_matrix_up_down(A,B):
    result = np.concatenate((A, B), axis=0)
    return result

def get_need_ST_reconstruction(input_adata:list):
    #can change to_array() to to_dense()
    if isspmatrix(input_adata[0].X):
        ST_need_reconstruction_matrix = input_adata[0].X.toarray()
    else:
        ST_need_reconstruction_matrix = input_adata[0].X
    for i in range(1,len(input_adata)):
        if isspmatrix(input_adata[i].X):
            ST_need_reconstruction_matrix = connect_matrix_up_down(ST_need_reconstruction_matrix,input_adata[i].X.toarray())
        else:
            ST_need_reconstruction_matrix = connect_matrix_up_down(ST_need_reconstruction_matrix,input_adata[i].X)

    return ST_need_reconstruction_matrix

def get_spatial_net(input_adata:list,                           # a series of aligned sliced data
                    coordinates_label:str = 'align_spatial',    # keywords for aligned coordinates
                    n_neighbors:int = 10,                       # number of neighbours per spot in the adjacency matrix
                    add_net_words:str = 'adj_spatial_net'       # keywords for the adjacency matrix
                    ):

    # Adding internal networks to each slice
    for index in range(len(input_adata)):
        coor = pd.DataFrame(input_adata[index].obsm[coordinates_label])
        coor.index = input_adata[index].obs.index
        coor.columns = ['x', 'y']
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors,algorithm='ball_tree').fit(coor)
        distances,indices = nbrs.kneighbors(coor)
        adj_row_indices = np.tile(indices[:,0],len(indices[0]))
        adj_col_indices = np.ravel(indices,order='F')
        adj_values = np.tile(1,len(adj_col_indices))
        adj_shape = (input_adata[index].X.shape[0],input_adata[index].X.shape[0]) 
        adj_matrix = sp.coo_matrix((adj_values, (adj_row_indices, adj_col_indices)), shape=adj_shape)
        input_adata[index].obsm[add_net_words] = adj_matrix
    
    return input_adata

def get_all_spatial_new(input_adata:list,
                        concact_method = 'diag'):
    
    for index in range(len(input_adata)):
        if index == 0:
            all_spatial_net = input_adata[index].obsm['adj_spatial_net']
        else:
            all_spatial_net = sp.block_diag((all_spatial_net,input_adata[index].obsm['adj_spatial_net']))
    
    return all_spatial_net

def get_cross_slice_spatial_net(
    ref_adata:AnnData,                      # One of the adata that needs to be created for the neighbourhood network
    align_adata:AnnData,                    # Another adata that needs to create a neighbouring network
    mapping_type = 'batch_effect_mapping'   # Keywords added after pairing
):
    # Functions set for single template alignment
    adj_row_indices = np.array(align_adata.obsm[mapping_type])
    adj_col_indices = np.arange(0,align_adata.X.shape[0],1)
    adj_values = np.tile(1,len(adj_row_indices))
    cross_slice_ref_to_align = sp.coo_matrix((adj_values, (adj_row_indices, adj_col_indices)), shape=(ref_adata.X.shape[0],align_adata.X.shape[0]))
    cross_slice_align_to_ref = sp.coo_matrix((adj_values, (adj_col_indices, adj_row_indices)), shape=(align_adata.X.shape[0],ref_adata.X.shape[0]))
    return cross_slice_ref_to_align,cross_slice_align_to_ref

def creat_null_coo(
        ref_adata,
        ali_adata
):
    empty_coo_matrix = sp.coo_matrix(([], ([], [])), shape=(ref_adata.X.shape[0], ali_adata.X.shape[0]))
    return empty_coo_matrix

def get_all_cross_slice_spatial_net(
    input_adata : list,                 # List of aligned spatial transcriptomic data
    ref_label:int = 0,                  # Alignment templates for single template alignment
    ref_ali_all_slices: bool = True,    # Whether to pass neighbor information to the global
    no_cross = False                    # No global adjacency
):
    # ref_adata = input_adata[ref_label]

    net_list = np.zeros((len(input_adata),len(input_adata)))
    net_list = net_list.tolist()

    for i in range(len(input_adata)):
        for j in range(len(input_adata)):
            if i == ref_label:
                if j == i:
                    net_list[i][j] = input_adata[i].obsm['adj_spatial_net']
                else:
                    cross_slice_ref_to_align,cross_slice_align_to_ref = get_cross_slice_spatial_net(input_adata[i],input_adata[j])
                    net_list[i][j] = cross_slice_ref_to_align
            else:
                if j == i:
                    net_list[i][j] = input_adata[i].obsm['adj_spatial_net']
                if j == ref_label:
                    cross_slice_ref_to_align,cross_slice_align_to_ref = get_cross_slice_spatial_net(input_adata[j],input_adata[i])
                    net_list[i][j] = cross_slice_align_to_ref
                if j != i and j != ref_label:
                    empty_coo_matrix = creat_null_coo(ref_adata=input_adata[i],ali_adata = input_adata[j])
                    net_list[i][j] = empty_coo_matrix
    
    if no_cross == True:
        for index in range(len(net_list)):
            if index == 0:
                all_spatial_net = net_list[index][index]
            else:
                all_spatial_net = sp.block_diag((all_spatial_net,net_list[index][index]))
        return all_spatial_net
    
    if ref_ali_all_slices:
        for i in range(len(net_list)):
            for j in range(len(net_list[i])):
                if i != ref_label and j != ref_label and i != j and i < j:
                    coo_ref = net_list[ref_label][i].toarray()
                    coo_ali = net_list[ref_label][j].toarray()
                    cross_slice_array = np.zeros((coo_ref.shape[1],coo_ali.shape[1]))
                    for row in range(len(coo_ref)):
                        row_label = np.nonzero(coo_ref[row]) 
                        col_label = np.nonzero(coo_ali[row])
                        for new_row in row_label[0]:
                            for new_col in col_label[0]:
                                cross_slice_array[new_row][new_col] = 1 
                    
                    cross_slice_array_i_j = cross_slice_array
                    cross_slice_array_j_i = cross_slice_array.T
                    cross_slice_array_i_j = sp.coo_matrix(cross_slice_array_i_j)
                    cross_slice_array_j_i = sp.coo_matrix(cross_slice_array_j_i)

                    net_list[i][j] = cross_slice_array_i_j
                    net_list[j][i] = cross_slice_array_j_i
    

    row_coo_matrix = []
    for i in net_list:
        for j in range(len(i)):
            if j == 0:
                empty_variable = i[j].tocsr()
                continue
            else:
                empty_variable = sp.hstack([empty_variable, i[j].tocsr()])
        row_coo_matrix.append(empty_variable)
    
    for i in range(len(row_coo_matrix)):
        if i == 0:
            empty_variable = row_coo_matrix[i]
            continue
        else:
            empty_variable = sp.vstack([empty_variable, row_coo_matrix[i]])
    
    return empty_variable.tocoo()

def adj_slices_to_net(
        target_adata : AnnData,                     # Spatial transcriptomic data needed to generate neighbor-joining matrix information
        adj_adata : AnnData,                        # Its neighboring slices
        spatial_label : str = 'align_spatial'       # Keywords for aligned coordinates
):
    target_spatial = np.array(target_adata.obsm[spatial_label],dtype=np.float32)
    adj_spatial = np.array(adj_adata.obsm[spatial_label],dtype=np.float32)

    distance_matrix = cdist(target_spatial,adj_spatial,metric='euclidean')
    min_indices_per_row = np.argmin(distance_matrix, axis=1)    
    min_indices_per_row = np.column_stack((np.arange(len(min_indices_per_row)),min_indices_per_row))

    adj_row_indices = min_indices_per_row[:,0].ravel()
    adj_col_indices = min_indices_per_row[:,1].ravel()
    adj_values = np.tile(1,len(adj_row_indices))

    target_adjslice_adj = sp.coo_matrix((adj_values, (adj_row_indices, adj_col_indices)), shape=(target_spatial.shape[0],adj_spatial.shape[0]))

    return target_adjslice_adj

def adata_preprocess_adjnet(
        input_adata : list,                                 # A series of aligned sliced data
        align_model : str = 'single_template_alignment',    # Alignment model selected for alignment
        ref_label : int = 0,                                # Alignment template for single template alignment model
        spatial_label : str = 'align_spatial',              # Keywords for aligned coordinates
        add_net_keywords_self : str = 'adj_spatial_net',    # Keywords for the adjacency matrix for each slices
        n_neighbors : int = 10,                             # Number of neighbours selected by the adjacency matrix at the time of its creation
        no_cross = False):                                  # If True, No global adjacency
    
    # Process a series of slices to get cross-slice information, with intra-slice information
    
    input_adata = input_adata.copy()

    if align_model.lower() not in ['single_template_alignment','sequential_alignment']:
        raise ValueError(f"Invalid flavor '{align_model}'. Please choose 'single_template_alignment' or 'sequential_alignment'.") 

    if align_model.lower() == 'single_template_alignment':
        input_adata = adata_preprocess_dim(input_adata=input_adata,ref_label=ref_label)
        input_adata = get_spatial_net(input_adata = input_adata,coordinates_label = spatial_label,
                                      n_neighbors=n_neighbors,
                                      add_net_words=add_net_keywords_self)
        slice_matrix = get_need_ST_reconstruction(input_adata=input_adata)
        spatial_net = get_all_cross_slice_spatial_net(
                                input_adata = input_adata,
                                ref_label = ref_label,
                                ref_ali_all_slices = True,
                                no_cross=no_cross)
        
        return slice_matrix,spatial_net

    if align_model.lower() == 'sequential_alignment':
        slice_matrix = get_need_ST_reconstruction(input_adata=input_adata)
        input_adata = get_spatial_net(input_adata = input_adata,coordinates_label = spatial_label,
                                      n_neighbors=n_neighbors,
                                      add_net_words=add_net_keywords_self)
        
        net_list = np.zeros((len(input_adata),len(input_adata)))
        net_list = net_list.tolist()

        for index_i in range(len(input_adata)):
            if index_i == 0:
                for index_j in range(len(input_adata)):
                    if index_j == index_i:
                        net_list[index_i][index_j] = input_adata[index_i].obsm[add_net_keywords_self]
                        continue
                    if index_j == (index_i + 1):
                        net_list[index_i][index_j] = adj_slices_to_net(
                            target_adata=input_adata[index_i],
                            adj_adata=input_adata[index_j],
                            spatial_label=spatial_label
                        )
                        continue
                    net_list[index_i][index_j] = sp.coo_matrix(([], ([], [])), 
                                                               shape=(input_adata[index_i].X.shape[0], input_adata[index_j].X.shape[0]))

            if index_i != 0 and index_i != len(input_adata) - 1:
                for index_j in range(len(input_adata)):
                    if index_j == index_i:
                        net_list[index_i][index_j] = input_adata[index_i].obsm[add_net_keywords_self]
                        continue
                    if index_j == (index_i + 1):
                        net_list[index_i][index_j] = adj_slices_to_net(
                            target_adata=input_adata[index_i],
                            adj_adata=input_adata[index_j],
                            spatial_label=spatial_label
                        )
                        continue
                    if index_j == (index_i - 1):
                        net_list[index_i][index_j] = adj_slices_to_net(
                            target_adata=input_adata[index_i],
                            adj_adata=input_adata[index_j],
                            spatial_label=spatial_label
                        )
                        continue
                    net_list[index_i][index_j] = sp.coo_matrix(([], ([], [])), 
                                                               shape=(input_adata[index_i].X.shape[0], input_adata[index_j].X.shape[0]))

            if index_i == len(input_adata) - 1:
                for index_j in range(len(input_adata)):
                    if index_j == index_i:
                        net_list[index_i][index_j] = input_adata[index_i].obsm[add_net_keywords_self]
                        continue

                    if index_j == (index_i - 1):
                        net_list[index_i][index_j] = adj_slices_to_net(
                            target_adata=input_adata[index_i],
                            adj_adata=input_adata[index_j],
                            spatial_label=spatial_label
                        )
                        continue

                    net_list[index_i][index_j] = sp.coo_matrix(([], ([], [])), 
                                                               shape=(input_adata[index_i].X.shape[0], input_adata[index_j].X.shape[0]))
        
        row_coo_matrix = []
        for i in net_list:
            for j in range(len(i)):
                if j == 0:
                    empty_variable = i[j].tocsr()
                    continue
                else:
                    empty_variable = sp.hstack([empty_variable, i[j].tocsr()])
            row_coo_matrix.append(empty_variable)
    
        for i in range(len(row_coo_matrix)):
            if i == 0:
                empty_variable = row_coo_matrix[i]
                continue
            else:
                empty_variable = sp.vstack([empty_variable, row_coo_matrix[i]])
        
        return slice_matrix,empty_variable.tocoo()

class GP_VAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GP_VAE, self).__init__()

        # encode
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True,
                            dropout = 0.1, add_self_loops= True, bias=False)
        self.gat2 = GATConv(hidden_channels , hidden_channels, heads=num_heads, concat=True,
                            dropout = 0.1, add_self_loops= True, bias=False)
        
        # or Direct GAT generation, similar to VGAE's GCN generation?
        self.fc_mu = nn.Linear(hidden_channels, out_channels)
        self.fc_logvar = nn.Linear(hidden_channels, out_channels)

        #decode
        self.gat3 = GATConv(out_channels, hidden_channels, heads=num_heads, concat=True,
                            dropout = 0.1, add_self_loops= True, bias=False)  
        self.gat4 = GATConv(hidden_channels , in_channels, heads=num_heads, concat=True,
                            dropout = 0.1, add_self_loops= True, bias=False)
        
    def encode(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return self.fc_mu(x), self.fc_logvar(x)
    
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    # def reparametrize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()  
    #     eps = torch.randn_like(std)   
    #     return eps.mul(std).add_(mu)  

    def decode(self, z, edge_index):
        h3 = F.relu(self.gat3(z, edge_index))
        h4 = self.gat4(h3,edge_index)
        return h4
    
    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparametrize(mu, logvar)
        return self.decode(z,edge_index), mu, logvar, z

class GP_VAE_all(nn.Module):
    '''
    All built using graph attention for variational inference
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GP_VAE_all, self).__init__()

        # encode
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True,
                            dropout = 0.1, add_self_loops= True, bias=False)
        self.gat2 = GATConv(hidden_channels , hidden_channels, heads=num_heads, concat=True,
                            dropout = 0.1, add_self_loops= True, bias=False)
        
        # or Direct GAT generation, similar to VGAE's GCN generation?
        self.fc_mu = GATConv(hidden_channels, out_channels, heads=num_heads, concat=True,
                            dropout = 0.1, add_self_loops= True, bias=False)
        self.fc_logvar = GATConv(hidden_channels, out_channels, heads=num_heads, concat=True,
                            dropout = 0.1, add_self_loops= True, bias=False)

        #decode
        self.gat3 = GATConv(out_channels, hidden_channels, heads=num_heads, concat=True,
                            dropout = 0.1, add_self_loops= True, bias=False)  
        self.gat4 = GATConv(hidden_channels , in_channels, heads=num_heads, concat=True,
                            dropout = 0.1, add_self_loops= True, bias=False)
        
    def encode(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return self.fc_mu(x,edge_index), self.fc_logvar(x,edge_index)
    
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    # def reparametrize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()  
    #     eps = torch.randn_like(std)   
    #     return eps.mul(std).add_(mu)  

    def decode(self, z, edge_index):
        h3 = F.relu(self.gat3(z, edge_index))
        h4 = self.gat4(h3,edge_index)
        return h4
    
    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparametrize(mu, logvar)
        return self.decode(z,edge_index), mu, logvar, z

def train_stVGP(
        ST_need_reconstruction_matrix,
        all_spatial_net,
        lr = 0.001,
        weight_decay = 0.0001,
        training_epoch = 1500,
        num_heads = 1,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') ,
        save_model = False,
        save_model_path = 'path',
        hidden_embedding = [512,32],
        random_seed = 512,
        optimize_method = 'adam',
        whether_gradient_clipping = False,
        gradient_clipping = 5.0,
        all_gat = False,
        ):
    
    '''
    Args:
        ST_need_reconstruction_matrix: 
            the splice matrix of all processed count matrices.
        all_spatial_net : 
            Adjacency network matrix constructed by all slices of all spots.
        lr : 
            learning rate.
        weight_decay : 
            weight_decay.
        training_epoch : 
            training_epoch.
        num_heads : 
            Number of GATconv heads.
        device : 
            If gpu is available, use gpu acceleration, if not, choose cpu
            save_model : Whether to save model parameters at the end of model training,
            if True, Please provide the save path to the parameter save_model_path.
        save_model_path : 
            Save path for model parameters.
        hidden_embedding : 
            Hidden layer dimension and embedding dimension. The model decoder and encoder 
            are both two-layer structures, please provide a list of length 2.
        optimize_method :
            Optimiser selection parameters, which optimiser to choose for optimisation.
        whether_gradient_clipping:
            Whether to set parameters to prevent gradient explosion.
        gradient_clipping:
            Gradient explosion prevention parameters
        all_gat:
            Whether or not all of them are built using the graph attention mechanism
    '''
    seed = random_seed

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    hidden_dims = [ST_need_reconstruction_matrix.shape[1]] + hidden_embedding
    X_tensor = torch.tensor(ST_need_reconstruction_matrix, dtype=torch.float32)
    edge_list = []
    edge_list.append(all_spatial_net.row.tolist())
    edge_list.append(all_spatial_net.col.tolist())
    adj_tensor = torch.LongTensor(edge_list)

    data = Data(x=X_tensor,edge_index=adj_tensor)
    data = data.to(device)

    in_channels, hidden_channels, out_channels = hidden_dims[0],hidden_dims[1],hidden_dims[2]
    num_heads = 1

    if all_gat :
        model = GP_VAE_all(in_channels = in_channels, hidden_channels = hidden_channels, 
                    out_channels = out_channels, num_heads = num_heads).to(device)
    else:
        model = GP_VAE(in_channels = in_channels, hidden_channels = hidden_channels, 
                    out_channels = out_channels, num_heads = num_heads).to(device)
    
    reconstruction_function = nn.MSELoss(reduction='sum')

    def loss_function(recon_x, x, mu, logvar):
        BCE = reconstruction_function(recon_x, x)  # mse loss
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        return BCE + KLD
    
    if optimize_method.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimize_method.lower() == 'rprop':
        optimizer = torch.optim.Rprop(model.parameters(), lr=lr)
    
    print('Model training')

    for epoch in tqdm(range(training_epoch)):
        model.train()
        optimizer.zero_grad()
        recon_x, mu, logvar, z = model(data.x, data.edge_index)
        loss = loss_function(recon_x, data.x, mu, logvar)
        loss.backward()
        if whether_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    
    model.eval()
    model_params = model.state_dict()

    if save_model:
        torch.save(model_params, save_model_path)

    with torch.no_grad():
        recon_x, embedding_mu, logvar,embedding_sample = model(data.x, data.edge_index)

    recon_x = recon_x.to('cpu').detach().numpy()
    embedding = embedding_mu.to('cpu').detach().numpy()
    logvar = logvar.to('cpu').detach().numpy()
    
    return recon_x,embedding,model_params,logvar

def get_3D_prediction(train_coordinates:NDArray,            # aligned coordinate
                      embedding:NDArray,                    # embedded layer representation
                      spatial_pred:NDArray,                 # Coordinates of predictive expressions
                      noise = False,                        # whether to introduce white noise
                      noise_value = 0.00001,                # noise level
                      constant_value = 1.0,                 # Constant washout for Gaussian processes
                      Rbf_value = 512):                     # Smoothness of Gaussian processes
    
    if noise:
        embedding_noise = np.random.normal(loc=0, scale = np.sqrt(noise_value), size = embedding.shape)
        embedding = embedding + embedding_noise

    else:
        embedding = embedding
        kernel = ConstantKernel(constant_value, constant_value_bounds="fixed") * RBF(Rbf_value, length_scale_bounds="fixed")
        gaussian = GaussianProcessRegressor(kernel=kernel)
        fiting = gaussian.fit(train_coordinates,embedding)
        d = fiting.predict(spatial_pred)

    return d

def gene_prediction(
        slice_matrix,
        prediction_embedding,               # Embedded expression after prediction
        adj_matrix,                         # Neighborhood information for spatial transcriptomes
        checkpoint,                         # Model parameters for stVGP(after training)
        model_layer,                        # Model parameters for each layer
        all_gat,                            # Whether all GAT structures are used
        logvar,                             # The variance information generated during training, if lost, can be regenerated using the original data
        device,                             # If gpu is available, use gpu acceleration, if not, choose cpu,please be as consistent as possible with the training
):
    in_channels, hidden_channels, out_channels,num_heads = model_layer[0],model_layer[1],model_layer[2],model_layer[3]  
    
    slice_matrix = slice_matrix.to(device)
    adj_matrix = adj_matrix.to(device)

    if all_gat:
        if logvar == None:

            model = GP_VAE_all(in_channels = in_channels, hidden_channels = hidden_channels, 
                        out_channels = out_channels, num_heads = num_heads).to(device)
            model.load_state_dict(checkpoint)

            # logvar needs to be generated when not provided
            mu, logvar = model.encode(slice_matrix,adj_matrix)
        
            data_pred = Data(x=prediction_embedding,edge_index=adj_matrix)
            data_pred = data_pred.to(device)

            with torch.no_grad():
                z = model.reparametrize(data_pred.x,logvar)
                recon_gene = model.decode(z,data_pred.edge_index)
            recon_gene = recon_gene.to('cpu').detach().numpy()

            return recon_gene
        
        else:
            model = GP_VAE_all(in_channels = in_channels, hidden_channels = hidden_channels, 
                        out_channels = out_channels, num_heads = num_heads).to(device)
            
            model.load_state_dict(checkpoint)
     
            data_pred = Data(x=prediction_embedding,edge_index=adj_matrix)
            data_pred = data_pred.to(device)

            logvar = torch.tensor(dtype=torch.float32)
            logvar = logvar.to(device)

            with torch.no_grad():
                z = model.reparametrize(data_pred.x,logvar)
                recon_gene = model.decode(z,data_pred.edge_index)
            recon_gene = recon_gene.to('cpu').detach().numpy()

            return recon_gene

    else:
        if logvar == None:
            model = GP_VAE(in_channels = in_channels, hidden_channels = hidden_channels, 
                        out_channels = out_channels, num_heads = num_heads).to(device)

            model.load_state_dict(checkpoint)

            data_pred = Data(x=prediction_embedding,edge_index=adj_matrix)
            data_pred = data_pred.to(device)

            # logvar needs to be generated when not provided
            mu, logvar = model.encode(slice_matrix,adj_matrix)

            with torch.no_grad():
                z = model.reparametrize(data_pred.x,logvar)
                recon_gene = model.decode(z,data_pred.edge_index)
            recon_gene = recon_gene.to('cpu').detach().numpy()

            return recon_gene
        
        else:
            model = GP_VAE(in_channels = in_channels, hidden_channels = hidden_channels, 
                        out_channels = out_channels, num_heads = num_heads).to(device)

            model.load_state_dict(checkpoint)

            logvar = torch.tensor(dtype=torch.float32)
            logvar = logvar.to(device)

            data_pred = Data(x=prediction_embedding,edge_index=adj_matrix)
            data_pred = data_pred.to(device)
            
            with torch.no_grad():
                z = model.reparametrize(data_pred.x,logvar)
                recon_gene = model.decode(z,data_pred.edge_index)
            recon_gene = recon_gene.to('cpu').detach().numpy()

            return recon_gene

    