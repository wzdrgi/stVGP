library(Seurat)

slice_num = 8
# Maximum serial number of the code generation file

for (index_slice in 0:slice_num){ 
  spotmarker_data = read.table(sprintf("C:/Users/wzd/Desktop/Alignment Domain Prediction/hh/select_gene/%s/select_gene_4.txt",index_slice),header = T,check.names = FALSE)
  
  cluster_class = spotmarker_data["marker_cluster",]
  cluster_class = t(cluster_class)
  cluster_class = as.data.frame(cluster_class)
  
  spatial_coordinates = spotmarker_data[c('x','y'),]
  spatial_coordinates = t(spatial_coordinates)
  spatial_coordinates = as.data.frame(spatial_coordinates)
  
  spotmarker_data <- spotmarker_data[!(rownames(spotmarker_data) == "marker_cluster"),]
  spotmarker_data <- spotmarker_data[!(rownames(spotmarker_data) == "x"),]
  spotmarker_data <- spotmarker_data[!(rownames(spotmarker_data) == "y"),]
  
  new_class_vector <- as.vector(cluster_class)
  my_vector <- unlist(new_class_vector)
  
  levels <- unique(my_vector)
  levels <- as.character(levels)
  
  my_vector22 <- factor(my_vector, levels = levels)
  
  seurat_obj <- CreateSeuratObject(counts = spotmarker_data)
  Idents(seurat_obj) <- factor(my_vector22)
  
  clusters1 <- Idents(seurat_obj)
  head(clusters1)
  
  seurat_obj <- FindVariableFeatures(seurat_obj)
  all_markers <- FindAllMarkers(seurat_obj, only.pos = TRUE)
  
  gene_name = all_markers$gene
  gene_name = unique(gene_name)
  spotmarker_data = t(spotmarker_data)
  spotmarker_data = as.data.frame(spotmarker_data)
  spotmarker_data$x = spatial_coordinates$x
  spotmarker_data$y = spatial_coordinates$y
  spotmarker_data$marker_cluster = cluster_class$marker_cluster
  
  morans_file <- spotmarker_data
  cluster_index <- unique(morans_file$marker_cluster)
  index_row_col <- trunc(sqrt(max(cluster_index)))
  
  
  library(sp)
  library(spdep)
  library(progress)
  result_gene_morans <- data.frame(stringsAsFactors = FALSE)
  
  pb <- progress_bar$new(total = length(gene_name))
  
  for (gene in gene_name){
    result_matrix <- matrix(NA, nrow = index_row_col, ncol = index_row_col)
    for (index in cluster_index){
      subset_morans_file <- morans_file[morans_file$marker_cluster == index,]
      coordinates <- subset_morans_file[, c("x", "y")]
      neighbors <- knn2nb(knearneigh(coordinates, k = 6)) 
      W <- nb2listw(neighbors)
      if (all(subset_morans_file[[gene]] == 0)){
        result_matrix[index] <- 0
        next
      }
      moran_result<- moran.test(subset_morans_file[[gene]], W)
      moran_result <- moran.mc(subset_morans_file[[gene]], W, nsim=999, zero.policy=TRUE)
      result_matrix[index] <- moran_result$statistic
    }
    new_matrix <- result_matrix - mean(result_matrix)
    new_matrix <- new_matrix * new_matrix
    var_value <- sum(new_matrix)
    new_row <- data.frame(gene = gene, Value = var_value)
    result_gene_morans <- rbind(result_gene_morans, new_row)
    pb$tick()
  }
  
  write.table(result_gene_morans,sprintf("C:/Users/wzd/Desktop/Alignment Domain Prediction/hh/select_gene/%s/gene_morans_4.txt",index_slice),row.names = F, quote = F, sep="\t",col.names = T)
}