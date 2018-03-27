# Sci-LMM
Sparse Cholesky factorIzation Linear Mixed model

## generate pedigree
SimulatePedigrees.py - creates 3 files. 
1. rel.npz - a compressed csr_matrix representing a pedigree via an adjacency matrix. can be loaded via load_sparse_csr
2. sex.npy - generated sex (0/1) for the pedigree
3. gen_ind.npy - each individual's generation (e.g. if you wish to generate yob for the pedigree)

### parameters
--sample_size - Size of the cohort
--sparsity_factor - Number of nonzero entries in the IBD matrix'
--gen_exp - Gen size = gen_exp X prev gen size
--init_keep_rate - 1 - number of edges to remove before iteration begins')
--save_folder - which folder it should save the output to

### example
python --sample_size 100000 --sparsity_factor 0.001 --gen_exp 1.4 --init_keep_rate 0.8 --save_folder Examples/100K



## Create covariance matrices
python create_matrices.py --epis --dom --main_folder . --temp_folder .
python compute_heritability.py --reml --matrices_paths_list ./NumeratorMatrix.npz ./EpistatisMatrix.npz ./DominanceMatrix.npz --phenotype_path ./y.npy

This example creates the following pedigree (the first example taken from http://www.genetic-genealogy.co.uk/Toc115570135.htmlhttp://www.genetic-genealogy.co.uk/Toc115570135.html)
Creates the numerator/epistasis/dominance matrices for only [7, 8, 9].
Generates phenotypes and computes HE and REML for it.


create_matrices.py arguments
--epis - generate Epistasis matrix
--dom - generate Dominance matrix
--rel_mat - the location of the .npy file containing the adjacency matrix of the pedigree
--indices_array - a subset of individuals that are of higher interest
--main_folder - where all the matrices, covariates and phenotype will be stored
--temp_folder - where all the temp files are created. Highly recommended to have a large disk space for that folder.
--verbose - verbose output.


## Compute heritability
compute_heritability.py arguments
--reml - when finding the maximum likelihood, take the restricted maximum likelihood.
--sim_num - number of vectors to simulate in-order to approximate the covarinace matrix (see Loh. 2015)
--pc_num - number of principal components to use as covariates
--matrices_paths_list - the location of the .npz files of all the covariance matrices SciLMM would take into consideration
--covariates_path - the .npy of the covariates
--phenotype_path - location of the phenotype (single phenotype only for now) of the cohort. saved as a .npy
--simulate_y_sigmas - if you generated sigmas somewhere else.
--simulate_y_fixed_effects - same as above.
--verbose - verbose output.




If needed, please contact Tal Shor at talihow@gmail.com.
