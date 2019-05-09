# Sci-LMM
Sparse Cholesky factorIzation Linear Mixed Model.

Sci-LMM is a modeling framework for studying population-scale family trees that combines techniques from the animal and plant breeding literature and from human genetics literature. 
The framework can construct a matrix of relationships between trillions of pairs of individuals and fit the corresponding Linear Mixed Models. 
Sci-LMM provides a unified framework for investigating the epidemiological history of human populations via genealogical records.

## Usage
The code has two separate programs:
- Identity By Descent (IBD) computing: used to compute a kinship matrix from a pedigree. 
- Sparse Cholesky: used to fit a linear mixed model from given IBD matrices and covariate matrices to phenotypes.

## Identity By Descent computing
In order to run this you can either use the `IBDCompute` class, or run from terminal `python IBDCompute.py [args]`. 

### Parameters

#### General parameters

`--output_folder OUTPUT_FOLDER` - which folder it should save the output to.


#### Pedigree simulation
`--simulate` - Run simulations

`--sample_size SAMPLE_SIZE` - Size of the cohort

`--sparsity_factor SPARSITY_FACTOR` - Number of nonzero entries in the IBD matrix

`--gen_exp GEN_EXP`  - Gen size = gen_exp X prev gen size

`--init_keep_rate INIT_KEEP_RATE` - 1 - number of edges to remove before iteration begins


### IBD computing parameters
`--fam FAM` - .fam file representing the pedigree. the phenotype column contains all 0 if everyone is of interest, or if only a subset is of interest the're phenotype will contain 1

`--remove_cycles` - Remove cycles from relationship matrix. WARNING: there should no be any cycles. All nodes in cycles will be removed.

`--remove_access_parents` - Remove relations of nodes with too many parents. WARNING: All individuals should have no more than 2 parents. Access edges will be removed, not nodes.'

## Sparse Cholesky
In order to run this section run from terminal `python SparseCholesky.py [args]`.

### Parameters
`--A` - A path of an IBD matrix in .mm form.

`--phe` - A path of a table with phenotypes (columns should be id and phenotype with no header).

`--cov` - A path of covariate matrix. Should have no header. 

`--reml` - If added uses the REML method instead of Haseman-Elston.


## Tips
Creating IBD matrix can also be done using R's Nadiv package.

For anything bigger than 200K-250K individuals, I prefer HE. It's really really really fast, pretty accurate, and you don't have to mess around with the CHOLMOD library.

Creating IBD matrices for 2-3 million individuals works great, from some reason - 11 million involves large numerical errors - will update on this when I understand this better.

If needed, please contact Iris Kalka at iris.kalka@weizmann.ac.il.
