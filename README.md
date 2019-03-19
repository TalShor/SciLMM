# Sci-LMM
Sparse Cholesky factorIzation Linear Mixed Model.

Sci-LMM is a modeling framework for studying population-scale family trees that combines techniques from the animal and plant breeding literature and from human genetics literature. 
The framework can construct a matrix of relationships between trillions of pairs of individuals and fit the corresponding Linear Mixed Models. 
Sci-LMM provides a unified framework for investigating the epidemiological history of human populations via genealogical records.

## Usage
The code has two separate programs:
- Identity By Descent (IBD) computing: used to compute a correlation matrix from a pedigree. 
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


## Test the system

The results of the following code should be similar (up to the REML simulated vectors method)

```
mkdir Examples/100K_2

python SciLMM.py --simulate --sample_size 100000 --sparsity_factor 0.001 --output_folder Examples/100K_2 --IBD --Epistasis --generate_y --HE --LMM --REML --fit_intercept

python SciLMM.py --output_folder Examples/100K_2 --fam Examples/100K_2/rel.fam --IBD_exists --Epis_exists --y Examples/100K_2/y.npy --HE --LMM --REML --fit_intercept
```


## Examples
### 2 million individuals, HE only

```
python SciLMM.py --output_folder Examples/2M --fam Examples/2M/rel.fam --IBD 

python SciLMM.py --output_folder Examples/2M --IBD_exists --HE --fit_intercept --y Examples/2M/y.npy --fam Examples/2M/rel.fam --covariates Examples/2M/cov.npy

```
Resulting in a $\sigma^2_g$ of ~0.6 
(The IBD matrix here is pretty big - have at least 200G of memory + 100G of storage)

### 100K, HE + REML
```
python SciLMM.py --output_folder Examples/100K --fam Examples/100K/rel.fam --IBD --Epistasis --Dominance

python SciLMM.py  --IBD_exists --Epis_exists --Dom_exists --HE --LMM --REML --fit_intercept --output_folder Examples/100K --fam Examples/100K/rel.fam --y  Examples/100K/y.npy
```

It has been simulated as:
 sigmas are \[0.4723, 0.1752, 0.0745,0.2781\]
 intercept is 0.0092 and the sex coefficient is 0.009

note that the dominance matrix lowers the percision of the results by quite a lot due to it's high resemblance to I.


## Tips
For anything bigger than 200K-250K individuals, I prefer HE. It's really really really fast, pretty accurate, and you don't have to mess around with the CHOLMOD library.

Creating IBD matrices for 2-3 million individuals works great, from some reason - 11 million involves large numerical errors - will update on this when I understand this better.

If needed, please contact Tal Shor at talihow@gmail.com.
