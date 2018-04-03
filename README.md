# Sci-LMM
Sparse Cholesky factorIzation Linear Mixed model


## Parameters

### General parameters
`--verbose` - prints more information along the run.

`--output_folder OUTPUT_FOLDER` - which folder it should save the output to.


### Pedigree simulation
`--simulate` - Run simulations

`--sample_size SAMPLE_SIZE` - Size of the cohort

`--sparsity_factor SPARSITY_FACTOR` - Number of nonzero entries in the IBD matrix

`--gen_exp GEN_EXP`  - Gen size = gen_exp X prev gen size

`--init_keep_rate INIT_KEEP_RATE - 1 - number of edges to remove before iteration begins

`--fam FAM` - .fam file representing the pedigree. the phenotype column contains all 0 if everyone is of interest, or if only a subset is of interest the're phenotype will contain 1


### Create covariance matrices
`--IBD` - Create IBD matrix

`--Epistasis` - Create pairwise-epistasis matrix

`--Dominance` - Create dominance matrix

`--IBD_exists` - existence of the .npz file for the IBD matrix (if you already had build the matrix via the --IBD option)

`--Epis_exists` - existence of the .npz file for the Epistasis matrix (if you already had build the matrix via the --Epistasis option)

`--Dom_exists` - existence of the .npz file for the Dominance matrix (if you already had build the matrix via the --Dominance option)


### Estimation
`--generate_y` - Generate a random y

`--y Y` - the phenotype (npy file containing an n sized numpy array)

`--covariates COV` - the covaraites, not including sex (npy file containing an nxc sized numpy array)

`--HE` - Estimate fixed effects and covariance coefficients via Haseman-Elston

`--LMM` - Estimate fixed effects and covariance coefficients via Linear mixed models

`--REML` - Use REML instead of simple maximum likelihood

`--sim_num SIM_NUM` - Number of simulated vectors

`--fit_intercept` - Use an intercept as a covariate


## Test the system

The results of the following code should be similar (up to the REML simulated vectors method)

```
mkdir Examples/100K_2

python SciLMM.py --simulate --sample_size 100000 --sparsity_factor 0.001 --output_folder Examples/100K_2 --IBD --Epistasis --generate_y --HE --LMM --REML --fit_intercept

python SciLMM.py`--output_folder Examples/100K_2 --fam Examples/100K_2/rel.fam --IBD_exists --Epis_exists --y Examples/100K_2/y.npy`--HE --LMM --REML --fit_intercept
```


If needed, please contact Tal Shor at talihow@gmail.com.
