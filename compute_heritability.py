"""
Copyright (C) 2017 Tal Shor and Omer Weissbrod

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""





import numpy as np
import scipy.stats as stats
import scipy.linalg as la
import scipy.optimize as optimize
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from scipy.sparse import csr_matrix, lil_matrix, spdiags, eye
import sys
import time
from datetime import datetime
import os
from os.path import exists
np.set_printoptions(precision=4, linewidth=200)
from sksparse.cholmod import cholesky
from sklearn.linear_model import LinearRegression
from SideAlgos.SparseFunctions import load_sparse_csr, save_sparse_csr
import argparse
from itertools import combinations_with_replacement,  product
from sklearn.metrics import mean_squared_error as mse
import json


class SparseLMM:
	"""
	This is the main class for the paper. It takes covariance matrices, covariates and a phenotype and calculates
	HE, REML and the certinty of those results

	Positional arguments:
	data - a Data instance containing all the matrices, covariates and y of the test
	sim_num - number of y_v vector to simulate V
	"""
	def __init__(self, data, sim_num):
		self._sim_num = sim_num
		self._data = data
	
	def covariate_significance(self, sig2g_array):
		"""
		Compute the p-values for the covariates given variances

		Positional arguments:
		sig2g_array - all the sigmas
		"""
		V = self._data.matrices_weighted_sum(sig2g_array)

		#Compute the Cholesky factorization of V, and permute all the other matrices accordingly
		factor = cholesky(V)
		
		#compute fixed effects
		C = self._data.covariates
		invV_C = factor(C)
		CT_invV_C = C.T.dot(invV_C)
		L_CT_invV_C = la.cho_factor(CT_invV_C)
		fixed_effects = la.cho_solve(L_CT_invV_C,  C.T.dot(factor(self._data.y)))		
		var_fixedeffects = la.cho_solve(L_CT_invV_C,  np.eye(C.shape[1]))
		
		test_stats = fixed_effects**2 / np.diag(var_fixedeffects)
		p_values = stats.f(1, self._data.y.shape[0]-1).sf(test_stats)
		
		return fixed_effects, p_values	
	
	def compute_sig_of_sig(self, sig2_arr):
		"""
		Compute the std of the sigmas

                Positional arguments:
                sig2_arr - all the sigmas
		"""	
		matrices = self._data.matrices
		X = self._data.covariates
		n = self._data.num_matrices
		y = self._data.y		

		#print "Computing V ... ",
		V = self._data.matrices_weighted_sum(sig2_arr)
		#print "Successfully computed V"


		#print "Cholesky on V ... ",
		factor = cholesky(V)
		#print "Successfully Choleskied V"
	        

		#print "Computing P ...", 
		V_inv_y = factor(y)
		V_inv_X = factor(X)
		X_t_V_inv_X__inv = np.linalg.inv(X.T.dot(V_inv_X))
		P = V_inv_y - V_inv_X.dot(X_t_V_inv_X__inv.dot(X.T.dot(V_inv_y)))
		#print "Successfully Computed P"


		#print "Calculating Hess ... ",
		hess = np.empty((n, n))
		factor_array = [factor(matrices[j].dot(P)) for j in range(n)]
		for i, j in combinations_with_replacement(range(n), 2):
			hess[i,j] = -0.5 * y.T.dot(factor(matrices[i].dot(factor_array[j])))
			hess[j,i] = hess[i,j]
		#print "Successfully calulated Hess"

		inv_neg_hess = la.inv(-hess)
		return np.sqrt((inv_neg_hess * (1+1.0/ self._sim_num))[range(n), range(n)])
		
	def bolt_gradient_estimation(self, log_sig2g_array, reml, verbose):
		"""
		The main function for the article. Finds the gradient of the REML / MLE for optimization.
		1000 simulations shows great results.

		Positional arguments:
		log_sig2g_array - the current sigmas (but logged). This is the optimization value.	
		verbose - A boolean indicating to print more or less information...
		reml - Either a REML run or a MLE 
		"""
		sig2g_array = np.exp(log_sig2g_array)
	
		if verbose:
			t0 = time.time()
			print 'estimating nll and its gradient at:', sig2g_array
			
		V = self._data.matrices_weighted_sum(sig2g_array)

		#Compute the Cholesky factorization of V, and permute all the other matrices accordingly
  		factor = cholesky(V)
		P = factor.P()
		p_inv = np.argsort(P)
				
		#compute fixed effects
		C = self._data.covariates
		invV_C = factor(C)
		L_CT_invV_C = la.cho_factor(C.T.dot(invV_C))
		fixed_effects = la.cho_solve(L_CT_invV_C,  C.T.dot(factor(self._data.y)))
		mu = self._data.covariates.dot(fixed_effects)

		#compute invV.dot(y-mu)
		invV_y = factor(self._data.y-mu)
	
		# Sound Math
		#compute the negative log likelihood
		nll_numer = (self._data.y-mu).dot(invV_y)
		nll_denom = (self._data.n*np.log(2*np.pi) + factor.logdet())	
		nll = 0.5 * (nll_numer + nll_denom)				
		
		if reml:
			# TODO: maybe it's - and no need for 2*?
			# Sound Math
			nll  += 0.5 * 2*np.sum(np.log(np.diag(L_CT_invV_C[0])))	#ignoring some irrelevant constants

		factor_L = factor.L()
		y_simu = factor_L.dot(np.random.randn(self._data.n, self._sim_num))
		y_simu = y_simu[p_inv]
		invV_ysimu = factor(y_simu)

		#compute the gradient
		grad = np.zeros(len(sig2g_array))
		for grad_i in xrange(len(sig2g_array)):
			# Sound math
			comp1 = np.mean(np.sum(self._data.matrices[grad_i].dot(invV_ysimu) * invV_ysimu, axis=0))
			comp2 = invV_y.dot(self._data.matrices[grad_i].dot(invV_y))
			grad[grad_i] = 0.5 * (comp1 - comp2)	#this is the gradient of the *negative* log likelihood
			
			if reml:
				# Sound math
				vec = invV_C.T.dot(self._data.matrices[grad_i].dot(invV_C))
				temp = la.cho_solve(L_CT_invV_C, vec)
				grad[grad_i] -= 0.5 * np.trace(temp)
				if verbose:
					print -0.5 * np.trace(temp),

		# TODO: check if this is relevant	
		#compute the gradient with respect to log_sig2g (instead of sig2g directly), using the chain rule
		grad *= sig2g_array		

		if verbose:
			print "grad : ", grad
			print 'nll: %0.8e   computation time: %0.2f seconds'%(nll, time.time()-t0)
	
		return nll, grad

	def  compute_HE(self, with_intercept):
		"""
		An efficient computation of Haseman-Elston for the sparse matrices

                Positional arguments:
		with_intercept - include an intercept in the calculations.
		"""
		matrix_list = [mat.copy() for mat in self._data.matrices[0:-1]]
	        m = self._data.num_matrices - 1
		y = self._data.y
	        y = (y - y.mean()) / y.std()
		n = self._data.n
		for i in range(m):
			mat = matrix_list[i]
			matrix_list[i] = (mat - mat.multiply(eye(n))).copy()


	        xtx = np.zeros((m+1, m+1) if with_intercept else (m, m))
	        for i, j in product(range(m),range(m)):
	                if i <= j:
	                        xtx[i, j] = matrix_list[i].multiply(matrix_list[j]).sum()
	                        xtx[j, i] = xtx[i, j]
	
	        n = matrix_list[0].shape[0]
	        if with_intercept:
	                for i in range(m):
	                        xtx[i, m] = matrix_list[i].sum()
	                        xtx[m, i] = xtx[i, m]
	                xtx[m, m] = n ** 2 - n
	
	        xty = np.zeros((m + 1 if with_intercept else m))
	        for i in range(m):
	                xty[i] = (y * matrix_list[i]).dot(y)
	
	        if with_intercept:
	                xty[m] = y.sum() ** 2 - y.dot(y).sum()
	
		coef  = np.linalg.inv(xtx).dot(xty)[0:-1]
		coef = np.append(coef, 1 - coef.sum())
		return coef


class Data(object):
	"""
	This class contains all the matrices and covariates.
	
	Positional arguments:

	matrices_paths - a list of paths to the npz files of the covariances matrices
	y_loc - the path to the npy file of the phenotypes 
	reml - Restricted ML flag (y would be regressed accordingly)
	

	Keyword arguments:
	cov_loc - if it has known covaraites 
	pc_num - number of principal components to add to the covariates
	verbose - print more information
	"""

	def __init__(self, matrices_paths, y_loc, reml,
			cov_loc=None, pc_num=0, verbose=False):
		
		if len(matrices_paths) < 1:
			raise Exception("Dat must have at l covariance matrix")

		for mat_path in matrices_paths:
			if mat_path is None or not exists(mat_path):
				raise Exception("Not a valid correlation matrix (.npz) location")		

		if y_loc is None or not exists(y_loc):
			raise Exception("Not a valid phenotype (.npy) path")

		self._verbose = verbose
		
		# covariance matrices
		self._matrices = [self._load_mat(mat_path) for mat_path in matrices_paths]
		
		# phenotype
		self._y = self._load_vec(y_loc)

		# fixed effects
		if cov_loc:
			cov = self._load_vec(cov_loc)
		else:
			cov = np.zeros((self._y.shape[0], 1))

		if pc_num > 0:
			# TODO: This is not the real PCA! you have to multiply it by X (if our correlation is XTX)
			s, U = sla.eigsh(self._matrices[0], k=pc_num)
			U = U[:, np.arange(pc_num)[::-1]]
			self._covariates = np.concatenate((cov, U), axis=1)
		else:
			self._covariates = cov

		# If non reml, simply regress covariates from phenotype
		if (not reml):
			if self._verbose:
				print 'normalizing covariates...'
			linreg = LinearRegression(fit_intercept=True)
			linreg.fit(self._covariates, self._y)
			if self._verbose:
				print 'linear regression coefficients:', linreg.coef_
			self._y -= linreg.predict(self._covariates)
			self._y /= self._y.std()
		

	
	def _load_mat(self, mat_name):
		if self._verbose:
			print "Loading", mat_name, "...", 
			start_time = time.time()

		mat = load_sparse_csr(mat_name)
		# For matrices that are packed to the brim with 1e-8 values removing them won't effect PD, and cholesky wastes time over them. 
		# those quick 2 actions remove enough - but don't damage the results
		mat = np.round(mat, 8)
		mat = mat.multiply(mat > 0.000001)

		if self._verbose:
			print "Successfully loaded in ", time.time() - start_time, "seconds"
		
		return mat
		
	def _load_vec(self, vec_name):
		return np.load(vec_name)
		
	@property
	def n(self):
		return self._matrices[0].shape[0]

	@property
	def y(self):
		return self._y
		
	@property
	def num_matrices(self):
		return 1 + len(self._matrices)

	@property	
	def matrices(self):
		return self._matrices + [eye(self.n).tocsr()]
	
	@property
	def covariates(self):
		if np.all(self._covariates == np.zeros((self.n, 1))):
			return np.ones((self.n, 1))

		# adds an intercept
		return np.concatenate((np.ones((self.n))[:,np.newaxis], 
								self._covariates), axis=1).copy()
	
	def matrices_weighted_sum(self, sig2g_array):
		matrices_list = self.matrices
		V = sig2g_array[0] * matrices_list[0]
		for i in xrange(1, len(sig2g_array)): V += sig2g_array[i] * matrices_list[i]
		return sparse.csc_matrix(V)


# Do not include intercept and environment in paths but include in sigams and fixed effects
def simulate_phenotype(matrices_paths, cov_path, sigmas, fixed_effects):
	if len(sigmas) != len(matrices_paths) + 1:
		raise Exception("Incorrect number of sigmas")

	# variance
	factors = [cholesky(load_sparse_csr(mat)) for mat in matrices_paths]
	n = factors[0].L().shape[0]

	sim = [factor.L().dot(np.random.randn(n)) for i, factor in enumerate(factors)]
	sim = [sim[i][np.argsort(factor.P())] for i, factor in enumerate(factors)]
	sim += [np.random.randn(n)]
	sim = np.array(sim).T
	sim = (sim - sim.mean(axis=0)) / sim.std(axis=0)
	y = sim.dot(np.sqrt(np.array(sigmas)))	
		
	# covariates
	if cov_path is None or len(cov_path) == 0:
		covs =  np.ones((n, 1))
	else:
		covs = np.hstack((np.ones((n, 1)), np.load(cov_path)))
	y += covs.dot(fixed_effects)
	return y


def compute_sigmas(matrices_paths, y_loc, cov_path=None, reml=True, sim_num=100, pc_num=0, verbose=False):
	ds = Data(matrices_paths, y_loc, reml, verbose=verbose, cov_loc=cov_path, pc_num=pc_num)
	x0 = np.random.random(ds.num_matrices); x0 /= x0.sum()
        x0 = np.log(x0)

	lmm = SparseLMM(ds, sim_num)

	log_bounds = tuple(((None, 0) for i in range(x0.shape[0])))

	bolt_time = datetime.now()
	optObj_bfgs = optimize.minimize(lmm.bolt_gradient_estimation, x0 , args=(reml, verbose), jac=True, method='L-BFGS-B', options={'eps':1e-5, 'ftol':1e-7})
	bolt_time = datetime.now() - bolt_time

	value = np.exp(optObj_bfgs.x)

	fe_calc_time = datetime.now()
	fixed_effects, fe_p_values = lmm.covariate_significance(value)
	fe_calc_time = datetime.now() - fe_calc_time

	sig_calc_time = datetime.now()
	sig_of_sig = lmm.compute_sig_of_sig(value)
	sig_calc_time = datetime.now() - sig_calc_time	

	he_time = datetime.now()
	he = lmm.compute_HE(True)
        he_time = datetime.now() - he_time

	# so it is a standad JSON object
	return {"bolt_sigmas":list(value),
		"bolt_sigmas_time":bolt_time.total_seconds(),
		"bolt_fixed_effects":list(fixed_effects),
		"bolt_fixed_effects_p_values":list(fe_p_values),
		"bolt_fixed_effects_time":fe_calc_time.total_seconds(),
		"bolt_sig_of_sig":list(sig_of_sig),
		"bolt_sig_of_sig_time":sig_calc_time.total_seconds(),
		"he_sigmas":list(he),
		"he_time":he_time.total_seconds()}


if __name__ == '__main__':
 	parser = argparse.ArgumentParser(description='Compute sigams')
 	parser.add_argument('--reml', dest='reml', action='store_true', default=True)
	parser.add_argument('--verbose', dest='verbose', action='store_true', default=True)
 	parser.add_argument('--sim_num', dest='sim_num', type=int, default=100)
	parser.add_argument('--pc_num', dest='pc_num', type=int, default=0)
	parser.add_argument('--matrices_paths_list', nargs='+', type=str, dest='matrices')
	parser.add_argument('--covariates_path', type=str, dest='cov_path', default=None)
	parser.add_argument('--phenotype_path', dest='y_path')	
	parser.add_argument('--simulate_y_sigmas', nargs='+', type=float, dest='sigmas')
	parser.add_argument('--simulate_y_fixed_effects', nargs='+', type=float, dest='fixed_effects')

 	args = parser.parse_args()
	if args.sigmas is not None:
		np.save(args.y_path, simulate_phenotype(args.matrices, args.cov_path, args.sigmas, args.fixed_effects))

	print json.dumps(compute_sigmas(args.matrices, args.y_path, 
		cov_path=args.cov_path, reml=args.reml, pc_num=args.pc_num,
		sim_num=args.sim_num, verbose=args.verbose), indent=4)

