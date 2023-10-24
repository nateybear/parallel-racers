import numpy as np
from numba import jit
from numba import njit
import timeit

'''
Initial multinomial choice problem speed test - using jit compilation through numba
numba does not accept axis calls e.g. np.mean(X,axis=1), and does not accept matrix multiplication in dim > 2
so cannot replace np.count or sum to replace mean by mat-multiplying that dim by a vector of 1's
only option is to collapse consumers and choices into a single vector s.t. we then have 2D tensor with sim draws
'''
## 100,000 consumers (each with scalar characteristic W)
N_cons = 100000
## 10 choices (each with scalar product characteristic X)
N_choices = 10
## 1000 sim draws (S)
N_sims = 1000


## Then for some arbitrary fixed parameters beta_0, beta_1, beta_2, 
## mean utility for the ith consumer for the jth product for the sth simulation draw is:
## U_{ijs} = beta_0 + (beta_1+beta_2*W[i]+ S[s])*X[j]

'''
generate data
'''
## consumer chars (100k) 
W = np.random.uniform(0,1,N_cons)
## product chars (10) (in a real implementation, these should be sorted by product IDs 1-10, so they correspond with correct rows in the choices matrix)
X = np.random.uniform(0,1,N_choices)[:,None]
## random utility (1000 draws)
S = np.random.uniform(0,1,N_sims)[None,:]
## fake choices Y, corresponding to product IDs (for the 100k consumers, no explicit outside option for now)
## move up by one integer so as not to divide by zero in broadcasting below to compute likelihood for each consumer over choices
Y = np.random.randint(0,10,N_cons) + 1
## sorted product_IDs
prod_IDs = np.arange(10).astype(int) + 1
## auxiliary matrix
## divide broadcast individual choices over the product IDs to get a 1 for each consumer's actual choice
## in a matrix of n_prods by n_cons
aux = Y[None,:] / prod_IDs[:,None]
choices = np.zeros((aux.shape[0],aux.shape[1]))
choices[aux==1] = 1

## generate consumer*simulant by consumer matrix
## (NS x N) to be right multiplied onto the product by consumer*simulant matrix (X x NS)
# ns_n = np.zeros((N_cons*N_sims,))

'''
fake parameters
'''
## beta is a vector consisting of
## constant utility,
## mean utility for characteristic
## agent-specific utility
## random taste for each product
beta = np.random.uniform(0,1,4)


'''
compute CCPs and total log likelihood across choices and consumers
--- numba decorators have to immediately precede function
--- numba does not accept matmul, only @
'''
## numba doesn't support axis calls (e.g. in np.sum)
## can't do matrix-vector multiplication to generate sums because numba does not accept 3D arrays for matrix multiplication
## see https://numba.readthedocs.io/en/0.51.2/reference/numpysupported.html



# @jit(nopython=True, parallel=True)
# @njit(parallel=True)
# @jit(parallel=True)
@jit(nopython=True)
def LL_jit(beta,W,X,S,choices):
    LL = 0
    ## loop over consumers and consumer characteristics
    for i in range(W.shape[0]):
        e_util_ijs = np.exp(beta[0] + (beta[1] + beta[2]*W[i] + beta[3]*S)*X)
        ## replace sum and mean with matmul by vectors of ones since numba does not accept axis calls
        # e_util_is_sumj = np.matmul(np.transpose(e_util_ijs),np.ones(e_util_ijs.shape[0])[:,None]).flatten()
        # e_util_is_sumj = (np.transpose(e_util_ijs)@np.ones(e_util_ijs.shape[0])[:,None]).flatten()
        ## make array contiguous (C (row) order) because transpose flips to column (Fortran) order
        e_util_is_sumj = (np.ascontiguousarray(np.transpose(e_util_ijs))@np.ones(e_util_ijs.shape[0])[:,None]).flatten()
        CCP_ijs = e_util_ijs/e_util_is_sumj[None,:]
        ## integrate-out simulants
        # CCP_ij = np.matmul(CCP_ijs,np.ones(CCP_ijs.shape[1])[:,None]) / CCP_ijs.shape[1]
        CCP_ij = (CCP_ijs@np.ones(CCP_ijs.shape[1])[:,None]) / CCP_ijs.shape[1]
        choice_ij = choices[:,i]
        loglike = -1*np.sum(np.log(choice_ij[:,None]*CCP_ij + (1-choice_ij[:,None])*(1-CCP_ij)))
        ## cum sum total LL
        LL += loglike
    return LL

## start timer
start = timeit.default_timer()


total_LL = LL_jit(beta,W,X,S,choices)

## timer stop and print runtime
stop = timeit.default_timer()
print('Time: ', (stop - start), 'seconds')
print(total_LL)

'''
Time to run:
Serial (uses only one core): 52 seconds
@jit(nopython=True) (still only one core, looks like maybe a tiny bit of two others):  49-50 seconds
@jit(nopython=True, parallel=True): gives error of sizes of S, X not matching
@njit(parallel=True): same error
@jit(parallel=True): same error
'''




