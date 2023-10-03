import numpy as np
from numba import jit
import timeit

'''
Initial multinomial choice problem speed test - using jit compilation through numba
numba does not accept axis calls e.g. np.mean(X,axis=1), and does not accept matrix multiplication in dim > 2
so cannot replace np.count or sum to replace mean by mat-multiplying that dim by a vector of 1's
only option is to collapse consumers and choices into a single vector s.t. we then have 2D tensor with sim draws
'''
## 100,000 consumers (each with scalar characteristic W)
N_cons = 1000
## 10 choices (each with scalar product characteristic X)
N_choices = 10
## 1000 sim draws (S)
N_sims = 100


## Then for some arbitrary fixed parameters beta_0, beta_1, beta_2, 
## mean utility for the ith consumer for the jth product for the sth simulation draw is:
## U_{ijs} = beta_0 + (beta_1+beta_2*W[i]+ S[s])*X[j]

'''
generate data
'''
## consumer chars (100k) 
W = np.random.uniform(0,1,N_cons)[:,None]
## product chars (10) (in a real implementation, these should be sorted by product IDs 1-10, so they correspond with correct rows in the choices matrix)
X = np.random.uniform(0,1,N_choices)[None,:]
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
ns_n = np.zeros((N_cons*N_sims,))

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
'''

# ## the following does not accept @jit decorator because numba does not allow axis calls
# def LL(beta,W,X,S,choices):
#     e_util_ijs = np.exp(beta[0] + (beta[1] + beta[2]*W + beta[3]*S)*X)
#     ## generate CCP for each consumer and choice (and sim draw)
#     CCP_ijs = e_util_ijs/np.sum(e_util_ijs,axis=0)
#     ## integrate-out simulants
#     CCP_ij = np.mean(CCP_ijs,axis=2)
#     return -1*np.sum(np.log(choices*CCP_ij + (1-choices)*(1-CCP_ij)))

# ## start timer
# start = timeit.default_timer()

# total_LL = LL(beta,W,X,S,choices)

# ## timer stop and print runtime
# stop = timeit.default_timer()

# print('Time: ', (stop - start), 'seconds')
# print(total_LL)


## modified version to accept @jit
## nevermind, can't do matrix-vector multiplication to generate sums because numba does not accept 3D arrays for matrix multiplication
## see https://numba.readthedocs.io/en/0.51.2/reference/numpysupported.html
## need to collapse to 2D array (collapse consumers and product choices dimensions into one)
## numba does not accept matmul, only @
## supports flatten but only order C (row major)

## try jax.jit -- seems to support

# @jit(nopython=True)
# @jit(parallel=True)
# @jit(nopython=True)
def LL_jit(beta,W,X,S,choices):
    ## to allow for jit compilation through numba, add the W and S components then flatten, and combine with the X component
    ## use vector which is 1 for each consumer and 0 for everyone else to get mean over simulants
    inner = beta[1] + beta[2]*W + beta[3]*S
    # e_util_ijs = np.exp(beta[0] + (inner)*X)
    # ## replace sum of exponentials with matrix multiplication on dimension X by vector of 1s of length X
    # CCP_ijs = e_util_ijs/np.sum(e_util_ijs,axis=0)
    # e_util_sum_axis_pre = np.transpose(e_util_ijs)
    # e_util_sum_axis_0 = np.matmul(np.transpose(e_util_ijs),np.ones((e_util_ijs.shape[0],1)))
    
    # test = np.moveaxis(e_util_ijs,0,2)
    # CCP_ij = np.mean(CCP_ijs,axis=2)
    # return -1*np.sum(np.log(choices*CCP_ij + (1-choices)*(1-CCP_ij)))
#     return e_util_ijs, CCP_ijs,e_util_sum_axis_pre, e_util_sum_axis_0
    return inner

# e_util_ijs, CCP_ijs,e_util_sum_axis_pre,e_util_sum_axis_0 = LL_jit(beta,W,X,S,choices)
inner = LL_jit(beta,W,X,S,choices)



A = np.ones((100000,100))
B = np.ones((1000,100))
F = np.ones((10,100,1000))

@jit(nopython=True)
def test_num(A,B,F):
    # C = A@B
    # D = np.matmul(A,B)
    C = F@B
    return C

C = test_num(A,B,F)

Q = F.flatten()


