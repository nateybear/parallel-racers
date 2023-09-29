import numpy as np
import pandas as pd
from numba import jit
import timeit

'''
Initial multinomial choice problem speed test
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
W = np.random.uniform(0,1,N_cons)[None,:,None]
## product chars (10) (assume sorted by product IDs 1-10)
X = np.random.uniform(0,1,N_choices)[:,None,None]
## random utility (1000 draws)
S = np.random.uniform(0,1,N_sims)[None,None,:]
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


'''
fake parameters
'''
## beta is a vector consisting of
## constant utility,
## mean utility for characteristic
## agent-specific utility
## random taste for each product
beta = np.random.uniform(0,1,4)

### compute CCPs and total log likelihood across choices and consumers ###

# @jit
def LL(beta,W,X,S,choices):
    e_util_ijs = np.exp(beta[0] + (beta[1] + beta[2]*W + beta[3]*S)*X)
    CCP_ijs = e_util_ijs/np.sum(e_util_ijs,axis=0)
    CCP_ij = np.mean(CCP_ijs,axis=2)
    return -1*np.sum(np.log(choices*CCP_ij + (1-choices)*(1-CCP_ij)))



## start timer
start = timeit.default_timer()

total_LL = LL(beta,W,X,S,choices)

## timer stop and print runtime
stop = timeit.default_timer()

print('Time: ', (stop - start), 'seconds')
print(total_LL)

    


