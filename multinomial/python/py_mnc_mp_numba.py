import numpy as np
import timeit
from numba import jit
from numba import njit
from multiprocessing import Pool
import functools

'''
Initial multinomial choice problem speed test - multiprocess pool (loop over consumer dimension)
WITH NUMBA JIT COMPILATION
Define functions first, run operations conditional on __name__ == '__main__':
'''

def data():
    ## 100,000 consumers (each with scalar characteristic W)
    N_cons = 100000
    ## 10 choices (each with scalar product characteristic X)
    N_choices = 10
    ## 1000 sim draws (S)
    N_sims = 1000


    ## Then for some arbitrary fixed parameters beta_0, beta_1, beta_2, 
    ## mean utility for the ith consumer for the jth product for the sth simulation draw is:
    ## U_{ijs} = beta_0 + (beta_1+beta_2*W[i]+ beta_3*S[s])*X[j]

    '''
    generate data
    '''
    ## set random seed
    np.random.seed(1)
    ## consumer chars (100k) 
    W = np.random.uniform(0,1,N_cons)
    ## product chars (10)  (in a real implementation, these should be sorted by product IDs 1-10, so they correspond with correct rows in the choices matrix)
    X = np.random.uniform(0,1,N_choices)[:,None]
    ## random utility (1000 draws)
    S = np.random.normal(0,1,N_sims)[None,:]
    ## fake choices Y, corresponding to product IDs (for the 100k consumers, no explicit outside option for now)
    ## move up by one integer so as not to divide by zero in broadcasting below to compute likelihood for each consumer over choices
    Y = np.random.randint(0,10,N_cons) + 1
    ## sorted product_IDs
    prod_IDs = np.arange(10).astype(int) + 1
    ## auxiliary matrix
    ## divide broadcast individual choices over the product IDs to get a 1 for each consumer's actual choice
    ## in a matrix of n_cons by n_prods (to match dims of vstack in p.map below)
    aux = Y[:,None] / prod_IDs[None,:]
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

    ## send consumer array to list iterable for p.map
    W_list = W.tolist()

    return W_list, X, S, choices, beta

'''
compute CCPs and total log likelihood across choices and consumers
define iterable function for parallelized likelihood calculation
'''

## jit decorator: (njit is equivalent to nopython=True)
@njit()
def comp_CCP(X,S,beta,W_list):
    e_util_ijs = np.exp(beta[0] + (beta[1] + beta[2]*W_list + beta[3]*S)*X)
    ## generate CCP for each consumer and choice (and sim draw)
    CCP_ijs = e_util_ijs/np.sum(e_util_ijs,axis=0)
    ## integrate-out simulants, without using np.mean
    ## numba only supports axis calls with np.sum
    CCP_ij = np.sum(CCP_ijs/CCP_ijs.shape[1],axis=1)
    ## outputs length N_choices vector of CCPs for consumer i
    return CCP_ij


def total_LL(n_processes):

    W_list, X, S, choices, beta = data()

    ### freeze comp_CCP inputs (except iterable) into function signature as positional arguments, to generate partial function
    partial_comp_CCP = functools.partial(comp_CCP,X,S,beta)

    with Pool(n_processes) as p:
        ## start timer
        start = timeit.default_timer()

        ## p.map takes a list of consumer characteristics as iterable and generates a list of the outputs over iteration
        CCP_ij = np.vstack(p.map(partial_comp_CCP,W_list))
        total_LL = np.sum(choices*np.log(CCP_ij) + (1-choices)*np.log(1-CCP_ij))

        ## timer stop and print runtime
        stop = timeit.default_timer()
        print('Time: ', (stop - start), 'seconds')

    return total_LL


if __name__ == '__main__':
    ## allocate number of cores (out of the 40 available)
    n_processes = 20
    tot_LL = total_LL(n_processes)
    print(tot_LL)


'''
Takes about 4.6 seconds on 20 cores
extra time is due to numba overhead, since the calculation of each individual's contribution to the likelihood is really fast
'''
