import numpy as np
import pandas as pd
from numpy import linalg as la
from multiprocessing import Pool
import timeit
import functools

'''
Method 3: multiprocess pool
'''

## to call in master process, run under:
## __name__ = '__main__' 
## otherwise calls will be duplicated under child processes
## https://docs.python.org/3/library/multiprocessing.html
## https://stackoverflow.com/questions/31858352/why-does-pool-run-the-entire-file-multiple-times


##############################################
## Call outputs of value function iteration ##
##############################################

## call results of value function iteration
CVFs = pd.read_csv(r'inputs/CVFs.csv')
V1_new = CVFs['V1'].values
V0_new = CVFs['V0'].values

##########################################
## Define functions for parallelization ##
##########################################

## parallelized log likelihood, computing CCPs within loop (with as many exponents as possible) to make it take as long as possible
## input is a tuple (at,it)
## output of p.map is list of log likelihoods, to be converted to an array
def par_ll(V1,V0,at_it):
    ## compute CCPs and likelihood
    a_t = at_it[0]
    i_t = at_it[1]
    CCP_i1 = np.exp(V1) / (np.exp(V0) + np.exp(V1))
    CCP_i0 = np.exp(V0) / (np.exp(V0) + np.exp(V1))
    log_like = -np.log((i_t * CCP_i1[int(a_t) - 1]) + ((1-i_t) * CCP_i0[int(a_t) - 1]))
    return log_like

## freeze V1, V0 into parallel function signature since p.map can only accept iterable input
## use functools to generate partial function
partial_par_ll = functools.partial(par_ll,V1_new,V0_new)

#########################################
## multiprocess likelihood calculation ##
#########################################

def rust_mp(dat_list,results,n_processes):
    ## method
    results['Method'] = 3
    # loop over data list (list of dfs of increasing expansion of original)
    for n_idx in range(len(dat_list)):
        ## run for current df, in parallel
        ## define tuple of (a_t,i_t) as iterable for pool
        at_it = tuple(map(tuple,dat_list[n_idx]))
        ## initiate (and close/join) multiprocess pool
        with Pool(n_processes) as p:
            ## start timer
            tic = timeit.default_timer()
            ll_mp = np.array(p.map(partial_par_ll,at_it))
            ## timer stop
            toc = timeit.default_timer()
        ll_tot = ll_mp.sum()
        print(ll_tot)
        ## seconds elapsed
        results.loc[n_idx,'Elapsed (s)'] = toc - tic
        ## number of observations
        results.loc[n_idx,'Size (N)'] = dat_list[n_idx].shape[0]
        # return results
        print(results)
    results.to_csv(f'outputs\\results3_p{n_processes}.csv',index=False)

if __name__ == '__main__':

    ################################################
    ## Call data generation (run within program)  ##
    ################################################

    with open("rust_datagen_py.py") as f:
        exec(f.read())

    #######################
    ## define results df ##
    #######################

    results_3 = pd.DataFrame(np.zeros((len(dat_list),3)),columns=['Method','Size (N)', 'Elapsed (s)'])

    ###########################################################
    ## call multiprocess likelihood computation for dat_list ##
    ###########################################################

    rust_mp(dat_list,results_3,n_processes=2)
    rust_mp(dat_list,results_3,n_processes=4)
    rust_mp(dat_list,results_3,n_processes=8)
    rust_mp(dat_list,results_3,n_processes=16)
