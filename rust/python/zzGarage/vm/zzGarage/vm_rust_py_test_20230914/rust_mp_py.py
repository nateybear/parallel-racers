import numpy as np
import pandas as pd
import math
from datetime import datetime
from numpy import linalg as la
from multiprocessing import Pool
import timeit


'''
Method 3: multiprocess pool
'''

#############################################################################################################
## Call data generation and value function iteration (assuming "rust_data_DP_py.py" is in the same folder) ##
#############################################################################################################

with open("rust_data_DP_py.py") as f:
    exec(f.read())
    
##########################################
## Define functions for parallelization ##
##########################################

## parallelized log likelihood
def par_ll(at_it):
    a_t = at_it[0]
    i_t = at_it[1]
    log_like = -np.log((i_t * CCP[int(a_t) - 1]) + ((1-i_t) * (1-CCP[int(a_t) - 1])))
    return log_like

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
            ll_mp = np.array(p.map(par_ll,at_it))
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