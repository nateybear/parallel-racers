import sys
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

##################################################################
## Call outputs of data generation and value function iteration ##
##################################################################
## note: need to call data generation and value function iteration outside and import the results, since the process pool runs all prior scripts on main
## then use subprocess in res_stack program to run all programs independently (vs exec which runs program within the script)
## avoid doing non-pool operations in mp program
## instead, run program separately and call into this one because the multiprocessing is re-calling the data generation and VFI every time (when using exec) and boosting the time counters by about 35 seconds (which is roughly the data generation time)
## see stack overflow re: entire file being called multiple times, need to place function calls in if __name__ = '__main__' to only run once
## https://stackoverflow.com/questions/31858352/why-does-pool-run-the-entire-file-multiple-times
## might need to break this into multiple programs????

# with open("rust_data_DP_py.py") as f:
#     exec(f.read())

## put open above in if name = main and reset rust_DP to output the value functions and dat_list, also remove 10^8, it seemed like there was gains before that, now not seeing anything remotely like gains


def read_data():
    # ## read in conditional (choice specific) value functions
    # CVFs = pd.read_csv(r'inputs/CVFs.csv')
    # V1_new = CVFs['V1'].values
    # V0_new = CVFs['V0'].values

    ## read in data and put in dat_list

    dat_3 = pd.read_csv(r'inputs/df_order3.csv').values
    dat_4 = pd.read_csv(r'inputs/df_order4.csv').values
    dat_5 = pd.read_csv(r'inputs/df_order5.csv').values
    dat_6 = pd.read_csv(r'inputs/df_order6.csv').values
    dat_7 = pd.read_csv(r'inputs/df_order7.csv').values
    # dat_8 = pd.read_csv(r'inputs/df_order8.csv').values

    # dat_list = [dat_3,dat_4,dat_5,dat_6,dat_7,dat_8]
    dat_list = [dat_3,dat_4,dat_5,dat_6,dat_7]

    return dat_list

##########################################
## Define functions for parallelization ##
##########################################

## parallelized log likelihood, computing CCPs within loop to make it take as long as possible
def par_ll(at_it):
    ## read in conditional (choice specific) value functions
    CVFs = pd.read_csv(r'inputs/CVFs.csv')
    V1_new = CVFs['V1'].values
    V0_new = CVFs['V0'].values

    ## compute CCPs and likelihood
    a_t = at_it[0]
    i_t = at_it[1]
    CCP_i1 = np.exp(V1_new) / (np.exp(V0_new) + np.exp(V1_new))
    CCP_i0 = np.exp(V0_new) / (np.exp(V0_new) + np.exp(V1_new))
    log_like = -np.log((i_t * CCP_i1[int(a_t) - 1]) + ((1-i_t) * CCP_i0[int(a_t) - 1]))
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



def main(argv):
    

    
#     # V1_new, V0_new, dat_list = read_data()

    dat_list = read_data()

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


if __name__ == '__main__':
    main(sys.argv)


    # dat_list = read_data()

    # #######################
    # ## define results df ##
    # #######################

    # results_3 = pd.DataFrame(np.zeros((len(dat_list),3)),columns=['Method','Size (N)', 'Elapsed (s)'])

    # ###########################################################
    # ## call multiprocess likelihood computation for dat_list ##
    # ###########################################################

    # rust_mp(dat_list,results_3,n_processes=2)
    # rust_mp(dat_list,results_3,n_processes=4)
    # rust_mp(dat_list,results_3,n_processes=8)
    # rust_mp(dat_list,results_3,n_processes=16)

    ##################
    ## call in data ##
    ##################

    # V1_new, V0_new, dat_list = read_data()

    # #######################
    # ## define results df ##
    # #######################

    # results_3 = pd.DataFrame(np.zeros((len(dat_list),3)),columns=['Method','Size (N)', 'Elapsed (s)'])

    # ###########################################################
    # ## call multiprocess likelihood computation for dat_list ##
    # ###########################################################

    # rust_mp(dat_list,results_3,n_processes=2)
    # rust_mp(dat_list,results_3,n_processes=4)
    # rust_mp(dat_list,results_3,n_processes=8)
    # rust_mp(dat_list,results_3,n_processes=16)