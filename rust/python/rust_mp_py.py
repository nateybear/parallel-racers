'''
Call packages, data generation and value function iteration (assuming "rust_data_DP_py.py" is in the same folder)
'''

with open("rust_data_DP_py.py") as f:
    exec(f.read())


'''
Method 3: multiprocess,Results DFs and function
'''

## parallelized log likelihood
def par_ll(at_it):
    a_t = at_it[0]
    i_t = at_it[1]
    log_like = -np.log((i_t * CCP[int(a_t) - 1]) + ((1-i_t) * (1-CCP[int(a_t) - 1])))
    return log_like

results_3 = pd.DataFrame(np.zeros((len(dat_list),3)),columns=['Method','Size (N)', 'Elapsed (s)'])

n_processes = 2

## method
results_3['Method'] = 3

if __name__ == '__main__':
    ## loop over data list
    for n_idx in range(len(dat_list)):
        ## run for current df, in parallel
        ## define tuple of (a_t,i_t) as iterable for pool
        at_it = tuple(map(tuple,dat_list[n_idx]))
        ## start timer
        tic = timeit.default_timer()
        with Pool(n_processes) as p:
            ll_mp = np.array(p.map(par_ll,at_it))
        ll_tot = ll_mp.sum()
        print(ll_tot)
        ## timer stop and print runtime
        toc = timeit.default_timer()
        ## seconds elapsed
        results_3.loc[n_idx,'Elapsed (s)'] = toc - tic
        ## number of observations
        results_3.loc[n_idx,'Size (N)'] = dat_list[n_idx].shape[0]
        # return results
        print(results_3)
    results_3.to_csv(r'outputs/results3.csv',index=False)

# if __name__ == '__main__':
#     ## loop over data list
#     for n_idx in range(len(dat_list)):
#         ## run for current df, in parallel
#         ## define tuple of (a_t,i_t) as iterable for pool
#         at_it = tuple(map(tuple,dat_list[n_idx]))
#         with Pool(n_processes) as p:
#             ## start timer
#             tic = timeit.default_timer()
#             ll_mp = np.array(p.map(par_ll,at_it))
#             ll_tot = ll_mp.sum()
#             print(ll_tot)
#             ## timer stop and print runtime
#             toc = timeit.default_timer()
#             ## seconds elapsed
#             results_3.loc[n_idx,'Elapsed (s)'] = toc - tic
#             ## number of observations
#             results_3.loc[n_idx,'Size (N)'] = dat_list[n_idx].shape[0]
#             # return results
#             print(results_3)
#     results_3.to_csv(r'outputs/results3.csv',index=False)

### below is one iteration of the above, just for 10^7 obs, to compare speed. 10^7 takes about 104 seconds with two processes vs. in the for loop about 90 seconds--comparable results for 10^6 also ###

# ## run for current df, in parallel
# ## define tuple of (a_t,i_t) as iterable for pool
# at_it = tuple(map(tuple,dat_list[4]))

# ## start timer
# tic = timeit.default_timer()

# if __name__ == '__main__':
#     with Pool(n_processes) as p:
#         ll_mp = np.array(p.map(par_ll,at_it))
#     ll_tot = ll_mp.sum()
#     print(ll_tot)

# ## timer stop and print runtime
# toc = timeit.default_timer()

# ## seconds elapsed
# results_3.loc[4,'Elapsed (s)'] = toc - tic

# ## number of observations
# results_3.loc[4,'Size (N)'] = dat_list[4].shape[0]

# print(results_3)

