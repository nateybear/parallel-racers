'''
Call packages, data generation and value function iteration (assuming "rust_data_DP_py.py" is in the same folder)
'''

with open("rust_data_DP_py.py") as f:
    exec(f.read())

'''
Results DFs and function
'''

results_temp = pd.DataFrame(np.zeros((len(dat_list),3)),columns=['Method','Size (N)', 'Elapsed (s)'])

def results_1(results):
    ## method
    results['Method'] = 1
    for n_idx in range(len(dat_list)):
        ## start timer
        tic = timeit.default_timer()
        ## run for current df
        ll_tot = vect_ll(dat_list[n_idx]).sum()
        ## timer stop and print runtime
        toc = timeit.default_timer()
        ## seconds elapsed
        results.loc[n_idx,'Elapsed (s)'] = toc - tic
        ## number of observations
        results.loc[n_idx,'Size (N)'] = dat_list[n_idx].shape[0]
    return results

def results_2(results):
    ## method
    results['Method'] = 2
    for n_idx in range(len(dat_list)):
        ## start timer
        tic = timeit.default_timer()
        ## run for current df
        ll_tot = serial_ll(dat_list[n_idx]).sum()
        ## timer stop and print runtime
        toc = timeit.default_timer()
        ## seconds elapsed
        results.loc[n_idx,'Elapsed (s)'] = toc - tic
        ## number of observations
        results.loc[n_idx,'Size (N)'] = dat_list[n_idx].shape[0]
    return results


'''
Method 1: vectorized application to compute LL
'''
def vect_ll(df):
    a_t = df[:,0]
    i_t = df[:,1]
    log_like = np.zeros(a_t.shape[0])
    for t in range(a.shape[0]):
        log_like[a_t==(t+1)] = -np.log((i_t[a_t==(t+1)] * CCP[t]) + ((1-i_t[a_t==(t+1)]) * (1-CCP[t])))
    return log_like
        
results_1 = results_1(results_temp.copy())
print(results_1)
results_1.to_csv(r'outputs/results1.csv',index=False)

'''
Method 2: serial for-loop
'''
## TRY NUMBA OR JNP for jit

def serial_ll(df):
    log_like = np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        a_t = df[i,0]
        i_t = df[i,1]
        log_like[i] = -np.log((i_t * CCP[int(a_t) - 1]) + ((1-i_t) * (1-CCP[int(a_t) - 1])))
    return log_like

results_2 = results_2(results_temp.copy())
print(results_2)
results_2.to_csv(r'outputs/results2.csv',index=False)