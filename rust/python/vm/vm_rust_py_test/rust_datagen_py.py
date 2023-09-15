import numpy as np
import pandas as pd
import timeit

################################################################################
### Call this program to solve DP and compute CCPs given guess of parameters ###
################################################################################

## set decimal float

float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


### import data (assumes its in the same folder) ###
dat = np.loadtxt("data.asc")

## generate stacked data, duplicated 10^exp times
## we want from 1,000 to 100M observations
## start timer for datagen
start_t0 = timeit.default_timer()

def datagen(df,n_sets):
    dat_list = [0]*n_sets
    ## call 6 to get exp of 5 which will iterate 6 times, producing 1000-100M obs
    for exp in range(n_sets):
        # df_big = pd.DataFrame(np.block([[df]]*(10**exp)),columns=['a_t','i_t'])
        # df_big.to_csv(f'inputs\\df_order{exp+3}.csv',index=False)

        df_big = np.block([[df]]*(10**exp))
        dat_list[exp] = df_big

    return dat_list

dat_list = datagen(dat,n_sets=6)

## timer stop and print runtime
stop_t0 = timeit.default_timer()
print('Time: ', stop_t0 - start_t0, 'seconds for data generation')
