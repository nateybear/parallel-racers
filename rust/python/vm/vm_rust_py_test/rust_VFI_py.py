import numpy as np
import pandas as pd
import math
from datetime import datetime
from numpy import linalg as la
import timeit

###############################################################################################################
### Call this program to solve DP and compute choice-specific value functions for given guess of parameters ###
###############################################################################################################

## set decimal float

float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

'''
set parameters
'''
beta = 0.9
theta_1 = -1.1
R = -4.4
theta = np.array([theta_1,R])
a = np.arange(1,6)
a_prime = np.arange(1,6)

## transition matrices
T_0 = np.zeros((a.shape[0],a.shape[0]))
T_1 = np.zeros((a.shape[0],a.shape[0]))

T_0[1,0] = 1
T_0[2,1] = 1
T_0[3,2] = 1
T_0[4,3] = 1
T_0[4,4] = 1

T_1[0] = 1

## set random seed
np.random.seed(1)

def solveDP(a,a_prime,theta):
    # initialize value functions
    u_0t = theta[0]*a
    u_1t = theta[1]


    # initialize conditional continuation value functions
    V0_new = np.zeros(a_prime.shape[0])
    V1_new = np.zeros(a_prime.shape[0])

    tol = 100
    iter = 0

    '''
    VFI
    '''
    print("Started Value Function Iteration {}".format(datetime.now()))
    while (tol > 1e-6):
        ## calc new conditional value function
        V0_old = V0_new
        V1_old = V1_new

        V0_new = u_0t + beta*(0.5772 + np.log(np.exp(V0_old@T_0) + np.exp(V1_old@T_0)))
        V1_new = u_1t + beta*(0.5772 + np.log(np.exp(V0_old@T_1) + np.exp(V1_old@T_1)))

        ## calc tolerance using sup norm

        tol_0 = np.max(np.abs(V0_new - V0_old))
        tol_1 = np.max(np.abs(V1_new - V1_old))

        if tol_0 > tol_1:
            tol = tol_0
        else:
            tol = tol_1

        ## print output every 50 iterations
        iter += 1
        if iter % 50 == 0:
            print("Iteration {} --- Sup Norm {}".format(iter, tol))
    print("Finished Value Function Iteration {}".format(datetime.now()))

    return V0_new + np.random.uniform(0,1,size=a_prime.shape[0]), V1_new + np.random.uniform(0,1,size=a_prime.shape[0])

V0_new, V1_new = solveDP(a,a_prime,theta)

## write to csv (cannot execute within multiprocess program because of the execution within pool)
cond_val_fns = pd.DataFrame(np.hstack((V0_new[:,None], V1_new[:,None])), columns=['V0','V1'])

cond_val_fns.to_csv('inputs\CVFs.csv',index=False)
