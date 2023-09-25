'''
Write certain numpy operations with and without threads
Only some numpy operations are threaded
Number of threads can be controlled with environ variable
See: https://superfastpython.com/multithreaded-numpy-functions/
Determine which operations slow down due to forced single-threading vs. not
Try single thread, regular numpy, 16 thread (one per core)
'''
# import thread environment controls
from os import environ
environ['OMP_NUM_THREADS'] = '1'

import timeit
import numpy as np
import pandas as pd

## define test functions

def rng(n):
    tic = timeit.default_timer()
    a = np.random.rand(n)
    toc = timeit.default_timer()
    sec = toc - tic
    return a, sec

def square_py(a):
    tic = timeit.default_timer()
    x = a**2
    toc = timeit.default_timer()
    sec = toc - tic
    return sec

def square_np(a):
    tic = timeit.default_timer()
    x = np.square(a)
    toc = timeit.default_timer()
    sec = toc - tic
    return sec

def exp(a):
    tic = timeit.default_timer()
    x = np.exp(a)
    toc = timeit.default_timer()
    sec = toc - tic
    return sec

def cube1(a):
    tic = timeit.default_timer()
    x = a**3
    toc = timeit.default_timer()
    sec = toc - tic
    return sec

def cube2(a):
    tic = timeit.default_timer()
    x = (a**2)*a
    toc = timeit.default_timer()
    sec = toc - tic
    return sec

def cube3(a):
    tic = timeit.default_timer()
    x = np.square(a)*a
    toc = timeit.default_timer()
    sec = toc - tic
    return sec

def Q1(a):
    tic = timeit.default_timer()
    x = a**4
    toc = timeit.default_timer()
    sec = toc - tic
    return sec

def Q2(a):
    tic = timeit.default_timer()
    b = a**2
    x = b**2
    toc = timeit.default_timer()
    sec = toc - tic
    return sec

def Q3(a):
    tic = timeit.default_timer()
    x = np.square(a)*np.square(a)
    toc = timeit.default_timer()
    sec = toc - tic
    return sec

def atoa(a):
    tic = timeit.default_timer()
    x = a**a
    toc = timeit.default_timer()
    sec = toc - tic
    return sec

def expcube1(a):
    tic = timeit.default_timer()
    x = np.exp(a**3)
    toc = timeit.default_timer()
    sec = toc - tic
    return sec

def expcube2(a):
    tic = timeit.default_timer()
    x = np.exp((a**2)*a)
    toc = timeit.default_timer()
    sec = toc - tic
    return sec

def expcube3(a):
    tic = timeit.default_timer()
    x = np.exp(np.square(a)*a)
    toc = timeit.default_timer()
    sec = toc - tic
    return sec


## call (14) test functions, iter times, for n (1 bill) obs, and average over run times

def thread_test(iter,n):
    results_iter = np.zeros((14,iter))
    for t in range(iter):
        a, results_iter[0,t] = rng(n)
        results_iter[1,t] = square_py(a)
        results_iter[2,t] = square_np(a)
        results_iter[3,t] = exp(a)
        results_iter[4,t] = cube1(a)
        results_iter[5,t] = cube2(a)
        results_iter[6,t] = cube3(a)
        results_iter[7,t] = Q1(a)
        results_iter[8,t] = Q2(a)
        results_iter[9,t] = Q3(a)
        results_iter[10,t] = atoa(a)
        results_iter[11,t] = expcube1(a)
        results_iter[12,t] = expcube2(a)
        results_iter[13,t] = expcube3(a)

    results_df = pd.DataFrame(['rng','square_py','square_np','exp','cube1','cube2','cube3','Q1','Q2','Q3','atoa','expcube1','expcube2','expcube3'],columns=['test'])
    results_df['singlethread_s'] = np.mean(results_iter,axis=1)

    results_df.to_csv(r'interim/results_single.csv',index=False)
    
    return results_df

results_df = thread_test(iter=10,n=1000000000)
