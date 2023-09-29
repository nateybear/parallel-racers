'''
Write certain numpy operations with and without threads
Only some numpy operations are threaded
Number of threads can be controlled with environ variable
See: https://superfastpython.com/multithreaded-numpy-functions/
Determine which operations slow down due to forced single-threading vs. not
Try single thread, regular numpy, 16 thread (one per core)

note: jit needs to decorate functions that use only numpy/base python behavior, e.g. does not accept default_timer
'''
# import thread environment controls
# from os import environ
# environ['OMP_NUM_THREADS'] = '1'
from numba import jit
import timeit
import numpy as np
import pandas as pd

## define test functions

@jit
def rng(n):
    a = np.random.rand(n)
    return a

@jit
def square1(a):
    x = a**2
    return x

@jit
def square2(a):
    x = np.square(a)
    return x

@jit
def square3(a):
    x = a*a
    return x

@jit
def exp(a):
    x = np.exp(a)
    return x

@jit
def cube1(a):
    x = a**3
    return x

@jit
def cube2(a):
    x = (a**2)*a
    return x

@jit
def cube3(a):
    x = np.square(a)*a
    return x

@jit
def cube4(a):
    x = a*a*a
    return x

@jit
def Q1(a):
    x = a**4
    return x

@jit
def Q2(a):
    b = a**2
    x = b**2
    return x

@jit
def Q3(a):
    x = np.square(a)*np.square(a)
    return x

@jit
def Q4(a):
    x = a*a*a*a
    return x

@jit
def atoa(a):
    x = a**a
    return x

@jit
def expcube1(a):
    x = np.exp(a**3)
    return x

@jit
def expcube2(a):
    x = np.exp((a**2)*a)
    return x

@jit
def expcube3(a):
    x = np.exp(np.square(a)*a)
    return x

@jit
def expcube4(a):
    x = np.exp(a*a*a)
    return x


## call (14) test functions, iter times, for n (1 bill) obs, and average over run times

def thread_test(iter,n):
    results_iter = np.zeros((18,iter))
    for t in range(iter):

        tic_0 = timeit.default_timer()
        a = rng(n)
        toc_0 = timeit.default_timer()
        results_iter[0,t] = toc_0 - tic_0

        tic_1 = timeit.default_timer()
        x1 = square1(a)
        toc_1 = timeit.default_timer()
        results_iter[1,t] = toc_1 - tic_1

        tic_2 = timeit.default_timer()
        x2 = square2(a)
        toc_2 = timeit.default_timer()
        results_iter[2,t] = toc_2 - tic_2 

        tic_3 = timeit.default_timer()
        x3 = square3(a)
        toc_3 = timeit.default_timer()
        results_iter[3,t] = toc_3 - tic_3             

        tic_4 = timeit.default_timer()
        x4 = exp(a)
        toc_4 = timeit.default_timer()
        results_iter[4,t] = toc_4 - tic_4    

        tic_5 = timeit.default_timer()
        x5 = cube1(a)
        toc_5 = timeit.default_timer()
        results_iter[5,t] = toc_5 - tic_5    

        tic_6 = timeit.default_timer()
        x6 = cube2(a)
        toc_6 = timeit.default_timer()
        results_iter[6,t] = toc_6 - tic_6    

        tic_7 = timeit.default_timer()
        x7 = cube3(a)
        toc_7 = timeit.default_timer()
        results_iter[7,t] = toc_7 - tic_7  

        tic_8 = timeit.default_timer()
        x8 = cube4(a)
        toc_8 = timeit.default_timer()
        results_iter[8,t] = toc_8 - tic_8    

        tic_9 = timeit.default_timer()
        x9 = Q1(a)
        toc_9 = timeit.default_timer()
        results_iter[9,t] = toc_9 - tic_9    

        tic_10 = timeit.default_timer()
        x10 = Q2(a)
        toc_10 = timeit.default_timer()
        results_iter[10,t] = toc_10 - tic_10    

        tic_11 = timeit.default_timer()
        x11 = Q3(a)
        toc_11 = timeit.default_timer()
        results_iter[11,t] = toc_11 - tic_11

        tic_12 = timeit.default_timer()
        x12 = Q4(a)
        toc_12 = timeit.default_timer()
        results_iter[12,t] = toc_12 - tic_12

        tic_13 = timeit.default_timer()
        x13 = atoa(a)
        toc_13 = timeit.default_timer()
        results_iter[13,t] = toc_13 - tic_13   

        tic_14 = timeit.default_timer()
        x14 = expcube1(a)
        toc_14 = timeit.default_timer()
        results_iter[14,t] = toc_14 - tic_14    

        tic_15 = timeit.default_timer()
        x15 = expcube2(a)
        toc_15 = timeit.default_timer()
        results_iter[15,t] = toc_15 - tic_15    

        tic_16 = timeit.default_timer()
        x16 = expcube3(a)
        toc_16 = timeit.default_timer()
        results_iter[16,t] = toc_16 - tic_16  

        tic_17 = timeit.default_timer()
        x17 = expcube4(a)
        toc_17 = timeit.default_timer()
        results_iter[17,t] = toc_17 - tic_17   


    results_df = pd.DataFrame(['rng','square1','square2','square3','exp','cube1','cube2','cube3','cube4','Q1','Q2','Q3','Q4','atoa','expcube1','expcube2','expcube3','expcube4'],columns=['test'])
    results_df['jit_multithread_default_s'] = np.mean(results_iter,axis=1)

    results_df.to_csv(r'interim/results_multi_jit.csv',index=False)
    
    return results_df

results_df = thread_test(iter=10,n=1000000000)
# results_df = thread_test(iter=10,n=100)


