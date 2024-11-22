'''
Test efficiency of np.exp in vectorization
relative to manually 'vectorized' math.exp
'''

import numpy as np
import timeit
import functools
import math

### take a bunch of uniform draws and np.exp the entire vector
## vs. map the math.exp to the vector (manual vectorization)
## vs. explicit for loop
## time 10 runs of each and take mean
## measure in milliseconds (thousandths 10^-3 of a second)

rng = np.random.default_rng(1)
draws = rng.uniform(0,1,10**6)
iter = 100

## numpy vectorized
def npexp(draws,iter):
    ## start timer
    tic = timeit.default_timer()
    obj = np.exp(draws)
    ## timer stop and print runtime
    toc = timeit.default_timer()
    # print('Runtime: ', (toc - tic), f'milliseconds for estimation')
    return np.array(toc-tic)*(10**3)

## using math.exp mapped
def pyexp_map(draws,iter):
    ## start timer
    tic = timeit.default_timer()
    obj = map(math.exp,draws)
    ## timer stop and print runtime
    toc = timeit.default_timer()
    # print('Runtime: ', (toc - tic), f'milliseconds for estimation')
    return np.array(toc-tic)*(10**3)

## math.exp with loop
def pyexp_for(draws,iter):
    obj = np.zeros(draws.shape[0])
    ## start timer
    tic = timeit.default_timer()
    for i in range(draws.shape[0]):
        obj[i] = math.exp(draws[i])
    ## timer stop and print runtime
    toc = timeit.default_timer()
    # print('Runtime: ', (toc - tic), f'milliseconds for estimation')
    return np.array(toc-tic)*(10**3)

## comparison
def comp(draws,iter):
    npexp_partial = functools.partial(npexp,draws)
    np_times = np.hstack(list(map(npexp_partial,range(iter))))
    mean_np_time = np.mean(np_times)

    pyexp_map_partial = functools.partial(pyexp_map,draws)
    py_map_times = np.hstack(list(map(pyexp_map_partial,range(iter))))
    mean_py_map_time = np.mean(py_map_times)

    pyexp_for_partial = functools.partial(pyexp_for,draws)
    py_for_times = np.hstack(list(map(pyexp_for_partial,range(iter))))
    mean_py_for_time = np.mean(py_for_times)

    print(f'np.exp time over 100 runs, 10^6 draws: {mean_np_time} milliseconds')
    print(f'mapped math.exp time over 100 runs, 10^6 draws: {mean_py_map_time} milliseconds')
    print(f'for-loop math.exp time over 100 runs, 10^6 draws: {mean_py_for_time} milliseconds')

    return mean_np_time, mean_py_map_time, mean_py_for_time

mean_np_time, mean_py_map_time, mean_py_for_time = comp(draws,iter)
