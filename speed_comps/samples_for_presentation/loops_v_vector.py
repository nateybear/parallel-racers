'''
Loops vs. implicit parallelization; vectorization + broadcasting
'''

import numpy as np
import timeit


#### Examples of loops vs. vectorization/implicit parallelization ####
## square

x = np.random.normal(0,1,10**7)
## start timer
tic_0 = timeit.default_timer()
y = x**2
## timer stop and print runtime
toc_0 = timeit.default_timer()
print('Runtime: ', (toc_0 - tic_0), f'seconds for estimation')

## start timer
tic_alt0 = timeit.default_timer()
y = np.square(x)
## timer stop and print runtime
toc_alt0 = timeit.default_timer()
print('Runtime: ', (toc_alt0 - tic_alt0), f'seconds for estimation')

## start timer
tic_1 = timeit.default_timer()
y = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    y[i] = x[i]**2
toc_1 = timeit.default_timer()
print('Runtime: ', (toc_1 - tic_1), f'seconds for estimation')

## sum

## start timer
tic_2 = timeit.default_timer()
y = np.sum(x)
## timer stop and print runtime
toc_2 = timeit.default_timer()
print('Runtime: ', (toc_2 - tic_2), f'seconds for estimation')

## start timer
tic_3 = timeit.default_timer()
y = 0
for i in range(x.shape[0]):
    y += x[i]
toc_3 = timeit.default_timer()
print('Runtime: ', (toc_3 - tic_3), f'seconds for estimation')


### BROADCASTING EXAMPLE ###
# X = np.ones((10,10**7))*2.0
# Y = np.random.normal(0,1,10)

X = np.random.uniform(1,2,size=(10,10**7))
Y = np.random.uniform(2,10,size=10)

Z = Y**X  ## ValueError: operands could not be broadcast together with shapes (10,) (10,10000000)

## start timer
tic_4 = timeit.default_timer()
Z = Y[:,None]**X
toc_4 = timeit.default_timer()
print('Runtime: ', (toc_4 - tic_4), f'seconds for estimation')

## start timer
tic_5 = timeit.default_timer()
Z = np.zeros((X.shape))
for i in range(X.shape[1]):
    Z[:,i] = Y**X[:,i]
toc_5 = timeit.default_timer()
print('Runtime: ', (toc_5 - tic_5), f'seconds for estimation')
