Multinomial choice implementation - python

Compute conditional choice probabilities and total log-likelihood given guess of parameters.

py_mnc_base.py: base implementation in numpy, using broadcast operations over arrays
py_mnc_jax.py: jit-compiled version of 'py_mnc_base.py'
py_mnc_numba.py: reduce to two-dimensional problem by looping over consumers, to circumvent numba limitations. jit compile entire loop.


