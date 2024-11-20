Multinomial choice implementation - python

Compute conditional choice probabilities and total log-likelihood given guess of parameters.

py_mnc_base.py: base implementation in numpy, using broadcast operations over arrays
py_mnc_jax.py: jax.jit-compiled version of 'py_mnc_base.py'
py_mnc_numba.py: reduce to two-dimensional problem by looping over consumers, to circumvent numba limitations. jit compile entire loop. This doesn't work very well due to the limitations (see notes in script and numba documentation, if interested).
py_mnc_mp.py: multiprocess pool example, again looping over consumers in two-dimensions (product x simulants), but distributing this iteration across 20 subprocesses.
py_mnc_serial.py: serial implementation without jit or multiprocessing, looping over consumers one-by-one.

The ``base'' (3 dimensional broadcast) version takes about 30 seconds to run, while the jax jit implementation (also in 3 dimensions) takes about 4.6 seconds
The serial implementation in two-dimensions (looping over consumers, the third dimension) is actually faster (takes about 12 seconds) than the broadcast version in three-dimensions, but not as fast as the jitted three-dimension version.
The multiprocess pool implementation in two-dimensions takes about 1.1 seconds (20 cores).
Numba does not work well for this implementation (though a more clever workaround may be possible), taking about 50 seconds.

Note that methods like jax.jit and multiprocess.pool can be combined if needed.

