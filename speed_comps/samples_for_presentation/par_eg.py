'''
Simple parallel computing example, passing objects to parallel function
Example of INCORRECT WAY TO PASS RNG IN PARALLEL
Correct way is with SeedSequence
'''

import numpy as np, functools
from multiprocessing import Pool

def par_func(rng,a,i):
    return np.mean(rng.normal(size=a))*i

def outer(n_procs):
    rng = np.random.default_rng(1)    # seed RNG
    a = 4
    with Pool(n_procs) as p:
        p.map(functools.partial(par_func,rng,a),np.arange(1,100).tolist())  # embed fixed inputs into function signature for map, generate list iterable

if __name__ == '__main__':
    outer(4)
