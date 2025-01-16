import numpy as np
import timeit
import jax
import jax.numpy as jnp

## jax defaults to run on GPU, can set the following to coerce CPU compilation
jax.config.update('jax_platform_name', 'cpu')
## other ways:
## https://github.com/jax-ml/jax/discussions/10399
## https://stackoverflow.com/questions/74537026/execute-function-specifically-on-cpu-in-jax

'''
Initial multinomial choice problem speed test - JIT compilation
'''
## 100,000 consumers (each with scalar characteristic W)
N_cons = 100000
## 10 choices (each with scalar product characteristic X)
N_choices = 10
## 1000 sim draws (S)
N_sims = 1000


## Then for some arbitrary fixed parameters beta_0, beta_1, beta_2, 
## mean utility for the ith consumer for the jth product for the sth simulation draw is:
## U_{ijs} = beta_0 + (beta_1+beta_2*W[i]+ beta_3*S[s])*X[j]

'''
generate data
'''
## set random seed
## if necessary, use jax's methods for pseudo-RNG that generate RNs that are reproducible, parallelizable, vectorisable
## https://jax.readthedocs.io/en/latest/random-numbers.html
## https://jax.readthedocs.io/en/latest/jax.random.html#module-jax.random
## mirrors (and may have advantages over) numpy's new best-practice for RNG in parallel
np.random.seed(1)
## consumer chars (100k) 
W = np.random.uniform(0,1,N_cons)[:,None,None]
## product chars (10)  (in a real implementation, these should be sorted by product IDs 1-10, so they correspond with correct rows in the choices matrix)
X = np.random.uniform(0,1,N_choices)[None,:,None]
## random utility (1000 draws)
S = np.random.normal(0,1,N_sims)[None,None,:]
## fake choices Y, corresponding to product IDs (for the 100k consumers, no explicit outside option for now)
## move up by one integer so as not to divide by zero in broadcasting below to compute likelihood for each consumer over choices
Y = np.random.randint(0,10,N_cons) + 1
## sorted product_IDs
prod_IDs = np.arange(10).astype(int) + 1
## auxiliary matrix
## divide broadcast individual choices over the product IDs to get a 1 for each consumer's actual choice
## in a matrix of n_cons by n_prods
aux = Y[:,None] / prod_IDs[None,:]
choices = np.zeros((aux.shape[0],aux.shape[1]))
choices[aux==1] = 1


'''
fake parameters
'''
## beta is a vector consisting of
## constant utility,
## mean utility for characteristic
## agent-specific utility
## random taste for each product
beta = np.random.uniform(0,1,4)


'''
compute CCPs and total log likelihood across choices and consumers
'''

## jnp and then jax.jit calls
def LL(beta,W,X,S,choices):
    e_util_ijs = jnp.exp(beta[0] + (beta[1] + beta[2]*W + beta[3]*S)*X)
    ## generate CCP for each consumer and choice (and sim draw)
    CCP_ijs = e_util_ijs/jnp.sum(e_util_ijs,axis=1)[:,None,:]
    ## integrate-out simulants
    CCP_ij = jnp.mean(CCP_ijs,axis=2)
    return jnp.sum(choices*jnp.log(CCP_ij) + (1-choices)*jnp.log(1-CCP_ij))

## call just in time compilation
## need block until ready required due to jax's asynchronous execution model
## https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html
## there is no GPU on VM, tell it to compile to CPU
LL_jit = jax.jit(LL)

## start timer
start = timeit.default_timer()

total_LL = LL_jit(beta,W,X,S,choices).block_until_ready()

## timer stop and print runtime
stop = timeit.default_timer()

print('Time: ', (stop - start), 'seconds')
print(total_LL)

'''
About 3.8 seconds in jax
'''
