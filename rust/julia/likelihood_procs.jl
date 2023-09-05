### Multiproc libraries
using Distributed
using Base.Iterators
using SharedArrays

### Numerical Libraries
using LinearAlgebra
using Statistics
using Distributions: Logistic, cdf

#### Utility Libraries
using Pipe: @pipe
using Chain: @chain
using PartialFunctions

let n = nprocs()
    if n < 2
        new_procs = addprocs(2 - n)
        @everywhere new_procs include("findFixedPoint.jl")
    end
end

# this is only *called* on the master process
# memory efficient because of batching over
# an iterator
likelihood(data) = let d = collect(zip(data...))
    @sync @distributed (+) for (a, i) in d
        p = cdf(Logistic(), v₁[a] - v₀[a])
        i * log(p) + (1 - i) * log(1 - p)
    end
end
