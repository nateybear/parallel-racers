include("findFixedPoint.jl")

using Distributions
using LinearAlgebra

likelihood((a, i)) =
    let p = cdf.(Logistic(), v₁[a] .- v₀[a]) # probability of replacing
        sum(@. i * log(p) + (1 - i) * log(1 - p)) # log-likelihood
    end
