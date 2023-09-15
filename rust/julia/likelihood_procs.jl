@everywhere begin
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

    include("findFixedPoint.jl")
end

likelihood((as, is)) = let d = collect(zip(as, is))
    @sync @distributed (+) for (a, i) in d
        p = cdf(Logistic(), v₁[a] - v₀[a])
        i * log(p) + (1 - i) * log(1 - p)
    end
end
