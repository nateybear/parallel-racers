using LinearAlgebra
using Statistics
using Distributions: Logistic, cdf
using Lazy
using PartialFunctions

function contract((V₀, V₁); θ, R, X)
    V_replace =
        let v₀ = exp(V₀[1]),
            v₁ = exp(V₁[1])

            @> (R + 0.9 * (0.5772 + log(v₀ + v₁))) fill(size(X))
        end

    V_keep =
        let x = min.(X .+ 1, 5),
            v₀ = exp.(V₀[x]),
            v₁ = exp.(V₁[x])

            @. θ * x + 0.9 * (0.5772 + log(v₀ + v₁))
        end

    V_keep, V_replace
end

V(v₀, v₁) = max.(v₀, v₁) # value function

distance(((V₀ₙ, V₁ₙ), (V₀ₙ₊₁, V₁ₙ₊₁))) = norm(V(V₀ₙ, V₁ₙ) .- V(V₀ₙ₊₁, V₁ₙ₊₁))

function findFixedPoint((θ, R), tol=1e-7)
    X = 1:5 # state space

    V₀ = V₁ = zeros(size(X))

    contractions = iterated(contract $ (; θ, R, X), (V₀, V₁))

    @>> contractions begin
        partition(2) # make windows of 2
        takeuntil((<)(tol) ∘ distance) 
        last # last window of 2
        last # last element of window
    end
end

findFixedPoint() = findFixedPoint([-1.0, -3.0] .+ randn(2))

v₀, v₁ = findFixedPoint()
