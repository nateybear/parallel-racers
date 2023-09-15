using LinearAlgebra
using Statistics
using Distributions: Logistic, cdf
using Pipe

function findFixedPoint((θ, R), tol=1e-7)
    X = 1:5 # state space

    V₀ = V₁ = zeros(size(X))

    V(v₀, v₁) = max.(v₀, v₁) # value function

    V_replace() =
        let v₀ = exp(V₀[1]),
            v₁ = exp(V₁[1])

            @pipe (R + 0.9 * (0.5772 + log(v₀ + v₁))) |>
                  fill(_, size(X),)
        end

    V_keep() =
        let x = min.(X .+ 1, 5),
            v₀ = exp.(V₀[x]),
            v₁ = exp.(V₁[x])

            @. θ * x + 0.9 * (0.5772 + log(v₀ + v₁))
        end

    while true
        W₀, W₁ = V_keep(), V_replace()

        if norm(V(V₀, V₁) - V(W₀, W₁)) < tol
            return W₀, W₁
        end

        V₀, V₁ = W₀, W₁
    end
end

findFixedPoint() = findFixedPoint([-1.0, -3.0] .+ randn(2))

v₀, v₁ = findFixedPoint()
