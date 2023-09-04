using Base.Threads
using LinearAlgebra
using Statistics
using Distributions: Logistic, cdf
using Pipe: @pipe

##### Solving the DP

const KEEP = Val{0}()
const REPLACE = Val{1}()

function findFixedPoint((θ, R), tol=1e-7)
    X = 1:5 # state space
    
    V₀ = V₁ = X # value grid (random initial guess)

    V(v₀, v₁) = max.(v₀, v₁) # value function

    function step(REPLACE)
        v₀ = exp(V₀[1])
        v₁ = exp(V₁[1])
        R + 0.9 * (0.5772 + log(v₀ + v₁))
    end

    function step(KEEP, x)
        v₀ = exp(V₀[min(x + 1, 5)])
        v₁ = exp(V₁[min(x + 1, 5)])
        θ * x + 0.9 * (0.5772 + log(v₀ + v₁))
    end

    while true
        W₀, W₁ = step.(KEEP, X), fill(step(REPLACE), size(X))

        if norm(V(V₀, V₁) - V(W₀, W₁)) < tol
            return W₀, W₁
        end

        V₀, V₁ = W₀, W₁
    end
end

##### Data Functions
read_dat(f::String) = @pipe readlines(f) .|>
                            strip .|>
                            split |>
                            hcat(_...) |>
                            parse.(Float64, _) |>
                            transpose |>
                            convert(Matrix{Float64}, _)

raw = @pipe read_dat("rust_data.asc") .|> convert(Int8, _) |> eachcol

multiple = parse(Int, get(ARGS, 1, "1"))
const as, is = repeat.(raw, multiple)

##### Likelihood Function

function likelihood(params) 
    v₀, v₁ = findFixedPoint(params)

    # NOTE: threading out instead of brodcasting (.*) means
    #       that autodiff fails and you need to use FiniteDiff
    out = zeros(length(as))
    @threads for j in eachindex(as)
        a = as[j]
        i = is[j]
        p = cdf(Logistic(), v₁[a] - v₀[a]) # probability of replacing
        
        out[j] = i * log(p) + (1 - i) * log(1 - p) # log-likelihood
    end

    return out
end

# get initial compile out of the way
likelihood([-1.0, -3.0])

@time likelihood([-1.0, -3.0])
