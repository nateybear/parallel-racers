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


##### Data Functions
read_dat(f::String) = @pipe readlines(f) .|>
                            strip .|>
                            split |>
                            hcat(_...) |>
                            parse.(Float64, _) |>
                            transpose |>
                            convert(Matrix{Float64}, _)


# raw is an iterator
raw::ColumnSlices = @pipe read_dat("rust_data.asc") .|> convert(Int8, _) |> eachcol

### Do it this way to give multiple and init type-stable definitions at the top
### level. Improves performance.
N::Int16 = (first ∘ size ∘ first)(raw)
multiple::Int = 1


#### Control how big the data is
#### N.B. this does not *materialize* the data
#### Each subprocess will only materialize the data it needs
function set_multiple!(m)
    global multiple = m
end

##### Solving the DP function

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

const fp = findFixedPoint([-1.0, -3.0])

##### Likelihood Function

# this is shared everywhere
function likelihood((v₀, v₁), (a, i))
    p = cdf.(Logistic(), v₁[a] .- v₀[a])
    
    @. i * log(p) + (1 - i) * log(1 - p)
end

# this is only *called* on the master process
# memory efficient because of batching over
# an iterator
function likelihood(params, out::SharedArray)
    # get the list of workers we can dispatch to
    w = procs(out)

    # partition the data into batches, one batch per worker
    batches = @chain begin
        Base.OneTo(multiple)
        partition(_, ceil(Int, length(_) / length(w)))
        length.()
        enumerate
    end

    # save the futures here
    futures = Future[]

    # asynchoronously push off and materialize
    # N * batch_size observations per worker
    for (iworker, idata) in batches
        f = @spawnat w[iworker] begin
            l = @chain raw begin
                repeat.(idata)
                Iterators.flatten.(_)
                collect.()
                likelihood(fp, _)
            end
            
            out[iworker] = sum(l)
        end
        push!(futures, f)
    end

    # here we wait for the futures we just created
    for f in futures
        wait(f)
    end

    sum(out)
end
