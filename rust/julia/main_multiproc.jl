### This file runs just the likelihood_procs.jl
### run like
#### julia --project=. --procs=4 rust/julia/main_multiproc.jl > rust/out/multiproc_4_julia.csv

### Numerical Libraries
using LinearAlgebra
using Statistics
using Distributions: Logistic, cdf

### DataFrame libraries
using DataFrames
using CSV

#### Utility Libraries
using Pipe: @pipe
using Chain: @chain
using BenchmarkTools

cd("rust/julia")

##### Data Functions
read_dat(f::String) = @pipe readlines(f) .|>
                            strip .|>
                            split |>
                            hcat(_...) |>
                            parse.(Float64, _) |>
                            transpose |>
                            convert(Matrix{Float64}, _)


const raw = @pipe read_dat("../data.asc") .|> convert(Int8, _) |> eachcol
const N = (first ∘ size ∘ first)(raw)


#### INCLUDE THE MULTIPROC LIKELIHOOD
include("likelihood_procs.jl")

const df = DataFrame(nprocs=UInt8[], elapsed=Float64[], size=Int64[])

for multiple in [1, 10, 100, 1000, 10000]
    @info "Running likelihood_procs.jl with multiple=$multiple"
    t = @belapsed likelihood($(repeat.(raw, multiple)))
    push!(df, (nworkers(), t, multiple * N))
end

CSV.write(stdout, df)
