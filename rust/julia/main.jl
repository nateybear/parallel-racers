### This file does benchmarks for different data sizes
### Run for single core as
#### julia --project=. rust/julia/main.jl
### Run for multiple cores like
#### julia --project=. --procs=XXX rust/julia/main.jl --multiproc

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

#### INCLUDE THE LIKELIHOOD
if length(ARGS) > 0 && ARGS[1] == "--multiproc"
    include("likelihood_procs.jl")
    P = nworkers()
else
    include("likelihood_implicit.jl")
    P = 1
end

### Compile benchmarks into df
const df = DataFrame(nprocs=UInt8[], elapsed=Float64[], size=Int64[])

for multiple in [1, 10, 100, 1000, 10000]
    @info "Running rust likelihood with multiple=$multiple"
    t = @belapsed likelihood($(repeat.(raw, multiple)))
    push!(df, (P, t, multiple * N))
end

CSV.write(stdout, df)
