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


# raw is an iterator
const raw = @pipe read_dat("../data.asc") .|> convert(Int8, _) |> eachcol
const N = (first ∘ size ∘ first)(raw)


#### SAVE THE FIXED POINT AND USE EVERYWHERE
include("findFixedPoint.jl")

const LIKELIHOOD_FILES = [
    "likelihood_implicit.jl",
    "likelihood_serial.jl",
    "likelihood_procs.jl"
]

const df = DataFrame(method=UInt8[], elapsed=Float64[], size=Int64[])

for (i, f) in enumerate(LIKELIHOOD_FILES),
    multiple in [1, 10, 100, 1000, 10000]

    include(f)
    @info "Running $f with multiple=$multiple"
    t = @belapsed likelihood($(repeat.(raw, multiple)))
    push!(df, (i, t, multiple * N))
end

CSV.write(stdout, df)
