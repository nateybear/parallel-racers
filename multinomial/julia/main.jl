### This file does benchmarks for different data sizes
### Run for single core as
#### julia --project=. multinomial/julia/main.jl
### Run for multiple cores like
#### julia --project=. --procs=XXX multinomial/julia/main.jl --multiproc

using CSV
using DataFrames
using Distributed
using BenchmarkTools
@everywhere begin
    using LinearAlgebra
    using Statistics
end

cd("multinomial/julia")
include("DGP.jl")

#### INCLUDE THE LIKELIHOOD
if length(ARGS) > 0 && ARGS[1] == "--multiproc"
    include("distributed_MNC.jl")
    P = nworkers()
else
    include("naive_MNC.jl")
    P = 1
end

### Compile benchmarks into df
const df = DataFrame(nprocs=UInt8[], elapsed=Float64[], size=Int64[])

for capN in [1, 10, 100, 1_000, 10_000]
    @info "Running multinomial choice with N = $capN"
    global w, y = gen_data(capN)
    global mat = [ reshape(w, :) y ]
    t = @belapsed L(ones(4))
    push!(df, (P, t, capN))
end

CSV.write(stdout, df)
