### This file does benchmarks for different data sizes
### Run for single core as
#### julia --project=. multinomial/julia/main.jl

# Run with multiproc as
#### julia --project=. multinomial/julia/main.jl multiproc

# Run for a specific number N like
### julia --project=. multinomial/julia/main.jl 10000

# Run with no JIT specialization/compilation
#### julia --project=. multinomial/julia/main.jl nojit

# Or any combination of the above...

multiproc, hamstrung, nojit, N = (
    any(==("multiproc"), ARGS),
    any(==("hamstrung"), ARGS),
    any(==("nojit"), ARGS),
    get(filter(!isnothing, tryparse.(Int64, ARGS)), 1, nothing)
)

# if multiproc, enable all CPU cores, change BLAS config to avoid oversubscription
using Distributed
using LinearAlgebra
if multiproc
    addprocs(20)
    @everywhere begin
        using LinearAlgebra
        BLAS.set_num_threads(1)
    end
end


# enable fast Intel-specific BLAS implementation
if !hamstrung
    @everywhere using MKL
end

# log system state
cpus = let io = IOBuffer()
    Sys.cpu_summary(io)
    String(take!(io))
end
@info "CPU Summary\n$cpus\n"
@info multiproc ? "Distributing i across $(nworkers()) cores" : "Running one main process"
@info "BLAS Config (per core)" nthreads = BLAS.get_num_threads() library = basename(BLAS.get_config().loaded_libs[1].libname)

macro maybe_hamstring(src)
    src = Base.eval(Main, src)
    e = read(src, String) |> Meta.parse
    if nojit
        quote
            @nospecialize
            @noinline $(esc(e))
            @specialize
        end 
    else
        esc(e)
    end
end

using CSV
using DataFrames
using BenchmarkTools
@everywhere using Statistics

cd("multinomial/julia")
include("DGP.jl")

#### INCLUDE THE LIKELIHOOD
@maybe_hamstring multiproc ? "distributed_MNC.jl" : "naive_MNC.jl"
P = nprocs()

### Compile benchmarks into df
const df = DataFrame(P=UInt8[], N=Int64[], hamstrung=Bool[], nojit=Bool[], milliseconds=Float64[])

Ns = isnothing(N) ? [10, 100, 1_000, 10_000] : [N]
for N in Ns
    @info "\nBenchmarking N = $N"
    global w, y = gen_data(N)
    global mat = [ reshape(w, :) y ]
    b = @benchmark L(ones(4))
    show(stderr, MIME("text/plain"), b)
    println()
    milliseconds = b.times ./ 1e6 # times are in ns
    append!(df, DataFrame(; P, N, hamstrung, nojit, milliseconds))
end

CSV.write(stdout, df)
