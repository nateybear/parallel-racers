using Distributed
using Base.Iterators
using Pipe
using DataFrames
using CSV
using BenchmarkTools
using ProgressMeter

include("rust_likelihood_procs.jl")
likelihood([-1.0, -3.0], SharedVector{Float64}(1))

### managing the workers here

function load!(new_procs)
    @everywhere new_procs begin
        include("rust_likelihood_procs.jl")
        fp = Base.invokelatest(findFixedPoint, [-1.0, -3.0])
        Base.invokelatest(likelihood, fp, raw)
    end
    return nothing
end

function set_workers!(n, reset=false)
    if reset
        set_workers!(0)
    end
    if nworkers() < n
        Δn = n - nworkers()
        new_procs = addprocs(Δn, enable_threaded_blas=true)
        load!(new_procs)
    elseif nworkers() > n
        rmprocs(setdiff(workers()[n+1:end], [1]))
    end

    return nothing
end

reset_workers!() = set_workers!(nworkers(), true)

reset_workers!()


##### This is the main loop

df = DataFrame(cores=Int[], elapsed_time=Float64[], multiple=Int[])

@warn "Do not compare runtimes across workers. The @benchmark macro may run different numbers of trials at each invocation."

for nprocs in [0, 4, 8, 16, 32]
    @info nprocs > 0 ? "Running on $nprocs workers" : "Running on master process"
    set_workers!(nprocs)
    out = SharedArray{Float64}(max(nprocs, 1))
    @showprogress for multiple in [1, 10, 100, 1000, 10000, 100000]
        set_multiple!(multiple)
        b = @benchmark likelihood([-1.0, -3.0], $out)
        Base.GC.gc()
        for i in Base.OneTo(length(b))
            elapsed_time = time(b[i]) / 1e9
            push!(df, (nprocs, elapsed_time, multiple))
        end
    end
end

CSV.write(stdout, df)
