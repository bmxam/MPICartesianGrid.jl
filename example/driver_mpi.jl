module DriverMPI
using MPI

function run_mpi(;nprocs=(1,1), filename)
    dir = @__DIR__
    repodir = joinpath(dir,".")
    println(repodir)
    nprocs_tot = prod(nprocs)
    nprocx, nprocy = nprocs
    mpiexec() do cmd
        run(`$cmd -n $nprocs_tot $(Base.julia_cmd()) --project=$repodir $filename $nprocx $nprocy`)
    end
end

filename = length(ARGS) >= 1 ? ARGS[1] : "./demo_2D.jl"
nprocs = length(ARGS) >= 3 ? parse.(Int, (ARGS[2], ARGS[3])) : (1,1)

# Usage:
# Run a script "demo_2D.jl" on 2x2 mpi ranks :
# $> julia driver_mpi.jl demo_2D.jl 2 2
run_mpi(nprocs = nprocs, filename = filename)
end