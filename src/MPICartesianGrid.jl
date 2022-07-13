module MPICartesianGrid
    using MPI

    include("grid.jl")
    export DistributedGrid, update_halo!, create_buffers, finalize_grid, get_rank, get_nelts, nelts_with_halo, local_indices
end