module MPIStructuredMesh
    using MPI

    include("mesh.jl")
    export DistributedMesh, update_halo!, create_buffers, finalize_mesh, get_rank, get_nelts, nelts_with_halo, local_indices
end