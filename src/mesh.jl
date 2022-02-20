"""
The structure could be mush lighter if we interrogate MPI to query infos (rank, etc) instead of storing
them here.
"""
struct DistributedMesh{D}
    comm::MPI.Comm
    ndims::NTuple{D,Int} # Number of process in each dimension
    coords::NTuple{D,Int} # Coordinates in the global mesh (1-based)
    nelts::NTuple{D,Int} # Number of elts of the local mesh in each direction, without halo
    noverlaps::NTuple{D,Int} # Number of overlap elements in each direction
    coords2rank::AbstractArray{Int,D} # MPI coordinates (1-based) to MPI rank (identical to MPI.Cart_rank)
    rank::Int # MPI rank (0-based)
end

@inline get_comm(mesh::DistributedMesh) = mesh.comm
@inline get_coords(mesh::DistributedMesh) = mesh.coords
@inline get_ndims(mesh::DistributedMesh) = mesh.ndims
@inline get_ndims(mesh::DistributedMesh, d::Int) = mesh.ndims[d]

@inline get_noverlaps(mesh::DistributedMesh) = mesh.noverlaps
@inline get_noverlap(mesh::DistributedMesh, d::Int) = mesh.noverlaps[d]

@inline get_rank(mesh::DistributedMesh) = mesh.rank
@inline get_rank(coords::Vector{Int}, mesh::DistributedMesh) = mesh.coords2rank[coords]
@inline get_rank(coords::NTuple{D,Int}, mesh::DistributedMesh{D}) where D = mesh.coords2rank[coords...]
@inline neighbors(::DistributedMesh{D}) where D = ntuple(d -> -1:1, D)

@inline get_nelts(mesh::DistributedMesh) = mesh.nelts
@inline get_nelts(mesh::DistributedMesh, d::Int) = mesh.nelts[d]

@inline nelts_with_halo(mesh::DistributedMesh) = mesh.nelts .+ 2 .* mesh.noverlaps
@inline nelts_with_halo(mesh::DistributedMesh, d::Int) = mesh.nelts[d] + 2*mesh.noverlaps[d]

@inline local_indices(mesh::DistributedMesh{D}) where D = ntuple(d -> range(mesh.noverlaps[d]+1, mesh.nelts[d] + mesh.noverlaps[d]), D)
@inline local_indices(mesh::DistributedMesh{D}, d::Int) where D = range(mesh.noverlaps[d]+1, mesh.nelts[d] + mesh.noverlaps[d])


function DistributedMesh(ndims::NTuple{D,Int}, nelts::NTuple{D,Int}, noverlaps::NTuple{D,Int}; init_MPI = true) where D
    # Init MPI if necessary
    init_MPI && MPI.Init()

    # Create comm
    comm = MPI.Cart_create(MPI.COMM_WORLD, [d for d in ndims], [0 for _ in ndims], false)

    # Build mesh
    DistributedMesh(comm, ndims, nelts, noverlaps)
end
DistributedMesh(ndims::Int, nelts::Int, noverlaps::Int; init_MPI = true) = DistributedMesh((ndims,), (nelts,), (noverlaps,);  init_MPI)

function DistributedMesh(comm::MPI.Comm, ndims::NTuple{D,Int}, nelts::NTuple{D,Int}, noverlaps::NTuple{D,Int}) where D

    # Determine map between coords and rank (MPI knows this map but I am not sure if it's fast or not)
    coords2rank = zeros(Int, ndims...)

    # Loop over all possible coordinates
    for coords in Iterators.product(range.(1, ndims)...)
        coords_target = [c - 1 for c in coords] # need 0-based for MPI
        coords2rank[coords...] = MPI.Cart_rank(comm, coords_target)
    end

    # Determine current proc coordinates
    coords = MPI.Cart_coords(comm) # 0-based
    coords = ntuple(d -> coords[d] + 1, D) # NTuple, 1-based

    # Get rank
    rank = MPI.Comm_rank(comm)

    DistributedMesh{D}(comm, ndims, coords, nelts, noverlaps, coords2rank, rank)
end
DistributedMesh(comm::MPI.Comm, ndims::Int, nelts::Int, noverlaps::Int; init_MPI = true) = DistributedMesh(comm, (ndims,), (nelts,), (noverlaps,);  init_MPI)

function create_buffers(type, mesh::DistributedMesh{D}) where D
    coords = get_coords(mesh)
    recv_buffer = Dict{Int,Vector{type}}()

    #- Loop over dimensions
    for stencil in Iterators.product(neighbors(mesh)...)
        neighbor = coords .+ stencil

        # Jump to next neighbor if the present neighbor is not a real one
        _is_true_neighbor(neighbor, stencil, mesh) || continue

        # Compute buffer size
        n = 1
        for d in 1:D
            n *= stencil[d] == 0 ? get_nelts(mesh, d) : get_noverlap(mesh, d)
        end

        # Determine src rank
        src = get_rank(neighbor, mesh)

        # Allocate and store buffer
        recv_buffer[src] = zeros(type, n)
    end

    return recv_buffer, copy(recv_buffer)
end

function update_halo!(array::AbstractArray{T,D}, mesh::DistributedMesh{D}) where {T,D}
    recv_buffer, send_buffer = create_buffers(T, mesh)
    update_halo!(array, recv_buffer, send_buffer, mesh)
end

function update_halo!(array::AbstractArray{T,D}, recv_buffer::Dict{Int,Vector{T}}, send_buffer::Dict{Int,Vector{T}}, mesh::DistributedMesh{D}) where {T,D}
    comm = get_comm(mesh)
    coords = get_coords(mesh)

    # Async receive
    recv_reqs = MPI.Request[]

    #- Loop over dimensions
    for stencil in Iterators.product(neighbors(mesh)...)
        neighbor = coords .+ stencil

        # Jump to next neighbor if the present neighbor is not a real one
        _is_true_neighbor(neighbor, stencil, mesh) || continue

        # Execute request
        src = get_rank(neighbor, mesh)
        buffer = recv_buffer[src]
        push!(recv_reqs, MPI.Irecv!(buffer, src, 0, comm))
    end

    # Async send
    send_reqs = MPI.Request[]

    #- Loop over dimensions
    for stencil in Iterators.product(neighbors(mesh)...)
        neighbor = coords .+ stencil

        # Jump to next neighbor if the present neighbor is not a real one
        _is_true_neighbor(neighbor, stencil, mesh) || continue

        # Fill buffer
        dst = get_rank(neighbor, mesh)
        buffer = send_buffer[dst]
        _array2buffer!(buffer, array, stencil, mesh)

        # Execute request
        push!(send_reqs, MPI.Isend(buffer, dst, 0, comm))
    end

    # Wait for all requests to terminate
    MPI.Waitall!(vcat(recv_reqs, send_reqs))

    # Copy from received buffer to array
    #- Loop over dimensions
    for stencil in Iterators.product(neighbors(mesh)...)
        neighbor = coords .+ stencil

        # Jump to next neighbor if the present neighbor is not a real one
        _is_true_neighbor(neighbor, stencil, mesh) || continue

        # Execute request
        src = get_rank(neighbor, mesh)
        buffer = recv_buffer[src]
        _buffer2array!(buffer, array, stencil, mesh)
    end
end

"""
Copy a part of the `array` into the sending `buffer`. The target is defined by the stencil.
"""
function _array2buffer!(buffer::Vector{T}, array::AbstractArray{T,D}, stencil::NTuple{D,Int}, mesh::DistributedMesh{D}) where {D,T}
    n = get_nelts(mesh)
    noverlaps = get_noverlaps(mesh)

    # Gather all elements index to copy into buffer
    rh = ntuple(
        d ->
        stencil[d] < 0 ? range(noverlaps[d] + 1; length = noverlaps[d]) :
        stencil[d] > 0 ? range(n[d] + 1, n[d] + noverlaps[d]) :
        local_indices(mesh,d),
        D
    )

    # Copy to buffer
    for (i,ind) in enumerate(Iterators.product(rh...))
        buffer[i] = array[ind...]
    end
end

"""
Copy the content of the received `buffer` into the `array`. The source is designated by the `stencil`.
"""
function _buffer2array!(buffer::Vector{T}, array::AbstractArray{T,D}, stencil::NTuple{D,Int}, mesh::DistributedMesh{D}) where {D,T}
    nelts_h = nelts_with_halo(mesh)

    # Gather all elements index to copy into array
    rh = ntuple(
        d ->
        stencil[d] < 0 ? range(1,get_noverlap(mesh,d)) :
        stencil[d] > 0 ? range(nelts_h[d] - get_noverlap(mesh,d) + 1, nelts_h[d]) :
        local_indices(mesh, d),
        D
    )

    # Copy from buffer
    for (i,ind) in enumerate(Iterators.product(rh...))
        array[ind...] = buffer[i]
    end
end

"""
Indicate if the `neighbor`, obtained with the given `stencil`, is a "true" neighbor : i.e inside the global mesh
and different from the local mesh.
"""
function _is_true_neighbor(neighbor::NTuple{D,Int}, stencil::NTuple{D,Int}, mesh::DistributedMesh{D}) where D
    return all(d -> 1 <= neighbor[d] <= get_ndims(mesh,d), 1:D) && any(d -> stencil[d] != 0, 1:D)
end

finalize_mesh(::DistributedMesh; finalize_MPI = true) = finalize_MPI && MPI.Finalize()