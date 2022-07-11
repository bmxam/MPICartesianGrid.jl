"""
Remark : the structure could lighter if we interrogate MPI to query infos (rank, etc) instead of storing
them here.
"""
struct DistributedMesh{D}
    comm::MPI.Comm
    ndims::NTuple{D,Int} # Number of process in each dimension
    coords::NTuple{D,Int} # Coordinates in the global mesh (1-based)
    nelts::NTuple{D,Int} # Number of elts of the local mesh in each direction, without halo
    noverlaps::NTuple{D,Int} # Number of overlap elements in each direction
    coords2rank::AbstractArray{Int,D} # MPI coordinates (1-based) to MPI rank (identical to MPI.Cart_rank)
end

@inline get_comm(mesh::DistributedMesh) = mesh.comm
@inline get_coords(mesh::DistributedMesh) = mesh.coords
@inline get_ndims(mesh::DistributedMesh) = mesh.ndims
@inline get_ndims(mesh::DistributedMesh, d::Int) = mesh.ndims[d]

@inline get_noverlaps(mesh::DistributedMesh) = mesh.noverlaps
@inline get_noverlap(mesh::DistributedMesh, d::Int) = mesh.noverlaps[d]

@inline get_rank(mesh::DistributedMesh) = MPI.Comm_rank(get_comm(mesh))
@inline get_rank(coords::Vector{Int}, mesh::DistributedMesh) = mesh.coords2rank[coords]
@inline get_rank(coords::NTuple{D,Int}, mesh::DistributedMesh{D}) where D = mesh.coords2rank[coords...]
@inline neighbors(::DistributedMesh{D}) where D = ntuple(d -> -1:1, D)

@inline get_nelts(mesh::DistributedMesh) = mesh.nelts
@inline get_nelts(mesh::DistributedMesh, d::Int) = mesh.nelts[d]

@inline nelts_with_halo(mesh::DistributedMesh) = mesh.nelts .+ 2 .* mesh.noverlaps
@inline nelts_with_halo(mesh::DistributedMesh, d::Int) = mesh.nelts[d] + 2*mesh.noverlaps[d]

@inline local_indices(mesh::DistributedMesh{D}) where D = ntuple(d -> range(mesh.noverlaps[d]+1, mesh.nelts[d] + mesh.noverlaps[d]), D)
@inline local_indices(mesh::DistributedMesh{D}, d::Int) where D = range(mesh.noverlaps[d]+1, mesh.nelts[d] + mesh.noverlaps[d])

"""
    DistributedMesh(ndims::NTuple{D,Int}, nelts::NTuple{D,Int}, noverlaps::NTuple{D,Int}; init_MPI = true) where D

`D` is the number of spatial dimensions. `ndims` is the number of processors in each spatial dimension. `nelts` is the number of grid elements
in each spatial dimension on each core (hence the total number of grid elements is `sum(ndims .* nelts)`). Finally,
`noverlaps` is the number of elements overlaps (on both side of the spatial direction) in each space direction.
"""
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

    DistributedMesh{D}(comm, ndims, coords, nelts, noverlaps, coords2rank)
end
DistributedMesh(comm::MPI.Comm, ndims::Int, nelts::Int, noverlaps::Int; init_MPI = true) = DistributedMesh(comm, (ndims,), (nelts,), (noverlaps,);  init_MPI)

"""
    create_buffers(type, mesh::DistributedMesh{D}, narrays::Int = 1) where D

Create send/recv buffers for MPI exchange of `Array`'s of type `type`.
"""
function create_buffers(mesh::DistributedMesh{D}, type, narrays::Int = 1) where D
    coords = get_coords(mesh)
    recv_buffer = Dict{Int,Vector{type}}()

    #- Loop over dimensions
    for stencil in Iterators.product(neighbors(mesh)...)
        neighbor = coords .+ stencil

        # Jump to next neighbor if the present neighbor is not a real one
        _is_true_neighbor(neighbor, stencil, mesh) || continue

        # Compute buffer size
        bufferSize = 1
        for d in 1:D
            bufferSize *= stencil[d] == 0 ? get_nelts(mesh, d) : get_noverlap(mesh, d)
        end

        # Determine src rank
        src = get_rank(neighbor, mesh)

        # Allocate and store buffer
        recv_buffer[src] = zeros(type, bufferSize * narrays)
    end

    return recv_buffer, copy(recv_buffer)
end
create_buffers(mesh::DistributedMesh{D}, ::NTuple{N,AbstractArray{T,D}}) where {D,T,N} = create_buffers(mesh, T, N)
create_buffers(mesh::DistributedMesh{D}, ::AbstractArray{T,D}) where {D,T} = create_buffers(mesh, T)

"""
    update_halo!(arrays::NTuple{N,AbstractArray{T,D}}, mesh::DistributedMesh{D}) where {T,D,N}

Update the halo (=border, in each dimension) of each array of the input `arrays`.

No buffer are required, they will be created. If you want to reuse buffers, check the other `update_halo!`
functions.
"""
function update_halo!(arrays::NTuple{N,AbstractArray{T,D}}, mesh::DistributedMesh{D}) where {T,D,N}
    recv_buffer, send_buffer = create_buffers(mesh, T, N)
    update_halo!(arrays, recv_buffer, send_buffer, mesh)
end

"""
    update_halo!(array::AbstractArray{T,D}, mesh::DistributedMesh{D}) where {T,D}

Update the halo (=border, in each dimension) of the input `array`.

No buffer are required, they will be created. If you want to reuse buffers, check the other `update_halo!`
functions.
"""
function update_halo!(array::AbstractArray{T,D}, mesh::DistributedMesh{D}) where {T,D}
    update_halo!((array, ), mesh)
end

"""
    update_halo!(arrays::NTuple{N,AbstractArray{T,D}}, recv_buffer::Dict{Int,Vector{T}}, send_buffer::Dict{Int,Vector{T}}, mesh::DistributedMesh{D}) where {T,D,N}

Update the halo (=border, in each dimension) of the input `array`.

Buffers need to be provided.
"""
function update_halo!(arrays::NTuple{N,AbstractArray{T,D}}, recv_buffer::Dict{Int,Vector{T}}, send_buffer::Dict{Int,Vector{T}}, mesh::DistributedMesh{D}) where {T,D,N}
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
        _arrays2buffer!(buffer, arrays, stencil, mesh)

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
        _buffer2arrays!(buffer, arrays, stencil, mesh)
    end
end

"""
Copy a part of the `arrays` (same for all arrays) into the sending `buffer`. The target is defined by the stencil.
"""
function _arrays2buffer!(buffer::Vector{T}, arrays::NTuple{N,AbstractArray{T,D}}, stencil::NTuple{D,Int}, mesh::DistributedMesh{D}) where {D,T,N}
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

    # Copy to buffer. Depending on the stencil, there might be zero, one, several elements of each dimension
    # of each array to copy in the buffer. Array elements are interlaced this way in the buffer:
    # buffer = [array1[1,2], array2[1,2], array3[1,2], array1[7,4], array2[7,4], array3[7,4], ....]
    for (i,ind) in enumerate(Iterators.product(rh...))
        for (iarray, array) in enumerate(arrays)
            buffer[iarray + (i-1)*N] = array[ind...]
        end
    end
end

"""
Copy the content of the received `buffer` into the `arrays`. The source is designated by the `stencil`.
"""
function _buffer2arrays!(buffer::Vector{T}, arrays::NTuple{N,AbstractArray{T,D}}, stencil::NTuple{D,Int}, mesh::DistributedMesh{D}) where {D,T,N}
    nelts_h = nelts_with_halo(mesh)

    # Gather all elements index to copy into array
    rh = ntuple(
        d ->
        stencil[d] < 0 ? range(1,get_noverlap(mesh,d)) :
        stencil[d] > 0 ? range(nelts_h[d] - get_noverlap(mesh,d) + 1, nelts_h[d]) :
        local_indices(mesh, d),
        D
    )

    # Copy from buffer. Array elements are interlaced this way in the buffer:
    # buffer = [array1[1,2], array2[1,2], array3[1,2], array1[7,4], array2[7,4], array3[7,4], ....]
    for (i,ind) in enumerate(Iterators.product(rh...))
        for (iarray, array) in enumerate(arrays)
            array[ind...] = buffer[iarray + (i-1)*N]
        end
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