"""
Remark : the structure could lighter if we interrogate MPI to query infos (rank, etc) instead of storing
them here.
"""
struct DistributedGrid{D}
    comm::MPI.Comm
    ndims::NTuple{D,Int} # Number of process in each dimension
    coords::NTuple{D,Int} # Coordinates in the global grid (1-based)
    nelts::NTuple{D,Int} # Number of elts of the local grid in each direction, without halo
    noverlaps::NTuple{D,Int} # Number of overlap elements in each direction
    coords2rank::AbstractArray{Int,D} # MPI coordinates (1-based) to MPI rank (identical to MPI.Cart_rank)
end

@inline get_comm(grid::DistributedGrid) = grid.comm
@inline get_coords(grid::DistributedGrid) = grid.coords
@inline get_ndims(grid::DistributedGrid) = grid.ndims
@inline get_ndims(grid::DistributedGrid, d::Int) = grid.ndims[d]

@inline get_noverlaps(grid::DistributedGrid) = grid.noverlaps
@inline get_noverlap(grid::DistributedGrid, d::Int) = grid.noverlaps[d]

@inline get_rank(grid::DistributedGrid) = MPI.Comm_rank(get_comm(grid))
@inline get_rank(coords::Vector{Int}, grid::DistributedGrid) = grid.coords2rank[coords]
@inline get_rank(coords::NTuple{D,Int}, grid::DistributedGrid{D}) where D = grid.coords2rank[coords...]
@inline neighbors(::DistributedGrid{D}) where D = ntuple(d -> -1:1, D)

@inline get_nelts(grid::DistributedGrid) = grid.nelts
@inline get_nelts(grid::DistributedGrid, d::Int) = grid.nelts[d]

@inline nelts_with_halo(grid::DistributedGrid) = grid.nelts .+ 2 .* grid.noverlaps
@inline nelts_with_halo(grid::DistributedGrid, d::Int) = grid.nelts[d] + 2*grid.noverlaps[d]

@inline local_indices(grid::DistributedGrid{D}) where D = ntuple(d -> range(grid.noverlaps[d]+1, grid.nelts[d] + grid.noverlaps[d]), D)
@inline local_indices(grid::DistributedGrid{D}, d::Int) where D = range(grid.noverlaps[d]+1, grid.nelts[d] + grid.noverlaps[d])

"""
    DistributedGrid(ndims::NTuple{D,Int}, nelts::NTuple{D,Int}, noverlaps::NTuple{D,Int}; init_MPI = true) where D

`D` is the number of topological dimensions. `ndims` is the number of processors in each topological dimension. `nelts` is the number of grid elements
in each topological dimension on each core (hence the total number of grid elements is `sum(ndims .* nelts)`). Finally,
`noverlaps` is the number of elements overlaps (on both side of the topological direction) in each topological direction.
"""
function DistributedGrid(ndims::NTuple{D,Int}, nelts::NTuple{D,Int}, noverlaps::NTuple{D,Int}; init_MPI = true) where D
    # Init MPI if necessary
    init_MPI && MPI.Init()

    # Create comm
    comm = MPI.Cart_create(MPI.COMM_WORLD, [d for d in ndims], [0 for _ in ndims], false)

    # Build grid
    DistributedGrid(comm, ndims, nelts, noverlaps)
end
DistributedGrid(ndims::Int, nelts::Int, noverlaps::Int; init_MPI = true) = DistributedGrid((ndims,), (nelts,), (noverlaps,);  init_MPI)

function DistributedGrid(comm::MPI.Comm, ndims::NTuple{D,Int}, nelts::NTuple{D,Int}, noverlaps::NTuple{D,Int}) where D
    @assert all(noverlaps .> 0) "Minimum overlap value is 1"

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

    DistributedGrid{D}(comm, ndims, coords, nelts, noverlaps, coords2rank)
end
DistributedGrid(comm::MPI.Comm, ndims::Int, nelts::Int, noverlaps::Int; init_MPI = true) = DistributedGrid(comm, (ndims,), (nelts,), (noverlaps,);  init_MPI)

"""
    create_buffers(type, grid::DistributedGrid{D}, narrays::Int = 1) where D

Create send/recv buffers for MPI exchange of `Array`'s of type `type`.
"""
function create_buffers(grid::DistributedGrid{D}, type, narrays::Int = 1) where D
    coords = get_coords(grid)
    recv_buffer = Dict{Int,Vector{type}}()

    #- Loop over dimensions
    for stencil in Iterators.product(neighbors(grid)...)
        neighbor = coords .+ stencil

        # Jump to next neighbor if the present neighbor is not a real one
        _is_true_neighbor(neighbor, stencil, grid) || continue

        # Compute buffer size
        bufferSize = 1
        for d in 1:D
            bufferSize *= stencil[d] == 0 ? get_nelts(grid, d) : get_noverlap(grid, d)
        end

        # Determine src rank
        src = get_rank(neighbor, grid)

        # Allocate and store buffer
        recv_buffer[src] = zeros(type, bufferSize * narrays)
    end

    return recv_buffer, copy(recv_buffer)
end
create_buffers(grid::DistributedGrid{D}, ::NTuple{N,AbstractArray{T,D}}) where {D,T,N} = create_buffers(grid, T, N)
create_buffers(grid::DistributedGrid{D}, ::AbstractArray{T,D}) where {D,T} = create_buffers(grid, T)

"""
    update_halo!(arrays::NTuple{N,AbstractArray{T,D}}, grid::DistributedGrid{D}) where {T,D,N}

Update the halo (=border, in each dimension) of each array of the input `arrays`.

No buffer are required, they will be created. If you want to reuse buffers, check the other `update_halo!`
functions.
"""
function update_halo!(arrays::NTuple{N,AbstractArray{T,D}}, grid::DistributedGrid{D}) where {T,D,N}
    recv_buffer, send_buffer = create_buffers(grid, T, N)
    update_halo!(arrays, recv_buffer, send_buffer, grid)
end

"""
    update_halo!(array::AbstractArray{T,D}, grid::DistributedGrid{D}) where {T,D}

Update the halo (=border, in each dimension) of the input `array`.

No buffer are required, they will be created. If you want to reuse buffers, check the other `update_halo!`
functions.
"""
function update_halo!(array::AbstractArray{T,D}, grid::DistributedGrid{D}) where {T,D}
    update_halo!((array, ), grid)
end

"""
    update_halo!(arrays::NTuple{N,AbstractArray{T,D}}, recv_buffer::Dict{Int,Vector{T}}, send_buffer::Dict{Int,Vector{T}}, grid::DistributedGrid{D}) where {T,D,N}

Update the halo (=border, in each dimension) of the input `array`.

Buffers need to be provided.
"""
function update_halo!(arrays::NTuple{N,AbstractArray{T,D}}, recv_buffer::Dict{Int,Vector{T}}, send_buffer::Dict{Int,Vector{T}}, grid::DistributedGrid{D}) where {T,D,N}
    comm = get_comm(grid)
    coords = get_coords(grid)

    # Async receive
    recv_reqs = MPI.Request[]

    #- Loop over dimensions
    for stencil in Iterators.product(neighbors(grid)...)
        neighbor = coords .+ stencil

        # Jump to next neighbor if the present neighbor is not a real one
        _is_true_neighbor(neighbor, stencil, grid) || continue

        # Execute request
        src = get_rank(neighbor, grid)
        buffer = recv_buffer[src]
        push!(recv_reqs, MPI.Irecv!(buffer, src, 0, comm))
    end

    # Async send
    send_reqs = MPI.Request[]

    #- Loop over dimensions
    for stencil in Iterators.product(neighbors(grid)...)
        neighbor = coords .+ stencil

        # Jump to next neighbor if the present neighbor is not a real one
        _is_true_neighbor(neighbor, stencil, grid) || continue

        # Fill buffer
        dst = get_rank(neighbor, grid)
        buffer = send_buffer[dst]
        _arrays2buffer!(buffer, arrays, stencil, grid)

        # Execute request
        push!(send_reqs, MPI.Isend(buffer, dst, 0, comm))
    end

    # Wait for all requests to terminate
    MPI.Waitall!(vcat(recv_reqs, send_reqs))

    # Copy from received buffer to array
    #- Loop over dimensions
    for stencil in Iterators.product(neighbors(grid)...)
        neighbor = coords .+ stencil

        # Jump to next neighbor if the present neighbor is not a real one
        _is_true_neighbor(neighbor, stencil, grid) || continue

        # Execute request
        src = get_rank(neighbor, grid)
        buffer = recv_buffer[src]
        _buffer2arrays!(buffer, arrays, stencil, grid)
    end
end

"""
Copy a part of the `arrays` (same for all arrays) into the sending `buffer`. The target is defined by the stencil.
"""
function _arrays2buffer!(buffer::Vector{T}, arrays::NTuple{N,AbstractArray{T,D}}, stencil::NTuple{D,Int}, grid::DistributedGrid{D}) where {D,T,N}
    n = get_nelts(grid)
    noverlaps = get_noverlaps(grid)

    # Gather all elements index to copy into buffer
    rh = ntuple(
        d ->
        stencil[d] < 0 ? range(noverlaps[d] + 1; length = noverlaps[d]) :
        stencil[d] > 0 ? range(n[d] + 1, n[d] + noverlaps[d]) :
        local_indices(grid,d),
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
function _buffer2arrays!(buffer::Vector{T}, arrays::NTuple{N,AbstractArray{T,D}}, stencil::NTuple{D,Int}, grid::DistributedGrid{D}) where {D,T,N}
    nelts_h = nelts_with_halo(grid)

    # Gather all elements index to copy into array
    rh = ntuple(
        d ->
        stencil[d] < 0 ? range(1,get_noverlap(grid,d)) :
        stencil[d] > 0 ? range(nelts_h[d] - get_noverlap(grid,d) + 1, nelts_h[d]) :
        local_indices(grid, d),
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
Indicate if the `neighbor`, obtained with the given `stencil`, is a "true" neighbor : i.e inside the global grid
and different from the local grid.
"""
function _is_true_neighbor(neighbor::NTuple{D,Int}, stencil::NTuple{D,Int}, grid::DistributedGrid{D}) where D
    return all(d -> 1 <= neighbor[d] <= get_ndims(grid,d), 1:D) && any(d -> stencil[d] != 0, 1:D)
end

finalize_grid(::DistributedGrid; finalize_MPI = true) = finalize_MPI && MPI.Finalize()