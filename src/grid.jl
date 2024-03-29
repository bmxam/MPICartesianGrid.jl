"""
A distributed cartesian grid. `D` is the number of spatial dimensions.

Remark : the structure could lighter if we interrogate MPI to query infos (rank, etc) instead of storing
them here.
"""
struct DistributedGrid{D}
    comm::MPI.Comm
    ndims::NTuple{D,Int} # Number of process in each dimension
    coords::NTuple{D,Int} # Coordinates in the global grid (1-based)
    nelts::NTuple{D,Int} # Number of elts of the local grid in each direction, without halo
    noverlaps::NTuple{D,Int} # Number of overlap elements in each direction
    coords2rank::Array{Int,D} # MPI coordinates (1-based) to MPI rank (identical to MPI.Cart_rank)
end

@inline get_comm(grid::DistributedGrid) = grid.comm
@inline get_coords(grid::DistributedGrid) = grid.coords
@inline get_ndims(grid::DistributedGrid) = grid.ndims
@inline get_ndims(grid::DistributedGrid, d::Int) = grid.ndims[d]

@inline get_noverlaps(grid::DistributedGrid) = grid.noverlaps
@inline get_noverlap(grid::DistributedGrid, d::Int) = grid.noverlaps[d]

@inline get_rank(grid::DistributedGrid) = MPI.Comm_rank(get_comm(grid))
@inline get_rank(coords::Vector{Int}, grid::DistributedGrid) = grid.coords2rank[coords]
@inline get_rank(coords::NTuple{D,Int}, grid::DistributedGrid{D}) where {D} = grid.coords2rank[coords...]

"""
    neighbors(::DistributedGrid{D}) where D

Iterators with neighbors stencil (in local reference, i.e `-1:1`) including the current position.
"""
@inline neighbors(::DistributedGrid{D}) where {D} = Iterators.product(ntuple(d -> -1:1, D)...)

@inline get_nelts(grid::DistributedGrid) = grid.nelts
@inline get_nelts(grid::DistributedGrid, d::Int) = grid.nelts[d]

@inline nelts_with_halo(grid::DistributedGrid) = grid.nelts .+ 2 .* grid.noverlaps
@inline nelts_with_halo(grid::DistributedGrid, d::Int) = grid.nelts[d] + 2 * grid.noverlaps[d]

@inline local_indices(grid::DistributedGrid{D}) where {D} = ntuple(d -> range(grid.noverlaps[d] + 1, grid.nelts[d] + grid.noverlaps[d]), D)
@inline local_indices(grid::DistributedGrid{D}, d::Int) where {D} = range(grid.noverlaps[d] + 1, grid.nelts[d] + grid.noverlaps[d])

"""
    DistributedGrid(ndims::NTuple{D,Int}, nelts::NTuple{D,Int}, noverlaps::NTuple{D,Int}; init_MPI = true) where D

`D` is the number of topological dimensions. `ndims` is the number of processors in each topological dimension. `nelts` is the number of grid elements
in each topological dimension on each core (hence the total number of grid elements is `sum(ndims .* nelts)`). Finally,
`noverlaps` is the number of elements overlaps (on both side of the topological direction) in each topological direction.
"""
function DistributedGrid(ndims::NTuple{D,Int}, nelts::NTuple{D,Int}, noverlaps::NTuple{D,Int}; init_MPI=true) where {D}
    # Init MPI if necessary
    init_MPI && MPI.Init()

    # Preliminary check
    navail = MPI.Comm_size(MPI.COMM_WORLD)
    nexpected = prod(ndims)
    @assert navail == nexpected "The existed number of processors is $navail, the expected number is $nexpected (according to `ndims`)"

    # Create comm
    #comm = MPI.Cart_create(MPI.COMM_WORLD, [d for d in ndims]) # future version of MPI.jl
    comm = MPI.Cart_create(MPI.COMM_WORLD, [d for d in ndims], [0 for _ in ndims], false)

    # Build grid
    DistributedGrid(comm, ndims, nelts, noverlaps)
end
DistributedGrid(ndims::Int, nelts::Int, noverlaps::Int; init_MPI=true) = DistributedGrid((ndims,), (nelts,), (noverlaps,); init_MPI)

function DistributedGrid(comm::MPI.Comm, ndims::NTuple{D,Int}, nelts::NTuple{D,Int}, noverlaps::NTuple{D,Int}) where {D}
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
DistributedGrid(comm::MPI.Comm, ndims::Int, nelts::Int, noverlaps::Int; init_MPI=true) = DistributedGrid(comm, (ndims,), (nelts,), (noverlaps,); init_MPI)

"""
    create_buffers(grid::DistributedGrid{D}, type::Type = Float64, narrays::Int = 1) where D

Create send/recv buffers for MPI exchange of `Array`'s of type `type`.

`narrays` is the number of arrays that will be exchanged.
"""
function create_buffers(grid::DistributedGrid{D}, type::Type=Float64, narrays::Int=1) where {D}
    coords = get_coords(grid)
    recv_buffer = Dict{Int,Vector{type}}()

    #- Loop over dimensions
    for stencil in neighbors(grid)
        neighbor = coords .+ stencil

        # Jump to next neighbor if the present neighbor is not a real one
        _is_true_neighbor(neighbor, stencil, grid) || continue

        # Compute buffer size
        # For each dimension:
        # * if the `stencil[d] == 0`, i.e the considered neighbor is,
        #   for this dimension, on the same "level", then the buffer size equals the number
        #   of elements in this direction.
        # * otherwise, the buffer is just the overlap
        # Then the final size is the product of the different buffer sizes.
        #
        # Draw the situation in 2D to understand :
        # * for neighbors in the x-direction (left/right neighbors), the number of elements to exchange
        #   in the y-direction is ny. However for the number of elements to exhange in the x-direction is
        #   just the overlap in the x-direction.
        # * same way for neighbors in x-direction
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
    update_halo!((array,), grid)
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
    for stencil in neighbors(grid)
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
    for stencil in neighbors(grid)
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
    for stencil in neighbors(grid)
        neighbor = coords .+ stencil

        # Jump to next neighbor if the present neighbor is not a real one
        _is_true_neighbor(neighbor, stencil, grid) || continue

        # Copy to array from received buffer
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
            stencil[d] < 0 ? range(noverlaps[d] + 1; length=noverlaps[d]) :
            stencil[d] > 0 ? range(n[d] + 1, n[d] + noverlaps[d]) :
            local_indices(grid, d),
        D
    )

    # Copy to buffer. Depending on the stencil, there might be zero, one, several elements of each dimension
    # of each array to copy in the buffer. Array elements are interlaced this way in the buffer:
    # buffer = [array1[1,2], array2[1,2], array3[1,2], array1[7,4], array2[7,4], array3[7,4], ....]
    for (i, ind) in enumerate(Iterators.product(rh...))
        for (iarray, array) in enumerate(arrays)
            buffer[iarray+(i-1)*N] = array[ind...]
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
            stencil[d] < 0 ? range(1, get_noverlap(grid, d)) :
            stencil[d] > 0 ? range(nelts_h[d] - get_noverlap(grid, d) + 1, nelts_h[d]) :
            local_indices(grid, d),
        D
    )

    # Copy from buffer. Array elements are interlaced this way in the buffer:
    # buffer = [array1[1,2], array2[1,2], array3[1,2], array1[7,4], array2[7,4], array3[7,4], ....]
    for (i, ind) in enumerate(Iterators.product(rh...))
        for (iarray, array) in enumerate(arrays)
            array[ind...] = buffer[iarray+(i-1)*N]
        end
    end
end

"""
Indicate if the `neighbor`, obtained with the given `stencil`, is a "true" neighbor : i.e inside the global grid
and different from the local grid.
"""
function _is_true_neighbor(neighbor::NTuple{D,Int}, stencil::NTuple{D,Int}, grid::DistributedGrid{D}) where {D}
    return all(d -> 1 <= neighbor[d] <= get_ndims(grid, d), 1:D) && any(d -> stencil[d] != 0, 1:D)
end

finalize_grid(::DistributedGrid; finalize_MPI=true) = finalize_MPI && MPI.Finalize()

"""
    gather_array(a, grid::DistributedGrid, root::Integer = 0)

Establish parallel communication to gather a distributed array (or a tuple
of distributed arrays) `a` on the `root` processor.
"""
function gather_array(array::AbstractArray{T,D}, grid::DistributedGrid{D}, root::Integer=0) where {T,D}

    # gather data on root
    all_nelts = MPI.Gather(get_nelts(grid), root, get_comm(grid))
    all_coords = MPI.Gather(get_coords(grid), root, get_comm(grid))
    if get_rank(grid) == root
        # limitation : we use `Gather` for now, then we must check that
        # all local array have the same size.
        if !all(map(x -> all_nelts[1] == x, all_nelts))
            @show all_nelts
            error("ERROR: all local arrays must have the same size.")
        end
        neltsGlobal = _compute_nelts_total(all_nelts, all_coords, grid)
        arrayGlobal = zeros(eltype(array), neltsGlobal...)
    end

    # `Gather` is valid if all local array have the same size.
    # We should implement `Gatherv` in the general case.
    buffer = MPI.Gather(array[local_indices(grid)...], root, get_comm(grid))

    if get_rank(grid) == root
        offset_buffer = 0
        globalIndices = _global_indices(all_nelts, all_coords, grid)
        for rank in 1:length(all_nelts)
            current_size = prod(all_nelts[rank])
            arrayGlobal[globalIndices[all_coords[rank]...]...] .= reshape(buffer[offset_buffer.+(1:current_size)], all_nelts[rank]...)
            offset_buffer = offset_buffer + current_size
        end
        return arrayGlobal
    else
        return nothing
    end
end

function gather_array(t::NTuple{N,<:AbstractArray}, args...) where {N}
    map(a -> gather_array(a, args...), t)
end

function _compute_nelts_per_rank_per_dim(neltsPerRank::AbstractVector{<:NTuple{D}}, coordsPerRank::AbstractVector{<:NTuple{D}}, grid::DistributedGrid{D}) where {D}
    map(1:D) do idim
        _nelts = zeros(Int, get_ndims(grid))
        for (coord_k, nelts_k) in zip(coordsPerRank, neltsPerRank)
            _nelts[coord_k...] = nelts_k[idim]
        end
        _nelts
    end
end

"""
Return the size of the global array obtain by gathering
"""
function _compute_nelts_total(neltsPerRank::AbstractVector{<:NTuple{D}}, coordsPerRank::AbstractVector{<:NTuple{D}}, grid::DistributedGrid{D}) where {D}
    neltsPerRankPerDim = _compute_nelts_per_rank_per_dim(neltsPerRank, coordsPerRank, grid)
    map(1:D) do idim
        sum(neltsPerRankPerDim[idim], dims=idim)[1]
    end
end

function _global_indices(neltsPerRank::AbstractVector{<:NTuple{D}}, coordsPerRank::AbstractVector{<:NTuple{D}}, grid::DistributedGrid{D}) where {D}
    neltsPerRankPerCoord = _compute_nelts_per_rank_per_dim(neltsPerRank, coordsPerRank, grid)
    rangePerRankPerCoord = zero.(neltsPerRankPerCoord)
    globalIndicePerCoord = Array{NTuple{D,UnitRange}}(undef, size(neltsPerRankPerCoord[1])...)
    for (idim, (_nelts, _range)) in enumerate(zip(neltsPerRankPerCoord, rangePerRankPerCoord))
        _range .= accumulate(+, _nelts, dims=idim)
    end
    for i in eachindex(globalIndicePerCoord)
        globalIndicePerCoord[i] = ntuple(idim -> rangePerRankPerCoord[idim][i]-neltsPerRankPerCoord[idim][i]+1:rangePerRankPerCoord[idim][i], Val(D))
    end
    globalIndicePerCoord
end
