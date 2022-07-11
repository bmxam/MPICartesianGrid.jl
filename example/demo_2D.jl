"""
Run this 2D-demo with 4 procs : mpirun -n 4 julia demo_2D.jl

This demo synchronizes only one array. To synchronize multiple arrays, just use the same `update_halo!` method
with a `Tuple` of `Array`.
"""
module demo
using MPICartesianGrid
using MPI # this is not necessary, only for demo display purpose

"""
MPI display is bad, this is just a display function for the demo
"""
function display_array(array, grid)
    println("\n")
    nprocs = MPI.Comm_size(grid.comm)
    for r in 0:nprocs-1
        if get_rank(grid) == r
            println(grid.coords) # print the grid coordinates in the cartesian grid
            display(transpose(array)) # display array
            println("\n")
        end
        MPI.Barrier(grid.comm)
    end
end

# Build grid
n = (4, 3) # Number of elements in each space directions on each processor
ndims = (2, 2) # Number of processors in each space direction
noverlaps = (2,1) # Number of elements overlap (on both side) in each space direction
grid = DistributedGrid(ndims, n, noverlaps)

# Build an array for demo. Only interior elements are initialized (they received a unique integer identifier)
array = zeros(Int, nelts_with_halo(grid))
irange, jrange = local_indices(grid)
nx, ny = get_nelts(grid)
for i in 1:nx, j in 1:ny
    array[irange[i],jrange[j]] = (get_rank(grid) + 1) * 100 + (j - 1) * nx + i
end

# Print the array before exchange
println("\nBefore update_halo!\n")
display_array(array, grid)

# Update halo !
update_halo!(array, grid)

# Print array after exchange
println("\nAfter update_halo!\n")
display_array(array, grid)

# The end
finalize_grid(grid)

end