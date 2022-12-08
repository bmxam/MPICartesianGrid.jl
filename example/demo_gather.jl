"""
Run this 2D-demo with 4 procs : mpirun -n 4 julia demo_2D.jl

This demo synchronizes only one array. To synchronize multiple arrays, just use the same `update_halo!` method
with a `Tuple` of `Array`.
"""
module demo
#include(string(@__DIR__, "/../src/MPICartesianGrid.jl"))
#using .MPICartesianGrid
using MPICartesianGrid
using MPI # this is not necessary, only for demo display purpose

"""
MPI display is bad, this is just a display function for the demo
"""
function display_array(array, grid)
    get_rank(grid) == 0 && println("\n")
    nprocs = MPI.Comm_size(grid.comm)
    for r in 0:nprocs-1
        if get_rank(grid) == r
            println(grid.coords) # print the grid coordinates in the cartesian grid
            if array!==nothing
                display(transpose(array)) # display array
            else
                print("nothing")
            end
            println("\n")
        end
        MPI.Barrier(grid.comm)
    end
end

# Build grid
n = (4, 3) # Number of elements in each space directions on each processor
ndims = length(ARGS) >= 2 ? parse.(Int, (ARGS[1], ARGS[2])) : (2, 2)   # Number of processors in each space direction
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
get_rank(grid) == 0 && println("\nBefore gather_array\n")
display_array(array, grid)

globalArray = gather_array(array, grid)

# Print array after exchange
get_rank(grid) == 0 &&  println("\nAfter gather_array\n")
display_array(globalArray, grid)

# The end
finalize_grid(grid)

end