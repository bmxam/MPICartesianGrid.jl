"""
Run this 2D-demo with 4 procs : mpirun -n 4 julia demo_2D.jl
"""
module demo
using MPIStructuredMesh
using MPI # this is not necessary, only for demo display purpose

"""
MPI display is bad, this is just a display function for the demo
"""
function display_array(array, mesh)
    println("\n")
    nprocs = MPI.Comm_size(mesh.comm)
    for r in 0:nprocs-1
        if get_rank(mesh) == r
            println(mesh.coords) # print the mesh coordinates in the cartesian grid
            display(transpose(array)) # display array
            println("\n")
        end
        MPI.Barrier(mesh.comm)
    end
end

# Build mesh
n = (4, 3) # Number of elements in each space directions
ndims = (2, 2) # Number of processors in each space direction
noverlaps = (2,1) # Number of elements overlap (on both side) in each space direction
mesh = DistributedMesh(ndims, n, noverlaps)

# Build an array for demo. Only interior elements are initialized (they received a unique integer identifier)
array = zeros(Int, nelts_with_halo(mesh))
irange, jrange = local_indices(mesh)
nx, ny = get_nelts(mesh)
for i in 1:nx, j in 1:ny
    array[irange[i],jrange[j]] = (get_rank(mesh) + 1) * 100 + (j - 1) * nx + i
end

# Print the array before exchange
println("\nBefore update_halo!\n")
display_array(array, mesh)

# Update halo !
update_halo!(array, mesh)

# Print array after exchange
println("\nAfter update_halo!\n")
display_array(array, mesh)

# The end
finalize_mesh(mesh)

end