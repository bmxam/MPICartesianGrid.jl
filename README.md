# MPICartesianGrid.jl
Distributed structured grid with MPI support. Check out the example(s) in the `example/` folder.

## Limitation(s)
* the function `update_halo!` only supports whose size matches the number of grid elements (but quite easy to improve to any array size)