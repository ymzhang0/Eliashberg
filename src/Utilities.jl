module Utilities

using StaticArrays
using ..Eliashberg: KGrid

export generate_1d_kgrid, generate_2d_kgrid, generate_3d_kgrid

"""
    generate_1d_kgrid(Nk::Int; kmin=-π, kmax=π)

Generate a 1D k-grid with `Nk` points from `kmin` to `kmax`.
"""
function generate_1d_kgrid(Nk::Int; kmin=-π, kmax=π)
    points = [SVector{1, Float64}(k) for k in range(kmin, kmax, length=Nk)]
    weights = fill(1.0/Nk, Nk)
    return KGrid(points, weights)
end

"""
    generate_2d_kgrid(Nx::Int, Ny::Int; kmin=(-π, -π), kmax=(π, π))

Generate a 2D k-grid with `Nx` * `Ny` points from `kmin` to `kmax`.
"""
function generate_2d_kgrid(Nx::Int, Ny::Int; kmin=(-π, -π), kmax=(π, π))
    points = [SVector{2, Float64}(kx, ky) 
              for kx in range(kmin[1], kmax[1], length=Nx) 
              for ky in range(kmin[2], kmax[2], length=Ny)]
    weights = fill(1.0/(Nx*Ny), Nx*Ny)
    return KGrid(points, weights)
end

"""
    generate_3d_kgrid(Nx::Int, Ny::Int, Nz::Int; kmin=(-π, -π, -π), kmax=(π, π, π))

Generate a 3D k-grid with `Nx` * `Ny` * `Nz` points from `kmin` to `kmax`.
"""
function generate_3d_kgrid(Nx::Int, Ny::Int, Nz::Int; kmin=(-π, -π, -π), kmax=(π, π, π))
    points = [SVector{3, Float64}(kx, ky, kz) 
              for kx in range(kmin[1], kmax[1], length=Nx) 
              for ky in range(kmin[2], kmax[2], length=Ny)
              for kz in range(kmin[3], kmax[3], length=Nz)]
    weights = fill(1.0/(Nx*Ny*Nz), Nx*Ny*Nz)
    return KGrid(points, weights)
end

end # module Utilities
