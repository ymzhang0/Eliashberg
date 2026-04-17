# src/Models/calculator.jl

using ..Eliashberg
using StaticArrays
using LinearAlgebra

"""
    calculate_bands(model::Dispersion, sampling)

Unified interface for calculating band structures, energy surfaces, and volumes.
Dispatch is based on the type and dimensionality of `sampling`.

# Samplings
- `KPath`: Returns `BandStructureData`.
- `AbstractKGrid{2}`: Returns `DispersionSurfaceData`.
- `AbstractKGrid{3}`: Returns `FermiSurfaceData`.
"""
function calculate_bands end

# --- 1. Band Structure over a K-Path ---

function calculate_bands(disp::Dispersion, kpath::KPath{D}) where {D}
    return with_stage_log(
        "Calculate bands (KPath)";
        context=(model=disp, kpath=kpath),
        summarize_result=identity,
    ) do
        points = path_points(kpath)
        # Calculate raw band data
        bands_raw = [ε(k, disp) for k in points]
        
        # Determine number of bands from first point
        num_bands = (bands_raw[1] isa AbstractVector) ? length(bands_raw[1]) : 1
        
        # Reshape to Matrix {k_points x num_bands}
        band_matrix = fill(NaN, length(points), num_bands)
        for band_idx in 1:num_bands
            if num_bands == 1
                band_matrix[:, 1] = [v for v in bands_raw]
            else
                band_matrix[:, band_idx] = [v[band_idx] for v in bands_raw]
            end
        end

        return BandStructureData(
            kpath=kpath,
            bands=band_matrix,
            num_bands=num_bands
        )
    end
end

# --- 2. Energy Surface over a 2D K-Grid ---

function calculate_bands(disp::Dispersion, kgrid::AbstractKGrid{2})
    return with_stage_log(
        "Calculate bands (2D Surface)";
        context=(model=disp, grid=kgrid),
        summarize_result=data -> (nx=length(data.kxs), ny=length(data.kys)),
    ) do
        # Extract axes for a Cartesian-aligned grid
        kxs = unique(sort([k[1] for k in kgrid.points]))
        kys = unique(sort([k[2] for k in kgrid.points]))
        
        # Mappings for matrix insertion
        ix = Dict(kx => idx for (idx, kx) in enumerate(kxs))
        iy = Dict(ky => idx for (idx, ky) in enumerate(kys))
        
        energy_matrix = zeros(Float64, length(kxs), length(kys))

        for k in kgrid.points
            val = ε(k, disp)
            # Surface data typically plots the first band
            energy_matrix[ix[k[1]], iy[k[2]]] = (val isa AbstractVector) ? val[1] : val
        end

        return DispersionSurfaceData(kxs, kys, energy_matrix)
    end
end

# --- 3. Energy Volume over a 3D K-Grid ---

function calculate_bands(disp::Dispersion, kgrid::AbstractKGrid{3})
    return with_stage_log(
        "Calculate bands (3D Volume)";
        context=(model=disp, grid=kgrid),
        summarize_result=data -> (nx=length(data.kxs), ny=length(data.kys), nz=length(data.kzs)),
    ) do
        kxs = unique(sort([k[1] for k in kgrid.points]))
        kys = unique(sort([k[2] for k in kgrid.points]))
        kzs = unique(sort([k[3] for k in kgrid.points]))
        
        ix = Dict(kx => idx for (idx, kx) in enumerate(kxs))
        iy = Dict(ky => idx for (idx, ky) in enumerate(kys))
        iz = Dict(kz => idx for (idx, kz) in enumerate(kzs))
        
        energy_volume = zeros(Float32, length(kxs), length(kys), length(kzs))

        for k in kgrid.points
            val = ε(k, disp)
            energy_volume[ix[k[1]], iy[k[2]], iz[k[3]]] = Float32((val isa AbstractVector) ? val[1] : val)
        end

        return FermiSurfaceData(kxs, kys, kzs, energy_volume)
    end
end

# --- 4. High-level Fermi Surface Helper ---

"""
    calculate_fermi_surface(model::Dispersion, system::AbstractSystem; n_pts::Integer=100)

High-level interface to calculate the 3D Fermi surface volume of a system.
Constructs a Cartesian bounding box in reciprocal space and samples it.
"""
function calculate_fermi_surface(disp::Dispersion{3}, system::AbstractSystem; n_pts::Integer=100)
    # Determine sampling range from reciprocal lattice
    B = reciprocal_lattice(system)
    
    # Calculate bounding box vertices
    vertices = [B * SVector{3,Float64}(i, j, k) for i in 0:1, j in 0:1, k in 0:1]
    k_min = SVector{3,Float64}(minimum(v[i] for v in vertices) for i in 1:3)
    k_max = SVector{3,Float64}(maximum(v[i] for v in vertices) for i in 1:3)
    
    # Expand slightly to avoid boundary cutoff
    pad = (k_max - k_min) * 0.05
    kxs = range(k_min[1] - pad[1], k_max[1] + pad[1], length=n_pts)
    kys = range(k_min[2] - pad[2], k_max[2] + pad[2], length=n_pts)
    kzs = range(k_min[3] - pad[3], k_max[3] + pad[3], length=n_pts)
    
    # Create Cartesian Grid points
    points = [SVector{3,Float64}(kx, ky, kz) for kx in kxs, ky in kys, kz in kzs]
    grid = KGrid(vec(points), fill(1.0/length(points), length(points)))
    
    return calculate_bands(disp, grid)
end
