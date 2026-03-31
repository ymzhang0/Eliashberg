using Eliashberg
using StaticArrays
using CairoMakie

# Function to compute visualization data, render, and save it
function test_viz(name, model, grid; kwargs...)
    println("Testing $name...")
    try
        fig = if grid isa KPath
            band_data = compute_path_band_data(model, grid)
            plot_band_structure(grid, band_data.bands; kwargs...)
        elseif grid isa KGrid{2}
            surface_data = compute_dispersion_surface_data(model, grid)
            plot_dispersion_surface(surface_data.kxs, surface_data.kys, surface_data.energy_matrix; kwargs...)
        elseif grid isa KGrid{3}
            fermi_surface = compute_fermi_surface_volume(model; n_pts=60)
            plot_fermi_surface(
                fermi_surface.kxs,
                fermi_surface.kys,
                fermi_surface.kzs,
                fermi_surface.energy_volume;
                kwargs...
            )
        elseif grid isa KGrid{1}
            curve_data = compute_dispersion_curve_data(model, grid)
            plot_dispersion_curves(curve_data.coordinates, curve_data.bands; kwargs...)
        else
            throw(ArgumentError("Unsupported grid type $(typeof(grid)) for visualization test."))
        end
        save("$(name).png", fig)
        println("  SUCCESS: Saved $(name).png")
    catch e
        println("  FAILURE: $name failed with error: $e")
        rethrow(e)
    end
end

# 1. Test 1D Visualization
# model = TightBinding(ChainLattice(1.0), 1.0, 0.0)
# N = 100
# ks = range(-π, π, length=N)
# grid1d = KGrid([SVector{1}(k) for k in ks], ones(N)/N)
# test_viz("viz_1d", model, grid1d; E_Fermi=0.5)

# 2. Test 2D Visualization
lattice2d = SquareLattice(1.0)
model2d = TightBinding(lattice2d, 1.0, 0.2, 0.0)
N2 = 30
ks2 = range(-π, π, length=N2)
grid2d = KGrid([SVector{2}(kx, ky) for kx in ks2 for ky in ks2], ones(N2^2)/N2^2)
test_viz("viz_2d", model2d, grid2d; E_Fermi=-0.5)

# 3. Test 3D KPath (Band Structure)
lattice3d = CubicLattice(1.0)
model3d = TightBinding(lattice3d, 1.0, 0.0)
nodes = [SVector{3}(0,0,0), SVector{3}(π,0,0), SVector{3}(π,π,0), SVector{3}(0,0,0)]
labels = ["Γ", "X", "M", "Γ"]
kpath = generate_kpath(nodes, labels; n_pts_per_segment=20)
test_viz("viz_3d_kpath", model3d, kpath; E_Fermi=1.0)

# 4. Test 3D KGrid (Isosurface)
N3 = 20
ks3 = range(-π, π, length=N3)
grid3d = KGrid([SVector{3}(kx, ky, kz) for kx in ks3 for ky in ks3 for kz in ks3], ones(N3^3)/N3^3)
test_viz("viz_3d_isosurface", model3d, grid3d; E_Fermi=0.0)

println("\nAll visualizations tested.")
