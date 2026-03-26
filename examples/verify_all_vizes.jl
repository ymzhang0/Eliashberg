using Eliashberg
using StaticArrays
using CairoMakie

# Function to run a visualization and save it
function test_viz(name, model, grid; kwargs...)
    println("Testing $name...")
    try
        fig = visualize_dispersion(model, grid; kwargs...)
        save("$(name).png", fig)
        println("  SUCCESS: Saved $(name).png")
    catch e
        println("  FAILURE: $name failed with error: $e")
        rethrow(e)
    end
end

# 1. Test 1D Visualization
# model = TightBinding{1}(1.0, 0.0)
# N = 100
# ks = range(-π, π, length=N)
# grid1d = KGrid{1}([SVector{1}(k) for k in ks], ones(N)/N)
# test_viz("viz_1d", model, grid1d; E_Fermi=0.5)

# 2. Test 2D Visualization
model2d = TightBinding{2}(1.0, 0.2, 0.0)
N2 = 30
ks2 = range(-π, π, length=N2)
grid2d = KGrid{2}([SVector{2}(kx, ky) for kx in ks2 for ky in ks2], ones(N2^2)/N2^2)
test_viz("viz_2d", model2d, grid2d; E_Fermi=-0.5)

# 3. Test 3D KPath (Band Structure)
model3d = TightBinding{3}(1.0, 0.0)
nodes = [SVector{3}(0,0,0), SVector{3}(π,0,0), SVector{3}(π,π,0), SVector{3}(0,0,0)]
labels = ["Γ", "X", "M", "Γ"]
kpath = generate_kpath(nodes, labels; n_pts_per_segment=20)
test_viz("viz_3d_kpath", model3d, kpath; E_Fermi=1.0)

# 4. Test 3D KGrid (Isosurface)
N3 = 20
ks3 = range(-π, π, length=N3)
grid3d = KGrid{3}([SVector{3}(kx, ky, kz) for kx in ks3 for ky in ks3 for kz in ks3], ones(N3^3)/N3^3)
test_viz("viz_3d_isosurface", model3d, grid3d; E_Fermi=0.0)

println("\nAll visualizations tested.")
