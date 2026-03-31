using Eliashberg
using StaticArrays
using CairoMakie

# 1. Define a 3D Tight-Binding model for a simple cubic lattice
# -2 * t * (cos(kx) + cos(ky) + cos(kz)) - EF
# t = 1.0, EF = 0.0 (Half-filling)
lattice = CubicLattice(1.0)
model = TightBinding(lattice, 1.0, 0.0)

# 2. Generate a 50x50x50 KGrid in the Brillouin Zone [-π, π]^3
# This represents a dense volume mesh as required.
N = 50
ks = range(-π, π, length=N)
points = [SVector{3}(kx, ky, kz) for kx in ks for ky in ks for kz in ks]
weights = ones(length(points)) / length(points)
kgrid = KGrid(points, weights)

# 3. Visualize the Fermi Surface (E = 0.0)
# Compute the volume data first, then render it through the pure plotting layer.
println("Generating 3D Fermi Surface plot...")
fermi_surface = compute_fermi_surface_volume(model; n_pts=60)
fig = plot_fermi_surface(
    fermi_surface.kxs,
    fermi_surface.kys,
    fermi_surface.kzs,
    fermi_surface.energy_volume;
    E_Fermi=0.0
)

# 4. Display or save
# display(fig) # Uncomment if running in an interactive session
save("fermi_surface_3d.png", fig)
println("Saved Fermi Surface plot to fermi_surface_3d.png")
