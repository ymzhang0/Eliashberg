using Pkg
Pkg.activate(".")
using StaticArrays
using LinearAlgebra
using Eliashberg

println("Testing Eliashberg dimensionality-agnostic refactor...")

# 1. Initialize a 2D KGrid
N = 4
kx = range(-π, π, length=N)
ky = range(-π, π, length=N)
points = [SVector{2, Float64}(x, y) for x in kx for y in ky]
weights = fill((2π/N)^2, length(points))

grid = KGrid{2}(points, weights)
println("Initialized 2D KGrid with ", length(grid), " points.")

# 2. Evaluate TightBinding{2} model
tb = TightBinding{2}(1.0, 0.0) # t=1.0, EF=0.0
k_test = SVector{2, Float64}(0.0, 0.0)
H_k = ε(k_test, tb)
println("Hamiltonian at k=(0,0): ", H_k)

bz = band_structure(tb, k_test)
println("Band energy at k=(0,0): ", bz.values)

# 3. Calculate static RPA polarization χ(q)
sm = FermiDiracSmearing(0.1, 0.0) # T=0.1, μ=0.0
polarization = StaticRPAPolarization(tb, sm)

q_test = SVector{2, Float64}(0.1, 0.1)
chi_q = χ(q_test, grid, polarization)
println("Static RPA Polarization χ(q=(0.1, 0.1)): ", chi_q)

# 4. Solve BCS gap equation with ConstantInteraction
inter = ConstantInteraction{2}(0.5)
vals, vecs = solve_bcs(grid, tb, inter)

println("BCS Eigenvalues (top 3): ", sort(vals, by=real, rev=true)[1:min(3, length(vals))])
println("Test script completed successfully!")
