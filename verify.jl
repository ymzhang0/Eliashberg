using Pkg
Pkg.activate(".")
using Eliashberg
using StaticArrays

println("Testing grid generation utilities...")

try
    grid1d = generate_1d_kgrid(10)
    println("1D grid generated: length $(length(grid1d))")
    
    grid2d = generate_2d_kgrid(16, 16)
    println("2D grid generated: length $(length(grid2d))")
    
    grid3d = generate_3d_kgrid(4, 4, 4)
    println("3D grid generated: length $(length(grid3d))")
    
    println("SUCCESS: Grid generation utilities are available and working.")
catch e
    println("FAILURE: Grid generation utilities missing or error: $e")
    rethrow(e)
end

# Existing verification tests
model = TightBinding{2}(1.0, 0.0, 0.0)
grid = generate_2d_kgrid(16, 16) # use new utility
q = SVector{2, Float64}(pi, pi)
cdw = ChargeDensityWave(q)
action = EffectiveAction(model, cdw, grid, 2.0)

F_exact = evaluate(action, 0.1, ExactTrLn())
println("ExactTrLn F(0.1) = ", F_exact)

F_rpa = evaluate(action, 0.1, RPA())
println("RPA F(0.1) = ", F_rpa)

phi_gs = solve_ground_state(action, RPA(); phi_guess=0.1)
println("Ground state phi (RPA) = ", phi_gs)
