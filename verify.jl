using Pkg
Pkg.activate(".")
using Eliashberg
using StaticArrays

model = TightBinding{2}(1.0, 0.0, 0.0)
grid = KGrid{2}([SVector(0.0,0.0), SVector(pi,pi)], [0.5, 0.5])

q = SVector{2, Float64}(pi, pi)
cdw = ChargeDensityWave(q)

action = EffectiveAction(model, cdw, grid, 2.0)

F_exact = evaluate(action, 0.1, ExactTrLn())
println("ExactTrLn F(0.1) = ", F_exact)

F_rpa = evaluate(action, 0.1, RPA())
println("RPA F(0.1) = ", F_rpa)

phi_gs = solve_ground_state(action, RPA(); phi_guess=0.1)
println("Ground state phi (RPA) = ", phi_gs)
