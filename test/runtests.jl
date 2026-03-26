using Test
using Eliashberg
using StaticArrays

@testset "Eliashberg.jl Core" begin
    # 1. Grid Generation
    grid = KGrid([SVector{1, Float64}(0.0)], [1.0])
    @test length(grid) == 1
    
    # 2. Dispersion
    tb = TightBinding{1}(1.0, 0.0)
    @test ε(SVector(0.0), tb).data[1] == -2.0
    
    # 3. Solver basics
    inter = ConstantInteraction{1}(0.5)
    vals, vecs = solve_bcs(grid, tb, inter)
    @test length(vals) == 1
    @test real(vals[1]) ≈ -1.5 # -2.0 + 0.5 * 1.0 (weight)
end

