using Test
using Eliashberg
using StaticArrays
using AtomsBase

@testset "Eliashberg.jl Core" begin
    lat = ChainLattice(1.0)
    @test lat isa PeriodicCell{1}
    @test periodicity(lat) == (true,)
    grid = KGrid([SVector{1,Float64}(0.0)], [1.0])
    @test length(grid) == 1

    model = TightBinding(lat, 1.0, 0.0)
    @test ε(SVector{1,Float64}(0.0), model).data[1] ≈ -2.0

    inter = ConstantInteraction(0.5)
    vals, vecs = solve_bcs(grid, model, inter)
    @test length(vals) == 1
    @test size(vecs) == (1, 1)
    @test real(vals[1]) ≈ -1.5

    free_electron = FreeElectron{1}(0.5)
    @test ε(SVector{1,Float64}(1.0), free_electron).data[1] ≈ 0.0
end

include("test_refactor.jl")
include("test_plot_dispatch.jl")
include("test_quantum_espresso.jl")
