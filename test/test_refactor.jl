using Test
using Eliashberg
using StaticArrays

@testset "Dimensionality-Agnostic Refactor" begin
    grid = generate_2d_kgrid(4, 4)
    @test length(grid) == 16

    lat = SquareLattice(1.0)
    model = TightBinding(lat, 1.0, 0.0, 0.0)

    k_test = SVector{2,Float64}(0.0, 0.0)
    H_k = ε(k_test, model)
    @test H_k[1, 1] ≈ -4.0

    bands = band_structure(model, k_test)
    @test bands.values[1] ≈ -4.0

    field = ChargeDensityWave(SVector{2,Float64}(0.0, 0.0))
    chi = GeneralizedSusceptibility(model, grid, field, 0.1)
    chi_q = chi(SVector{2,Float64}(0.1, 0.1))
    @test isfinite(real(chi_q))
    @test isfinite(imag(chi_q))

    landscape = scan_instability_landscape(model, grid, grid; T=0.1, η=1e-3)
    @test landscape isa Matrix{Float64}
    @test size(landscape) == (4, 4)

    inter = ConstantInteraction(0.5)
    vals, vecs = solve_bcs(grid, model, inter)
    @test length(vals) == length(grid)
    @test size(vecs) == (length(grid), length(grid))
    @test all(isfinite, real.(vals))
end
