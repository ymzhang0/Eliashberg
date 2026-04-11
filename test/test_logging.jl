using Test
using Logging
using Eliashberg

@testset "Logging Hooks" begin
    lattice = ChainLattice(1.0)
    model = TightBinding(lattice, 1.0, 0.0)
    grid = generate_1d_kgrid(4)
    path = generate_kpath(
        [first(grid.points), last(grid.points)],
        ["-π", "π"];
        n_pts_per_segment=length(grid.points) - 1,
    )
    interaction = ConstantInteraction(0.5)

    band_data = @test_logs (
        :info,
        r"Compute band data started",
    ) (
        :info,
        r"Compute band data finished",
    ) compute_band_data(model, path)
    @test band_data.num_bands == 1

    sparse_bcs = @test_logs min_level=Logging.Warn match_mode=:any (
        :warn,
        r"Sparse BCS matrix is using dense eigensolver fallback",
    ) solve_bcs(grid, model, interaction; matrix_format=:sparse)
    @test length(first(sparse_bcs)) == length(grid)

    phi = @test_logs min_level=Logging.Debug match_mode=:any (
        :debug,
        r"Solve ground state iteration",
    ) solve_ground_state(BCSReducedPairing(), model, interaction, grid, ExactTrLn(); phi_guess=0.2, T=0.1)
    @test phi isa Real
end
