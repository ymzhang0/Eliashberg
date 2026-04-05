using Test
using Eliashberg
using StaticArrays
using Distributed
using LinearAlgebra
using SparseArrays
using Makie

@testset "Dimensionality-Agnostic Refactor" begin
    grid = generate_2d_kgrid(4, 4)
    @test length(grid) == 16

    line_grid = generate_1d_kgrid(8)
    direct_integral = sum((k[1]^2) * w for (k, w) in zip(line_grid.points, line_grid.weights))
    @test integrate_grid(k -> k[1]^2, line_grid) ≈ direct_integral
    @test Eliashberg.Engine.integrate_grid(k -> k[1]^2, line_grid) ≈ direct_integral
    samples = grid_samples(line_grid)
    @test samples[1] isa GridSample
    @test samples[1].value == line_grid.points[1]
    @test samples[1].weight == line_grid.weights[1]

    mapped = distributed_map_grid((x, y) -> x + 10y, [1, 2], [3, 4, 5])
    @test mapped == [31 41 51; 32 42 52]
    @test Eliashberg.Engine.distributed_map_grid((x, y) -> x + 10y, [1, 2], [3, 4, 5]) == mapped
    assembled_vector = assemble_grid_vector(sample -> sample.index + sample.weight, samples)
    @test length(assembled_vector) == length(samples)
    assembled_matrix = assemble_grid_matrix((left, right) -> left.index == right.index ? 1.0 : 0.0, samples, samples)
    @test assembled_matrix == Matrix{Float64}(I, length(samples), length(samples))
    sparse_matrix = assemble_sparse_grid_matrix((left, right) -> left.index == right.index ? 2.0 : 0.0, samples, samples)
    @test issparse(sparse_matrix)
    @test Matrix(sparse_matrix) == 2.0 * Matrix{Float64}(I, length(samples), length(samples))

    block_layout = UniformBlockLayout(2, 2)
    dense_block_matrix = assemble_block_grid_matrix(
        (left, right) -> left.index == right.index ? [1.0 0.0; 0.0 1.0] : zeros(2, 2),
        samples,
        samples,
        block_layout
    )
    @test size(dense_block_matrix) == (2 * length(samples), 2 * length(samples))
    @test dense_block_matrix == Matrix{Float64}(I, 2 * length(samples), 2 * length(samples))

    sparse_block_matrix = assemble_sparse_block_grid_matrix(
        (left, right) -> left.index == right.index ? [3.0 0.0; 0.0 3.0] : zeros(2, 2),
        samples,
        samples,
        block_layout
    )
    @test issparse(sparse_block_matrix)
    @test Matrix(sparse_block_matrix) == 3.0 * Matrix{Float64}(I, 2 * length(samples), 2 * length(samples))

    variable_layout = VariableBlockLayout([1, 2, 1], [2, 1, 2])
    variable_row_axis = samples[1:3]
    variable_col_axis = samples[1:3]
    dense_variable_blocks = assemble_block_grid_matrix(
        (left, right) -> begin
            row_size = variable_layout.row_axis.block_sizes[left.index]
            col_size = variable_layout.col_axis.block_sizes[right.index]
            left.index == right.index ? fill(Float64(left.index), row_size, col_size) : zeros(row_size, col_size)
        end,
        variable_row_axis,
        variable_col_axis,
        variable_layout
    )
    @test size(dense_variable_blocks) == (sum(variable_layout.row_axis.block_sizes), sum(variable_layout.col_axis.block_sizes))
    @test dense_variable_blocks[1, 1] == 1.0
    @test dense_variable_blocks[2:3, 3] == fill(2.0, 2)
    @test dense_variable_blocks[4, 4:5] == fill(3.0, 2)

    sparse_variable_blocks = assemble_sparse_block_grid_matrix(
        (left, right) -> begin
            row_size = variable_layout.row_axis.block_sizes[left.index]
            col_size = variable_layout.col_axis.block_sizes[right.index]
            left.index == right.index ? fill(Float64(2 * left.index), row_size, col_size) : zeros(row_size, col_size)
        end,
        variable_row_axis,
        variable_col_axis,
        variable_layout
    )
    @test issparse(sparse_variable_blocks)
    @test Matrix(sparse_variable_blocks)[1, 1] == 2.0
    @test Matrix(sparse_variable_blocks)[2:3, 3] == fill(4.0, 2)
    @test Matrix(sparse_variable_blocks)[4, 4:5] == fill(6.0, 2)

    dense_block_diagonal = assemble_block_diagonal_matrix(
        sample -> [Float64(sample.index) 0.0; 0.0 Float64(sample.index)],
        samples,
        UniformBlockLayout(2, 2)
    )
    @test size(dense_block_diagonal) == (2 * length(samples), 2 * length(samples))
    @test dense_block_diagonal[1:2, 1:2] == [1.0 0.0; 0.0 1.0]
    @test dense_block_diagonal[3:4, 1:2] == zeros(2, 2)

    sparse_block_diagonal = assemble_sparse_block_diagonal_matrix(
        sample -> [Float64(sample.index) 0.0; 0.0 Float64(sample.index)],
        samples,
        UniformBlockLayout(2, 2)
    )
    @test issparse(sparse_block_diagonal)
    @test Matrix(sparse_block_diagonal) == dense_block_diagonal

    dense_spectrum = solve_assembled_eigensystem(assembled_matrix)
    @test dense_spectrum isa AssemblySpectrum
    @test length(dense_spectrum.values) == length(samples)

    sparse_hook = SparseEigenSolverHook((matrix; kwargs...) -> eigen(Matrix(matrix)))
    sparse_spectrum = solve_assembled_eigensystem(sparse_matrix; solver=sparse_hook)
    @test sparse_spectrum isa AssemblySpectrum
    @test length(sparse_spectrum.values) == length(samples)

    graphene = Graphene(1.0, 0.0)
    graphene_assembly = assemble_sampled_hamiltonian(grid, graphene)
    @test graphene_assembly isa SampledHamiltonianAssembly
    @test graphene_assembly.layout isa VariableBlockLayout
    @test size(graphene_assembly.matrix) == (2 * length(grid), 2 * length(grid))

    graphene_spectrum = solve_sampled_hamiltonian(grid, graphene)
    @test graphene_spectrum isa AssemblySpectrum
    @test length(graphene_spectrum.values) == 2 * length(grid)

    one_d_model = TightBinding(ChainLattice(1.0), 1.0, 0.0)
    bdg_dispersion = MeanFieldDispersion(one_d_model, BCSReducedPairing(), 0.2)
    bdg_assembly = assemble_sampled_hamiltonian(line_grid, bdg_dispersion; matrix_format=:sparse)
    @test bdg_assembly.layout isa VariableBlockLayout
    @test issparse(bdg_assembly.matrix)
    @test size(bdg_assembly.matrix) == (2 * length(line_grid), 2 * length(line_grid))

    bdg_spectrum = solve_sampled_hamiltonian(line_grid, bdg_dispersion; matrix_format=:sparse, eigensolver=sparse_hook)
    @test bdg_spectrum isa AssemblySpectrum
    @test length(bdg_spectrum.values) == 2 * length(line_grid)

    existing_workers = Set(Distributed.workers())
    bootstrapped_workers = Int[]
    try
        bootstrapped_workers = bootstrap_engine_workers!(1)
        @test length(bootstrapped_workers) >= 1

        mapped_with_bootstrap = distributed_map_grid(
            (x, y) -> x + 10y,
            [1, 2],
            [3, 4, 5];
            bootstrap_workers=true,
            n_workers=1
        )
        @test mapped_with_bootstrap == mapped
    finally
        new_workers = setdiff(bootstrapped_workers, collect(existing_workers))
        !isempty(new_workers) && Distributed.rmprocs(new_workers)
    end

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

    spinor_line_model = SpinorDispersion(one_d_model)
    spinor_hk = ε(SVector{1,Float64}(0.1), spinor_line_model)
    bare_energy = ε(SVector{1,Float64}(0.1), one_d_model)[1, 1]
    @test size(spinor_hk) == (2, 2)
    @test Matrix(spinor_hk) ≈ [bare_energy 0.0; 0.0 bare_energy]

    @test vertex_matrix(one_d_model, DirectChannel()) == 1.0
    @test vertex_matrix(one_d_model, ExchangeChannel()) == 1.0
    @test_throws ArgumentError vertex_matrix(one_d_model, ExchangeChannel(:x))
    @test Matrix(vertex_matrix(spinor_line_model, DirectChannel())) == Matrix(σ₀)
    @test Matrix(vertex_matrix(spinor_line_model, ExchangeChannel(:x))) == Matrix(σ₁)
    @test Matrix(vertex_matrix(spinor_line_model, ExchangeChannel(:y))) == Matrix(σ₂)
    @test Matrix(vertex_matrix(spinor_line_model, ExchangeChannel(:z))) == Matrix(σ₃)

    transverse_chi = GeneralizedSusceptibility(spinor_line_model, line_grid, ExchangeChannel(:transverse), 0.1)
    transverse_val = transverse_chi(SVector{1,Float64}(0.2))
    @test isfinite(real(transverse_val))
    @test isfinite(imag(transverse_val))

    sdw = SpinDensityWave(SVector{1,Float64}(0.2), :x)
    sdw_basis = normal_state_basis(one_d_model, sdw)
    @test sdw_basis isa Eliashberg.ParticleHoleNormalDispersion
    @test sdw_basis.bare isa SpinorDispersion
    sdw_dispersion = MeanFieldDispersion(one_d_model, sdw, 0.2)
    @test size(ε(SVector{1,Float64}(0.1), sdw_dispersion)) == (4, 4)

    landscape = scan_instability_landscape(model, grid, grid; T=0.1, η=1e-3)
    @test landscape isa Matrix{Float64}
    @test size(landscape) == (4, 4)

    qpath = generate_kpath(lat; n_pts_per_segment=4)
    omegas = collect(range(0.0, 1.0, length=5))
    spectral = scan_spectral_function(model, grid, qpath, omegas; T=0.1, η=0.02)
    @test size(spectral) == (length(qpath), length(omegas))
    spectral_fig = plot_spectral_function(qpath, omegas, spectral)
    @test spectral_fig isa Figure

    line_path = KPath(line_grid.points, line_grid.weights, [1, length(line_grid.points)], ["-π", "π"])
    line_band_data = compute_band_data(one_d_model, line_path)
    @test size(line_band_data.bands, 1) == length(line_grid)
    dispersion_fig = plot(line_band_data)
    @test dispersion_fig isa Figure

    path = generate_kpath(ChainLattice(1.0); n_pts_per_segment=4)
    path_band_data = compute_band_data(one_d_model, path)
    @test size(path_band_data.bands, 1) == length(path)
    band_fig = plot(path_band_data)
    @test band_fig isa Figure

    line_landscape_data = compute_landscape_line_data(line_grid, collect(1.0:length(line_grid)))
    @test length(line_landscape_data.qs) == length(line_grid)
    landscape_fig = plot_landscape(Val(1), line_landscape_data.qs, line_landscape_data.values)
    @test landscape_fig isa Figure

    phase_data = compute_phase_transition_data(
        collect(range(0.0, 0.2, length=3)),
        [0.1, 0.2],
        BCSReducedPairing(:s_wave),
        one_d_model,
        ConstantInteraction(-1.0),
        line_grid
    )
    @test size(phase_data.condensation_energy) == (3, 2)
    phase_fig = plot(phase_data)
    @test phase_fig isa Figure

    renormalized_data = compute_renormalized_band_data(
        [0.1],
        BCSReducedPairing(:s_wave),
        one_d_model,
        ConstantInteraction(-1.0),
        line_grid,
        path
    )
    @test size(renormalized_data.bare_bands, 1) == length(path)
    @test size(renormalized_data.renormalized_bands, 1) == length(path)
    @test size(renormalized_data.renormalized_bands, 3) == 1
    renormalized_fig = plot(renormalized_data)
    @test renormalized_fig isa Figure

    collective_data = compute_collective_mode_spectral_data(
        0.1,
        BCSReducedPairing(:s_wave),
        one_d_model,
        ConstantInteraction(-1.0),
        line_grid,
        path;
        n_omegas=4
    )
    @test size(collective_data.spectral_matrix) == (length(path), 4)
    collective_fig = plot(collective_data)
    @test collective_fig isa Figure

    inter = ConstantInteraction(0.5)
    vals, vecs = solve_bcs(grid, model, inter)
    @test length(vals) == length(grid)
    @test size(vecs) == (length(grid), length(grid))
    @test all(isfinite, real.(vals))

    vals_bootstrapped, vecs_bootstrapped = solve_bcs(grid, model, inter; bootstrap_workers=true, n_workers=1)
    @test length(vals_bootstrapped) == length(grid)
    @test size(vecs_bootstrapped) == size(vecs)

    vals_sparse, vecs_sparse = solve_bcs(
        grid,
        model,
        inter;
        matrix_format=:sparse,
        eigensolver=sparse_hook
    )
    @test length(vals_sparse) == length(grid)
    @test size(vecs_sparse) == size(vecs)
end
