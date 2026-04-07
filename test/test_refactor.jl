using Test
using Eliashberg
using StaticArrays
using Distributed
using LinearAlgebra
using SparseArrays
using Makie
using AtomsBase
using Unitful

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
    @test one_d_model.lattice isa SMatrix{1,1,Float64,1}
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
    @test lat isa PeriodicCell{2}
    @test periodicity(lat) == (true, true)
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

    cdw_from_vector = ChargeDensityWave([π, π])
    @test cdw_from_vector.q == SVector{2,Float64}(π, π)

    cdw_from_tuple = ChargeDensityWave((π, π))
    @test cdw_from_tuple.q == SVector{2,Float64}(π, π)

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

    sdw_from_vector = SpinDensityWave([0.2], :y)
    @test sdw_from_vector isa SpinDensityWave{1,:y}
    @test sdw_from_vector.q == SVector{1,Float64}(0.2)

    fflo_from_tuple = FFLOPairing((π, 0.0), 1)
    @test fflo_from_tuple.q == SVector{2,Float64}(π, 0.0)
    @test fflo_from_tuple.h == 1.0

    pdw_from_vector = PairDensityWave([π, 0.0])
    @test pdw_from_vector.q == SVector{2,Float64}(π, 0.0)

    static_from_tuple = StaticMeanField((0, 0))
    @test static_from_tuple.q == SVector{2,Float64}(0.0, 0.0)

    fluct_from_vector = DynamicalFluctuation([π], 2)
    @test fluct_from_vector.q == SVector{1,Float64}(π)
    @test fluct_from_vector.ω == 2.0

    landscape = scan_instability_landscape(model, grid, grid; T=0.1, η=1e-3)
    @test landscape isa Matrix{Float64}
    @test size(landscape) == (4, 4)

    qpath = generate_kpath(lat; n_pts_per_segment=4)
    omegas = collect(range(0.0, 1.0, length=5))
    spectral = scan_spectral_function(model, grid, qpath, omegas; T=0.1, η=0.02)
    @test size(spectral) == (length(qpath), length(omegas))
    spectral_fig = plot_spectral_function(qpath, omegas, spectral)
    @test spectral_fig isa Figure

    line_path = generate_kpath(
        [first(line_grid.points), last(line_grid.points)],
        ["-π", "π"];
        n_pts_per_segment=length(line_grid.points) - 1,
    )
    line_band_data = compute_band_data(one_d_model, line_path)
    @test size(line_band_data.bands, 1) == length(line_grid)
    dispersion_fig = plot(line_band_data)
    @test dispersion_fig isa Figure

    hr_dir = mktempdir()
    hr_file = joinpath(hr_dir, "mock_hr.dat")
    hr_io = open(hr_file, "w")
    try
        write(hr_io, """
Generated for testing
1
1
1
0 0 0 1 1 -0.5 0.0
""")
        close(hr_io)

        num_wann, hoppings = parse_wannier90_hr(hr_file)
        @test num_wann == 1
        @test length(hoppings) == 1
        @test hoppings[1][1] == 1
        @test hoppings[1][2] == 1
        @test hoppings[1][3] == SVector{3,Int}(0, 0, 0)
        @test hoppings[1][4] == -0.5 + 0.0im

        crystal_3d = Crystal(CubicLattice(1.0), [:H => [0.0, 0.0, 0.0]])
        wannier_model = build_model_from_wannier90(hr_file, crystal_3d, 0.0)
        @test wannier_model isa MultiOrbitalTightBinding{3}
        @test length(wannier_model.hoppings) == 1
    finally
        isopen(hr_io) && close(hr_io)
        rm(hr_dir; force=true, recursive=true)
    end

    tb_dir = mktempdir()
    tb_file = joinpath(tb_dir, "mock_tb.dat")
    tb_io = open(tb_file, "w")
    try
        write(tb_io, """
Generated for testing
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
2
2
1 2

0 0 0
1 1 0.0 0.0
2 1 1.0e-7 0.0
1 2 1.0 0.0
2 2 0.0 0.0

1 0 0
1 1 2.0 0.0
2 1 0.0 0.0
1 2 0.0 0.0
2 2 0.0 0.0

0 0 0
1 1 0.0 0.0 0.0 0.0 0.0 0.0
2 1 0.0 0.0 0.0 0.0 0.0 0.0
1 2 0.0 0.0 0.0 0.0 0.0 0.0
2 2 0.2 0.0 0.0 0.0 0.0 0.0

1 0 0
1 1 0.0 0.0 0.0 0.0 2.0 0.0
2 1 0.0 0.0 0.0 0.0 0.0 0.0
1 2 0.0 0.0 0.0 0.0 0.0 0.0
2 2 0.0 0.0 0.0 0.0 0.0 0.0
""")
        close(tb_io)

        tb_data = parse_wannier90_tb(tb_file)
        @test tb_data.num_wann == 2
        @test tb_data.lattice_vectors[1] == SVector{3,Float64}(1.0, 0.0, 0.0)
        @test tb_data.lattice_vectors[2] == SVector{3,Float64}(0.0, 1.0, 0.0)
        @test tb_data.lattice_vectors[3] == SVector{3,Float64}(0.0, 0.0, 1.0)

        @test length(tb_data.hoppings) == 2
        @test tb_data.hoppings[1] == (1, 2, SVector{3,Int}(0, 0, 0), 1.0 + 0.0im)
        @test tb_data.hoppings[2] == (1, 1, SVector{3,Int}(1, 0, 0), 1.0 + 0.0im)

        @test length(tb_data.position_matrices) == 2
        @test tb_data.position_matrices[1] == (
            2,
            2,
            SVector{3,Int}(0, 0, 0),
            SVector{3,ComplexF64}(0.2 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im),
        )
        @test tb_data.position_matrices[2] == (
            1,
            1,
            SVector{3,Int}(1, 0, 0),
            SVector{3,ComplexF64}(0.0 + 0.0im, 0.0 + 0.0im, 1.0 + 0.0im),
        )

        tb_cell = cell_from_wannier90_tb(tb_file)
        @test Eliashberg.primitive_vectors(tb_cell) == @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        @test periodicity(tb_cell) == (true, true, true)

        @test Eliashberg.primitive_vectors(tb_data.cell) == Eliashberg.primitive_vectors(tb_cell)
        @test reduce(hcat, tb_data.lattice_vectors) == Eliashberg.primitive_vectors(tb_cell)

        tb_periodic_cell = periodic_cell_from_wannier90_tb(tb_file; periodicity=(true, true, false))
        @test periodicity(tb_periodic_cell) == (true, true, false)

        slab_path = generate_kpath(tb_periodic_cell; n_pts_per_segment=4)
        @test slab_path isa KPath{3}
        @test all(isapprox(k[3], 0.0; atol=1e-10) for k in Eliashberg.path_points(slab_path))
        slab_node_indices, slab_node_labels = Eliashberg.path_node_metadata(slab_path)
        @test first(slab_node_labels) == "Γ"
        @test last(slab_node_labels) == "Γ"
        @test Set(slab_node_labels) == Set(["Γ", "X", "M"])
        @test length(slab_node_indices) == 4

        slab_grid = generate_reciprocal_lattice(tb_periodic_cell, 4, 3)
        @test slab_grid isa KGrid{3}
        @test length(slab_grid) == 12
        @test all(isapprox(k[3], 0.0; atol=1e-10) for k in slab_grid.points)

        xz_periodic_cell = periodic_cell_from_wannier90_tb(tb_file; periodicity=(true, false, true))
        xz_grid = generate_reciprocal_lattice(xz_periodic_cell, 4, 3)
        @test xz_grid isa KGrid{3}
        @test all(isapprox(k[2], 0.0; atol=1e-10) for k in xz_grid.points)

        xz_path = generate_kpath(xz_periodic_cell; n_pts_per_segment=4)
        @test xz_path isa KPath{3}
        @test all(isapprox(k[2], 0.0; atol=1e-10) for k in Eliashberg.path_points(xz_path))

        carbon_species = [ChemicalSpecies(:C), ChemicalSpecies(:C)]
        slab_system = FastSystem(
            tb_periodic_cell,
            [
                SVector(0.0u"Å", 0.0u"Å", 0.0u"Å"),
                SVector(0.5u"Å", 0.5u"Å", 0.0u"Å"),
            ],
            carbon_species,
            mass.(carbon_species),
        )
        @test periodic_rank(slab_system) == 2
        system_path = generate_kpath(slab_system; n_pts_per_segment=4)
        @test system_path isa KPath{3}
        @test all(isapprox(k[3], 0.0; atol=1e-10) for k in Eliashberg.path_points(system_path))

        wannier_tb_model = build_model_from_wannier90(tb_file, 0.0)
        @test wannier_tb_model isa MultiOrbitalTightBinding{3}
        @test wannier_tb_model.cell isa PeriodicCell{3}
        @test periodicity(wannier_tb_model) == (true, true, true)
        @test length(wannier_tb_model.hoppings) == 2
        @test Eliashberg.primitive_vectors(wannier_tb_model) == Eliashberg.primitive_vectors(tb_cell)
        @test wannier_tb_model.num_orbitals == 2

        wannier_slab_model = build_model_from_wannier90(tb_file, 0.0, (true, true, false))
        @test periodicity(wannier_slab_model) == (true, true, false)
        @test periodicity(wannier_slab_model.cell) == (true, true, false)

        wannier_system_model = MultiOrbitalTightBinding(slab_system, tb_data.hoppings, 0.0)
        @test periodicity(wannier_system_model) == (true, true, false)
        @test Eliashberg.primitive_vectors(wannier_system_model) == Eliashberg.primitive_vectors(tb_cell)
        @test wannier_system_model.num_orbitals == length(slab_system)

        model_grid_rank = generate_reciprocal_lattice(wannier_system_model, 4, 3)
        @test model_grid_rank isa KGrid{3}
        @test all(isapprox(k[3], 0.0; atol=1e-10) for k in model_grid_rank.points)

        model_grid_ambient = generate_reciprocal_lattice(wannier_system_model, 4, 3, 1)
        @test model_grid_ambient isa KGrid{3}
        @test length(model_grid_ambient) == length(model_grid_rank)
        @test all(isapprox(k[3], 0.0; atol=1e-10) for k in model_grid_ambient.points)

        model_path = generate_kpath(wannier_system_model; n_pts_per_segment=4)
        @test model_path isa KPath{3}
        @test all(isapprox(k[3], 0.0; atol=1e-10) for k in Eliashberg.path_points(model_path))

        graphene_tb = joinpath(dirname(@__DIR__), "examples", "graphene", "graphene_tb.dat")
        graphene_model = build_model_from_wannier90(graphene_tb, -2.3042, (true, true, false))
        @test periodicity(graphene_model) == (true, true, false)
        @test periodicity(graphene_model.cell) == (true, true, false)

        graphene_grid = generate_reciprocal_lattice(graphene_model.cell, 5, 5, 1)
        @test graphene_grid isa KGrid{3}
        @test length(graphene_grid) == 25
        @test all(isapprox(k[3], 0.0; atol=1e-10) for k in graphene_grid.points)

        graphene_path = generate_kpath(graphene_model.cell; n_pts_per_segment=6)
        graphene_node_indices, graphene_node_labels = Eliashberg.path_node_metadata(graphene_path)
        @test graphene_path isa KPath{3}
        @test all(isapprox(k[3], 0.0; atol=1e-10) for k in Eliashberg.path_points(graphene_path))
        @test graphene_node_labels == ["Γ", "M", "K", "Γ"]
        @test length(graphene_node_indices) == 4
    finally
        isopen(tb_io) && close(tb_io)
        rm(tb_dir; force=true, recursive=true)
    end

    bravais_examples = [
        (:aP, ibrav(14, 1.0; celldm2=1.1, celldm3=1.2, celldm4=0.15, celldm5=0.2, celldm6=0.25)),
        (:mP, ibrav(12, 1.0; celldm2=1.2, celldm3=1.3, celldm4=0.2)),
        (:mC, ibrav(13, 1.0; celldm2=1.2, celldm3=1.3, celldm4=0.2)),
        (:oP, ibrav(8, 1.0; celldm2=1.2, celldm3=1.4)),
        (:oC, ibrav(9, 1.0; celldm2=1.2, celldm3=1.4)),
        (:oF, ibrav(10, 1.0; celldm2=1.2, celldm3=1.4)),
        (:oI, ibrav(11, 1.0; celldm2=1.2, celldm3=1.4)),
        (:tP, ibrav(6, 1.0; celldm3=1.4)),
        (:tI, ibrav(7, 1.0; celldm3=1.4)),
        (:hP, ibrav(4, 1.0; celldm3=1.6)),
        (:hR, ibrav(5, 1.0; celldm4=0.2)),
        (:cP, ibrav(1, 1.0)),
        (:cF, ibrav(2, 1.0)),
        (:cI, ibrav(3, 1.0)),
    ]

    for (expected_bravais, lat3d) in bravais_examples
        @test lat3d isa PeriodicCell{3}
        @test bravais_lattice(lat3d) == expected_bravais
        path3d = generate_kpath(lat3d; n_pts_per_segment=4)
        @test path3d isa KPath{3}
        node_indices, node_labels = Eliashberg.path_node_metadata(path3d)
        @test length(node_indices) == length(node_labels)
        @test length(path3d) > length(node_labels)
    end

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

    phase_data_parallel = compute_phase_transition_data(
        collect(range(0.0, 0.2, length=3)),
        [0.1, 0.2],
        BCSReducedPairing(:s_wave),
        one_d_model,
        ConstantInteraction(-1.0),
        line_grid;
        warm_start=false
    )
    @test size(phase_data_parallel.condensation_energy) == (3, 2)
    @test length(phase_data_parallel.order_parameters) == 2

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

    renormalized_data_parallel = compute_renormalized_band_data(
        [0.1],
        BCSReducedPairing(:s_wave),
        one_d_model,
        ConstantInteraction(-1.0),
        line_grid,
        path;
        warm_start=false
    )
    @test size(renormalized_data_parallel.renormalized_bands, 3) == 1
    @test length(renormalized_data_parallel.gaps) == 1

    coexistence_field = CompositeField(ChargeDensityWave([0.0]), BCSReducedPairing(:s_wave))
    coexistence_interaction = CompositeInteraction(ConstantInteraction(-1.0), ConstantInteraction(-1.0))
    coexistence_dispersion = MeanFieldDispersion(one_d_model, coexistence_field, [0.1, 0.2])
    @test size(ε(SVector{1,Float64}(0.1), coexistence_dispersion)) == (4, 4)

    coexistence_action = evaluate_action(
        [0.05, 0.1],
        coexistence_field,
        one_d_model,
        coexistence_interaction,
        line_grid,
        ExactTrLn();
        T=0.1
    )
    @test isfinite(coexistence_action)

    coexistence_solution = solve_ground_state(
        coexistence_field,
        one_d_model,
        coexistence_interaction,
        line_grid,
        ExactTrLn();
        phi_guess=[0.05, 0.1],
        T=0.1
    )
    @test length(coexistence_solution) == 2
    @test all(isfinite, coexistence_solution)

    coexistence_band_data = compute_renormalized_band_data(
        [0.1],
        coexistence_field,
        one_d_model,
        coexistence_interaction,
        line_grid,
        path;
        phi_guess=[0.05, 0.1]
    )
    @test size(coexistence_band_data.gaps) == (2, 1)
    @test size(coexistence_band_data.renormalized_bands, 3) == 1
    @test plot(coexistence_band_data) isa Figure

    coexistence_landscape = compute_coexistence_landscape(
        [-0.1, 0.0, 0.1],
        [-0.2, 0.0, 0.2],
        coexistence_field,
        one_d_model,
        coexistence_interaction,
        line_grid;
        T=0.1
    )
    @test coexistence_landscape isa CoexistenceLandscapeData
    @test size(coexistence_landscape.free_energy) == (3, 3)
    @test coexistence_landscape.field_1_type == string(typeof(coexistence_field[1]))
    @test coexistence_landscape.field_2_type == string(typeof(coexistence_field[2]))

    p_wave_vertex = vertex_matrix(
        NormalNambuDispersion(one_d_model),
        SVector{1,Float64}(0.1),
        BCSReducedPairing(:p_wave)
    )
    @test size(parent(p_wave_vertex)) == (2, 2)

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
