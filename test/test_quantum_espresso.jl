using Eliashberg
using StaticArrays
using Test
using AtomsBase
using Unitful

@testset "Quantum ESPRESSO bands parser" begin
    dir = joinpath(@__DIR__, "..", "examples", "niobium")
    parsed = parse_quantum_espresso_bands(dir, "Nb")

    @test parsed.num_bands == 14
    @test length(parsed.kpath) == 181
    @test size(parsed.bands) == (181, 14)

    node_indices, node_labels = Eliashberg.path_node_metadata(parsed.kpath)
    @test node_indices == [1, 21, 41, 61, 81, 101, 121, 141, 161, 181]
    @test node_labels == ["Γ", "H", "P", "N", "Γ", "H", "N", "Γ", "P", "N"]

    points = Eliashberg.path_points(parsed.kpath)
    @test points[1] ≈ SVector(0.0, 0.0, 0.0)
    @test points[21] ≈ SVector(1.903707107159504, 0.0, 0.0)
    @test points[end] ≈ SVector(0.951853553579752, 0.0, 0.951853553579752)

    @test isapprox(parsed.bands[1, 1], -36.317; atol=1e-3)
    @test isapprox(parsed.bands[1, 14], 43.103; atol=1e-3)
    @test isapprox(parsed.bands[21, 1], -35.894; atol=1e-3)
end

@testset "Quantum ESPRESSO high-symmetry matching with large k-point gap branches" begin
    dir = joinpath(@__DIR__, "..", "examples", "silicon")
    parsed = parse_quantum_espresso_bands(dir, "Si")

    @test size(parsed.bands) == (82, 14)
    @test length(parsed.kpath) == 82
    @test length(Eliashberg.path_branches(parsed.kpath)) == 2
    @test length.(Eliashberg.path_branches(parsed.kpath)) == [41, 41]

    node_indices, node_labels = Eliashberg.path_node_metadata(parsed.kpath)
    @test node_indices == [1, 21, 41, 42, 62, 82]
    @test node_labels == ["L", "Γ", "X", "X", "K", "Γ"]

    distances = Eliashberg.path_distances(parsed.kpath)
    @test distances[41] ≈ distances[42]
    @test isapprox(distances[62], 2.5678557104; atol=1e-6)
    @test isapprox(distances[82], 3.7949454113; atol=1e-6)
end

@testset "Wannier90 band parser uses k-point gap branches" begin
    dir = joinpath(@__DIR__, "..", "examples", "silicon")
    parsed = parse_wannier90_band_dat(dir, "Si", "silicon")

    @test size(parsed.bands) == (380, 8)
    @test length(parsed.kpath) == 380
    @test length.(Eliashberg.path_branches(parsed.kpath)) == [216, 164]

    node_indices, node_labels = Eliashberg.path_node_metadata(parsed.kpath)
    @test node_indices == [1, 101, 216, 217, 258, 380]
    @test node_labels == ["L", "Γ", "X", "X", "K", "Γ"]

    distances = Eliashberg.path_distances(parsed.kpath)
    @test distances[216] ≈ distances[217]
    @test isapprox(distances[258], 2.5678557104; atol=1e-6)
    @test isapprox(distances[380], 3.7949454113; atol=1e-6)
end

@testset "Quantum ESPRESSO KPath builder" begin
    cell = @SMatrix [
        2.46 -1.23 0.0;
        0.0 2.130422 0.0;
        0.0 0.0 15.0
    ]

    kpoints = [SVector(0.0, 0.0, 0.0), SVector(0.5, 0.0, 0.0)]
    kpath = kpath_from_quantum_espresso_bands(
        kpoints;
        cell=cell,
        node_labels=["Γ", "M"],
    )

    @test length(kpath) == 2
    @test Eliashberg.path_node_metadata(kpath) == ([1, 2], ["Γ", "M"])

    periodic_cell = PeriodicCell(
        ;
        cell_vectors=(
            SVector(2.46u"Å", 0.0u"Å", 0.0u"Å"),
            SVector(-1.23u"Å", 2.130422u"Å", 0.0u"Å"),
            SVector(0.0u"Å", 0.0u"Å", 15.0u"Å"),
        ),
        periodicity=(true, true, false),
    )
    periodic_kpath = kpath_from_quantum_espresso_bands(
        kpoints;
        cell=periodic_cell,
        node_labels=["Γ", "M"],
    )
    @test length(periodic_kpath) == 2
end
