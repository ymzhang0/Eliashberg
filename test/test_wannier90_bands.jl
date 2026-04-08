using Eliashberg
using Test
using Makie
using StaticArrays

@testset "Wannier90 band parser and plotting" begin
    bands_file = joinpath(@__DIR__, "..", "examples", "graphene", "graphene_band.dat")
    kpoints_file = joinpath(@__DIR__, "..", "examples", "graphene", "graphene_band.kpt")
    labelinfo_file = joinpath(@__DIR__, "..", "examples", "graphene", "graphene_band.labelinfo.dat")
    tb_file = joinpath(@__DIR__, "..", "examples", "graphene", "graphene_tb.dat")

    parsed = parse_wannier90_band_dat(bands_file)
    @test parsed.num_kpoints == 274
    @test parsed.num_bands == 8
    @test size(parsed.bands) == (274, 8)
    @test parsed.distances[1] ≈ 0.0
    @test parsed.distances[end] ≈ 4.028774
    @test parsed.bands[1, 1] ≈ -21.911754
    @test parsed.bands[end, 8] ≈ 10.338948

    labels = parse_wannier90_labelinfo(labelinfo_file)
    @test labels.node_labels == ["Γ", "M", "K", "Γ"]
    @test labels.node_indices == [1, 101, 159, 274]
    @test labels.dimension == 2

    parsed_kpoints = parse_wannier90_kpoints(kpoints_file)
    @test parsed_kpoints.num_kpoints == 274
    @test parsed_kpoints.kpoints[1] ≈ SVector(0.0, 0.0, 0.0)
    @test parsed_kpoints.kpoints[159] ≈ SVector(0.333333, 0.333333, 0.0) atol=1e-6
    @test all(isapprox.(parsed_kpoints.weights, 1.0; atol=1e-12))

    data = band_data_from_wannier90_bands(bands_file)
    @test data isa BandStructureData{2}
    @test size(data.bands) == (274, 8)
    @test length(data.kpath) == 274
    @test Eliashberg.path_node_metadata(data.kpath) == ([1, 101, 159, 274], ["Γ", "M", "K", "Γ"])

    model = build_model_from_wannier90(tb_file, -2.3042, (true, true, false))
    comparison = compare_wannier90_tb_to_bands(
        model,
        bands_file;
        kpoints_filename=kpoints_file,
        labelinfo_filename=labelinfo_file,
    )
    @test comparison isa Wannier90BandComparison{2}
    @test size(comparison.reference.bands) == (274, 8)
    @test size(comparison.model.bands) == (274, 8)
    @test comparison.energy_shift ≈ -2.3043553282257645 atol=1e-8
    @test comparison.rms_error < 0.01
    @test comparison.max_error < 0.03
    @test comparison.kpoints_fractional[159] ≈ SVector(0.333333, 0.333333, 0.0) atol=1e-6

    @test plot(data) isa Figure
    @test plot_wannier90_band_structure(bands_file) isa Figure
    @test plot(comparison) isa Figure
    @test plot_wannier90_tb_band_comparison(comparison) isa Figure
end
