using Test
using Eliashberg
using Eliashberg.Config
using JLD2
using HDF5
using Makie
using StaticArrays

function _write_test_kpath(group, name::AbstractString, kpath)
    path_group = create_group(group, name)
    points = Eliashberg.path_points(kpath)
    node_indices, node_labels = Eliashberg.path_node_metadata(kpath)
    branch_ranges = Eliashberg.path_branch_ranges(kpath)

    path_group["points"] = reduce(vcat, permutedims.(collect.(points)))
    path_group["node_indices"] = Int.(node_indices)
    path_group["node_labels"] = collect(String.(node_labels))
    path_group["branch_start"] = [first(range) for range in branch_ranges]
    path_group["branch_stop"] = [last(range) for range in branch_ranges]
    attrs(path_group)["dimension"] = length(first(points))
    attrs(path_group)["n_points"] = length(points)
    return path_group
end

@testset "Result File IO" begin
    qpath = generate_kpath(
        [SVector(0.0, 0.0), SVector(pi, 0.0), SVector(pi, pi)],
        ["Γ", "X", "M"];
        n_pts_per_segment=4,
    )
    omegas = collect(range(0.0, 1.0, length=5))
    spectral_matrix = [sin(i / 3) + cos(j / 4) for i in 1:length(qpath), j in 1:length(omegas)]
    spectral_data = SpectralMapData(qpath, omegas, spectral_matrix, 0.25, 0.5, 0.02)

    mktempdir() do tmpdir
        h5_path = joinpath(tmpdir, "plasmon_scan.h5")
        h5open(h5_path, "w") do file
            attrs(file)["format"] = "Eliashberg-HDF5"
            attrs(file)["format_version"] = "1.0"
            attrs(file)["result_type"] = string(typeof(spectral_data))
            attrs(file)["task_type"] = "compute_collective_mode_spectral_data"

            result_group = create_group(file, "result")
            _write_test_kpath(result_group, "qpath", qpath)
            result_group["omegas"] = omegas
            result_group["spectral_matrix"] = spectral_matrix
            attrs(result_group)["gap"] = spectral_data.gap
            attrs(result_group)["temperature"] = spectral_data.temperature
            attrs(result_group)["pair_breaking_edge"] = spectral_data.pair_breaking_edge
        end

        loaded_h5 = load_result(h5_path)
        @test loaded_h5 isa SpectralMapData{2}
        @test Eliashberg.path_points(loaded_h5.qpath) == Eliashberg.path_points(qpath)
        @test loaded_h5.omegas == omegas
        @test loaded_h5.spectral_matrix ≈ spectral_matrix
        @test loaded_h5.pair_breaking_edge ≈ spectral_data.pair_breaking_edge
        @test plot_result_file(h5_path) isa Figure
        @test plot_result_file(h5_path; mode=:spectral) isa Figure

        legacy_config = EliashbergConfig(
            geometry=SquareLatticeOption(a=1.0),
            kpoints=KpointsOptions(grid=[4, 4]),
            model=TightBindingOption(t=1.0, EF=0.0),
            interaction=ConstantInteractionOption(V0=1.0),
            field=DirectChannelOption(),
            task=ScanSpectralFunctionOption(
                qpath_points=[[0.0, 0.0], [pi, 0.0], [pi, pi]],
                qpath_labels=["Γ", "X", "M"],
                npoints_each_line=4,
                omega_range=[0.0, 1.0],
                omega_points=5,
                T_val=0.1,
                eta=0.02,
            ),
        )

        jld2_path = joinpath(tmpdir, "plasmon_scan.jld2")
        JLD2.jldsave(jld2_path; result=spectral_matrix, config=legacy_config)

        loaded_jld2 = load_result(jld2_path)
        @test loaded_jld2 isa SpectralMapData{2}
        @test loaded_jld2.omegas == omegas
        @test loaded_jld2.spectral_matrix ≈ spectral_matrix
        @test loaded_jld2.temperature ≈ 0.1
        @test loaded_jld2.gap == 0.0
        @test isnothing(loaded_jld2.pair_breaking_edge)
        @test plot_result_file(jld2_path) isa Figure

        fallback_h5_path = joinpath(tmpdir, "legacy_scan.h5")
        h5open(fallback_h5_path, "w") do file
            attrs(file)["format"] = "Eliashberg-HDF5"
            attrs(file)["format_version"] = "1.0"
            attrs(file)["result_type"] = string(typeof(spectral_data))
            attrs(file)["task_type"] = "scan_spectral_function"

            result_group = create_group(file, "result")
            _write_test_kpath(result_group, "qpath", qpath)
            result_group["omegas"] = omegas
            result_group["spectral_matrix"] = spectral_matrix
            attrs(result_group)["gap"] = 0.0
            attrs(result_group)["temperature"] = 0.1
        end

        fallback_jld2_path = joinpath(tmpdir, "legacy_scan.jld2")
        JLD2.jldsave(fallback_jld2_path; result=spectral_matrix)

        loaded_fallback = load_result(fallback_jld2_path)
        @test loaded_fallback isa SpectralMapData{2}
        @test loaded_fallback.omegas == omegas
        @test loaded_fallback.spectral_matrix ≈ spectral_matrix
        @test loaded_fallback.temperature ≈ 0.1
    end
end
