using Test
using Eliashberg
using Configurations
using AtomsBase

@testset "Config Parsing" begin
    config_path = normpath(joinpath(@__DIR__, "..", "examples", "bcs", "bcs_1d_chain_s_wave.toml"))
    config = load_config(config_path)

    @test config.geometry isa Eliashberg.Config.ChainLatticeOption
    @test config.model isa Eliashberg.Config.TightBindingOption
    @test config.interaction isa Eliashberg.Config.ConstantInteractionOption
    @test config.field isa Eliashberg.Config.BCSReducedPairingOption
    @test config.task isa Eliashberg.Config.ComputePhaseTransitionDataOption

    params = build_from_config(config)
    @test params.geometry isa PeriodicCell{1}
    @test params.model isa TightBinding{1}
    @test params.interaction isa ConstantInteraction
    @test params.field isa BCSReducedPairing
    @test length(params.kpoints) == 200
end

@testset "Nested Polymorphic Config Parsing" begin
    interaction = from_dict(
        Eliashberg.Config.CompositeInteractionOption,
        Dict(
            "interactions" => Any[
                Dict("type" => "Constant", "V0" => -1.0),
                Dict("type" => "BareCoulomb", "cutoff" => 3.0),
            ],
        ),
    )

    @test interaction isa Eliashberg.Config.CompositeInteractionOption
    @test length(interaction.interactions) == 2
    @test interaction.interactions[1] isa Eliashberg.Config.ConstantInteractionOption
    @test interaction.interactions[2] isa Eliashberg.Config.BareCoulombInteractionOption
end

@testset "Task Path Resolution" begin
    task = from_dict(
        Eliashberg.Config.ScanSpectralFunctionOption,
        Dict(
            "qpath_points" => Any[[0.0], [1.0], [2.0]],
            "qpath_labels" => Any["Γ", "X", "M"],
            "npoints_each_line" => 7,
            "omega_range" => Any[0.0, 1.0],
            "omega_points" => 5,
            "T_val" => 0.02,
            "eta" => 0.03,
        ),
    )

    task_kwargs = Eliashberg.Config.build_task(task)
    @test length(task_kwargs.qpath) == 2 * task.npoints_each_line + 1
    @test length(task_kwargs.omegas) == 5
end
