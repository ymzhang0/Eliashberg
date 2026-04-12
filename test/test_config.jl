using Test
using Eliashberg
using Configurations
using AtomsBase

@testset "Config Parsing" begin
    config_path = normpath(joinpath(@__DIR__, "..", "examples", "configs", "bcs_1d_chain_s_wave.toml"))
    config = load_config(config_path)

    @test config.geometry isa Eliashberg.Config.ChainLatticeOption
    @test config.model isa Eliashberg.Config.TightBindingOption
    @test config.interaction isa Eliashberg.Config.ConstantInteractionOption
    @test config.field isa Eliashberg.Config.BCSReducedPairingOption
    @test config.task isa Eliashberg.Config.ComputePhaseTransitionDataOption

    params = build_from_config(config)
    @test params.cell isa PeriodicCell{1}
    @test params.model isa TightBinding{1}
    @test params.interaction isa ConstantInteraction
    @test params.field isa BCSReducedPairing
    @test length(params.kgrid) == 2000
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
