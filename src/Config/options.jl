# src/Config/options.jl

export SystemOptions
export AbstractGeometryOption, ChainLatticeOption, SquareLatticeOption, HexagonalLatticeOption, CubicLatticeOption, FCCLatticeOption, BCCLatticeOption, QELatticeOption
export KpointsOptions, PlotOptions
export AbstractModelOption, FreeElectronOption, TightBindingOption, MultiOrbitalTightBindingOption, KagomeLatticeOption, GrapheneOption, SSHModelOption, EinsteinModelOption, DebyeModelOption, PolaritonModelOption, MonoatomicLatticeModelOption
export AbstractInteractionOption, ConstantInteractionOption, LocalInteractionOption, YukawaInteractionOption, LimitedConstantInteractionOption, BareCoulombInteractionOption, ScreenedCoulombInteractionOption, CompositeInteractionOption
export AbstractFieldOption, ChargeDensityWaveOption, SpinDensityWaveOption, BCSReducedPairingOption, FFLOPairingOption, PairDensityWaveOption, MomentumDependentPairingOption, DirectChannelOption, ExchangeChannelOption
export AbstractTaskOption, SolveGroundStateOption, ScanInstabilityLandscapeOption, ScanSpectralFunctionOption, ComputePhaseTransitionDataOption, ComputeRenormalizedBandDataOption, ComputeZeemanPairingDataOption
export EliashbergConfig, load_config

# ---------------------------------------------------------
# 1. System Options
# ---------------------------------------------------------
@option struct SystemOptions
    n_workers::Int = 1
    bootstrap_workers::Bool = false
    project::Union{String,Nothing} = nothing
    restrict::Bool = true
end

# ---------------------------------------------------------
# 2. Geometry Options (Polymorphic)
# ---------------------------------------------------------
# 独立出来的 KGrid 数值参数
@option struct KpointsOptions
    grid::Union{Int,Vector{Int}} = 10
end

# 新增的绘图参数
@option struct PlotOptions
    enable::Bool = false
    save_path::Union{String,Nothing} = nothing
    title::Union{String,Nothing} = nothing
    colormap::String = "viridis"
    dpi::Int = 300
    # 未来还可以加 xlim, ylim, show_legend 等
end

abstract type AbstractGeometryOption end

@option "ChainLattice" struct ChainLatticeOption <: AbstractGeometryOption
    a::Float64 = 1.0
end

@option "SquareLattice" struct SquareLatticeOption <: AbstractGeometryOption
    a::Float64 = 1.0
end

@option "HexagonalLattice" struct HexagonalLatticeOption <: AbstractGeometryOption
    a::Float64 = 1.0
end

@option "CubicLattice" struct CubicLatticeOption <: AbstractGeometryOption
    a::Float64 = 1.0
end

@option "BCCLattice" struct BCCLatticeOption <: AbstractGeometryOption
    a::Float64 = 1.0
end

@option "FCCLattice" struct FCCLatticeOption <: AbstractGeometryOption
    a::Float64 = 1.0
end

@option "QE" struct QELatticeOption <: AbstractGeometryOption
    ibrav::Int
    a::Float64
    celldm2::Union{Float64,Nothing} = nothing
    celldm3::Union{Float64,Nothing} = nothing
    celldm4::Union{Float64,Nothing} = nothing
    celldm5::Union{Float64,Nothing} = nothing
    celldm6::Union{Float64,Nothing} = nothing
end

# ---------------------------------------------------------
# 3. Model Options (Polymorphic)
# ---------------------------------------------------------
abstract type AbstractModelOption end

@option struct HoppingDef
    R::Vector{Int}
    t::Float64
end

@option "FreeElectron" struct FreeElectronOption <: AbstractModelOption
    EF::Float64 = 0.0
    mass::Float64 = 1.0
end

@option "TightBinding" struct TightBindingOption <: AbstractModelOption
    EF::Float64 = 0.0
    hoppings::Vector{HoppingDef} = HoppingDef[]
    # Convenience scalar hop parameters for simple cells
    t::Union{Float64,Nothing} = nothing
    tp::Union{Float64,Nothing} = nothing
    use_spinor::Bool = false
end

@option "MultiOrbitalTightBinding" struct MultiOrbitalTightBindingOption <: AbstractModelOption
    EF::Float64 = 0.0
    num_orbitals::Int
    # Each basic hopping given as: [orbital_i, orbital_j, rx, ry, rz, t_real, t_imag]
    # Represented here as simplified structure
    hoppings::Vector{Vector{Float64}}
end

@option "KagomeLattice" struct KagomeLatticeOption <: AbstractModelOption
    t::Float64
    EF::Float64 = 0.0
end

@option "Graphene" struct GrapheneOption <: AbstractModelOption
    t::Float64
    EF::Float64 = 0.0
end

@option "SSHModel" struct SSHModelOption <: AbstractModelOption
    t1::Float64
    t2::Float64
    EF::Float64 = 0.0
end

@option "EinsteinModel" struct EinsteinModelOption <: AbstractModelOption
    omega_E::Float64
end

@option "DebyeModel" struct DebyeModelOption <: AbstractModelOption
    vs::Float64
    omega_D::Float64
end

@option "PolaritonModel" struct PolaritonModelOption <: AbstractModelOption
    omega_E::Float64
    vs::Float64
end

@option "MonoatomicLattice" struct MonoatomicLatticeModelOption <: AbstractModelOption
    K::Float64
    M::Float64
end

# ---------------------------------------------------------
# 4. Interaction Options (Polymorphic)
# ---------------------------------------------------------
abstract type AbstractInteractionOption end

@option "Constant" struct ConstantInteractionOption <: AbstractInteractionOption
    V0::Float64
end

@option "Local" struct LocalInteractionOption <: AbstractInteractionOption
    V0::Float64
    threshold::Float64
    fsthick::Float64
end

@option "Yukawa" struct YukawaInteractionOption <: AbstractInteractionOption
    V0::Float64
    lambda::Float64
end

@option "LimitedConstant" struct LimitedConstantInteractionOption <: AbstractInteractionOption
    V0::Float64
    omega_c::Float64
end

@option "BareCoulomb" struct BareCoulombInteractionOption <: AbstractInteractionOption
    cutoff::Float64
end

@option "ScreenedCoulomb" struct ScreenedCoulombInteractionOption <: AbstractInteractionOption
    bare::AbstractInteractionOption
end

@option "Composite" struct CompositeInteractionOption <: AbstractInteractionOption
    interactions::Vector{AbstractInteractionOption}
end

# ---------------------------------------------------------
# 5. Field Options (Polymorphic)
# ---------------------------------------------------------
abstract type AbstractFieldOption end

@option "DirectChannel" struct DirectChannelOption <: AbstractFieldOption end

@option "ExchangeChannel" struct ExchangeChannelOption <: AbstractFieldOption
    direction::String = "z"
end

@option "ChargeDensityWave" struct ChargeDensityWaveOption <: AbstractFieldOption
    q::Vector{Float64}
end

@option "SpinDensityWave" struct SpinDensityWaveOption <: AbstractFieldOption
    q::Vector{Float64}
    direction::String = "z"
end

@option "BCSReducedPairing" struct BCSReducedPairingOption <: AbstractFieldOption
    symmetry::String = "s_wave"
end

@option "FFLOPairing" struct FFLOPairingOption <: AbstractFieldOption
    q::Vector{Float64}
    symmetry::String = "s_wave"
    h::Float64 = 0.0
end

@option "PairDensityWave" struct PairDensityWaveOption <: AbstractFieldOption
    q::Vector{Float64}
    symmetry::String = "s_wave"
end

@option "MomentumDependentPairing" struct MomentumDependentPairingOption <: AbstractFieldOption
    seed::String = "random"
    amplitude::Float64 = 0.01
end

# ---------------------------------------------------------
# 6. Task Options (Polymorphic)
# ---------------------------------------------------------
abstract type AbstractTaskOption end

@option "solve_ground_state" struct SolveGroundStateOption <: AbstractTaskOption
    phi_guess::Float64 = 0.1
    T::Float64 = 1e-3
    approx::String = "ExactTrLn"
    warm_start::Bool = true
end

@option "scan_instability_landscape" struct ScanInstabilityLandscapeOption <: AbstractTaskOption
    qgrid_size::Union{Int,Vector{Int}} = 50
    T::Float64 = 0.001
    eta::Float64 = 0.001
end

@option "scan_spectral_function" struct ScanSpectralFunctionOption <: AbstractTaskOption
    qpath_points::Union{Vector{Vector{Float64}},Nothing} = nothing
    qpath_labels::Union{Vector{String},Nothing} = nothing
    omega_range::Union{Vector{Float64},Nothing} = nothing
    omega_points::Int = 100
    T::Float64 = 0.001
    eta::Float64 = 0.05
end

@option "compute_phase_transition_data" struct ComputePhaseTransitionDataOption <: AbstractTaskOption
    phi_range::Union{Vector{Float64},Nothing} = nothing
    phi_points::Int = 100
    T_range::Union{Vector{Float64},Nothing} = nothing
    T_points::Int = 10
    phi_guess::Float64 = 0.2
    approx::String = "ExactTrLn"
    warm_start::Bool = true
end

@option "compute_renormalized_band_data" struct ComputeRenormalizedBandDataOption <: AbstractTaskOption
    qpath_points::Union{Vector{Vector{Float64}},Nothing} = nothing
    qpath_labels::Union{Vector{String},Nothing} = nothing
    T_range::Union{Vector{Float64},Nothing} = nothing
    T_points::Int = 10
    phi_guess::Float64 = 0.5
    approx::String = "ExactTrLn"
    warm_start::Bool = true
end

@option "compute_zeeman_pairing_data" struct ComputeZeemanPairingDataOption <: AbstractTaskOption
    q_range::Union{Vector{Float64},Nothing} = nothing
    q_points::Int = 50
    h_val::Float64 = 0.1
    T_val::Float64 = 0.01
    phi_guess::Float64 = 0.4
    approx::String = "ExactTrLn"
    warm_start::Bool = true
end

# ---------------------------------------------------------
# Master Configuration Struct
# ---------------------------------------------------------
@option struct EliashbergConfig
    system::SystemOptions = SystemOptions()
    geometry::AbstractGeometryOption
    kpoints::KpointsOptions = KpointsOptions()
    model::AbstractModelOption
    interaction::AbstractInteractionOption
    field::AbstractFieldOption
    task::AbstractTaskOption
    plot::PlotOptions = PlotOptions()
end

"""
    load_config(path::AbstractString)

Reads a TOML configuration file, parses it into an `EliashbergConfig`, 
and performs cross-field validation.
"""
function load_config(path::AbstractString)::EliashbergConfig
    config = from_toml(EliashbergConfig, path)
    return config
end

const GEOMETRY_OPTION_TYPES = Dict{String,DataType}(
    "ChainLattice" => ChainLatticeOption,
    "SquareLattice" => SquareLatticeOption,
    "HexagonalLattice" => HexagonalLatticeOption,
    "CubicLattice" => CubicLatticeOption,
    "BCCLattice" => BCCLatticeOption,
    "FCCLattice" => FCCLatticeOption,
    "QE" => QELatticeOption,
)

const MODEL_OPTION_TYPES = Dict{String,DataType}(
    "FreeElectron" => FreeElectronOption,
    "TightBinding" => TightBindingOption,
    "MultiOrbitalTightBinding" => MultiOrbitalTightBindingOption,
    "KagomeLattice" => KagomeLatticeOption,
    "Graphene" => GrapheneOption,
    "SSHModel" => SSHModelOption,
    "EinsteinModel" => EinsteinModelOption,
    "DebyeModel" => DebyeModelOption,
    "PolaritonModel" => PolaritonModelOption,
    "MonoatomicLattice" => MonoatomicLatticeModelOption,
)

const INTERACTION_OPTION_TYPES = Dict{String,DataType}(
    "Constant" => ConstantInteractionOption,
    "Local" => LocalInteractionOption,
    "Yukawa" => YukawaInteractionOption,
    "LimitedConstant" => LimitedConstantInteractionOption,
    "BareCoulomb" => BareCoulombInteractionOption,
    "ScreenedCoulomb" => ScreenedCoulombInteractionOption,
    "Composite" => CompositeInteractionOption,
)

const FIELD_OPTION_TYPES = Dict{String,DataType}(
    "ChargeDensityWave" => ChargeDensityWaveOption,
    "SpinDensityWave" => SpinDensityWaveOption,
    "BCSReducedPairing" => BCSReducedPairingOption,
    "FFLOPairing" => FFLOPairingOption,
    "PairDensityWave" => PairDensityWaveOption,
    "MomentumDependentPairing" => MomentumDependentPairingOption,
)

const TASK_OPTION_TYPES = Dict{String,DataType}(
    "solve_ground_state" => SolveGroundStateOption,
    "scan_instability_landscape" => ScanInstabilityLandscapeOption,
    "scan_spectral_function" => ScanSpectralFunctionOption,
    "compute_phase_transition_data" => ComputePhaseTransitionDataOption,
    "compute_renormalized_band_data" => ComputeRenormalizedBandDataOption,
    "compute_zeeman_pairing_data" => ComputeZeemanPairingDataOption,
)

function _parse_polymorphic_option(
    ::Type{AbstractOption},
    x::AbstractDict{String,<:Any},
    option_types::Dict{String,DataType},
) where {AbstractOption}
    type_name = get(x, "type", nothing)
    isnothing(type_name) && error(
        "missing `type` field while parsing $(nameof(AbstractOption))",
    )

    concrete_type = get(option_types, type_name, nothing)
    isnothing(concrete_type) && error(
        "unknown $(nameof(AbstractOption)) type `$type_name`; expected one of $(collect(keys(option_types)))",
    )

    payload = Dict{String,Any}(k => v for (k, v) in x if k != "type")
    return from_dict(concrete_type, payload)
end

Configurations.from_dict(
    ::Type{OptionType},
    ::Type{AbstractGeometryOption},
    x::AbstractDict{String,<:Any},
) where {OptionType} = _parse_polymorphic_option(AbstractGeometryOption, x, GEOMETRY_OPTION_TYPES)

Configurations.from_dict(
    ::Type{OptionType},
    ::Type{AbstractModelOption},
    x::AbstractDict{String,<:Any},
) where {OptionType} = _parse_polymorphic_option(AbstractModelOption, x, MODEL_OPTION_TYPES)

Configurations.from_dict(
    ::Type{OptionType},
    ::Type{AbstractInteractionOption},
    x::AbstractDict{String,<:Any},
) where {OptionType} = _parse_polymorphic_option(AbstractInteractionOption, x, INTERACTION_OPTION_TYPES)

Configurations.from_dict(
    ::Type{OptionType},
    ::Type{AbstractFieldOption},
    x::AbstractDict{String,<:Any},
) where {OptionType} = _parse_polymorphic_option(AbstractFieldOption, x, FIELD_OPTION_TYPES)

Configurations.from_dict(
    ::Type{OptionType},
    ::Type{AbstractTaskOption},
    x::AbstractDict{String,<:Any},
) where {OptionType} = _parse_polymorphic_option(AbstractTaskOption, x, TASK_OPTION_TYPES)
