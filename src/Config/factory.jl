# src/Config/factory.jl

export build_geometry, build_model, build_interaction, build_field, build_from_config, extract_kwargs

"""
    extract_kwargs(opt; exclude=())

Convert a configuration struct into a `NamedTuple`, automatically filtering out any `nothing` fields.
Optionally excludes specific keys (useful when some fields are positional arguments).
"""
function extract_kwargs(opt; exclude=())
    # 巧妙利用 Generator 和 NamedTuple 构造，直接剔除 nothing 和 exclude 列表
    return (; (f => getproperty(opt, f) for f in propertynames(opt)
               if getproperty(opt, f) !== nothing && !(f in exclude))...)
end

# Helper to get dimensionality gracefully
_dim(cell) = size(primitive_vectors(cell), 1)
_dim(disp::ElectronicDispersion{D}) where {D} = D

# Helper for robust SVector conversion
_svec(D::Int, v::AbstractVector) = SVector{D,eltype(v)}(v)
_svec(D::Int, v::Tuple) = SVector{D,eltype(v)}(v)

# ---------------------------------------------------------
# Geometry Factory
# ---------------------------------------------------------
build_geometry(opt::ChainLatticeOption) = ChainLattice(opt.a)
build_geometry(opt::SquareLatticeOption) = SquareLattice(opt.a)
build_geometry(opt::HexagonalLatticeOption) = HexagonalLattice(opt.a)
build_geometry(opt::CubicLatticeOption) = CubicLattice(opt.a)
build_geometry(opt::FCCLatticeOption) = FCCLattice(opt.a)
build_geometry(opt::BCCLatticeOption) = BCCLattice(opt.a)

function build_geometry(opt::QELatticeOption)
    # 丝滑！直接用 extract_kwargs，排除位置参数 ibrav 和 a
    kwargs = extract_kwargs(opt, exclude=(:ibrav, :a))
    return qe_lattice(opt.ibrav, opt.a; kwargs...)
end

function build_kpoints(opt::KpointsOptions, geometry)
    D = _dim(geometry)
    sz = opt.grid

    # 统一转换网格大小为对应维度的数组或元组
    sz_array = sz isa Int ? fill(sz, D) : sz
    length(sz_array) == D || throw(DimensionMismatch("kpoints grid length must match lattice dimension $D"))

    # 如果 Eliashberg 导出过 generate_reciprocal_lattice，直接用它最优雅
    # 否则保持你的分支，但尽量用 Tuple 传参
    if D == 1
        return generate_1d_kgrid(sz_array[1])
    elseif D == 2
        return generate_2d_kgrid(sz_array)
    else
        return generate_3d_kgrid(sz_array)
    end
end

# ---------------------------------------------------------
# Model Factory
# ---------------------------------------------------------

build_model(opt::FreeElectronOption, cell) = FreeElectron{_dim(cell)}(opt.EF, opt.mass)
build_model(opt::EinsteinModelOption, cell) = EinsteinModel{_dim(cell)}(opt.omega_E)
build_model(opt::DebyeModelOption, cell) = DebyeModel{_dim(cell)}(opt.vs, opt.omega_D)
build_model(opt::PolaritonModelOption, cell) = PolaritonModel{_dim(cell)}(opt.omega_E, opt.vs)
build_model(opt::MonoatomicLatticeModelOption, cell) = MonoatomicLatticeModel{_dim(cell)}(cell, opt.K, opt.M)
function build_model(opt::TightBindingOption, cell)
    if !isnothing(opt.hoppings) && !isempty(opt.hoppings)
        D = _dim(cell)
        # 丝滑的列表推导式
        hops = [(_svec(D, h.R), h.t) for h in opt.hoppings]
        return TightBinding(cell, hops, opt.EF)
    end

    isnothing(opt.t) && throw(ArgumentError("TightBinding requires either `hoppings` or `t`."))
    return isnothing(opt.tp) ? TightBinding(cell, opt.t, opt.EF) : TightBinding(cell, opt.t, opt.tp, opt.EF)
end

function build_model(opt::MultiOrbitalTightBindingOption, cell)
    D = _dim(cell)
    hops = Tuple{Int,Int,SVector{D,Int},ComplexF64}[]

    for h in opt.hoppings
        length(h) != 4 + D && throw(ArgumentError("Invalid MultiOrbital hopping length. Expected $(4+D) elements."))

        R_vec = _svec(D, h[3:2+D])
        t_val = complex(h[end-1], h[end])
        push!(hops, (Int(h[1]), Int(h[2]), R_vec, t_val))
    end
    return MultiOrbitalTightBinding(cell, opt.num_orbitals, hops, opt.EF)
end

build_model(opt::KagomeLatticeOption, cell) = KagomeLattice(cell, opt.t, opt.EF)
build_model(opt::GrapheneOption, cell) = Graphene(cell, opt.t, opt.EF)
build_model(opt::SSHModelOption, cell) = SSHModel(cell, opt.t1, opt.t2, opt.EF)
# ---------------------------------------------------------
# Interaction Factory
# ---------------------------------------------------------
build_interaction(opt::ConstantInteractionOption, disp) = ConstantInteraction(opt.V0)
build_interaction(opt::LocalInteractionOption, disp) = LocalInteraction(opt.V0, opt.threshold, opt.fsthick)
build_interaction(opt::YukawaInteractionOption, disp) = YukawaInteraction(opt.V0, opt.lambda)
build_interaction(opt::BareCoulombInteractionOption, disp) = BareCoulombInteraction(opt.cutoff)

function build_interaction(opt::LimitedConstantInteractionOption, disp::ElectronicDispersion)
    return LimitedConstantInteraction{_dim(disp),typeof(disp)}(opt.V0, opt.omega_c, disp)
end

function build_interaction(opt::ScreenedCoulombInteractionOption, disp)
    throw(ArgumentError("ScreenedCoulombInteraction requires a dynamical polarization propagator not supported purely via scalar TOML."))
end

function build_interaction(opt::CompositeInteractionOption, disp)
    # Map is cleaner than for-loop here
    ints = map(sub -> build_interaction(sub, disp), opt.interactions)
    return CompositeInteraction(ints...)
end

# ---------------------------------------------------------
# Field Factory
# ---------------------------------------------------------
build_field(opt::DirectChannelOption, D::Int) = DirectChannel()
build_field(opt::ExchangeChannelOption, D::Int) = ExchangeChannel(Symbol(opt.direction))
build_field(opt::ChargeDensityWaveOption, D::Int) = ChargeDensityWave(_svec(D, opt.q))
build_field(opt::SpinDensityWaveOption, D::Int) = SpinDensityWave(_svec(D, opt.q), Symbol(opt.direction))
build_field(opt::BCSReducedPairingOption, D::Int) = BCSReducedPairing(Symbol(opt.symmetry))
build_field(opt::FFLOPairingOption, D::Int) = FFLOPairing(_svec(D, opt.q), opt.h)
build_field(opt::PairDensityWaveOption, D::Int) = PairDensityWave(_svec(D, opt.q))

function build_field(opt::MomentumDependentPairingOption, grid::AbstractKGrid)
    return MomentumDependentPairing(grid; seed=Symbol(opt.seed), amp=opt.amplitude)
end

# ---------------------------------------------------------
# Task Validation
# ---------------------------------------------------------
"""
    validate_task_config(task::AbstractTaskOption)

Validates whether mandatory fields are provided for the given task and throws a user-friendly `ArgumentError` if they are missing.
"""
function validate_task_config(task::AbstractTaskOption)
    # Default fallback: do nothing, meaning valid
end

function validate_task_config(task::ScanSpectralFunctionOption)
    isnothing(task.qpath_points) && throw(ArgumentError("Task 'scan_spectral_function' requires `qpath_points` in TOML."))
    isnothing(task.qpath_labels) && throw(ArgumentError("Task 'scan_spectral_function' requires `qpath_labels` in TOML."))
    isnothing(task.omega_range) && throw(ArgumentError("Task 'scan_spectral_function' requires `omega_range` in TOML."))
end

function validate_task_config(task::ComputePhaseTransitionDataOption)
    isnothing(task.phi_range) && throw(ArgumentError("Task 'compute_phase_transition_data' requires `phi_range` in TOML."))
    isnothing(task.T_range) && throw(ArgumentError("Task 'compute_phase_transition_data' requires `T_range` in TOML."))
end

function validate_task_config(task::ComputeRenormalizedBandDataOption)
    isnothing(task.qpath_points) && throw(ArgumentError("Task 'compute_renormalized_band_data' requires `qpath_points` in TOML."))
    isnothing(task.qpath_labels) && throw(ArgumentError("Task 'compute_renormalized_band_data' requires `qpath_labels` in TOML."))
    isnothing(task.T_range) && throw(ArgumentError("Task 'compute_renormalized_band_data' requires `T_range` in TOML."))
end

function validate_task_config(task::ComputeZeemanPairingDataOption)
    isnothing(task.q_range) && throw(ArgumentError("Task 'compute_zeeman_pairing_data' requires `q_range` in TOML."))
end

# ---------------------------------------------------------
# Task Builder
# ---------------------------------------------------------
_parse_approx(name::String) = name == "ExactTrLn" ? ExactTrLn() : name == "RPA" ? RPA() : error("Unknown approximation type: $name")

"""
    build_task(task::AbstractTaskOption)

Transforms configuration structs into ready-to-use keyword arguments / unpacked arrays 
expected by the solver engines.
"""
function build_task(task::AbstractTaskOption)
    # Default fallback: just extract clean kwargs
    return extract_kwargs(task, exclude=(:type,))
end

function build_task(task::SolveGroundStateOption)
    approx = _parse_approx(task.approx)
    return (; phi_guess=task.phi_guess, T=task.T, approx, warm_start=task.warm_start)
end

function build_task(task::ComputePhaseTransitionDataOption)
    phis = collect(range(task.phi_range[1], task.phi_range[2], length=task.phi_points))
    Ts = collect(range(task.T_range[1], task.T_range[2], length=task.T_points))
    approx = _parse_approx(task.approx)
    return (; phis, Ts, phi_guess=task.phi_guess, approx, warm_start=task.warm_start)
end

function build_task(task::ComputeRenormalizedBandDataOption)
    Ts = collect(range(task.T_range[1], task.T_range[2], length=task.T_points))
    approx = _parse_approx(task.approx)
    if !isnothing(task.qpath_points)
        D = length(task.qpath_points[1])
        nodes = [_svec(D, pt) for pt in task.qpath_points]
        kpath = generate_kpath(nodes, task.qpath_labels; n_pts_per_segment=50) # default segment
        return (; Ts, kpath, phi_guess=task.phi_guess, approx, warm_start=task.warm_start)
    end
    return (; Ts, phi_guess=task.phi_guess, approx, warm_start=task.warm_start)
end

function build_task(task::ScanSpectralFunctionOption)
    omegas = collect(range(task.omega_range[1], task.omega_range[2], length=task.omega_points))
    if !isnothing(task.qpath_points)
        D = length(task.qpath_points[1])
        nodes = [_svec(D, pt) for pt in task.qpath_points]
        qpath = generate_kpath(nodes, task.qpath_labels; n_pts_per_segment=50)
        return (; qpath, omegas, T_val=task.T, eta=task.eta)
    end
    return (; omegas, T_val=task.T, eta=task.eta)
end

function build_task(task::ComputeZeemanPairingDataOption)
    q_vals = collect(range(task.q_range[1], task.q_range[2], length=task.q_points))
    approx = _parse_approx(task.approx)
    return (; q_vals, h_val=task.h_val, T_val=task.T_val, phi_guess=task.phi_guess, approx, warm_start=task.warm_start)
end

# ---------------------------------------------------------
# High-level Constructor
# ---------------------------------------------------------
"""
    build_from_config(config::EliashbergConfig)

Instantiate all physics models, interactions, kpoints, and order parameter fields
from the provided TOML configurations. Returns a NamedTuple with `(system, geometry, kpoints, model, interaction, field, task, plot)`.
"""
function build_from_config(config::EliashbergConfig)
    # 1. Geometry & Kpoints
    geometry = build_geometry(config.geometry)
    kpoints = build_kpoints(config.kpoints, geometry)
    D = _dim(geometry)

    # 2. Model
    raw_model = build_model(config.model, geometry)
    model = (config.model isa TightBindingOption && config.model.use_spinor) ? SpinorDispersion(raw_model) : raw_model

    # 3. Interaction
    interaction = build_interaction(config.interaction, model)

    # 4. Field
    if config.field isa MomentumDependentPairingOption
        field = build_field(config.field, kpoints)
    else
        field = build_field(config.field, D) # Pass dimension to enforce SVector
    end

    # 5. Extract Clean Kwargs
    system_kwargs = extract_kwargs(config.system)
    task_kwargs = build_task(config.task)
    plot_kwargs = extract_kwargs(config.plot, exclude=(:enable, :save_path))

    return (; system=system_kwargs, geometry, kpoints, model, interaction, field, task=task_kwargs, plot=(enable=config.plot.enable, path=config.plot.save_path, kwargs=plot_kwargs))
end