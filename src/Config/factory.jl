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

function _build_configured_kpath(
    geometry::AbstractSystem,
    points::Union{AbstractVector{<:AbstractVector},Nothing},
    labels::Union{AbstractVector{<:AbstractString},Nothing},
    npoints_each_line::Integer,
)
    # Case 1: Neither provided -> Automatic from geometry
    if isnothing(points) && isnothing(labels)
        return generate_kpath(geometry; n_pts_per_segment=Int(npoints_each_line))
    end

    # Case 2: Only labels provided -> Look up from symmetry_path
    if isnothing(points)
        D = AtomsBase.n_dimensions(geometry)
        spath = symmetry_path(geometry)
        isnothing(spath) && throw(ArgumentError("No symmetry path metadata defined for this geometry. Please provide `qpath_points` explicitly."))

        # Check if all labels exist
        for l in labels
            haskey(spath.points, String(l)) || throw(ArgumentError("Symmetry point '$l' not found for this lattice. Available: $(keys(spath.points))"))
        end

        B = reciprocal_lattice(geometry)
        nodes = [B * spath.points[String(l)] for l in labels]
        return generate_kpath(nodes, collect(String.(labels)); n_pts_per_segment=Int(npoints_each_line))
    end

    # Case 3: Points (and optional labels) provided
    D = length(points[1])
    nodes = [_svec(D, pt) for pt in points]
    final_labels = isnothing(labels) ? ["pt$i" for i in 1:length(nodes)] : collect(String.(labels))
    return generate_kpath(nodes, final_labels; n_pts_per_segment=Int(npoints_each_line))
end

# ---------------------------------------------------------
# Geometry Factory
# ---------------------------------------------------------
# --- Internal Lattice Builders ---

build_lattice(opt::ChainLatticeOption) = ChainLattice(opt.a)
build_lattice(opt::SquareLatticeOption) = SquareLattice(opt.a)
build_lattice(opt::HexagonalLatticeOption) = HexagonalLattice2D(opt.a)
build_lattice(opt::CubicLatticeOption) = SimpleCubic(opt.a)
build_lattice(opt::FCCLatticeOption) = FaceCenteredCubic(opt.a)
build_lattice(opt::BCCLatticeOption) = BodyCenteredCubic(opt.a)

# --- Top-level Geometry Builders (Always returning AbstractSystem) ---

function build_geometry(opt::PredefinedStructureOption)
    name = opt.name
    if name == "atomic_chain"
        return atomic_chain(opt.a, opt.element)
    elseif name == "square_lattice"
        return square_lattice(opt.a, opt.element)
    elseif name == "diamond"
        return diamond(opt.element, opt.a)
    elseif name == "silicon"
        return silicon(opt.a)
    elseif name == "germanium"
        return germanium(opt.a)
    elseif name == "zincblende"
        return zincblende(opt.element, opt.element2, opt.a)
    elseif name == "sic"
        return sic(opt.a)
    elseif name == "nacl"
        return nacl(opt.a)
    elseif name == "graphene"
        return graphene(opt.a)
    elseif name == "graphite"
        return graphite(opt.a, opt.c)
    elseif name == "kagome"
        return kagome(opt.a)
    elseif name == "ssh_lattice"
        return ssh_lattice(opt.a)
    else
        throw(ArgumentError("Unknown predefined structure name: $name"))
    end
end

function build_geometry(opt::CustomStructureOption)
    lattice = build_lattice(opt.lattice)
    # Process atoms: TOML provides Vector of Dicts [ {element="C", position=[x,y]}, ... ]
    processed_atoms = [
        Symbol(a["element"]) => _svec(_dim(lattice), a["position"])
        for a in opt.atoms
    ]
    return periodic_system(processed_atoms, lattice; fractional=true)
end

# build_geometry(opt::QELatticeOption) removed.

function build_kpoints(opt::KpointsOptions, geometry)
    D = _dim(geometry)
    sz = opt.grid

    # 统一转换网格大小为对应维度的数组或元组
    sz_array = sz isa Int ? fill(sz, D) : sz
    length(sz_array) == D || throw(DimensionMismatch("kpoints grid length must match lattice dimension $D"))

    return generate_kgrid(geometry, Tuple(sz_array)...)
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

build_model(opt::KagomeModelOption, cell) = KagomeModel(opt.t, opt.EF)
build_model(opt::GrapheneModelOption, cell) = GrapheneModel(opt.t, opt.EF)
build_model(opt::SSHModelOption, cell) = SSHModel(opt.t1, opt.t2, opt.EF)
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

function validate_task_config(task::SpectralFunctionOption)
    isnothing(task.omega_range) && throw(ArgumentError("Task 'spectral_function' requires `omega_range` in TOML."))
end

function validate_task_config(task::PhaseTransitionOption)
    isnothing(task.phi_range) && throw(ArgumentError("Task 'phase_transition' requires `phi_range` in TOML."))
    isnothing(task.T_range) && throw(ArgumentError("Task 'phase_transition' requires `T_range` in TOML."))
end

function validate_task_config(task::BandRenormalizationOption)
    isnothing(task.T_range) && throw(ArgumentError("Task 'band_renormalization' requires `T_range` in TOML."))
end

function validate_task_config(task::ZeemanPairingOption)
    isnothing(task.qs_range) && throw(ArgumentError("Task 'Zeeman_pairing' requires `qs_range` in TOML."))
end

function validate_task_config(task::CollectiveModeSpectralOption)
    # Automatic path generation supported
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
function build_task(task::AbstractTaskOption, geometry=nothing)
    # Default fallback: just extract clean kwargs
    return extract_kwargs(task, exclude=(:type,))
end

function build_task(task::SolveGroundStateOption, geometry=nothing)
    approx = _parse_approx(task.approx)
    return (; phi_guess=task.phi_guess, T=task.temperature, approx, warm_start=task.warm_start)
end

function build_task(task::PhaseTransitionOption, geometry=nothing)
    phis = collect(range(task.phi_range[1], task.phi_range[2], length=task.phi_points))
    Ts = collect(range(task.temperature_range[1], task.temperature_range[2], length=task.temperature_points))
    approx = _parse_approx(task.approx)
    return (; phis, Ts, phi_guess=task.phi_guess, approx, warm_start=task.warm_start)
end

function build_task(task::BandRenormalizationOption, geometry)
    Ts = collect(range(task.temperature_range[1], task.temperature_range[2], length=task.temperature_points))
    approx = _parse_approx(task.approx)

    # Path is optional in options, but if we have labels or points or nothing (automatic), we build it.
    kpath = _build_configured_kpath(geometry, task.qpath_points, task.qpath_labels, task.npoints_each_line)
    return (; Ts, kpath, phi_guess=task.phi_guess, approx, warm_start=task.warm_start)
end

function build_task(task::SpectralFunctionOption, geometry)
    omegas = collect(range(task.omega_range[1], task.omega_range[2], length=task.omega_points))
    qpath = _build_configured_kpath(geometry, task.qpath_points, task.qpath_labels, task.npoints_each_line)
    return (; qpath, omegas, T=task.temperature, η=task.eta)
end

function build_task(task::ZeemanPairingOption, geometry=nothing)
    qs = collect(range(task.qs_range[1], task.qs_range[2], length=task.qs_points))
    approx = _parse_approx(task.approx)
    return (; qs, h=task.h, T=task.temperature, phi_guess=task.phi_guess, approx, warm_start=task.warm_start)
end

function build_task(task::CollectiveModeSpectralOption, geometry)
    approx = _parse_approx(task.approx)
    qpath = _build_configured_kpath(geometry, task.qpath_points, task.qpath_labels, task.npoints_each_line)
    return (;
        qpath,
        T=task.temperature,
        omega_max_factor=task.omega_max_factor,
        n_omegas=task.n_omegas,
        η=task.eta,
        phi_guess=task.phi_guess,
        approx,
    )
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
    validate_task_config(config.task)

    # 1. Geometry & Kpoints
    geometry = build_geometry(config.geometry)
    kpoints = build_kpoints(config.kpoints, geometry)
    D = _dim(geometry)

    # 2. Model
    raw_model = build_model(config.model, geometry)
    model = (config.model isa TightBindingOption && config.model.spinor) ? SpinorDispersion(raw_model) : raw_model

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
    task_kwargs = build_task(config.task, geometry)
    plot_kwargs = extract_kwargs(config.plot, exclude=(:enable, :save_path))

    return (; system=system_kwargs, geometry, kpoints, model, interaction, field, task=task_kwargs, plot=(enable=config.plot.enable, path=config.plot.save_path, kwargs=plot_kwargs))
end
