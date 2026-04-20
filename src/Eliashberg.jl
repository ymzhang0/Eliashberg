module Eliashberg

# 0. External Dependencies (Consolidated)
using LinearAlgebra
using StaticArrays
using AtomsBase
using HDF5
using JLD2
using AtomsIO
import AtomsBase: periodicity, PeriodicCell, FastSystem, ChemicalSpecies, mass,
    periodic_system, isolated_system, atomic_system, Atom,
    n_dimensions, position, atomic_symbol,
    atomic_number, species, cell_vectors
import Brillouin
using Spglib
using Unitful
using PhysicalConstants.CODATA2022
using QuadGK
using Optim
using Distributed
using SparseArrays
using Logging
using ProgressLogging
using TimerOutputs
using Makie: Makie
import Makie: plot

const TO = TimerOutput()

# 1. Fundamental Constants and Linear Algebra (Base Tier)
include("Numerics/Constants.jl")
include("Numerics/la.jl")

# 2. Abstract Tier (Topological Order - Only abstract types)
include("Geometry/types.jl")
include("Geometry/Lattices.jl")
include("Geometry/symmetry_points.jl")
include("Models/types.jl")
include("Correlations/types.jl")
include("Numerics/smearings.jl")
include("Solvers/types.jl")

# 2.5 Engine Tier (Pure execution and scheduling abstractions)
include("Engine/Engine.jl")
using .Engine: GridSample, BlockAxisLayout, UniformBlockLayout, VariableBlockLayout, AssemblySpectrum, DenseEigenSolver, SparseEigenSolverHook, bootstrap_engine_workers!, grid_samples, assemble_grid_vector, assemble_grid_matrix, assemble_sparse_grid_matrix, assemble_block_grid_matrix, assemble_sparse_block_grid_matrix, assemble_block_diagonal_matrix, assemble_sparse_block_diagonal_matrix, solve_assembled_eigensystem, integrate_grid, distributed_map_grid

# 3. Data Structure Tier (Concrete structs and constructors)
# Geometry
include("Geometry/crystal.jl")
include("Geometry/predefined_structures.jl")

# Models
include("Models/dispersions.jl")
include("Models/predefined_models.jl")
include("Models/calculator.jl")
include("Models/interactions.jl")

# Interfaces
include("Interfaces/Interfaces.jl")


# 4. Method & Logic Tier (Computational algorithms and dispatch)
# Geometry methods
include("Geometry/reciprocal_lattice.jl")

# Response data objects
include("Data/types.jl")
include("Data/factories.jl")

# Correlation data objects needed by model evaluators
include("Correlations/self_energies.jl")

# Models methods
include("Models/Hamiltonian.jl")

# Correlations methods
# Correlations
include("Correlations/mean_fields.jl")
include("Correlations/propagators.jl")
include("Correlations/vertex.jl")
include("Correlations/renormalization.jl")
include("Correlations/susceptibilities.jl")
include("Correlations/bosons.jl")

# Numerical solvers and algorithms
include("Solvers/Thermodynamics/self_consistency.jl")
include("Solvers/Thermodynamics/minimization.jl")
include("Solvers/Thermodynamics/bse.jl")
include("Solvers/sampled_hamiltonians.jl")
include("Solvers/Actions/exact.jl")
include("Solvers/Actions/rpa.jl")
include("Solvers/Observables/bands.jl")
include("Solvers/Observables/phase_transition.jl")

# 5. Visualization-adjacent pure utilities
include("Visualization/core_utils.jl")

# 5.5 Logging helpers
include("LoggingUtils.jl")

const _VISUALIZATION_LOADED = Ref(false)

function load_visualization!()
    _VISUALIZATION_LOADED[] && return nothing

    Base.include(@__MODULE__, joinpath(@__DIR__, "Visualization", "Visualization.jl"))
    _VISUALIZATION_LOADED[] = true
    return nothing
end

function _call_visualization(func::Symbol, args...; kwargs...)
    load_visualization!()
    visualization = Base.invokelatest(getproperty, @__MODULE__, :Visualization)
    plotter = Base.invokelatest(getproperty, visualization, func)
    return Base.invokelatest(plotter, args...; kwargs...)
end

_plot_dispersion_curves(args...; kwargs...) = _call_visualization(:_plot_dispersion_curves, args...; kwargs...)
_plot_dispersion_surface(args...; kwargs...) = _call_visualization(:_plot_dispersion_surface, args...; kwargs...)
_plot_band_structure(args...; kwargs...) = _call_visualization(:_plot_band_structure, args...; kwargs...)
_plot_wannier90_band_structure(args...; kwargs...) = _call_visualization(:_plot_wannier90_band_structure, args...; kwargs...)
_plot_wannier90_tb_band_comparison(args...; kwargs...) = _call_visualization(:_plot_wannier90_tb_band_comparison, args...; kwargs...)
_plot_fermi_surface(args...; kwargs...) = _call_visualization(:_plot_fermi_surface, args...; kwargs...)
_plot_renormalized_bands(args...; kwargs...) = _call_visualization(:_plot_renormalized_bands, args...; kwargs...)
_plot_landscape(args...; kwargs...) = _call_visualization(:_plot_landscape, args...; kwargs...)
_plot_spectral_function(args...; kwargs...) = _call_visualization(:_plot_spectral_function, args...; kwargs...)
_plot_phase_transition(args...; kwargs...) = _call_visualization(:_plot_phase_transition, args...; kwargs...)
_plot_zeeman_pairing_landscape(args...; kwargs...) = _call_visualization(:_plot_zeeman_pairing_landscape, args...; kwargs...)
_plot_collective_modes(args...; kwargs...) = _call_visualization(:_plot_collective_modes, args...; kwargs...)
_plot_lattice(args...; kwargs...) = _call_visualization(:_plot_lattice, args...; kwargs...)
_plot_reciprocal_space(args...; kwargs...) = _call_visualization(:_plot_reciprocal_space, args...; kwargs...)

# Define the set of types handled by the visualization module for lazy-loading
const VisualizationTypes = Union{
    BandStructureData,DispersionSurfaceData,FermiSurfaceData,
    LandscapeLineData,LandscapeSurfaceData,PhaseDiagramData,
    SpectralMapData,ZeemanPairingData,RenormalizedBandData,
    Wannier90BandComparison,PeriodicCell,AbstractSystem,AbstractKGrid
}

function Makie.plot(data::VisualizationTypes; kwargs...)
    return _call_visualization(:plot, data; kwargs...)
end

function Makie.plot(data::AbstractVector{<:VisualizationTypes}; kwargs...)
    return _call_visualization(:plot, data; kwargs...)
end
# 6. Backward Compatibility and Aliases
const LindhardSusceptibility = GeneralizedSusceptibility

# 7. Centralized Exports

# Constants & LA
export Å, a0, Ry, me, e, ε0, h, ħ, kB, c, Ry2J, Ry2eV, Ha2J, Ha2eV, kB2meV, kB2eV, kB2Ha, A2Bohr
export σ₀, σ₁, σ₂, σ₃, pauli_matrices, γ⁰, γ¹, γ², γ³, gamma_matrices, commutator, anticommutator

# Geometry
export ChainLattice, SquareLattice, HexagonalLattice2D, HexagonalLattice, RectangularLattice, CenteredRectangularLattice, ObliqueLattice
export SimpleCubic, FaceCenteredCubic, BodyCenteredCubic
export AbstractKGrid, KGrid, KPath
export periodic_system, isolated_system, atomic_system, Atom
export periodic_rank, primitive_vectors, bravais_lattice, generate_irreducible_kgrid, reciprocal_lattice, generate_kgrid, generate_kpath
export path_points, path_branches, path_node_metadata, symmetry_path
export load_system, save_system
export atomic_chain, square_lattice, cF, cI, cF8, aluminium, copper, iron, niobium, vanadium, diamond, silicon, germanium, zincblende, sic, nacl, graphene, graphite, kagome, ssh_lattice
export PREDEFINED_STRUCTURE_REGISTRY, register_structure!
export GrapheneModel, KagomeModel, SSHModel


# Models
export PhysicalModel, Dispersion, ElectronicDispersion, PhononDispersion, Interaction
export FreeElectron, TightBinding, SpinorDispersion, MultiOrbitalTightBinding, EinsteinModel, DebyeModel, PolaritonModel, MonoatomicLatticeModel
export CoulombInteraction, ElectronPhononInteraction, ScreenedInteraction, CombinedInteraction, CompositeInteraction
export ConstantInteraction, LocalInteraction, YukawaInteraction, LimitedConstantInteraction, BareCoulombInteraction, ScreenedCoulombInteraction, MediatedInteraction
export ε, ω, V
export parse_wannier90_hr, parse_wannier90_tb, cell_from_wannier90_tb, periodic_cell_from_wannier90_tb, build_model_from_wannier90
export parse_wannier90_band_dat, parse_wannier90_kpoints, parse_wannier90_labelinfo
export kpath_from_wannier90_bands, kpath_from_wannier90_kpoints, compare_wannier90_tb_to_bands
export parse_quantum_espresso_bands, parse_quantum_espresso_xml, parse_quantum_espresso_cell
export Wannier90BandComparison

# Correlations
export AuxiliaryField, StaticMeanField, DynamicalFluctuation, DirectChannel, ExchangeChannel, ChargeDensityWave, SpinDensityWave, BCSReducedPairing, FFLOPairing, PairDensityWave, CompositeField
export MeanFieldDispersion, NormalNambuDispersion, normal_state_basis, gap_form_factor
export Propagator, PhononPropagator, ElectronPropagator, GorkovPropagator, SelfEnergy, Smearing, Polarization
export GeneralizedSusceptibility, LindhardSusceptibility, vertex_matrix, H, D, ε, ω, diagonalize
export RPABoson, CachedBoson, evaluate_boson_propagator, materialize_boson
export BandStructureData, DispersionSurfaceData, FermiSurfaceData, LandscapeLineData, LandscapeSurfaceData
export compute_landscape_line_data, compute_landscape_surface_data
export PhaseDiagramData, RenormalizedBandData, SpectralMapData, ZeemanPairingData, CoexistenceLandscapeData, Wannier90BandComparison

# Solvers
export ApproximationLevel, ExactTrLn, RPA, TO
export Engine, GridSample, BlockAxisLayout, UniformBlockLayout, VariableBlockLayout, AssemblySpectrum, DenseEigenSolver, SparseEigenSolverHook, bootstrap_engine_workers!, grid_samples, assemble_grid_vector, assemble_grid_matrix, assemble_sparse_grid_matrix, assemble_block_grid_matrix, assemble_sparse_block_grid_matrix, assemble_block_diagonal_matrix, assemble_sparse_block_diagonal_matrix, solve_assembled_eigensystem, integrate_grid, distributed_map_grid
export SampledHamiltonianAssembly, assemble_sampled_hamiltonian, solve_sampled_hamiltonian
export evaluate_action, solve_bcs, solve_ground_state, scan_instability_landscape, spectral_function, scan_rpa_spectral_function_hpc
export calculate_bands, calculate_fermi_surface
export phase_transition, band_renormalization, Zeeman_pairing, collective_mode_spectral, compute_coexistence_landscape

# Visualization
export plot, dimensionality

# 8. Configuration System
include("Config/Config.jl")
using .Config: EliashbergConfig, load_config, build_from_config
export EliashbergConfig, load_config, build_from_config

# 9. Result File IO
include("IO/IO.jl")
export save, load

end # module Eliashberg
