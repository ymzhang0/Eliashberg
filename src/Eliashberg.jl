module Eliashberg

# 0. External Dependencies (Consolidated)
using LinearAlgebra
using StaticArrays
using AtomsBase
import AtomsBase: periodicity, PeriodicCell, FastSystem, ChemicalSpecies, mass
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

const TO = TimerOutput()

# 1. Fundamental Constants and Linear Algebra (Base Tier)
include("Constants.jl")
include("la.jl")

# 2. Abstract Tier (Topological Order - Only abstract types)
include("Geometry/types.jl")
include("Models/types.jl")
include("Responses/types.jl")
include("Solvers/types.jl")

# 2.5 Engine Tier (Pure execution and scheduling abstractions)
include("Engine/Engine.jl")
using .Engine: GridSample, BlockAxisLayout, UniformBlockLayout, VariableBlockLayout, AssemblySpectrum, DenseEigenSolver, SparseEigenSolverHook, bootstrap_engine_workers!, grid_samples, assemble_grid_vector, assemble_grid_matrix, assemble_sparse_grid_matrix, assemble_block_grid_matrix, assemble_sparse_block_grid_matrix, assemble_block_diagonal_matrix, assemble_sparse_block_diagonal_matrix, solve_assembled_eigensystem, integrate_grid, distributed_map_grid

# 3. Data Structure Tier (Concrete structs and constructors)
# Geometry
include("Geometry/crystal.jl")

# Models
include("Models/dispersions.jl")
include("Models/interactions.jl")

# Interfaces
include("Interfaces/Interfaces.jl")

# Responses
include("Responses/fields.jl")
include("Responses/propagators.jl")
include("Responses/smearings.jl")
include("Responses/self_energies.jl")
include("Responses/vertex.jl")

# 4. Method & Logic Tier (Computational algorithms and dispatch)
# Geometry methods
include("Geometry/reciprocal_lattice.jl")

# Response data objects
include("Responses/data_types.jl")

# Interface data objects
include("Interfaces/Wannier90/data_types.jl")

# Models methods
include("Models/evaluators.jl")

# Responses methods
include("Responses/mean_field.jl")
include("Responses/susceptibilities.jl")
include("Models/bosons.jl")

# Numerical solvers and algorithms
include("Solvers/integrals.jl")
include("Solvers/bcs_equations.jl")
include("Solvers/sampled_hamiltonians.jl")
include("Solvers/effective_action.jl")
include("Solvers/observables.jl")
include("Solvers/scanners.jl")
include("Solvers/spectra.jl")

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

plot_dispersion_curves(args...; kwargs...) = _call_visualization(:plot_dispersion_curves, args...; kwargs...)
plot_dispersion_surface(args...; kwargs...) = _call_visualization(:plot_dispersion_surface, args...; kwargs...)
plot_band_structure(args...; kwargs...) = _call_visualization(:plot_band_structure, args...; kwargs...)
plot_wannier90_band_structure(args...; kwargs...) = _call_visualization(:plot_wannier90_band_structure, args...; kwargs...)
plot_wannier90_tb_band_comparison(args...; kwargs...) = _call_visualization(:plot_wannier90_tb_band_comparison, args...; kwargs...)
plot_fermi_surface(args...; kwargs...) = _call_visualization(:plot_fermi_surface, args...; kwargs...)
plot_renormalized_bands(args...; kwargs...) = _call_visualization(:plot_renormalized_bands, args...; kwargs...)
plot_landscape(args...; kwargs...) = _call_visualization(:plot_landscape, args...; kwargs...)
plot_spectral_function(args...; kwargs...) = _call_visualization(:plot_spectral_function, args...; kwargs...)
plot_phase_transition(args...; kwargs...) = _call_visualization(:plot_phase_transition, args...; kwargs...)
plot_zeeman_pairing_landscape(args...; kwargs...) = _call_visualization(:plot_zeeman_pairing_landscape, args...; kwargs...)
plot_collective_modes(args...; kwargs...) = _call_visualization(:plot_collective_modes, args...; kwargs...)
visualize_dispersion(args...; kwargs...) = _call_visualization(:visualize_dispersion, args...; kwargs...)
visualize_landscape(args...; kwargs...) = _call_visualization(:visualize_landscape, args...; kwargs...)
visualize_spectral_function(args...; kwargs...) = _call_visualization(:visualize_spectral_function, args...; kwargs...)
visualize_phase_transition(args...; kwargs...) = _call_visualization(:visualize_phase_transition, args...; kwargs...)
visualize_renormalized_bands(args...; kwargs...) = _call_visualization(:visualize_renormalized_bands, args...; kwargs...)
visualize_zeeman_pairing_landscape(args...; kwargs...) = _call_visualization(:visualize_zeeman_pairing_landscape, args...; kwargs...)
visualize_collective_modes(args...; kwargs...) = _call_visualization(:visualize_collective_modes, args...; kwargs...)
visualize_lattice(args...; kwargs...) = _call_visualization(:visualize_lattice, args...; kwargs...)
visualize_reciprocal_space(args...; kwargs...) = _call_visualization(:visualize_reciprocal_space, args...; kwargs...)

# 6. Backward Compatibility and Aliases
const LindhardSusceptibility = GeneralizedSusceptibility

# 7. Centralized Exports

# Constants & LA
export Å, a0, Ry, me, e, ε0, h, ħ, kB, c, Ry2J, Ry2eV, Ha2J, Ha2eV, kB2meV, kB2eV, kB2Ha, A2Bohr
export σ₀, σ₁, σ₂, σ₃, pauli_matrices, γ⁰, γ¹, γ², γ³, gamma_matrices, commutator, anticommutator

# Geometry
export Crystal, ChainLattice, SquareLattice, HexagonalLattice, CubicLattice, FCCLattice, BCCLattice, AbstractKGrid, KGrid, KPath
export ibrav, qe_lattice, cubic_p_lattice, cubic_f_lattice, cubic_i_lattice, hexagonal_p_lattice, trigonal_r_lattice, tetragonal_p_lattice, tetragonal_i_lattice
export orthorhombic_p_lattice, orthorhombic_base_centered_lattice, orthorhombic_face_centered_lattice, orthorhombic_body_centered_lattice, monoclinic_p_lattice, monoclinic_base_centered_lattice, triclinic_lattice
export scaled_positions, positions, append_atom!, set_scaled_positions!, set_positions!, set_cell!, cartesian_basis, generate_1d_kgrid, generate_2d_kgrid, generate_3d_kgrid, reciprocal_vectors, generate_reciprocal_lattice, generate_kpath
export build_spglib_cell, bravais_lattice, generate_irreducible_kgrid, periodic_rank

# Models
export PhysicalModel, Dispersion, ElectronicDispersion, PhononDispersion, Interaction
export FreeElectron, TightBinding, SpinorDispersion, MultiOrbitalTightBinding, Graphene, KagomeLattice, SSHModel, EinsteinModel, DebyeModel, PolaritonModel, MonoatomicLatticeModel
export CoulombInteraction, ElectronPhononInteraction, ScreenedInteraction, CombinedInteraction, CompositeInteraction
export ConstantInteraction, LocalInteraction, YukawaInteraction, LimitedConstantInteraction, BareCoulombInteraction, ScreenedCoulombInteraction, MediatedInteraction
export ε, ω, V
export parse_wannier90_hr, parse_wannier90_tb, cell_from_wannier90_tb, periodic_cell_from_wannier90_tb, build_model_from_wannier90
export parse_wannier90_band_dat, parse_wannier90_kpoints, parse_wannier90_labelinfo
export kpath_from_wannier90_bands, kpath_from_wannier90_kpoints, band_data_from_wannier90_bands, compare_wannier90_tb_to_bands
export Wannier90BandComparison

# Responses
export AuxiliaryField, StaticMeanField, DynamicalFluctuation, DirectChannel, ExchangeChannel, ChargeDensityWave, SpinDensityWave, BCSReducedPairing, FFLOPairing, PairDensityWave, CompositeField
export MeanFieldDispersion, NormalNambuDispersion, normal_state_basis, gap_form_factor
export Propagator, PhononPropagator, ElectronPropagator, GorkovPropagator, SelfEnergy, Smearing, Polarization
export GeneralizedSusceptibility, LindhardSusceptibility, vertex_matrix, band_structure
export RPABoson, CachedBoson, evaluate_boson_propagator, materialize_boson
export BandStructureData, DispersionSurfaceData, FermiSurfaceData, LandscapeLineData, LandscapeSurfaceData
export PhaseDiagramData, RenormalizedBandData, SpectralMapData, ZeemanPairingData, CoexistenceLandscapeData

# Solvers
export ApproximationLevel, ExactTrLn, RPA, TO
export Engine, GridSample, BlockAxisLayout, UniformBlockLayout, VariableBlockLayout, AssemblySpectrum, DenseEigenSolver, SparseEigenSolverHook, bootstrap_engine_workers!, grid_samples, assemble_grid_vector, assemble_grid_matrix, assemble_sparse_grid_matrix, assemble_block_grid_matrix, assemble_sparse_block_grid_matrix, assemble_block_diagonal_matrix, assemble_sparse_block_diagonal_matrix, solve_assembled_eigensystem, integrate_grid, distributed_map_grid
export SampledHamiltonianAssembly, assemble_sampled_hamiltonian, solve_sampled_hamiltonian
export evaluate_action, solve_bcs, solve_ground_state, scan_instability_landscape, scan_spectral_function, scan_rpa_spectral_function_hpc
export compute_dispersion_surface_data, compute_band_data, compute_fermi_surface_volume
export compute_landscape_line_data, compute_landscape_surface_data, compute_landscape_axes
export compute_phase_transition_data, compute_renormalized_band_data, compute_zeeman_pairing_data, compute_collective_mode_spectral_data, compute_coexistence_landscape

# Visualization
export plot_dispersion_curves, plot_dispersion_surface, plot_band_structure, plot_wannier90_band_structure, plot_wannier90_tb_band_comparison, plot_fermi_surface, plot_renormalized_bands
export plot_landscape, plot_spectral_function, plot_phase_transition, plot_zeeman_pairing_landscape, plot_collective_modes
export visualize_dispersion, dimensionality, visualize_landscape, visualize_spectral_function, visualize_phase_transition, visualize_renormalized_bands, visualize_zeeman_pairing_landscape, visualize_collective_modes
export visualize_lattice, visualize_reciprocal_space

# 8. Configuration System
include("Config/Config.jl")
using .Config: EliashbergConfig, load_config, build_from_config
export EliashbergConfig, load_config, build_from_config

end # module Eliashberg
