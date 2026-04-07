module Eliashberg

# 0. External Dependencies (Consolidated)
using LinearAlgebra
using StaticArrays
using Makie
using AtomsBase
using Spglib
using Unitful
using PhysicalConstants.CODATA2022
using QuadGK
using Optim
using Distributed
using SparseArrays

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

# Models methods
include("Models/evaluators.jl")

# Responses methods
include("Responses/mean_field.jl")
include("Responses/susceptibilities.jl")

# Numerical solvers and algorithms
include("Solvers/integrals.jl")
include("Solvers/bcs_equations.jl")
include("Solvers/sampled_hamiltonians.jl")
include("Solvers/effective_action.jl")
include("Solvers/observables.jl")
include("Solvers/scanners.jl")
include("Solvers/spectra.jl")

# 5. Visualization Tier
include("Visualization/utils.jl")
include("Visualization/geometry.jl")
include("Visualization/bands.jl")
include("Visualization/responses.jl")

# 6. Backward Compatibility and Aliases
const LindhardSusceptibility = GeneralizedSusceptibility

# 7. Centralized Exports

# Constants & LA
export Å, a0, Ry, me, e, ε0, h, ħ, kB, c, Ry2J, Ry2eV, Ha2J, Ha2eV, kB2meV, kB2eV, kB2Ha, A2Bohr
export σ₀, σ₁, σ₂, σ₃, pauli_matrices, γ⁰, γ¹, γ², γ³, gamma_matrices, commutator, anticommutator

# Geometry
export Lattice, Crystal, ChainLattice, SquareLattice, HexagonalLattice, CubicLattice, FCCLattice, BCCLattice, AbstractKGrid, KGrid, KPath
export ibrav, qe_lattice, cubic_p_lattice, cubic_f_lattice, cubic_i_lattice, hexagonal_p_lattice, trigonal_r_lattice, tetragonal_p_lattice, tetragonal_i_lattice
export orthorhombic_p_lattice, orthorhombic_base_centered_lattice, orthorhombic_face_centered_lattice, orthorhombic_body_centered_lattice, monoclinic_p_lattice, monoclinic_base_centered_lattice, triclinic_lattice
export scaled_positions, positions, append_atom!, set_scaled_positions!, set_positions!, set_cell!, cartesian_basis, generate_1d_kgrid, generate_2d_kgrid, generate_3d_kgrid, reciprocal_vectors, generate_reciprocal_lattice, generate_kpath
export build_spglib_cell, generate_irreducible_kgrid

# Models
export PhysicalModel, Dispersion, ElectronicDispersion, PhononDispersion, Interaction
export FreeElectron, TightBinding, SpinorDispersion, MultiOrbitalTightBinding, Graphene, KagomeLattice, SSHModel, EinsteinModel, DebyeModel, PolaritonModel, MonoatomicLatticeModel
export CoulombInteraction, ElectronPhononInteraction, ScreenedInteraction, CombinedInteraction, CompositeInteraction
export ConstantInteraction, LocalInteraction, YukawaInteraction, LimitedConstantInteraction, BareCoulombInteraction, ScreenedCoulombInteraction
export ε, ω, V
export parse_wannier90_hr, parse_wannier90_tb, cell_from_wannier90_tb, build_model_from_wannier90

# Responses
export AuxiliaryField, StaticMeanField, DynamicalFluctuation, DirectChannel, ExchangeChannel, ChargeDensityWave, SpinDensityWave, BCSReducedPairing, FFLOPairing, PairDensityWave, CompositeField
export MeanFieldDispersion, NormalNambuDispersion, normal_state_basis, gap_form_factor
export Propagator, PhononPropagator, ElectronPropagator, GorkovPropagator, SelfEnergy, Smearing, Polarization
export GeneralizedSusceptibility, LindhardSusceptibility, vertex_matrix, band_structure
export BandStructureData, DispersionSurfaceData, FermiSurfaceData, LandscapeLineData, LandscapeSurfaceData
export PhaseDiagramData, RenormalizedBandData, SpectralMapData, ZeemanPairingData, CoexistenceLandscapeData

# Solvers
export ApproximationLevel, ExactTrLn, RPA
export Engine, GridSample, BlockAxisLayout, UniformBlockLayout, VariableBlockLayout, AssemblySpectrum, DenseEigenSolver, SparseEigenSolverHook, bootstrap_engine_workers!, grid_samples, assemble_grid_vector, assemble_grid_matrix, assemble_sparse_grid_matrix, assemble_block_grid_matrix, assemble_sparse_block_grid_matrix, assemble_block_diagonal_matrix, assemble_sparse_block_diagonal_matrix, solve_assembled_eigensystem, integrate_grid, distributed_map_grid
export SampledHamiltonianAssembly, assemble_sampled_hamiltonian, solve_sampled_hamiltonian
export evaluate_action, solve_bcs, solve_ground_state, scan_instability_landscape, scan_spectral_function, scan_rpa_spectral_function_hpc
export compute_dispersion_surface_data, compute_band_data, compute_fermi_surface_volume
export compute_landscape_line_data, compute_landscape_surface_data, compute_landscape_axes
export compute_phase_transition_data, compute_renormalized_band_data, compute_zeeman_pairing_data, compute_collective_mode_spectral_data, compute_coexistence_landscape

# Visualization
export plot_dispersion_curves, plot_dispersion_surface, plot_band_structure, plot_fermi_surface, plot_renormalized_bands
export plot_landscape, plot_spectral_function, plot_phase_transition, plot_zeeman_pairing_landscape, plot_collective_modes
export visualize_dispersion, dimensionality, visualize_landscape, visualize_spectral_function, visualize_phase_transition, visualize_renormalized_bands, visualize_zeeman_pairing_landscape, visualize_collective_modes
export visualize_lattice, visualize_reciprocal_space

end # module Eliashberg
