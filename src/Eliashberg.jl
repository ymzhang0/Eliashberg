module Eliashberg

# 0. External Dependencies (Consolidated)
using LinearAlgebra
using StaticArrays
using Makie
using Unitful
using PhysicalConstants.CODATA2022
using QuadGK
using Optim

# 1. Fundamental Constants and Linear Algebra
include("Constants.jl") # Also includes la.jl

# 2. Sharded Abstract and Base Types (Topological Order)
include("Geometry/types.jl")
include("Models/types.jl")
include("Responses/types.jl")
include("Solvers/types.jl")

# 3. Domain Implementations
# Geometry (Real and Reciprocal space)
include("Geometry/lattice.jl")
include("Geometry/reciprocal_lattice.jl")

# Physical models and bare dispersions
include("Models/dispersions.jl")
include("Models/interactions.jl")

# Many-body responses and Mean-Field logic
include("Responses/mean_field.jl")
include("Responses/propagators.jl")
include("Responses/smearings.jl")
include("Responses/self_energies.jl")
include("Responses/susceptibilities.jl")

# Numerical solvers and algorithms
include("Solvers/integrals.jl")
include("Solvers/bcs_equations.jl")
include("Solvers/effective_action.jl")
include("Solvers/observables.jl")
include("Solvers/scanners.jl")

# Synchronization of Visualization Dependencies (Consolidated at top)

# Visualization wrappers
include("Visualization/utils.jl")
include("Visualization/geometry.jl")
include("Visualization/bands.jl")
include("Visualization/responses.jl")

# 4. Backward Compatibility and Aliases
const LindhardSusceptibility = GeneralizedSusceptibility

# 5. Centralized Exports

# Constants & LA
export Å, a0, Ry, me, e, ε0, h, ħ, kB, c, Ry2J, Ry2eV, Ha2J, Ha2eV, kB2meV, kB2eV, kB2Ha, A2Bohr
export σ₀, σ₁, σ₂, σ₃, pauli_matrices, γ⁰, γ¹, γ², γ³, gamma_matrices, commutator, anticommutator

# Geometry
export Lattice, ChainLattice, SquareLattice, HexagonalLattice, CubicLattice, FCCLattice, BCCLattice, AbstractKGrid, KGrid, KPath
export generate_1d_kgrid, generate_2d_kgrid, generate_3d_kgrid, reciprocal_vectors, generate_reciprocal_lattice, generate_kpath

# Models
export PhysicalModel, Dispersion, ElectronicDispersion, PhononDispersion, Interaction
export FreeElectron, TightBinding, Graphene, KagomeLattice, SSHModel, EinsteinModel, DebyeModel, PolaritonModel, MonoatomicLatticeModel
export CoulombInteraction, ElectronPhononInteraction, ScreenedInteraction, CombinedInteraction
export ConstantInteraction, LocalInteraction, YukawaInteraction, LimitedConstantInteraction, BareCoulombInteraction, ScreenedCoulombInteraction
export ε, ω, V

# Responses
export AuxiliaryField, StaticMeanField, DynamicalFluctuation, ChargeDensityWave, BCSReducedPairing, FFLOPairing, PairDensityWave
export MeanFieldDispersion, NormalNambuDispersion, normal_state_basis, gap_form_factor
export Propagator, PhononPropagator, ElectronPropagator, GorkovPropagator, SelfEnergy, Smearing, Polarization
export GeneralizedSusceptibility, LindhardSusceptibility, vertex_matrix, band_structure

# Solvers
export ApproximationLevel, ExactTrLn, RPA
export evaluate_action, solve_bcs, solve_ground_state, scan_instability_landscape, scan_spectral_function

# Visualization
export visualize_dispersion, dimensionality, visualize_landscape, visualize_spectral_function, visualize_phase_transition, visualize_renormalized_bands, visualize_zeeman_pairing_landscape, visualize_collective_modes
export visualize_lattice, visualize_reciprocal_space

end # module Eliashberg
