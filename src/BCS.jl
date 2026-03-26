module BCS

using StaticArrays
using Statistics
using LinearAlgebra
using QuadGK
using ..Constants
using ..Eliashberg: PhysicalModel, AuxiliaryField, Dispersion, ElectronicDispersion, PhononDispersion, 
                   AbstractKGrid, KGrid, SelfEnergy, Interaction, Smearing, Propagator, ε, band_structure,
                   CoulombInteraction, ElectronPhononInteraction, ScreenedInteraction, Polarization,
                   GapFunction, SpectralFunction, ElectronSpectralFunction, PhononSpectralFunction,
                   ElectronPropagator, PhononPropagator, GorkovPropagator

# BCS-specific feature modules
include("interactions.jl")
include("propagators.jl")
include("self_energies.jl")
include("smearings.jl")
include("solvers.jl")


export  AbstractKGrid, KGrid
export  FreeElectron, TightBinding, RenormalizedDispersion, ε, band_structure
export  EinsteinModel, DebyeModel, PolaritonModel, MonoatomicLatticeModel, ω
export  ConstantInteraction, LocalInteraction, YukawaInteraction, 
        LimitedConstantInteraction, BareCoulombInteraction, ScreenedCoulombInteraction, 
        StaticRPAPolarization, DynamicalRPAPolarization, χ, V
export  DebyePhononPropagator, RetardedPhononPropagator, 
        FreeElectronPropagator, G, D
export FermiDiracSmearing, BoseEinsteinSmearing, GaussianSmearing, f
export solve_bcs

end # module