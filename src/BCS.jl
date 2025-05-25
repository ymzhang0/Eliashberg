module BCS

using LinearAlgebra
using QuadGK

include("types.jl")
include("dispersions.jl")
include("interactions.jl")
include("propagators.jl")
include("self_energies.jl")
include("smearings.jl")
include("solvers.jl")


export FreeElectron_1d, TightBinding_1d, RenormalizedDispersion, ε
export EinsteinModel, DebyeModel, PolaritonModel, MonoatomicLatticeModel, ω
export ConstantInteraction_1d, LocalInteraction_1d, YukawaInteraction_1d, LimitedConstantInteraction, BareCoulombInteraction, ScreenedCoulombInteraction, V
export DebyePhononPropagator, RetardedPhononPropagator, FreeElectronPropagator, G, D
export FermiDiracSmearing, BoseEinsteinSmearing, GaussianSmearing, f
export solve_bcs

end # module