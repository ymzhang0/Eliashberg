module Eliashberg

include("Constants.jl")

# Core new architecture additions
include("types.jl")
include("dispersions.jl")
include("fields.jl")
include("susceptibilities.jl")
include("effective_action.jl")
include("observables.jl")

include("BCS.jl")
include("Utilities.jl")
include("visualization.jl")

using .Constants
using .BCS
using .Utilities
using .Visualization

for mod in (Constants, BCS, Utilities, Visualization)
    for name in names(mod)
        if name != nameof(mod)
            @eval export $name
        end
    end
end

export PhysicalModel, AuxiliaryField, ApproximationLevel, ExactTrLn, RPA
export ChargeDensityWave, EffectiveAction
export reconstructed_bands, LindhardSusceptibility, evaluate, solve_ground_state
export Dispersion, ElectronicDispersion, PhononDispersion

end # module Eliashberg
