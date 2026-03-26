module Eliashberg

include("Constants.jl")

# Core new architecture additions
include("types.jl")
include("dispersions.jl")
include("fields.jl")
include("susceptibilities.jl")
include("scanners.jl")
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
export StaticMeanField, DynamicalFluctuation
export reconstructed_bands, LindhardSusceptibility, evaluate, solve_ground_state
export scan_instability_landscape, scan_spectral_function
export Dispersion, ElectronicDispersion, PhononDispersion, TightBinding
export KPath, KGrid, generate_kpath

end # module Eliashberg
