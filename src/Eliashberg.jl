module Eliashberg

# Basic utilities and constants
include("Constants.jl")

# Core architecture - Types and Dispersions defined early
include("types.jl")
include("dispersions.jl")

# Utilities and Feature modules
include("Utilities.jl")
include("susceptibilities.jl")
include("scanners.jl")
include("effective_action.jl")
include("observables.jl")

# Sub-packages (relying on above types)
include("BCS.jl")
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

const LindhardSusceptibility = GeneralizedSusceptibility

export PhysicalModel, AuxiliaryField, ApproximationLevel, ExactTrLn, RPA
export ChargeDensityWave, EffectiveAction
export StaticMeanField, DynamicalFluctuation
export GeneralizedSusceptibility, vertex_matrix, evaluate, solve_ground_state
export scan_instability_landscape, scan_spectral_function
export Dispersion, ElectronicDispersion, PhononDispersion, TightBinding, MeanFieldDispersion
export KPath, KGrid, generate_kpath
export SuperconductingPairing, gap_form_factor, NormalNambuDispersion, normal_state_basis
export LindhardSusceptibility

end # module Eliashberg
