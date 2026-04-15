module Visualization

using LinearAlgebra
using Makie
using StaticArrays

using ..Eliashberg: Dispersion, FreeElectron, TightBinding, MultiOrbitalTightBinding,
    RenormalizedDispersion, MeanFieldDispersion, KagomeLattice, Graphene, SSHModel,
    KGrid, KPath, AbstractKGrid, Crystal, PeriodicCell, AbstractSystem,
    DispersionSurfaceData, BandStructureData, FermiSurfaceData, LandscapeLineData,
    LandscapeSurfaceData, PhaseDiagramData, SpectralMapData, ZeemanPairingData,
    RenormalizedBandData, Wannier90BandComparison, primitive_vectors, reciprocal_vectors,
    path_points, path_branches, path_node_metadata, path_branch_ranges,
    with_stage_log, _validate_gap_storage,
    dimensionality, default_kgrid, path_distances, parse_wannier90_band_dat,
    parse_wannier90_labelinfo, band_data_from_wannier90_bands, PhysicalModel
using ..Eliashberg: periodic_rank

include("geometry.jl")
include("bands.jl")
include("responses.jl")

end # module Visualization
