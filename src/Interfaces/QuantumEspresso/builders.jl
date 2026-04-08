function _qe_reciprocal_basis(cell_like)
    reciprocal = reciprocal_vectors(primitive_vectors(cell_like))
    D = size(reciprocal, 1)
    return [SVector{D, Float64}(reciprocal[:, idx]) for idx in 1:D]
end

function _qe_cartesianize_kpoints(kpoints::AbstractVector{<:SVector{D, <:Real}}, basis::AbstractVector{<:SVector{D, Float64}}) where {D}
    basis_matrix = reduce(hcat, basis)
    return [SVector{D, Float64}(basis_matrix * k) for k in kpoints]
end

"""
    kpath_from_quantum_espresso_bands(
        kpoints;
        cell=nothing,
        coordinates=:fractional,
        node_labels=nothing,
    )

Build a single-branch `KPath` from Quantum ESPRESSO band-path samples. When a
`cell` is provided and `coordinates == :fractional`, the k-points are
interpreted as fractional coordinates in the reciprocal basis and converted to
Cartesian reciprocal-space vectors.
"""
function kpath_from_quantum_espresso_bands(
    kpoints::AbstractVector{<:SVector{D, <:Real}};
    cell=nothing,
    coordinates::Symbol=:fractional,
    node_labels::Union{Nothing, AbstractVector{<:AbstractString}}=nothing,
) where {D}
    coordinates in (:fractional, :cartesian) || throw(ArgumentError("`coordinates` must be either `:fractional` or `:cartesian`."))

    basis = cell === nothing ?
        [SVector{D, Float64}(ntuple(i -> i == j ? 1.0 : 0.0, D)) for j in 1:D] :
        _qe_reciprocal_basis(cell)

    path_points = if coordinates == :fractional && cell !== nothing
        _qe_cartesianize_kpoints(kpoints, basis)
    else
        [SVector{D, Float64}(point) for point in kpoints]
    end

    labels = Dict{Int, Symbol}()
    if node_labels !== nothing
        length(node_labels) == length(path_points) || throw(DimensionMismatch("`node_labels` must match the number of k-points."))
        for (idx, label) in pairs(node_labels)
            isempty(strip(label)) || (labels[idx] = Symbol(label))
        end
    end

    return KPath{D}([path_points], [labels], basis, Ref(Brillouin.CARTESIAN))
end

"""
    band_data_from_quantum_espresso_bands(
        bands_filename::String;
        cell=nothing,
        cell_filename=nothing,
        coordinates=:fractional,
        node_labels=nothing,
    )

Parse a Quantum ESPRESSO `bands.x` output file and wrap it as
`BandStructureData` for direct plotting with `plot_band_structure`.
"""
function band_data_from_quantum_espresso_bands(
    bands_filename::String;
    cell=nothing,
    cell_filename::Union{Nothing, AbstractString}=nothing,
    coordinates::Symbol=:fractional,
    node_labels::Union{Nothing, AbstractVector{<:AbstractString}}=nothing,
)
    parsed = parse_quantum_espresso_bands(bands_filename)
    resolved_cell = cell_filename === nothing ? cell : parse_quantum_espresso_cell(String(cell_filename))
    kpath = kpath_from_quantum_espresso_bands(
        parsed.kpoints;
        cell=resolved_cell,
        coordinates=coordinates,
        node_labels=node_labels,
    )

    return BandStructureData(kpath, parsed.bands, parsed.num_bands)
end
