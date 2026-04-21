function _qe_reciprocal_basis(cell_like)
    reciprocal = reciprocal_vectors(primitive_vectors(cell_like))
    D = size(reciprocal, 1)
    return [SVector{D, Float64}(reciprocal[:, idx]) for idx in 1:D]
end

function _qe_reciprocal_basis(cell_like::AbstractMatrix{<:Number})
    reciprocal = reciprocal_vectors(Matrix{Float64}(cell_like))
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
        branch_gap_factor=5.0,
        branch_gap_threshold=nothing,
    )

Build a `KPath` from Quantum ESPRESSO band-path samples. When a
`cell` is provided and `coordinates == :fractional`, the k-points are
interpreted as fractional coordinates in the reciprocal basis and converted to
Cartesian reciprocal-space vectors. Large adjacent k-point jumps are interpreted
as branch breaks.
"""
function kpath_from_quantum_espresso_bands(
    kpoints::AbstractVector{<:SVector{D, <:Real}};
    cell=nothing,
    coordinates::Symbol=:fractional,
    node_labels::Union{Nothing, AbstractVector{<:AbstractString}}=nothing,
    branch_gap_factor::Real=5.0,
    branch_gap_threshold::Union{Nothing,Real}=nothing,
) where {D}
    coordinates in (:fractional, :cartesian) || throw(ConfigurationError("coordinates", coordinates, "Must be either `:fractional` or `:cartesian`."))

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

    branched = _branch_kpath_at_large_point_gaps(
        path_points,
        labels;
        gap_factor=branch_gap_factor,
        gap_threshold=branch_gap_threshold,
    )

    return KPath{D}(branched.branches, branched.labels, basis, Ref(Brillouin.CARTESIAN))
end
