# src/Geometry/reciprocal_lattice.jl

using StaticArrays
using LinearAlgebra
using Spglib: SpglibCell, get_dataset
import Brillouin

# ---------------------------------------------------------
# K-Grid and K-Path Structs
# ---------------------------------------------------------

"""
    KGrid{D} <: AbstractKGrid{D}

A concrete generic implementation of a D-dimensional K-grid.
Contains the grid `points` as `SVector{D, Float64}` and corresponding 
integration `weights`.
"""
struct KGrid{D} <: AbstractKGrid{D}
    points::Vector{SVector{D,Float64}}
    weights::Vector{Float64}
end

Base.show(io::IO, g::KGrid{D}) where {D} = print(io, "KGrid (", D, "D, ", length(g), " points)")
Base.show(io::IO, ::MIME"text/plain", g::KGrid{D}) where {D} = show(io, g)

Base.show(io::IO, kp::Brillouin.KPathInterpolant{D}) where {D} = print(io, "KPath (", D, "D, ", length(path_points(kp)), " points)")

function Base.show(io::IO, ::MIME"text/plain", path::Brillouin.KPathInterpolant{D}) where D
    print(io, "KPath (", D, "D)")
    for (b_idx, labels) in enumerate(path.labels)
        sorted_indices = sort(collect(keys(labels)))
        for idx in sorted_indices
            label = labels[idx]
            point = path.kpaths[b_idx][idx]
            print(io, "\n  ", label, ": ", point)
        end
    end
end

function _format_path_label(path::Brillouin.KPathInterpolant)
    branch_labels = String[]
    for labels in path.labels
        isempty(labels) && continue
        sorted_indices = sort(collect(keys(labels)))
        push!(branch_labels, join([string(labels[idx]) for idx in sorted_indices], "-"))
    end
    full_path = join(branch_labels, "|")
    return replace(full_path, r"([^|]+)\|\1" => s"\1")
end

# ---------------------------------------------------------
# Reciprocal Lattice (Vectors)
# ---------------------------------------------------------

"""
    reciprocal_lattice(lattice_like)

Calculate the reciprocal lattice vectors (matrix columns) satisfying b_i ⋅ a_j = 2πδ_ij. 
Support is strictly limited to `AbstractBravaisLattice` and `AbstractMatrix`.
"""
function reciprocal_lattice(vectors::AbstractMatrix{<:Number})
    primitive = vectors
    D = size(primitive, 1)
    size(primitive, 2) == D || throw(ArgumentError("`vectors` must be a square matrix for reciprocal calculations."))
    return 2π * transpose(inv(primitive))
end

reciprocal_lattice(l::AbstractBravaisLattice) = reciprocal_lattice(primitive_cell(l))

# Backward compatibility alias
const reciprocal_vectors = reciprocal_lattice

# ---------------------------------------------------------
# K-Grid Sampling
# ---------------------------------------------------------

"""
    generate_kgrid(lattice, mesh::NTuple{N, Int}; is_shift=nothing)

Generate a uniform k-point grid in fractional coordinates of the reciprocal basis.
Uses the Monkhorst-Pack style logic: k_i = (n_i + shift_i/2) / N_i where n_i = 0, ..., N_i-1.
This is a dimension-agnostic port of the Spglib/QE grid logic.
`is_shift` should be a tuple of 0 or 1 (or Booleans).
"""
function generate_kgrid(lattice, mesh::NTuple{N,Int}; is_shift=nothing) where {N}
    B = reciprocal_lattice(lattice)
    D = size(B, 1)
    N == D || throw(ArgumentError("Sampling dimensions ($N) must match lattice dimensionality ($D)."))

    shifts = isnothing(is_shift) ? ntuple(_ -> 0.0, N) : Float64.(is_shift)
    points = Vector{SVector{D,Float64}}(undef, prod(mesh))
    index = 1

    # Generate points: k = Σ [ (n_i + s_i/2) / N_i ] * b_i
    for idxs in Iterators.product(ntuple(i -> 0:mesh[i]-1, N)...)
        k = zero(SVector{D,Float64})
        for i in 1:N
            frac = (idxs[i] + shifts[i] / 2) / mesh[i]
            k += frac * B[:, i]
        end
        points[index] = k
        index += 1
    end

    weights = fill(1.0 / length(points), length(points))
    return KGrid(points, weights)
end

# Convenience overloads
generate_kgrid(l::AbstractBravaisLattice, mesh::Vararg{Int,N}; kwargs...) where {N} = generate_kgrid(l, mesh; kwargs...)
generate_kgrid(m::AbstractMatrix, mesh::Vararg{Int,N}; kwargs...) where {N} = generate_kgrid(m, mesh; kwargs...)
generate_kgrid(s::AbstractSystem, mesh::NTuple{N,Int}; irreducible=false, kwargs...) where {N} =
    irreducible ? generate_irreducible_kgrid(s, mesh; kwargs...) : generate_kgrid(primitive_vectors(s), mesh; kwargs...)
generate_kgrid(s::AbstractSystem, mesh::Vararg{Int,N}; kwargs...) where {N} = generate_kgrid(s, mesh; kwargs...)

# ---------------------------------------------------------
# Symmetry-Aware Operations (3D Only)
# ---------------------------------------------------------

"""
    bravais_lattice(system::AbstractSystem)

Identify the Bravais lattice type of a 3D system using Spglib.
Returns a symbol like `:cP`, `:cF`, etc.
"""

"""
    generate_irreducible_kgrid(system::AbstractSystem, mesh::NTuple{3, Int}; is_shift=[0,0,0])

Generate a symmetry-reduced k-point grid for a 3D system using Spglib.
`is_shift` follows the Spglib convention (0 for no shift, 1 for 1/2nd mesh shift).
"""
function generate_irreducible_kgrid(system::AbstractSystem, mesh::NTuple{3,Int}; is_shift=[0, 0, 0])
    cell = Spglib.SpglibCell(system)

    # Spglib returns (ir_mapping_table, grid_address)
    mesh_result = Spglib.get_ir_reciprocal_mesh(cell, Int.(mesh); is_shift=map(x -> !iszero(x), is_shift))

    ir_mapping = mesh_result.ir_mapping_table
    grid_address = mesh_result.grid_address

    unique_indices = sort!(unique(ir_mapping))
    total_pts = prod(mesh)

    # Weights based on degeneracy
    degeneracies = Dict{Int,Int}()
    for idx in ir_mapping
        degeneracies[idx] = get(degeneracies, idx, 0) + 1
    end

    B = reciprocal_lattice(system)
    points = Vector{SVector{3,Float64}}(undef, length(unique_indices))
    weights = Vector{Float64}(undef, length(unique_indices))

    for (i, ir_idx) in enumerate(unique_indices)
        # grid_address is in units of 1/mesh
        # k = Σ (n_i + s_i/2)/N_i * b_i
        # Spglib address corresponds to n_i (if no shift) or n_i + 1/2 (if shifted)
        addr = grid_address[ir_idx]
        frac = ntuple(j -> (addr[j] + (is_shift[j] != 0 ? 0.5 : 0.0)) / mesh[j], 3)
        points[i] = B * SVector{3}(frac)
        weights[i] = degeneracies[ir_idx] / total_pts
    end

    return KGrid(points, weights)
end

# ---------------------------------------------------------
# K-Path Generation
# ---------------------------------------------------------

"""
    generate_kpath(lattice::AbstractBravaisLattice; n_pts_per_segment=50)

Generate a high-symmetry k-path using Brillouin.jl for 2D/3D and native 
definitions for 1D.
"""
function generate_kpath(l::AbstractBravaisLattice{1}; n_pts_per_segment=50)
    data = symmetry_path(l)
    if !isnothing(data)
        B = reciprocal_lattice(l)
        nodes = [B * data.points[label] for label in data.path]
        return generate_kpath(nodes, data.path; n_pts_per_segment)
    end

    throw(ArgumentError("No symmetry path defined for lattice $(typeof(l)). Please define it in `src/Geometry/symmetry_points.jl`."))
end

"""
    symmetry_path(system::AbstractSystem)

Return the `SymmetryPath` for a given atomic system by identifying its Bravais lattice.
For 3D systems, this uses Spglib and Brillouin.jl.
"""
function symmetry_path(s::AbstractSystem)
    D = AtomsBase.n_dimensions(s)
    if D == 3
        vectors = primitive_vectors(s)
        # Ensure we pass Int to Brillouin and use StaticArrays for vectors
        dataset = Spglib.get_dataset(Spglib.SpglibCell(Matrix{Float64}(vectors), [[0.0, 0.0, 0.0]], [1]))
        Rs = SVector{3}(SVector{3}(vectors[:, i]) for i in 1:3)
        kp = Brillouin.irrfbz_path(Int(dataset.spacegroup_number), Rs)
        points = Dict(String(k) => v for (k, v) in kp.points)
        # Use only the first branch for the "default" path. 
        # Multi-branch paths are handled by Brillouin.interpolate directly.
        path = [String(k) for k in kp.paths[1]] 
        return SymmetryPath{3}(points, path)
    end

    # 1D/2D Identification
    vectors = primitive_vectors(s)
    type = bravais_lattice(s)

    lattice = if D == 1 && type == :line
        ChainLattice(norm(vectors[:, 1]))
    elseif D == 2
        if type == :sqP
            SquareLattice(norm(vectors[:, 1]))
        elseif type == :hP
            HexagonalLattice2D(norm(vectors[:, 1]))
        elseif type == :rP
            RectangularLattice(norm(vectors[:, 1]), norm(vectors[:, 2]))
        elseif type == :obP
            v1, v2 = vectors[:, 1], vectors[:, 2]
            gamma = acos(clamp(dot(v1, v2) / (norm(v1) * norm(v2)), -1.0, 1.0))
            ObliqueLattice(norm(v1), norm(v2), gamma)
        else
            nothing
        end
    else
        nothing
    end

    return isnothing(lattice) ? nothing : symmetry_path(lattice)
end

function generate_kpath(l::AbstractBravaisLattice{2}; n_pts_per_segment=50)
    data = symmetry_path(l)
    if !isnothing(data)
        B = reciprocal_lattice(l)
        nodes = [B * data.points[label] for label in data.path]
        return generate_kpath(nodes, data.path; n_pts_per_segment)
    end

    # Brillouin.jl backup (only if it's a known Bravais type that Brillouin handles)
    sgnum = l isa SquareLattice ? 10 :
            l isa HexagonalLattice2D ? 13 :
            l isa RectangularLattice ? 3 :
            l isa CenteredRectangularLattice ? 5 : nothing

    if !isnothing(sgnum)
        primitive = [SVector{2}(primitive_cell(l)[:, i]) for i in 1:2]
        kp = Brillouin.irrfbz_path(sgnum, primitive, Val(2))
        return Brillouin.interpolate(Brillouin.cartesianize(kp), n_pts_per_segment)
    end

    throw(ArgumentError("No symmetry path defined for lattice $(typeof(l)). Please define it in `src/Geometry/symmetry_points.jl`."))
end

function generate_kpath(l::AbstractBravaisLattice{3}; n_pts_per_segment=50)
    primitive = Matrix{Float64}(primitive_cell(l))
    # Temporary SpglibCell for symmetry identification (full grid skip, but path needs sgnum)
    dataset = dataset = Spglib.get_dataset(Spglib.SpglibCell(primitive, [[0.0, 0.0, 0.0]], [1]))
    kp = Brillouin.irrfbz_path(dataset.spacegroup_number, [SVector{3}(primitive[:, i]) for i in 1:3])
    return Brillouin.interpolate(Brillouin.cartesianize(kp), n_pts_per_segment)
end

function generate_kpath(s::AbstractSystem; kwargs...)
    D = AtomsBase.n_dimensions(s)

    # 3D Path Generation (Spglib + Brillouin)
    if D == 3
        # Extract lattice vectors as a matrix
        vectors = primitive_vectors(s)
        # Identify spacegroup and generate path
        dataset = Spglib.get_dataset(Spglib.SpglibCell(Matrix{Float64}(vectors), [[0.0, 0.0, 0.0]], [1]))
        kp = Brillouin.irrfbz_path(dataset.spacegroup_number, [SVector{3}(vectors[:, i]) for i in 1:3])
        n_pts_per_segment = get(kwargs, :n_pts_per_segment, 50)
        return Brillouin.interpolate(Brillouin.cartesianize(kp), n_pts_per_segment)
    end

    # For 1D and 2D, we try to identify the lattice and use symmetry definitions
    vectors = primitive_vectors(s)
    type = bravais_lattice(s)

    lattice = if D == 1 && type == :line
        ChainLattice(norm(vectors[:, 1]))
    elseif D == 2
        if type == :sqP
            SquareLattice(norm(vectors[:, 1]))
        elseif type == :hP
            HexagonalLattice2D(norm(vectors[:, 1]))
        elseif type == :rP
            RectangularLattice(norm(vectors[:, 1]), norm(vectors[:, 2]))
        elseif type == :obP
            # For oblique, we need a1, a2 and the angle gamma
            v1, v2 = vectors[:, 1], vectors[:, 2]
            gamma = acos(clamp(dot(v1, v2) / (norm(v1) * norm(v2)), -1.0, 1.0))
            ObliqueLattice(norm(v1), norm(v2), gamma)
        else
            nothing
        end
    else
        nothing
    end

    if !isnothing(lattice)
        path_data = symmetry_path(lattice)
        if !isnothing(path_data)
            return generate_kpath(lattice; kwargs...)
        end
    end

    # No fallback allowed for 1D/2D as per user request
    error_msg = "No symmetry path defined for this $(D)D lattice (type: $type). " *
                "Please define it in `src/Geometry/symmetry_points.jl` by adding " *
                "a `symmetry_path(::$(isnothing(lattice) ? "UnknownLattice" : typeof(lattice)))` method."
    throw(ArgumentError(error_msg))
end


"""
    generate_kpath(nodes::Vector{SVector{D,Float64}}, labels::Vector{String}; n_pts_per_segment=50)

Generates a 1D path in K-space connecting high-symmetry nodes.
"""
function generate_kpath(nodes::Vector{SVector{D,Float64}}, labels::Vector{String}; n_pts_per_segment=50) where {D}
    length(nodes) == length(labels) || throw(DimensionMismatch("`nodes` and `labels` must have the same length."))
    points = SVector{D,Float64}[]
    node_labels = Dict{Int,Symbol}()
    current_idx = 1
    node_labels[current_idx] = Symbol(labels[1])

    for i in 1:(length(nodes)-1)
        for j in 1:n_pts_per_segment
            t = (j - 1) / n_pts_per_segment
            push!(points, nodes[i] + t * (nodes[i+1] - nodes[i]))
        end
        current_idx += n_pts_per_segment
        node_labels[current_idx] = Symbol(labels[i+1])
    end
    push!(points, nodes[end])
    cartesian_basis = [SVector{D,Float64}(ntuple(i -> i == j ? 1.0 : 0.0, D)) for j in 1:D]
    return KPath{D}([points], [node_labels], cartesian_basis, Ref(Brillouin.CARTESIAN))
end


# ---------------------------------------------------------
# Base Method Overloads for K-Grids
# ---------------------------------------------------------

Base.length(g::AbstractKGrid) = length(g.points)
Base.iterate(g::AbstractKGrid, state=1) = iterate(g.points, state)
Base.getindex(g::AbstractKGrid, i::Int) = g.points[i]
Base.firstindex(g::AbstractKGrid) = 1
Base.lastindex(g::AbstractKGrid) = length(g.points)
Base.eltype(::Type{<:AbstractKGrid{D}}) where {D} = SVector{D,Float64}

path_branches(kpath::KPath) = getfield(kpath, :kpaths)
function path_points(kpath::KPath{D}) where {D}
    points = SVector{D,Float64}[]
    for branch in path_branches(kpath)
        append!(points, branch)
    end
    return points
end

function path_node_metadata(kpath::KPath)
    node_indices = Int[]
    node_labels = String[]
    offset = 0

    for (branch, labels) in zip(path_branches(kpath), getfield(kpath, :labels))
        for idx in sort!(collect(keys(labels)))
            push!(node_indices, offset + idx)
            push!(node_labels, String(labels[idx]))
        end
        offset += length(branch)
    end

    return node_indices, node_labels
end

function path_branch_ranges(kpath::KPath)
    ranges = UnitRange{Int}[]
    start = 1
    for branch in path_branches(kpath)
        stop = start + length(branch) - 1
        push!(ranges, start:stop)
        start = stop + 1
    end
    return ranges
end

# ---------------------------------------------------------
# Model Overloads
# ---------------------------------------------------------

function generate_kgrid(model::MultiOrbitalTightBinding, sizes::Vararg{Int,N}; kwargs...) where {N}
    return generate_kgrid(periodic_cell(model).lattice, sizes...; kwargs...)
end

function generate_kpath(model::MultiOrbitalTightBinding; n_pts_per_segment=50)
    return generate_kpath(periodic_cell(model).lattice; n_pts_per_segment=n_pts_per_segment)
end
