# src/Geometry/types.jl

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

const KPath = Brillouin.KPathInterpolant

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



"""
    generate_1d_kgrid(Nk::Int; kmin=-π, kmax=π)

Generate a 1D k-grid with `Nk` points from `kmin` to `kmax`.
"""
function generate_1d_kgrid(Nk::Int; kmin=-π, kmax=π)
    points = [SVector{1,Float64}(k) for k in range(kmin, kmax, length=Nk)]
    weights = fill(1.0 / Nk, Nk)
    return KGrid(points, weights)
end

"""
    generate_2d_kgrid(Nx::Int, Ny::Int; kmin=(-π, -π), kmax=(π, π))

Generate a 2D k-grid with `Nx` * `Ny` points from `kmin` to `kmax`.
"""
function generate_2d_kgrid(Nx::Int, Ny::Int; kmin=(-π, -π), kmax=(π, π))
    points = [SVector{2,Float64}(kx, ky)
              for kx in range(kmin[1], kmax[1], length=Nx)
              for ky in range(kmin[2], kmax[2], length=Ny)]
    weights = fill(1.0 / (Nx * Ny), Nx * Ny)
    return KGrid(points, weights)
end

"""
    generate_3d_kgrid(Nx::Int, Ny::Int, Nz::Int; kmin=(-π, -π, -π), kmax=(π, π, π))

Generate a 3D k-grid with `Nx` * `Ny` * `Nz` points from `kmin` to `kmax`.
"""
function generate_3d_kgrid(Nx::Int, Ny::Int, Nz::Int; kmin=(-π, -π, -π), kmax=(π, π, π))
    points = [SVector{3,Float64}(kx, ky, kz)
              for kx in range(kmin[1], kmax[1], length=Nx)
              for ky in range(kmin[2], kmax[2], length=Ny)
              for kz in range(kmin[3], kmax[3], length=Nz)]
    weights = fill(1.0 / (Nx * Ny * Nz), Nx * Ny * Nz)
    return KGrid(points, weights)
end

# ---------------------------------------------------------
# 1D Reciprocal Space
# ---------------------------------------------------------

"""
    reciprocal_vectors(lattice::Lattice{1})

Calculates the 1D reciprocal lattice vector.
"""
function reciprocal_vectors(lattice::AbstractLattice{1})
    vectors = primitive_vectors(lattice)
    a = vectors[1, 1]
    return SMatrix{1,1,Float64,1}(2π / a)
end

"""
    generate_reciprocal_lattice(lattice::Lattice{1}, Nx::Int)

Generates a 1D KGrid spanning the first Brillouin Zone.
"""
function generate_reciprocal_lattice(lattice::AbstractLattice{1}, Nx::Int)
    B = reciprocal_vectors(lattice)
    b1 = B[:, 1]

    points = SVector{1,Float64}[]
    for i in 1:Nx+1
        u1 = (i - 1) / Nx - 0.5
        push!(points, u1 * b1)
    end

    weights = fill(1.0 / (Nx + 1), Nx + 1)
    return KGrid(points, weights)
end

# ---------------------------------------------------------
# 2D Reciprocal Space
# ---------------------------------------------------------

"""
    reciprocal_vectors(lattice::AbstractLattice{2})

Calculates and returns the 2D reciprocal lattice vectors `b_i` which satisfy 
the condition `a_i \\cdot b_j = 2\\pi \\delta_{ij}`.
"""
function reciprocal_vectors(lattice::AbstractLattice{2})
    vectors = primitive_vectors(lattice)
    a1 = vectors[:, 1]
    a2 = vectors[:, 2]
    area = a1[1] * a2[2] - a1[2] * a2[1]

    b1 = SVector{2}(2π * a2[2] / area, -2π * a2[1] / area)
    b2 = SVector{2}(-2π * a1[2] / area, 2π * a1[1] / area)
    return @SMatrix [b1[1] b2[1];
        b1[2] b2[2]]
end

"""
    generate_reciprocal_lattice(lattice::AbstractLattice{2}, Nx::Int, Ny::Int)

Generates a Monkhorst-Pack style KGrid spanning the first Brillouin Zone 
defined by the reciprocal vectors of the given real-space `lattice`.
"""
function generate_reciprocal_lattice(lattice::AbstractLattice{2}, Nx::Int, Ny::Int)
    B = reciprocal_vectors(lattice)
    b1, b2 = B[:, 1], B[:, 2]

    points = SVector{2,Float64}[]
    # Generate points across the reciprocal basis vectors
    for i in 1:Nx
        for j in 1:Ny
            # Map [0, 1) to [-0.5, 0.5) to center around Gamma point (0,0)
            u1 = (i - 1) / Nx - 0.5
            u2 = (j - 1) / Ny - 0.5
            push!(points, u1 * b1 + u2 * b2)
        end
    end

    weights = fill(1.0 / (Nx * Ny), Nx * Ny)
    return KGrid(points, weights)
end

# ---------------------------------------------------------
# 3D Reciprocal Space
# ---------------------------------------------------------

"""
    reciprocal_vectors(lattice::AbstractLattice{3})

Calculates the 3D reciprocal lattice vectors using the cross product formulas.
"""
function reciprocal_vectors(lattice::AbstractLattice{3})
    vectors = primitive_vectors(lattice)
    a1 = vectors[:, 1]
    a2 = vectors[:, 2]
    a3 = vectors[:, 3]

    # Cell volume V = a1 · (a2 × a3)
    V = dot(a1, cross(a2, a3))

    b1 = 2π * cross(a2, a3) / V
    b2 = 2π * cross(a3, a1) / V
    b3 = 2π * cross(a1, a2) / V

    return @SMatrix [b1[1] b2[1] b3[1];
        b1[2] b2[2] b3[2];
        b1[3] b2[3] b3[3]]
end

"""
    build_spglib_cell(crystal::Crystal{3})

Convert a `Crystal{3}` into the unit-cell format expected by `Spglib`. The
resulting cell stores a unitless `Float64` lattice, fractional basis
coordinates, and integer atom types.
"""
function build_spglib_cell(crystal::Crystal{3})
    symbol_to_type = Dict{Symbol,Int}()
    atom_types = [get!(symbol_to_type, symbol, length(symbol_to_type) + 1) for symbol in crystal.atomic_symbols]

    return Spglib.SpglibCell(
        Matrix{Float64}(primitive_vectors(crystal)),
        copy(crystal.fractional_positions),
        atom_types,
    )
end

"""
    build_spglib_cell(system::AbstractSystem{3})

Convert an `AtomsBase.AbstractSystem` into an `Spglib.SpglibCell` by first
constructing the internal `Crystal` representation.
"""
function build_spglib_cell(system::AbstractSystem{3})
    return build_spglib_cell(Crystal(system))
end

"""
    build_spglib_cell(lattice::AbstractLattice{3})

Convert a standalone Bravais lattice into an `Spglib.SpglibCell` by attaching a
single dummy atom at the origin. This is sufficient for space-group-based
Bravais-lattice identification.
"""
function build_spglib_cell(lattice::AbstractLattice{3})
    return Spglib.SpglibCell(
        Matrix{Float64}(primitive_vectors(lattice)),
        [SVector{3,Float64}(0.0, 0.0, 0.0)],
        [1],
    )
end

function _spacegroup_dataset(lattice::AbstractLattice{3})
    return Spglib.get_dataset(build_spglib_cell(lattice))
end

function _bravais_from_spacegroup(spacegroup_number::Integer, international_symbol::AbstractString)
    number = Int(spacegroup_number)
    centering = uppercase(first(strip(international_symbol)))

    if 1 <= number <= 2
        return :aP
    elseif 3 <= number <= 15
        return centering in ('A', 'B', 'C', 'I') ? :mC : :mP
    elseif 16 <= number <= 74
        return if centering == 'F'
            :oF
        elseif centering == 'I'
            :oI
        elseif centering in ('A', 'B', 'C')
            :oC
        else
            :oP
        end
    elseif 75 <= number <= 142
        return centering == 'I' ? :tI : :tP
    elseif 143 <= number <= 167
        return centering == 'R' ? :hR : :hP
    elseif 168 <= number <= 194
        return :hP
    elseif 195 <= number <= 230
        return if centering == 'F'
            :cF
        elseif centering == 'I'
            :cI
        else
            :cP
        end
    end

    error("Unsupported international space-group number $number.")
end

"""
    bravais_lattice(lattice::AbstractLattice{3})

Identify the 3D Bravais lattice symbol (`:aP`, `:mP`, `:mC`, `:oP`, `:oC`,
`:oI`, `:oF`, `:tP`, `:tI`, `:hP`, `:hR`, `:cP`, `:cI`, or `:cF`) using
Spglib's space-group analysis.
"""
function bravais_lattice(lattice::AbstractLattice{3})
    dataset = _spacegroup_dataset(lattice)
    return _bravais_from_spacegroup(dataset.spacegroup_number, dataset.international_symbol)
end

bravais_lattice(crystal::Crystal{3}) = bravais_lattice(Lattice(primitive_vectors(crystal)))
bravais_lattice(system::AbstractSystem{3}) = bravais_lattice(Crystal(system))

"""
    generate_irreducible_kgrid(crystal::Crystal{3}, mesh::Vector{Int}; is_shift=[0, 0, 0])

Generate a symmetry-reduced Brillouin-zone integration grid using
`Spglib.get_ir_reciprocal_mesh`. The irreducible-point weights are normalized
by the total number of points in the full mesh so that they sum to one.
"""
function generate_irreducible_kgrid(crystal::Crystal{3}, mesh::Vector{Int}; is_shift=[0, 0, 0])
    length(mesh) == 3 || throw(ArgumentError("`mesh` must contain exactly three integers."))
    length(is_shift) == 3 || throw(ArgumentError("`is_shift` must contain exactly three entries."))
    all(mesh .> 0) || throw(ArgumentError("`mesh` entries must be positive."))

    spglib_cell = build_spglib_cell(crystal)
    mesh_result = Spglib.get_ir_reciprocal_mesh(
        spglib_cell,
        Int.(mesh);
        is_shift=map(x -> !iszero(x), is_shift),
    )

    ir_mapping_table = Int.(mesh_result.ir_mapping_table)
    grid_address = mesh_result.grid_address
    irreducible_indices = sort!(unique(ir_mapping_table))
    degeneracies = zeros(Int, length(grid_address))

    for index in ir_mapping_table
        degeneracies[index] += 1
    end

    reciprocal = reciprocal_vectors(crystal)
    points = Vector{SVector{3,Float64}}(undef, length(irreducible_indices))
    weights = Vector{Float64}(undef, length(irreducible_indices))
    total_points = prod(mesh)

    for (point_index, ir_index) in pairs(irreducible_indices)
        fractional_k = SVector{3,Float64}(grid_address[ir_index])
        points[point_index] = reciprocal * fractional_k
        weights[point_index] = degeneracies[ir_index] / total_points
    end

    return KGrid(points, weights)
end

"""
    generate_irreducible_kgrid(system::AbstractSystem{3}, mesh::Vector{Int}; is_shift=[0, 0, 0])

Generate a symmetry-reduced Brillouin-zone integration grid directly from an
`AtomsBase.AbstractSystem`.
"""
function generate_irreducible_kgrid(system::AbstractSystem{3}, mesh::Vector{Int}; is_shift=[0, 0, 0])
    return generate_irreducible_kgrid(Crystal(system), mesh; is_shift)
end

"""
    generate_reciprocal_lattice(lattice::AbstractLattice{3}, Nx::Int, Ny::Int, Nz::Int)

Generates a 3D Monkhorst-Pack KGrid spanning the first Brillouin Zone.
"""
function generate_reciprocal_lattice(lattice::AbstractLattice{3}, Nx::Int, Ny::Int, Nz::Int)
    B = reciprocal_vectors(lattice)
    b1, b2, b3 = B[:, 1], B[:, 2], B[:, 3]

    points = SVector{3,Float64}[]
    for i in 1:Nx
        for j in 1:Ny
            for k in 1:Nz
                u1 = (i - 1) / Nx - 0.5
                u2 = (j - 1) / Ny - 0.5
                u3 = (k - 1) / Nz - 0.5
                push!(points, u1 * b1 + u2 * b2 + u3 * b3)
            end
        end
    end

    weights = fill(1.0 / (Nx * Ny * Nz), Nx * Ny * Nz)
    return KGrid(points, weights)
end

"""
    generate_kpath(nodes::Vector{SVector{D,Float64}}, labels::Vector{String}; n_pts_per_segment=50)

Generates a 1D path in K-space connecting high-symmetry nodes for band structure plotting.
"""
function generate_kpath(nodes::Vector{SVector{D,Float64}}, labels::Vector{String}; n_pts_per_segment=50) where {D}
    length(nodes) == length(labels) || throw(DimensionMismatch("`nodes` and `labels` must have the same length."))
    n_pts_per_segment >= 1 || throw(ArgumentError("`n_pts_per_segment` must be at least 1."))

    points = SVector{D,Float64}[]
    node_labels = Dict{Int,Symbol}()
    current_idx = 1
    node_labels[current_idx] = Symbol(labels[1])

    for i in 1:(length(nodes)-1)
        start_node = nodes[i]
        end_node = nodes[i+1]

        for j in 1:n_pts_per_segment
            t = (j - 1) / n_pts_per_segment
            k = start_node + t * (end_node - start_node)
            push!(points, k)
        end

        current_idx += n_pts_per_segment
        node_labels[current_idx] = Symbol(labels[i + 1])
    end

    push!(points, nodes[end])
    cartesian_basis = [SVector{D,Float64}(ntuple(i -> i == j ? 1.0 : 0.0, D)) for j in 1:D]
    return KPath{D}([points], [node_labels], cartesian_basis, Ref(Brillouin.CARTESIAN))
end

function generate_kpath(lat::ChainLattice; n_pts_per_segment=50)
    a = lat.a
    b1 = SVector{1,Float64}(2π / a)
    nodes = [0.0 * b1, 0.5 * b1]
    labels = ["Γ", "X"]
    return generate_kpath(nodes, labels; n_pts_per_segment=n_pts_per_segment)
end

function generate_kpath(lat::SquareLattice; n_pts_per_segment=50)
    a = lat.a
    b1 = SVector{2,Float64}(2π / a, 0.0)
    b2 = SVector{2,Float64}(0.0, 2π / a)
    nodes = [
        0.0 * b1 + 0.0 * b2, # Γ
        0.5 * b1 + 0.0 * b2, # X
        0.5 * b1 + 0.5 * b2, # M
        0.0 * b1 + 0.0 * b2  # Γ
    ]
    labels = ["Γ", "X", "M", "Γ"]
    return generate_kpath(nodes, labels; n_pts_per_segment=n_pts_per_segment)
end

function generate_kpath(lat::HexagonalLattice; n_pts_per_segment=50)
    a = lat.a
    b1 = SVector{2,Float64}(2π / a, 2π / (sqrt(3) * a))
    b2 = SVector{2,Float64}(0.0, 4π / (sqrt(3) * a))
    nodes = [
        0.0 * b1 + 0.0 * b2,             # Γ
        0.5 * b1 + 0.0 * b2,             # M
        (1.0 / 3.0) * b1 + (1.0 / 3.0) * b2, # K
        0.0 * b1 + 0.0 * b2              # Γ
    ]
    labels = ["Γ", "M", "K", "Γ"]
    return generate_kpath(nodes, labels; n_pts_per_segment=n_pts_per_segment)
end

function _standardized_direct_basis(lattice::AbstractLattice{3})
    dataset = _spacegroup_dataset(lattice)
    std_lattice = Matrix{Float64}(dataset.std_lattice)
    return [SVector{3,Float64}(std_lattice[i, 1], std_lattice[i, 2], std_lattice[i, 3]) for i in 1:3], Int(dataset.spacegroup_number)
end

function generate_kpath(lat::AbstractLattice{3}; n_pts_per_segment=50)
    n_pts_per_segment >= 1 || throw(ArgumentError("`n_pts_per_segment` must be at least 1."))

    direct_basis, sgnum = _standardized_direct_basis(lat)
    kp = Brillouin.irrfbz_path(sgnum, direct_basis)
    kp_cartesian = Brillouin.cartesianize(kp)
    return Brillouin.splice(kp_cartesian, n_pts_per_segment - 1)
end

generate_kpath(crystal::Crystal{3}; n_pts_per_segment=50) = generate_kpath(Lattice(primitive_vectors(crystal)); n_pts_per_segment=n_pts_per_segment)
