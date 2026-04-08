function _infer_wannier90_labelinfo_filename(bands_filename::AbstractString)
    endswith(lowercase(bands_filename), ".dat") || return nothing
    candidate = string(first(bands_filename, lastindex(bands_filename) - 4), ".labelinfo.dat")
    return isfile(candidate) ? candidate : nothing
end

function _infer_wannier90_kpoints_filename(bands_filename::AbstractString)
    endswith(lowercase(bands_filename), ".dat") || return nothing
    candidate = string(first(bands_filename, lastindex(bands_filename) - 4), ".kpt")
    return isfile(candidate) ? candidate : nothing
end

function _wannier90_node_labels(labelinfo::NamedTuple, num_kpoints::Integer)
    length(labelinfo.node_labels) == length(labelinfo.node_indices) ||
        throw(DimensionMismatch("Wannier90 node labels and indices must have the same length."))

    labels = fill("", Int(num_kpoints))
    for (idx, label) in zip(labelinfo.node_indices, labelinfo.node_labels)
        1 <= idx <= num_kpoints || throw(BoundsError(labels, idx))
        labels[idx] = label
    end

    return labels
end

function _wannier90_synthetic_path_points(distances::AbstractVector{<:Real}, dimension::Int)
    dimension >= 1 || throw(ArgumentError("Synthetic Wannier90 path dimension must be at least 1."))
    return [SVector{dimension, Float64}(distance, ntuple(_ -> 0.0, dimension - 1)...) for distance in distances]
end

"""
    cell_from_wannier90_tb(filename::String)

Build an `AtomsBase.PeriodicCell` directly from the lattice vectors stored in a
`wannier90_tb.dat` file.
"""
function cell_from_wannier90_tb(filename::String; periodicity=nothing)
    return parse_wannier90_tb(filename; periodicity).cell
end

function cell_from_wannier90_tb(tb_data::NamedTuple; periodicity=nothing)
    hasproperty(tb_data, :cell) || error("The provided Wannier90 TB data does not contain a cell field.")
    if isnothing(periodicity)
        return tb_data.cell
    end
    return periodic_cell_from_wannier90_tb(tb_data; periodicity)
end

"""
    periodic_cell_from_wannier90_tb(filename::String; periodicity=nothing)

Build an `AtomsBase.PeriodicCell` directly from the lattice vectors stored in a
`wannier90_tb.dat` file. Use `periodicity=(true, true, false)` for slab
systems, for example.
"""
function periodic_cell_from_wannier90_tb(filename::String; periodicity=nothing)
    return parse_wannier90_tb(filename; periodicity).cell
end

function periodic_cell_from_wannier90_tb(tb_data::NamedTuple; periodicity=nothing)
    if hasproperty(tb_data, :cell) && isnothing(periodicity)
        return tb_data.cell
    end
    hasproperty(tb_data, :lattice_vectors) || error("The provided Wannier90 TB data does not contain lattice vectors.")
    atomsbase_periodicity = isnothing(periodicity) ? (true, true, true) :
        periodicity isa Bool ? ntuple(_ -> periodicity, 3) : Tuple(periodicity)
    return PeriodicCell(
        ;
        cell_vectors=ntuple(i -> tb_data.lattice_vectors[i] .* u"Å", 3),
        periodicity=atomsbase_periodicity,
    )
end

"""
    build_model_from_wannier90(filename::String, cell, EF::Float64)

Construct a `MultiOrbitalTightBinding` model from either a `wannier90_hr.dat`
or `wannier90_tb.dat` file and a given cell description. The preferred cell
inputs are `AtomsBase.PeriodicCell`, `AtomsBase.AbstractSystem`, or a primitive
vector matrix.
"""
function build_model_from_wannier90(filename::String, cell::AbstractMatrix{<:Number}, EF::Float64)
    if endswith(lowercase(basename(filename)), "_tb.dat")
        parsed = parse_wannier90_tb(filename)
        return MultiOrbitalTightBinding(parsed.cell, parsed.num_wann, parsed.hoppings, EF)
    elseif endswith(lowercase(basename(filename)), "_hr.dat")
        num_wann, hoppings = parse_wannier90_hr(filename)
        return MultiOrbitalTightBinding(cell, num_wann, hoppings, EF)
    else
        error("Unrecognized Wannier90 file type for $filename. Expected seedname_hr.dat or seedname_tb.dat suffix.")
    end
end

function build_model_from_wannier90(filename::String, crystal::Crystal, EF::Float64)
    return build_model_from_wannier90(filename, primitive_vectors(crystal), EF)
end

function build_model_from_wannier90(filename::String, cell::PeriodicCell, EF::Float64)
    if endswith(lowercase(basename(filename)), "_tb.dat")
        parsed = parse_wannier90_tb(filename; periodicity=periodicity(cell))
        return MultiOrbitalTightBinding(parsed.cell, parsed.num_wann, parsed.hoppings, EF)
    elseif endswith(lowercase(basename(filename)), "_hr.dat")
        num_wann, hoppings = parse_wannier90_hr(filename)
        return MultiOrbitalTightBinding(cell, num_wann, hoppings, EF)
    end
    error("Unrecognized Wannier90 file type for $filename. Expected seedname_hr.dat or seedname_tb.dat suffix.")
end

function build_model_from_wannier90(filename::String, system::AbstractSystem, EF::Float64)
    if endswith(lowercase(basename(filename)), "_tb.dat")
        parsed = parse_wannier90_tb(filename; periodicity=periodicity(system))
        return MultiOrbitalTightBinding(parsed.cell, parsed.num_wann, parsed.hoppings, EF)
    elseif endswith(lowercase(basename(filename)), "_hr.dat")
        num_wann, hoppings = parse_wannier90_hr(filename)
        return MultiOrbitalTightBinding(system, num_wann, hoppings, EF)
    end
    error("Unrecognized Wannier90 file type for $filename. Expected seedname_hr.dat or seedname_tb.dat suffix.")
end

"""
    build_model_from_wannier90(filename::String, EF::Float64)

Construct a `MultiOrbitalTightBinding` model directly from a `wannier90_tb.dat`
file using the standalone cell stored in the TB data.
"""
function build_model_from_wannier90(filename::String, EF::Float64, periodicity=nothing)
    endswith(lowercase(basename(filename)), "_tb.dat") || error("A standalone cell can only be reconstructed from a Wannier90 TB file.")
    parsed = parse_wannier90_tb(filename; periodicity)
    return MultiOrbitalTightBinding(parsed.cell, parsed.num_wann, parsed.hoppings, EF)
end

"""
    kpath_from_wannier90_kpoints(
        kpoints;
        cell,
        labelinfo=nothing,
        coordinates=:fractional,
    )

Build a `KPath` from Wannier90 `*.kpt` path samples. When a cell is provided
and `coordinates == :fractional`, the k-points are interpreted in the
reciprocal basis of that cell, matching Wannier90's path convention.
"""
function kpath_from_wannier90_kpoints(
    kpoints::AbstractVector{<:SVector{3, <:Real}};
    cell,
    labelinfo::Union{Nothing, NamedTuple}=nothing,
    coordinates::Symbol=:fractional,
)
    node_labels = labelinfo === nothing ? nothing : _wannier90_node_labels(labelinfo, length(kpoints))
    return kpath_from_quantum_espresso_bands(
        kpoints;
        cell=cell,
        coordinates=coordinates,
        node_labels=node_labels,
    )
end

function kpath_from_wannier90_kpoints(
    kpoints_filename::String;
    cell,
    labelinfo_filename::Union{Nothing, AbstractString}=nothing,
    coordinates::Symbol=:fractional,
)
    parsed = parse_wannier90_kpoints(kpoints_filename)
    labelinfo = labelinfo_filename === nothing ? nothing : parse_wannier90_labelinfo(String(labelinfo_filename))
    return kpath_from_wannier90_kpoints(
        parsed.kpoints;
        cell=cell,
        labelinfo=labelinfo,
        coordinates=coordinates,
    )
end

"""
    kpath_from_wannier90_bands(distances; labelinfo=nothing)

Build a synthetic `KPath` compatible with `plot_band_structure` from the
distance axis stored in a Wannier90 `*_band.dat` file. Because the band file
contains only the cumulative path coordinate, the returned path embeds that
coordinate along a synthetic Cartesian axis while preserving symmetry labels
from `labelinfo` when available.
"""
function kpath_from_wannier90_bands(
    distances::AbstractVector{<:Real};
    labelinfo::Union{Nothing, NamedTuple}=nothing,
)
    isempty(distances) && throw(ArgumentError("`distances` must contain at least one Wannier90 path sample."))

    distance_values = collect(Float64.(distances))
    any(diff(distance_values) .< -1e-10) && throw(ArgumentError("Wannier90 path distances must be nondecreasing."))

    dimension = labelinfo === nothing ? 1 : Int(labelinfo.dimension)
    points = _wannier90_synthetic_path_points(distance_values, dimension)
    basis = [SVector{dimension, Float64}(ntuple(i -> i == axis ? 1.0 : 0.0, dimension)) for axis in 1:dimension]
    labels = Dict{Int, Symbol}()

    if labelinfo !== nothing
        length(labelinfo.node_labels) == length(labelinfo.node_indices) == length(labelinfo.node_distances) ||
            throw(DimensionMismatch("Wannier90 labelinfo metadata lengths must agree."))

        for (idx, label, distance) in zip(labelinfo.node_indices, labelinfo.node_labels, labelinfo.node_distances)
            1 <= idx <= length(distance_values) || throw(BoundsError(distance_values, idx))
            isapprox(distance_values[idx], distance; atol=1e-6, rtol=1e-6) ||
                error("Wannier90 label distance mismatch at index $idx: file gives $distance but the band path contains $(distance_values[idx]).")
            labels[idx] = Symbol(label)
        end
    end

    return KPath{dimension}([points], [labels], basis, Ref(Brillouin.CARTESIAN))
end

"""
    band_data_from_wannier90_bands(bands_filename::String; labelinfo_filename=nothing)

Parse Wannier90 `*_band.dat` output and wrap it as `BandStructureData` for
direct plotting. When `labelinfo_filename` is omitted, a sibling
`*.labelinfo.dat` file is used automatically when present.
"""
function band_data_from_wannier90_bands(
    bands_filename::String;
    labelinfo_filename::Union{Nothing, AbstractString}=nothing,
)
    parsed = parse_wannier90_band_dat(bands_filename)
    resolved_labelinfo = isnothing(labelinfo_filename) ? _infer_wannier90_labelinfo_filename(bands_filename) : String(labelinfo_filename)
    labelinfo = resolved_labelinfo === nothing ? nothing : parse_wannier90_labelinfo(resolved_labelinfo)
    kpath = kpath_from_wannier90_bands(parsed.distances; labelinfo)
    return BandStructureData(kpath, parsed.bands, parsed.num_bands)
end

"""
    compare_wannier90_tb_to_bands(
        model::MultiOrbitalTightBinding,
        bands_filename::String;
        kpoints_filename=nothing,
        labelinfo_filename=nothing,
        energy_shift=nothing,
    )

Compare a reconstructed Wannier90 tight-binding model against Wannier90's own
`*_band.dat` output on the exact `*.kpt` sampling path. By default a constant
energy offset is fitted as the mean band difference; pass `energy_shift=0.0`
to compare the raw energies without any additional alignment.
"""
function compare_wannier90_tb_to_bands(
    model::MultiOrbitalTightBinding,
    bands_filename::String;
    kpoints_filename::Union{Nothing, AbstractString}=nothing,
    labelinfo_filename::Union{Nothing, AbstractString}=nothing,
    energy_shift::Union{Nothing, Real}=nothing,
)
    resolved_labelinfo = isnothing(labelinfo_filename) ? _infer_wannier90_labelinfo_filename(bands_filename) : String(labelinfo_filename)
    resolved_kpoints = isnothing(kpoints_filename) ? _infer_wannier90_kpoints_filename(bands_filename) : String(kpoints_filename)
    resolved_kpoints === nothing &&
        error("Could not infer a Wannier90 `*.kpt` file from $bands_filename. Pass `kpoints_filename=...` explicitly.")

    labelinfo = resolved_labelinfo === nothing ? nothing : parse_wannier90_labelinfo(resolved_labelinfo)
    parsed_kpoints = parse_wannier90_kpoints(resolved_kpoints)
    reference = band_data_from_wannier90_bands(bands_filename; labelinfo_filename=resolved_labelinfo)
    model_kpath = kpath_from_wannier90_kpoints(
        parsed_kpoints.kpoints;
        cell=model.cell,
        labelinfo=labelinfo,
        coordinates=:fractional,
    )
    model_data = compute_band_data(model, model_kpath)

    size(reference.bands) == size(model_data.bands) ||
        throw(DimensionMismatch("Wannier90 reference bands and model bands must have the same shape for comparison."))

    remapped_model = BandStructureData(reference.kpath, copy(model_data.bands), model_data.num_bands)
    shift = if isnothing(energy_shift)
        total_difference = sum(reference.bands .- remapped_model.bands)
        total_difference / length(reference.bands)
    else
        Float64(energy_shift)
    end
    shifted_bands = remapped_model.bands .+ shift
    difference = reference.bands .- shifted_bands
    shifted_model = BandStructureData(reference.kpath, shifted_bands, remapped_model.num_bands)

    return Wannier90BandComparison(
        reference,
        remapped_model,
        shifted_model,
        parsed_kpoints.kpoints,
        shift,
        sqrt(sum(difference .^ 2) / length(difference)),
        maximum(abs.(difference)),
        difference,
    )
end

function compare_wannier90_tb_to_bands(
    tb_filename::String,
    bands_filename::String,
    EF::Real;
    periodicity=nothing,
    kpoints_filename::Union{Nothing, AbstractString}=nothing,
    labelinfo_filename::Union{Nothing, AbstractString}=nothing,
    energy_shift::Union{Nothing, Real}=nothing,
)
    model = build_model_from_wannier90(tb_filename, Float64(EF), periodicity)
    return compare_wannier90_tb_to_bands(
        model,
        bands_filename;
        kpoints_filename=kpoints_filename,
        labelinfo_filename=labelinfo_filename,
        energy_shift=energy_shift,
    )
end
