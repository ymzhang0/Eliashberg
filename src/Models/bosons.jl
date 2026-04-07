# bosons.jl

"""
    RPABoson{S<:GeneralizedSusceptibility,T<:Real}

Lazy emergent boson wrapper that dresses a bare susceptibility with the RPA
Dyson equation

    D(q) = chi0(q) / (1 - sign * U * Re[chi0(q)])

Use `sign = +1.0` for spin fluctuations and `sign = -1.0` for charge
fluctuations.
"""
struct RPABoson{S<:GeneralizedSusceptibility,T<:Real}
    chi_0::S
    U::T
    sign::Float64
end

RPABoson(chi_0::GeneralizedSusceptibility, U::T, sign::Real) where {T<:Real} =
    RPABoson{typeof(chi_0),T}(chi_0, U, Float64(sign))

RPABoson(chi_0::GeneralizedSusceptibility, U::Real; sign::Real=1.0) =
    RPABoson(chi_0, U, sign)

function evaluate_boson_propagator(
    q::SVector{D,Float64},
    boson::RPABoson,
) where {D}
    chi0_val = evaluate_boson_propagator(q, boson.chi_0)
    denominator = 1.0 - boson.sign * boson.U * real(chi0_val)
    return chi0_val / denominator
end

"""
    CachedBoson{D,G,A,L}

Materialized boson propagator values sampled over a fixed q-grid. An internal
point-to-index map keeps repeated lookup cheap inside self-consistent loops.
"""
struct CachedBoson{
    D,
    G<:AbstractKGrid{D},
    A<:AbstractVector,
    L<:AbstractDict{SVector{D,Float64},Int},
}
    grid::G
    values::A
    index_map::L
end

function CachedBoson(grid::AbstractKGrid{D}, values::AbstractVector) where {D}
    length(grid) == length(values) ||
        throw(DimensionMismatch("Cached boson values length must match the q-grid length."))

    index_map = Dict{SVector{D,Float64},Int}()
    sizehint!(index_map, length(grid))

    for idx in eachindex(grid.points)
        point = grid[idx]
        haskey(index_map, point) &&
            throw(ArgumentError("Cannot build a cached boson from a q-grid with duplicate points."))
        index_map[point] = idx
    end

    return CachedBoson{D,typeof(grid),typeof(values),typeof(index_map)}(grid, values, index_map)
end

function find_kpoint_index(
    q::SVector{D,Float64},
    qgrid::AbstractKGrid{D};
    atol::Float64=1e-10,
) where {D}
    for idx in eachindex(qgrid.points)
        q == qgrid[idx] && return idx
    end

    idx = _nearest_grid_index(q, qgrid)
    nearest_q = qgrid[idx]
    isapprox(q, nearest_q; atol=atol, rtol=0.0) && return idx

    throw(KeyError("Momentum $q is not represented on the supplied q-grid. Closest point is $nearest_q."))
end

function evaluate_boson_propagator(
    q::SVector{D,Float64},
    boson::CachedBoson{D},
) where {D}
    idx = get(boson.index_map, q, 0)
    idx == 0 && (idx = find_kpoint_index(q, boson.grid))
    return boson.values[idx]
end

function _same_grid_points(lhs::AbstractKGrid{D}, rhs::AbstractKGrid{D}) where {D}
    length(lhs) == length(rhs) || return false
    for idx in eachindex(lhs.points)
        lhs[idx] == rhs[idx] || return false
    end
    return true
end

function materialize_boson(
    boson::CachedBoson{D},
    qgrid::AbstractKGrid{D},
) where {D}
    _same_grid_points(boson.grid, qgrid) && return boson

    values = similar(boson.values, length(qgrid))
    Threads.@threads for idx in eachindex(qgrid.points)
        @inbounds values[idx] = evaluate_boson_propagator(qgrid[idx], boson)
    end

    return CachedBoson(qgrid, values)
end

function materialize_boson(
    boson,
    qgrid::AbstractKGrid{D},
) where {D}
    values = Vector{ComplexF64}(undef, length(qgrid))

    Threads.@threads for idx in eachindex(qgrid.points)
        @inbounds values[idx] = ComplexF64(evaluate_boson_propagator(qgrid[idx], boson))
    end

    return CachedBoson(qgrid, values)
end
