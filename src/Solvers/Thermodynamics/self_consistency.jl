const _ORDER_PARAMETER_TOLERANCE = 1e-4

_initial_phi_guess(::AuxiliaryField, phi_guess::Real) = Float64(phi_guess)
_initial_phi_guess(field::CompositeField, phi_guess::Real) = fill(Float64(phi_guess), length(field))

function _initial_phi_guess(field::CompositeField, phi_guess::AbstractVector{<:Real})
    length(field) == length(phi_guess) || throw(DimensionMismatch("Number of fields must match number of phi guesses."))
    return Float64.(phi_guess)
end

_regularize_order_parameter(phi::Real) = abs(phi) < _ORDER_PARAMETER_TOLERANCE ? 0.0 : Float64(phi)
_regularize_order_parameter(phis::AbstractVector{<:Real}) = [_regularize_order_parameter(phi) for phi in phis]

function _store_gap!(gaps::AbstractVector{<:Real}, idx::Int, phi::Real)
    gaps[idx] = Float64(phi)
    return gaps
end

function _store_gap!(gaps::AbstractMatrix{<:Real}, idx::Int, phis::AbstractVector{<:Real})
    size(gaps, 1) == length(phis) || throw(DimensionMismatch("Gap storage row count must match the number of composite fields."))
    gaps[:, idx] .= Float64.(phis)
    return gaps
end

_next_phi_guess(phi::Real, fallback::Real) = abs(phi) > 0.0 ? Float64(phi) : max(Float64(fallback), 0.05)

function _next_phi_guess(phis::AbstractVector{<:Real}, fallback::AbstractVector{<:Real})
    length(phis) == length(fallback) || throw(DimensionMismatch("Gap vector length must match the fallback guess length."))
    return [abs(Float64(phi)) > 0.0 ? Float64(phi) : max(Float64(fallback[idx]), 0.05) for (idx, phi) in enumerate(phis)]
end

function _solve_regularized_ground_state(
    field,
    model::ElectronicDispersion,
    interaction,
    kgrid::AbstractKGrid,
    approx::ApproximationLevel;
    phi_guess,
    T::Real,
    log_level::LogLevel=Logging.Debug
)
    phi = solve_ground_state(field, model, interaction, kgrid, approx; phi_guess=phi_guess, T=T, log_level=log_level)
    return _regularize_order_parameter(phi)
end

function _scan_parameter_axis(
    step!,
    axis::AbstractVector;
    warm_start::Bool,
    initial_state,
    progress_name::AbstractString,
)
    n_items = length(axis)

    if warm_start
        state = initial_state

        @withprogress name=progress_name begin
            for (idx, parameter) in enumerate(axis)
                @logprogress idx / n_items
                state = step!(idx, parameter, state)
            end
        end

        return state
    end

    @withprogress name="$(progress_name) (Threaded)" begin
        Threads.@threads for idx in eachindex(axis)
            @logprogress idx / n_items
            step!(idx, axis[idx], initial_state)
        end
    end

    return initial_state
end
