const _LOG_STAGE_GROUP = :eliashberg_stage
const _LOG_STAGE_DEPTH_KEY = :eliashberg_stage_depth

function _stage_log(level::LogLevel, message::AbstractString; kwargs...)
    logger = current_logger()
    Logging.min_enabled_level(logger) <= level || return nothing
    Logging.shouldlog(logger, level, @__MODULE__, _LOG_STAGE_GROUP, nothing) || return nothing
    Logging.handle_message(
        logger,
        level,
        message,
        @__MODULE__,
        _LOG_STAGE_GROUP,
        nothing,
        @__FILE__,
        0;
        kwargs...,
    )
    return nothing
end

function _format_namedtuple_to_string(nt::NamedTuple, indent=2)
    keys_str = string.(keys(nt))
    isempty(keys_str) && return ""
    max_len = maximum(length.(keys_str))
    
    io = IOBuffer()
    for (i, (k, v)) in enumerate(zip(keys(nt), values(nt)))
        # Prepend newline for each entry
        print(io, "\n", " "^indent, "• ", rpad(string(k), max_len), " : ")
        if v isa NamedTuple
            print(io, _format_namedtuple_to_string(v, indent + 2))
        else
            print(io, v)
        end
    end
    return String(take!(io))
end

function with_stage_log(f::F, stage::AbstractString; level::LogLevel=Logging.Info, context::NamedTuple=NamedTuple(), summarize_result=nothing) where {F}
    start_ns = time_ns()
    depth = get(task_local_storage(), _LOG_STAGE_DEPTH_KEY, 0)
    task_local_storage(_LOG_STAGE_DEPTH_KEY, depth + 1)
    
    formatted_context = _format_namedtuple_to_string(context)
    _stage_log(level, "$(stage) started$(formatted_context)"; stage, status=:start, context)

    try
        result = f()
        elapsed_s = (time_ns() - start_ns) / 1.0e9
        
        summary = isnothing(summarize_result) ? nothing : summarize_result(result)
        formatted_summary = summary isa NamedTuple ? _format_namedtuple_to_string(summary) : (isnothing(summary) ? "" : "\n  • summary     : $(summary)")
        
        _stage_log(level, "$(stage) finished$(formatted_summary)"; stage, status=:finish, elapsed_s, context, summary)
        return result
    catch err
        elapsed_s = (time_ns() - start_ns) / 1.0e9
        formatted_context = _format_namedtuple_to_string(context)
        if depth == 0
            _stage_log(Logging.Error, "$(stage) failed$(formatted_context)"; stage, status=:error, elapsed_s, context, exception=(err, catch_backtrace()))
        else
            _stage_log(
                Logging.Error,
                "$(stage) failed$(formatted_context)";
                stage,
                status=:error,
                elapsed_s,
                context,
                error_type=string(typeof(err)),
                error_message=sprint(showerror, err),
            )
        end
        rethrow()
    finally
        task_local_storage(_LOG_STAGE_DEPTH_KEY, depth)
    end
end

_safe_weight_sum(weights) = isempty(weights) ? 0.0 : sum(weights)

isfinite_value(x::Real) = isfinite(x)
isfinite_value(x::Complex) = isfinite(real(x)) && isfinite(imag(x))
isfinite_value(values::AbstractArray) = all(isfinite_value, values)
count_nonfinite(values) = count(value -> !isfinite_value(value), values)
