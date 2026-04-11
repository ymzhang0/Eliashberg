"""
    bootstrap_engine_workers!(n_workers::Integer=max(0, Threads.nthreads() - 1); project=Base.active_project(), restrict=true)

Ensure that the execution engine has at least `n_workers` distributed workers
available for parameter-space map operations. New workers inherit the active
project, and every available worker is activated on that project and loads
`Eliashberg` so typed engine tasks can execute without extra user setup.
"""
function bootstrap_engine_workers!(
    n_workers::Integer=max(0, Threads.nthreads() - 1);
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
)
    return _with_stage_log(
        "Bootstrap engine workers";
        context=(target_workers=max(0, Int(n_workers)), current_workers=Distributed.nworkers(), project=project, restrict=restrict),
        summarize_result=result -> (n_workers=length(result), worker_ids=collect(result)),
    ) do
        target = max(0, Int(n_workers))
        current = Distributed.nworkers()

        if current < target
            _add_engine_workers!(target - current, project, restrict)
        end

        worker_ids = Distributed.workers()
        _load_engine_runtime!(worker_ids, project)
        return worker_ids
    end
end

"""
    distributed_map_grid(f::F, grids...) where {F}

Map a callable `f` over an arbitrary Cartesian product of parameter axes.
Each axis may be a plain iterable or an `AbstractKGrid`, in which case the
grid samples are used as the axis values. The engine constructs the Cartesian
index space once, dispatches a typed map task through `pmap` when distributed
workers are available, and reshapes the flat result back to the parameter-space
array.

Set `bootstrap_workers=true` to provision or synchronize workers automatically
before dispatching the map workload.
"""
function distributed_map_grid(
    f::F,
    grids...;
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
) where {F}
    return _with_stage_log(
        "Distributed map";
        context=(
            kernel=string(typeof(f)),
            n_axes=length(grids),
            axes=[_axis_summary(_parameter_axis(grid)) for grid in grids],
            bootstrap_workers=bootstrap_workers,
            requested_workers=Int(n_workers),
            project=project,
            restrict=restrict,
        ),
        summarize_result=result -> (result_type=string(typeof(result)), size=size(result)),
    ) do
        @timeit TO "Distributed Map" begin
            length(grids) == 0 && throw(ArgumentError("distributed_map_grid requires at least one parameter axis."))

            axes_tuple = map(_parameter_axis, grids)
            dims = map(length, axes_tuple)
            map_task = ParameterMapTask(f, axes_tuple)
            values = _map_parameter_task(
                map_task,
                Tuple(dims);
                bootstrap_workers=bootstrap_workers,
                n_workers=n_workers,
                project=project,
                restrict=restrict
            )

            return reshape(values, dims...)
        end
    end
end

struct ParameterMapTask{F,A}
    f::F
    axes::A
end

function (task::ParameterMapTask)(index::CartesianIndex{N}) where {N}
    args = ntuple(dim -> task.axes[dim][index[dim]], Val(N))
    return task.f(args...)
end

_parameter_axis(grid::AbstractKGrid) = grid.points
_parameter_axis(axis) = axis

function _map_parameter_task(
    map_task,
    dims::Tuple;
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
)
    return _with_stage_log(
        "Map parameter task";
        context=(dims=dims, bootstrap_workers=bootstrap_workers, requested_workers=Int(n_workers), current_workers=Distributed.nworkers()),
        summarize_result=result -> (n_results=length(result), execution_mode=Distributed.nworkers() > 1 ? :distributed : :serial),
    ) do
        bootstrap_workers && bootstrap_engine_workers!(n_workers; project=project, restrict=restrict)
        index_space = collect(CartesianIndices(dims))

        if Distributed.nworkers() > 1
            return Distributed.pmap(map_task, index_space)
        end

        return map(map_task, index_space)
    end
end

function _add_engine_workers!(n_new::Int, project::Union{Nothing,AbstractString}, restrict::Bool)
    n_new <= 0 && return Int[]

    if project === nothing
        return Distributed.addprocs(n_new; restrict=restrict)
    end

    return Distributed.addprocs(n_new; restrict=restrict, exeflags="--project=$(project)")
end

function _load_engine_runtime!(worker_ids::AbstractVector{<:Integer}, project::Union{Nothing,AbstractString})
    for worker_id in worker_ids
        _load_engine_runtime!(worker_id, project)
    end

    return worker_ids
end

function _load_engine_runtime!(worker_id::Integer, project::Nothing)
    Distributed.remotecall_wait(Core.eval, worker_id, Main, quote
        using Eliashberg
    end)
end

function _load_engine_runtime!(worker_id::Integer, project::AbstractString)
    Distributed.remotecall_wait(Core.eval, worker_id, Main, quote
        try
            import Pkg
            Pkg.activate($project; io=Base.devnull)
        catch err
            if !(err isa ArgumentError)
                rethrow()
            end
        end
        using Eliashberg
    end)
end
