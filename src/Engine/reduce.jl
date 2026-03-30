"""
    integrate_grid(f::F, grid::AbstractKGrid) where {F}

Reduce a weighted sampling grid by evaluating `f` over each sample and
accumulating the weighted result. The engine performs a chunked
multithreaded reduction where each spawned task owns an independent partial
sum before the final serial combine step.
"""
function integrate_grid(f::F, grid::AbstractKGrid) where {F}
    n_items = length(grid)
    n_items == 0 && throw(ArgumentError("integrate_grid requires a non-empty grid."))

    schedule = ChunkedGridReduction(f, grid, min(n_items, max(1, Threads.nthreads())))
    tasks = Vector{Task}(undef, length(schedule.ranges))

    for idx in eachindex(schedule.ranges)
        chunk = schedule.ranges[idx]
        chunk_task = GridChunkTask(schedule.reducer, first(chunk), last(chunk))
        tasks[idx] = Threads.@spawn chunk_task()
    end

    total = fetch(tasks[1])
    for idx in 2:length(tasks)
        total += fetch(tasks[idx])
    end

    return total
end

struct WeightedGridReducer{F,G<:AbstractKGrid}
    f::F
    grid::G
end

struct ChunkedGridReduction{R}
    reducer::R
    ranges::Vector{UnitRange{Int}}
end

function ChunkedGridReduction(f::F, grid::G, n_chunks::Int) where {F,G<:AbstractKGrid}
    reducer = WeightedGridReducer(f, grid)
    ranges = _chunk_ranges(length(grid), n_chunks)
    return ChunkedGridReduction(reducer, ranges)
end

struct GridChunkTask{R}
    reducer::R
    chunk_start::Int
    chunk_stop::Int
end

(task::GridChunkTask)() = _chunk_reduce(task.reducer, task.chunk_start, task.chunk_stop)

function _chunk_ranges(n_items::Int, n_chunks::Int)
    chunk_size = cld(n_items, n_chunks)
    ranges = UnitRange{Int}[]

    for chunk_start in 1:chunk_size:n_items
        chunk_stop = min(chunk_start + chunk_size - 1, n_items)
        push!(ranges, chunk_start:chunk_stop)
    end

    return ranges
end

function _chunk_reduce(reducer::WeightedGridReducer, chunk_start::Int, chunk_stop::Int)
    points = reducer.grid.points
    weights = reducer.grid.weights

    @inbounds partial = reducer.f(points[chunk_start]) * weights[chunk_start]
    @inbounds for idx in (chunk_start + 1):chunk_stop
        partial += reducer.f(points[idx]) * weights[idx]
    end

    return partial
end
