using Eliashberg
using Test

@testset "Unified Plotting Interface" begin
    # 1. Check exports
    @test :plot in names(Eliashberg)
    @test !(:plot_band_structure in names(Eliashberg))
    @test !(:plot_lattice in names(Eliashberg))

    # 2. Mock BandStructureData
    # We need to construct it carefully as it's a typed struct
    # BandStructureData{D}(kpath::KPath, bands::Matrix{Float64}, num_bands::Int)
    
    # KPath(points::Vector{SVector{D, Float64}}, branches::Vector{Pair{Symbol, Symbol}}, 
    #       node_indices::Vector{Int}, branch_ranges::Vector{UnitRange{Int}})
    using StaticArrays
    kpath = KPath(
        [SVector{1, Float64}(0.0), SVector{1, Float64}(1.0)],
        [:Γ => :X],
        [1, 2],
        [1:2]
    )
    bands = rand(2, 2)
    band_data = BandStructureData(kpath, bands, 2)

    println("Attempting to call Eliashberg.plot(band_data)...")
    # This should trigger lazy loading of Visualization and succeed (it won't render without a backend, but it should reach the plotter)
    try
        f = plot(band_data)
        println("Success: plot(band_data) returned a $(typeof(f))")
    catch e
        if e isa MethodError && e.f === Eliashberg.Visualization.Makie.plot
            # This is expected if no Makie backend is loaded, but it means we REACHED the recipe!
            println("Caught expected MethodError (no Makie backend), but successfully reached the recipe dispatch.")
        else
            rethrow(e)
        end
    end

    # 3. Check internal access
    @test Eliashberg._plot_band_structure(band_data) isa Any
end
