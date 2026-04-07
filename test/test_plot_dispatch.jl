using Test
using Eliashberg
using Makie
using StaticArrays

@testset "Typed Plot Dispatch" begin
    lattice2d = SquareLattice(1.0)
    model2d = TightBinding(lattice2d, 1.0, 0.0, 0.0)
    grid2d = generate_2d_kgrid(4, 4)

    surface_data = compute_dispersion_surface_data(model2d, grid2d)
    @test surface_data isa DispersionSurfaceData
    @test size(surface_data.energy_matrix) == (4, 4)
    @test plot(surface_data) isa Figure

    model3d = TightBinding(CubicLattice(1.0), 1.0, 0.0)
    fermi_surface_data = compute_fermi_surface_volume(model3d; n_pts=4)
    @test fermi_surface_data isa FermiSurfaceData
    @test size(fermi_surface_data.energy_volume) == (4, 4, 4)
    @test plot(fermi_surface_data) isa Figure

    landscape = scan_instability_landscape(model2d, grid2d, grid2d; T=0.1, η=1e-3)
    line_grid = generate_1d_kgrid(8)
    line_landscape_data = compute_landscape_line_data(line_grid, collect(1.0:length(line_grid)))
    @test line_landscape_data isa LandscapeLineData
    @test plot(line_landscape_data) isa Figure

    surface_landscape_data = compute_landscape_surface_data(grid2d, landscape)
    @test surface_landscape_data isa LandscapeSurfaceData
    @test size(surface_landscape_data.landscape_matrix) == (4, 4)
    @test plot(surface_landscape_data) isa Figure

    cell2d = PeriodicCell(lattice2d; periodicity=(true, true))
    reciprocal_plot = visualize_reciprocal_space(cell2d, generate_reciprocal_lattice(cell2d, 4, 4))
    @test reciprocal_plot isa Figure
    @test visualize_lattice(Eliashberg.primitive_vectors(lattice2d)) isa Figure

    zeeman_data = ZeemanPairingData(
        collect(range(-0.3, 0.3, length=3)),
        [-0.1, -0.2, -0.05],
        [0.2, 0.3, 0.1],
        0.0,
        2
    )
    @test zeeman_data isa ZeemanPairingData
    @test plot(zeeman_data) isa Figure
end
