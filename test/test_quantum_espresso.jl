using Eliashberg
using StaticArrays
using Test
using AtomsBase
using Unitful

@testset "Quantum ESPRESSO bands parser" begin
    parsed = parse_quantum_espresso_bands(joinpath(@__DIR__, "..", "examples", "graphene", "graphene.bands.dat"))

    @test parsed.num_bands == 60
    @test parsed.num_kpoints == 274
    @test length(parsed.kpoints) == parsed.num_kpoints
    @test size(parsed.bands) == (parsed.num_kpoints, parsed.num_bands)

    @test parsed.kpoints[1] == SVector(0.0, 0.0, 0.0)
    @test parsed.kpoints[2] == SVector(0.005, 0.002887, 0.0)

    @test parsed.bands[1, 1] ≈ -21.912
    @test parsed.bands[1, 60] ≈ 35.875
    @test parsed.bands[2, 3] ≈ -5.382
    @test parsed.bands[2, 60] ≈ 35.951
end

@testset "Quantum ESPRESSO cell parser" begin
    input_file, input_io = mktemp()
    write(input_io, """
&SYSTEM
    ibrav = 0,
/
CELL_PARAMETERS angstrom
  2.460000  0.000000  0.000000
 -1.230000  2.130422  0.000000
  0.000000  0.000000 15.000000
""")
    close(input_io)

    parsed_input_cell = parse_quantum_espresso_cell(input_file)
    @test parsed_input_cell isa PeriodicCell{3}
    @test Eliashberg.primitive_vectors(parsed_input_cell)[:, 1] ≈ [2.46, 0.0, 0.0]
    @test Eliashberg.primitive_vectors(parsed_input_cell)[:, 3] ≈ [0.0, 0.0, 15.0]
    @test periodicity(parsed_input_cell) == (true, true, true)

    slab_input_cell = parse_quantum_espresso_cell(input_file; periodicity=(true, true, false))
    @test periodicity(slab_input_cell) == (true, true, false)

    output_file, output_io = mktemp()
    write(output_io, """
     bravais-lattice index     =            4
     lattice parameter (alat)  =      4.64970000  a.u.
     crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000000   0.000000000   0.000000000 )
               a(2) = (  -0.500000000   0.866025404   0.000000000 )
               a(3) = (   0.000000000   0.000000000   6.097565976 )
""")
    close(output_io)

    parsed_output_cell = parse_quantum_espresso_cell(output_file)
    @test Eliashberg.primitive_vectors(parsed_output_cell)[:, 1] ≈ [2.460515277535679, 0.0, 0.0]
    @test Eliashberg.primitive_vectors(parsed_output_cell)[:, 2] ≈ [-1.2302576387678394, 2.1308687372760087, 0.0]
    @test Eliashberg.primitive_vectors(parsed_output_cell)[:, 3] ≈ [0.0, 0.0, 15.003154239729753]
end

@testset "Quantum ESPRESSO band data wrapper" begin
    cell = @SMatrix [
        2.46 -1.23 0.0;
        0.0 2.130422 0.0;
        0.0 0.0 15.0
    ]

    data = band_data_from_quantum_espresso_bands(
        joinpath(@__DIR__, "..", "examples", "graphene", "graphene.bands.dat");
        cell=cell,
    )

    @test data isa BandStructureData{3}
    @test size(data.bands) == (274, 60)
    @test length(data.kpath) == 274
    @test isempty(getfield(data.kpath, :labels)[1])

    periodic_cell = PeriodicCell(
        ;
        cell_vectors=(
            SVector(2.46u"Å", 0.0u"Å", 0.0u"Å"),
            SVector(-1.23u"Å", 2.130422u"Å", 0.0u"Å"),
            SVector(0.0u"Å", 0.0u"Å", 15.0u"Å"),
        ),
        periodicity=(true, true, false),
    )
    periodic_data = band_data_from_quantum_espresso_bands(
        joinpath(@__DIR__, "..", "examples", "graphene", "graphene.bands.dat");
        cell=periodic_cell,
    )
    @test periodic_data isa BandStructureData{3}
    @test length(periodic_data.kpath) == 274
end
