using Eliashberg
using AtomsBase
using Unitful
using StaticArrays

println("--- Eliashberg Geometry Verification ---")

try
    # 1. Test lattice construction
    println("[1/4] testing ChainLattice...")
    lat1d = ChainLattice(1.5)
    println("      Success: $(typeof(lat1d))")

    # 2. Test crystal factory (atomic_chain)
    println("[2/4] testing atomic_chain...")
    sys1d = atomic_chain(1.5, :H)
    println("      Success: $(typeof(sys1d)) with $(length(sys1d)) atoms")
    
    # 3. Test k-grid generation
    println("[3/4] testing generate_kgrid...")
    kg = generate_kgrid(sys1d, 100)
    println("      Success: KGrid with $(length(kg.points)) points")

    # 4. Test k-path generation
    println("[4/5] testing generate_kpath...")
    kp = generate_kpath(sys1d; n_pts_per_segment=50)
    println("      Success: KPath with $(length(path_points(kp))) points")

    # 5. Test TightBinding model
    println("[5/5] testing TightBinding model...")
    tb = TightBinding(sys1d, 1.0, -0.3)
    println("      Success: TightBinding model created")

    println("\n--- ALL TESTS PASSED ---")
catch e
    println("\n--- TEST FAILED ---")
    rethrow(e)
end
