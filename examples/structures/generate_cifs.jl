using Eliashberg
using AtomsIO
using AtomsBase

# 1. Define the structures to export
structures = Dict(
    "Al" => Eliashberg.aluminium(),
    "Cu" => Eliashberg.copper(),
    "Fe" => Eliashberg.iron(),
    "Nb" => Eliashberg.niobium(),
    "V"  => Eliashberg.vanadium(),
    "C_diamond" => Eliashberg.diamond(),
    "Si" => Eliashberg.silicon(),
    "Ge" => Eliashberg.germanium()
)

# 2. Create output directory
output_dir = joinpath(@__DIR__, "cif")
mkpath(output_dir)

# 3. Export to CIF
println("Starting CIF export to $output_dir...")
for (name, sys) in structures
    filename = joinpath(output_dir, "$name.cif")
    # Note: AtomsIO.save handles CIF if .cif extension is provided
    try
        AtomsIO.save(filename, sys)
        println("  [SUCCESS] Saved $filename")
    catch e
        @warn "  [FAILED] Could not save $name: $e"
    end
end
println("Done.")
