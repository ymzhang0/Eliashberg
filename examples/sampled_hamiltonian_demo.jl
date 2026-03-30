using Eliashberg
using CairoMakie
using LinearAlgebra
using SparseArrays

function matrix_view_data(matrix)
    if issparse(matrix)
        return Float64.(Matrix(matrix .!= 0))
    end

    return log10.(abs.(Matrix(matrix)) .+ 1e-8)
end

function sampled_hamiltonian_figure(title_text, assembly, spectrum)
    fig = Figure(size=(1100, 420), fontsize=16)

    ax_matrix = Axis(
        fig[1, 1];
        title="$title_text Matrix",
        xlabel="Column Index",
        ylabel="Row Index"
    )

    matrix_data = matrix_view_data(assembly.matrix)
    hm = heatmap!(ax_matrix, matrix_data; colormap=:magma)
    Colorbar(fig[1, 2], hm, label=issparse(assembly.matrix) ? "Nonzero Pattern" : "log10(|H| + 1e-8)")

    ax_spec = Axis(
        fig[1, 3];
        title="$title_text Spectrum",
        xlabel="State Index",
        ylabel="Eigenvalue"
    )

    eigenvalues = sort(real.(spectrum.values))
    scatterlines!(ax_spec, collect(eachindex(eigenvalues)), eigenvalues; color=:royalblue, markersize=6, linewidth=2)
    hlines!(ax_spec, [0.0], color=:black, linestyle=:dash, alpha=0.4)

    return fig
end

function run_graphene_demo(output_dir::AbstractString)
    lattice = HexagonalLattice(1.0)
    model = Graphene(1.0, 0.0)
    kgrid = generate_2d_kgrid(6, 6)

    assembly = assemble_sampled_hamiltonian(kgrid, model)
    spectrum = solve_sampled_hamiltonian(kgrid, model)

    fig = sampled_hamiltonian_figure("Graphene Sampled Hamiltonian", assembly, spectrum)
    save(joinpath(output_dir, "graphene_sampled_hamiltonian.png"), fig)

    println("Saved graphene sampled-Hamiltonian demo to $(joinpath(output_dir, "graphene_sampled_hamiltonian.png"))")
    println("  block sizes = $(assembly.layout.row_axis.block_sizes[1:min(end, 8)])")
end

function run_bdg_demo(output_dir::AbstractString)
    lattice = ChainLattice(1.0)
    bare_model = TightBinding(lattice, 1.0, -0.3)
    field = BCSReducedPairing(:s_wave)
    bdg_model = MeanFieldDispersion(bare_model, field, 0.25)
    kgrid = generate_1d_kgrid(32)

    sparse_hook = SparseEigenSolverHook((matrix; kwargs...) -> eigen(Matrix(matrix)))

    assembly = assemble_sampled_hamiltonian(kgrid, bdg_model; matrix_format=:sparse)
    spectrum = solve_sampled_hamiltonian(kgrid, bdg_model; matrix_format=:sparse, eigensolver=sparse_hook)

    fig = sampled_hamiltonian_figure("BCS BdG Sampled Hamiltonian", assembly, spectrum)
    save(joinpath(output_dir, "bdg_sampled_hamiltonian.png"), fig)

    println("Saved BdG sampled-Hamiltonian demo to $(joinpath(output_dir, "bdg_sampled_hamiltonian.png"))")
    println("  block sizes = $(assembly.layout.row_axis.block_sizes[1:min(end, 8)])")
end

function main(output_dir::AbstractString)
    mkpath(output_dir)
    run_graphene_demo(output_dir)
    run_bdg_demo(output_dir)
end

output_dir = isempty(ARGS) ? string(@__DIR__) : ARGS[1]
main(output_dir)
