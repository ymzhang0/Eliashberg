# Eliashberg

`Eliashberg.jl` is a Julia package for lattice-based electronic structure, generalized susceptibilities, mean-field effective actions, and visualization of symmetry-breaking phases.

## API Note

The current API is built around concrete lattice types such as `ChainLattice`, `SquareLattice`, `HexagonalLattice`, `CubicLattice`, `FCCLattice`, and `BCCLattice`.

Please use constructors like:

```julia
lattice = SquareLattice(1.0)
model = TightBinding(lattice, 1.0, 0.2, 0.0)
```

Do not rely on older patterns such as `TightBinding{D}(...)` or helper names like `chain_lattice(...)`.

## Getting Started

From a local checkout:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Eliashberg
using StaticArrays
```

## Geometry and Grids

### 1D chain

```julia
lattice = ChainLattice(1.5)
kgrid = generate_reciprocal_lattice(lattice, 200)
kpath = generate_kpath(lattice; n_pts_per_segment=100)
```

### 2D square lattice

```julia
lattice = SquareLattice(1.0)
kgrid = generate_reciprocal_lattice(lattice, 80, 80)
kpath = generate_kpath(lattice; n_pts_per_segment=50)
```

### 3D cubic lattice

```julia
lattice = CubicLattice(1.0)
kgrid = generate_reciprocal_lattice(lattice, 24, 24, 24)
kpath = generate_kpath(lattice; n_pts_per_segment=30)
```

## Tight-Binding Models

`TightBinding` dispatches on the concrete lattice type:

- `TightBinding(ChainLattice(...), t, EF=0.0)`
- `TightBinding(SquareLattice(...), t, tp=0.0, EF=0.0)`
- `TightBinding(HexagonalLattice(...), t, EF=0.0)` for the triangular-lattice basis
- `TightBinding(CubicLattice(...), t, EF=0.0)`
- `TightBinding(FCCLattice(...), t, EF=0.0)`
- `TightBinding(BCCLattice(...), t, EF=0.0)`

Example:

```julia
lattice = SquareLattice(1.0)
model = TightBinding(lattice, 1.0, 0.2, 0.0)

kΓ = SVector{2, Float64}(0.0, 0.0)
HΓ = ε(kΓ, model)
bands = band_structure(model, kΓ)
```

You can also use the exported free-electron model:

```julia
free_model = FreeElectron{2}(0.5)
Hk = ε(SVector{2, Float64}(1.0, 2.0), free_model)
```

## Visualizing Lattices and Bands

```julia
using CairoMakie

lattice = SquareLattice(1.0)
model = TightBinding(lattice, 1.0, 0.2, 0.0)
kgrid = generate_reciprocal_lattice(lattice, 80, 80)
kpath = generate_kpath(lattice; n_pts_per_segment=50)

fig1 = visualize_lattice(lattice)
fig2 = visualize_reciprocal_space(lattice, kgrid)
fig3 = visualize_dispersion(model, kpath)
fig4 = visualize_dispersion(model, kgrid; E_Fermi=0.0)
```

## Mean-Field Action and Order Parameters

For superconducting or density-wave channels, use `evaluate_action` and `solve_ground_state` directly.

### BCS pairing

```julia
lattice = ChainLattice(1.5)
model = TightBinding(lattice, 1.0, -0.3)
kgrid = generate_reciprocal_lattice(lattice, 400)

field = BCSReducedPairing(:s_wave)
interaction = ConstantInteraction(-2.5)

phis = range(0.0, 1.0, length=80)
Ts = range(0.1, 0.4, length=10)

F_exact = evaluate_action(phis, field, model, interaction, kgrid, ExactTrLn(); T=0.1)
phi_gs = solve_ground_state(field, model, interaction, kgrid, ExactTrLn(); phi_guess=0.2, T=0.1)

fig = visualize_phase_transition(phis, Ts, field, model, interaction, kgrid)
bands_fig = visualize_renormalized_bands(Ts, field, model, interaction, kgrid, generate_kpath(lattice))
```

### Charge-density-wave channel

```julia
lattice = SquareLattice(1.0)
model = TightBinding(lattice, 1.0, 0.0, 0.0)
kgrid = generate_2d_kgrid(80, 80)

field = ChargeDensityWave(SVector{2, Float64}(pi, pi))
interaction = ConstantInteraction(2.0)

phis = range(0.0, 1.5, length=40)
F_rpa = evaluate_action(phis, field, model, interaction, kgrid, RPA(); T=0.1)
phi_gs = solve_ground_state(field, model, interaction, kgrid, ExactTrLn(); phi_guess=0.5, T=0.1)
```

## Sampled Hamiltonian Assembly

For multi-orbital Bloch models and BdG mean-field models, you can assemble the
sampled direct-sum Hamiltonian explicitly and then diagonalize it through the
engine solve layer.

### Multi-orbital example: Graphene

```julia
lattice = HexagonalLattice(1.0)
model = Graphene(1.0, 0.0)
kgrid = generate_2d_kgrid(6, 6)

assembly = assemble_sampled_hamiltonian(kgrid, model)
spectrum = solve_sampled_hamiltonian(kgrid, model)

size(assembly.matrix) == (2 * length(kgrid), 2 * length(kgrid))
assembly.layout.row_axis.block_sizes[1] == 2
```

### BdG example: 1D BCS mean-field Hamiltonian

```julia
bare_model = TightBinding(ChainLattice(1.0), 1.0, -0.3)
field = BCSReducedPairing(:s_wave)
bdg_model = MeanFieldDispersion(bare_model, field, 0.25)
kgrid = generate_1d_kgrid(32)

# In production you can replace this with a package-backed sparse eigensolver.
sparse_hook = SparseEigenSolverHook((matrix; kwargs...) -> eigen(Matrix(matrix)))

assembly = assemble_sampled_hamiltonian(kgrid, bdg_model; matrix_format=:sparse)
spectrum = solve_sampled_hamiltonian(
    kgrid,
    bdg_model;
    matrix_format=:sparse,
    eigensolver=sparse_hook
)

issparse(assembly.matrix)
length(spectrum.values) == 2 * length(kgrid)
```

A complete visualization demo for both cases lives in:

- `examples/sampled_hamiltonian_demo.jl`

## Susceptibilities and Spectral Scans

Static and dynamical response calculations use `GeneralizedSusceptibility`, `scan_instability_landscape`, and `scan_spectral_function`.

```julia
lattice = SquareLattice(1.0)
model = TightBinding(lattice, 1.0, 0.0, 0.0)
kgrid = generate_2d_kgrid(60, 60)
qgrid = generate_2d_kgrid(60, 60)

field = ChargeDensityWave(SVector{2, Float64}(0.0, 0.0))
chi0 = GeneralizedSusceptibility(model, kgrid, field, 0.1)
chi_q = chi0(SVector{2, Float64}(0.1, 0.1))

landscape = scan_instability_landscape(model, kgrid, qgrid; T=0.1, η=1e-3)
fig = visualize_landscape(Val(2), qgrid, landscape)
```

For a momentum-frequency spectral map:

```julia
lattice = ChainLattice(1.5)
model = TightBinding(lattice, 1.0, -0.3)
kgrid = generate_reciprocal_lattice(lattice, 200)

q_nodes = [SVector{1, Float64}(0.0), SVector{1, Float64}(pi)]
q_labels = ["Γ", "X"]
qpath = generate_kpath(q_nodes, q_labels; n_pts_per_segment=100)
omegas = range(0.0, 5.0, length=200)

spectral = scan_spectral_function(model, kgrid, qpath, omegas; T=0.01, η=0.02)
fig = visualize_spectral_function(qpath, omegas, spectral)
```

## Example Files

Current example entry points live in `examples/`:

- `examples/test_3d_viz.jl`
- `examples/verify_all_vizes.jl`
- `examples/Eliashberg_Dashboard.jl`
- `examples/sampled_hamiltonian_demo.jl`
- `examples/geometry.ipynb`
- `examples/BCS.ipynb`
- `examples/CDW.ipynb`
- `examples/plasmon.ipynb`

The test suite is also a good reference for the supported API:

- `test/runtests.jl`
- `test/test_refactor.jl`

## Current Limitations

- The recommended public geometry API is the concrete lattice hierarchy rooted at `AbstractLattice{D}`.
- `FFLOPairing` and `PairDensityWave` are present, but the corresponding normal-state basis helpers are still only partially implemented.
- Some advanced many-body pieces are still placeholders; the examples above stick to currently verified paths.
