# src/Models/predefined_models.jl

using StaticArrays

"""
    GrapheneModel(t::Real=2.8, EF::Real=0.0)
Convenience constructor for a 2-site Graphene Tight-Binding model.
Uses nearest-neighbor hopping `t` (default 2.8 eV).
"""
function GrapheneModel(t::Real=2.8, EF::Real=0.0)
    system = graphene()
    # Sites: 1 at (1/3, 2/3), 2 at (2/3, 1/3)
    # NN hoppings:
    # 1 -> 2 at R=(0,0)
    # 1 -> 2 at R=(0,1)
    # 1 -> 2 at R=(-1,1)
    hoppings = [
        (1, 2, SVector(0, 0), -t),
        (1, 2, SVector(0, 1), -t),
        (1, 2, SVector(-1, 1), -t),
    ]
    return MultiOrbitalTightBinding(system, 2, hoppings, EF)
end

"""
    KagomeModel(t::Real=1.0, EF::Real=0.0)
Convenience constructor for a 3-site Kagome Tight-Binding model.
Uses nearest-neighbor hopping `t`.
"""
function KagomeModel(t::Real=1.0, EF::Real=0.0)
    system = kagome()
    # Sites: 1 at (0.5, 0), 2 at (0, 0.5), 3 at (0.5, 0.5)
    # NN hoppings:
    # Inside cell: (1,3), (2,3)
    # Between cells: (1,2) at R=(1,-1), (1,2) at R=(0,0) ...
    # Standard Kagome NN pattern:
    hoppings = [
        # Intra-cell
        (1, 3, SVector(0, 0), -t),
        (2, 3, SVector(0, 0), -t),
        (1, 2, SVector(1, -1), -t),
        # Inter-cell neighbors
        (3, 1, SVector(0, 1), -t),
        (3, 2, SVector(1, 0), -t),
        (2, 1, SVector(0, 0), -t)
    ]
    return MultiOrbitalTightBinding(system, 3, hoppings, EF)
end

"""
    SSHModel(t1::Real, t2::Real, EF::Real=0.0)
Convenience constructor for the 1D Su-Schrieffer-Heeger (SSH) model.
`t1` is intra-cell hopping, `t2` is inter-cell hopping.
"""
function SSHModel(t1::Real, t2::Real, EF::Real=0.0)
    system = ssh_lattice()
    # Sites: 1 at (0.0), 2 at (0.5)
    hoppings = [
        (1, 2, SVector(0), -t1),
        (2, 1, SVector(1), -t2)
    ]
    return MultiOrbitalTightBinding(system, 2, hoppings, EF)
end

# --- Monoatomic Lattice Models (Phonon) ---

MonoatomicLatticeModel(::Val{1}, K::Real, M::Real, a::Real=1.0) = MonoatomicLatticeModel{1}(ChainLattice(a), Float64(K), Float64(M))
MonoatomicLatticeModel(::Val{2}, K::Real, M::Real, a::Real=1.0) = MonoatomicLatticeModel{2}(SquareLattice(a), Float64(K), Float64(M))
MonoatomicLatticeModel(::Val{3}, K::Real, M::Real, a::Real=1.0) = MonoatomicLatticeModel{3}(CubicLattice(a), Float64(K), Float64(M))

# Convenience dispatch for integers
MonoatomicLatticeModel(D::Int, K::Real, M::Real, a::Real=1.0) = MonoatomicLatticeModel(Val(D), K, M, a)
