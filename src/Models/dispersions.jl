# ---------------------------------------------------------
# Unified Tight-Binding Constructors (Symbol Dispatch)
# ---------------------------------------------------------

"""
    TightBinding(type::Symbol, [lat::Lattice], args...)

Unified constructor for all standard tight-binding models.
Available types: `:chain`, `:square`, `:triangular`, `:cubic`, `:fcc`, `:bcc`.

If a custom `Lattice` is provided, it applies the standard hopping topology 
to the given geometry (useful for modeling strain). Otherwise, an ideal 
lattice with `a=1.0` is generated automatically.
"""
TightBinding(sym::Symbol, args...) = TightBinding(Val(sym), args...)
TightBinding(sym::Symbol, lat::Lattice, args...) = TightBinding(Val(sym), lat, args...)

# --- 1D Chain ---
function TightBinding(::Val{:chain}, lat::Lattice{1}, t::Float64, EF::Float64=0.0)
    hops = [(SVector{1,Int}(1), -t)]
    return TightBinding{1}(lat, hops, EF)
end
TightBinding(::Val{:chain}, t::Float64, EF::Float64=0.0) = TightBinding(Val(:chain), chain_lattice(1.0), t, EF)

# --- 2D Square ---
function TightBinding(::Val{:square}, lat::Lattice{2}, t::Float64, tp::Float64=0.0, EF::Float64=0.0)
    hops = [
        (SVector{2,Int}(1, 0), -t),
        (SVector{2,Int}(0, 1), -t)
    ]
    if abs(tp) > 1e-10
        push!(hops, (SVector{2,Int}(1, 1), -tp))
        push!(hops, (SVector{2,Int}(-1, 1), -tp))
    end
    return TightBinding{2}(lat, hops, EF)
end

# --- 2D Triangular (Hexagonal Lattice) ---
function TightBinding(::Val{:triangular}, lat::Lattice{2}, t::Float64, EF::Float64=0.0)
    hops = [
        (SVector{2,Int}(1, 0), -t),
        (SVector{2,Int}(0, 1), -t),
        (SVector{2,Int}(-1, 1), -t)
    ]
    return TightBinding{2}(lat, hops, EF)
end
TightBinding(::Val{:triangular}, t::Float64, EF::Float64=0.0) = TightBinding(Val(:triangular), hexagonal_lattice(1.0), t, EF)

# --- 3D Simple Cubic ---
function TightBinding(::Val{:cubic}, lat::Lattice{3}, t::Float64, EF::Float64=0.0)
    hops = [
        (SVector{3,Int}(1, 0, 0), -t),
        (SVector{3,Int}(0, 1, 0), -t),
        (SVector{3,Int}(0, 0, 1), -t)
    ]
    return TightBinding{3}(lat, hops, EF)
end
TightBinding(::Val{:cubic}, t::Float64, EF::Float64=0.0) = TightBinding(Val(:cubic), cubic_lattice(1.0), t, EF)

# --- 3D Face-Centered Cubic (FCC) ---
function TightBinding(::Val{:fcc}, lat::Lattice{3}, t::Float64, EF::Float64=0.0)
    hops = [
        (SVector{3,Int}(1, 0, 0), -t),
        (SVector{3,Int}(0, 1, 0), -t),
        (SVector{3,Int}(0, 0, 1), -t),
        (SVector{3,Int}(1, -1, 0), -t),
        (SVector{3,Int}(0, 1, -1), -t),
        (SVector{3,Int}(-1, 0, 1), -t)
    ]
    return TightBinding{3}(lat, hops, EF)
end
TightBinding(::Val{:fcc}, t::Float64, EF::Float64=0.0) = TightBinding(Val(:fcc), fcc_lattice(1.0), t, EF)

# --- 3D Body-Centered Cubic (BCC) ---
function TightBinding(::Val{:bcc}, lat::Lattice{3}, t::Float64, EF::Float64=0.0)
    hops = [
        (SVector{3,Int}(1, 0, 0), -t),
        (SVector{3,Int}(0, 1, 0), -t),
        (SVector{3,Int}(0, 0, 1), -t),
        (SVector{3,Int}(1, 1, 1), -t)
    ]
    return TightBinding{3}(lat, hops, EF)
end
TightBinding(::Val{:bcc}, t::Float64, EF::Float64=0.0) = TightBinding(Val(:bcc), bcc_lattice(1.0), t, EF)

# ---------------------------------------------------------
# Universal Hamiltonian Evaluator (统一的核心求解器)
# ---------------------------------------------------------

"""
    ε(k::SVector{D, Float64}, model::TightBinding{D}) where {D}

Evaluates the tight-binding Hamiltonian at momentum `k` using Fourier transform
over the explicit real-space hopping vectors.
"""
function ε(k::SVector{D,Float64}, model::TightBinding{D}) where {D}
    # Start with the on-site energy (Fermi level shift)
    E_k = -model.EF

    # Sum over all hopping vectors
    for (R_idx, t_hop) in model.hoppings
        # Calculate real-space vector R = n1*a1 + n2*a2 ...
        R = model.lattice.vectors * R_idx

        # Add Hermitian pair contribution: t * exp(ikR) + t * exp(-ikR) = 2t * cos(k·R)
        E_k += 2 * t_hop * cos(dot(k, R))
    end

    return Hermitian(hcat(E_k))
end

# ---------------------------------------------------------
# Multi-Orbital Legacy / Convenience Constructors
# ---------------------------------------------------------

KagomeLattice(t::Float64, EF::Float64=0.0) = KagomeLattice(hexagonal_lattice(1.0), t, EF)

Graphene(t::Float64, EF::Float64=0.0) = Graphene(hexagonal_lattice(1.0), t, EF)

SSHModel(t1::Float64, t2::Float64, EF::Float64=0.0) = SSHModel(chain_lattice(1.0), t1, t2, EF)


# ---------------------------------------------------------
# Hamiltonian Evaluators (ε and ω)
# ---------------------------------------------------------

function ε(k::SVector{2,Float64}, model::KagomeLattice)
    a1 = model.lattice.vectors[:, 1]
    a2 = model.lattice.vectors[:, 2]

    # Fully vector-based, invariant to lattice scaling/rotation
    h12 = -2 * model.t * cos(dot(k, a1) / 2.0)
    h13 = -2 * model.t * cos(dot(k, a2) / 2.0)
    h23 = -2 * model.t * cos(dot(k, a1 - a2) / 2.0)

    H = @SMatrix [
        -model.EF h12 h13;
        h12 -model.EF h23;
        h13 h23 -model.EF
    ]
    return Hermitian(H)
end

"""
    ε(k::SVector{2, Float64}, model::Graphene)

Evaluates the 2x2 Dirac Hamiltonian for Graphene.
Uses the phase shifts strictly derived from the hexagonal primitive vectors.
"""
function ε(k::SVector{2,Float64}, model::Graphene)
    a1 = model.lattice.vectors[:, 1]
    a2 = model.lattice.vectors[:, 2]

    # The off-diagonal element connects the A and B sublattices
    f_k = 1.0 + exp(-im * dot(k, a1)) + exp(-im * dot(k, a2))
    h_AB = -model.t * f_k

    H = @SMatrix [
        -model.EF h_AB;
        conj(h_AB) -model.EF
    ]
    return Hermitian(H)
end

"""
    ε(k::SVector{1, Float64}, model::SSHModel)

Evaluates the 2x2 Hamiltonian for the 1D SSH topological chain.
"""
function ε(k::SVector{1,Float64}, model::SSHModel)
    a1 = model.lattice.vectors[:, 1]

    # t1 is intra-cell (no phase), t2 is inter-cell (phase shift by a1)
    h_AB = model.t1 + model.t2 * exp(-im * dot(k, a1))

    H = @SMatrix [
        -model.EF h_AB;
        conj(h_AB) -model.EF
    ]
    return Hermitian(H)
end

function ε(
    k::SVector{D,Float64},
    model::RenormalizedDispersion{D}
) where {D}
    bare_H = ε(k, model.bare_dispersion)
    Σ_H = Σ(k, model.self_energy)
    return Hermitian(bare_H + Σ_H)
end

# ---------------------------------------------------------
# Phonon Dispersions Evaluators
# ---------------------------------------------------------

MonoatomicLatticeModel{1}(K::Float64, M::Float64, a::Float64=1.0) = MonoatomicLatticeModel{1}(chain_lattice(a), K, M)
MonoatomicLatticeModel{2}(K::Float64, M::Float64, a::Float64=1.0) = MonoatomicLatticeModel{2}(square_lattice(a), K, M)
MonoatomicLatticeModel{3}(K::Float64, M::Float64, a::Float64=1.0) = MonoatomicLatticeModel{3}(cubic_lattice(a), K, M)

"""
    ω(q::SVector{D,Float64}, d::MonoatomicLatticeModel{D})

Computes the acoustic phonon dispersion strictly using the real-space lattice vectors.
"""
function ω(q::SVector{D,Float64}, d::MonoatomicLatticeModel{D}) where {D}
    val = 0.0
    # Sum over the primitive lattice vectors dynamically
    for i in 1:D
        a_vec = d.lattice.vectors[:, i]
        val += 1.0 - cos(dot(q, a_vec))
    end
    return sqrt(2 * d.K / d.M * val)
end

ω(q::SVector{D,Float64}, model::EinsteinModel) where {D} = model.ωE
ω(q::SVector{D,Float64}, model::DebyeModel) where {D} = model.vs * norm(q)
ω(q::SVector{D,Float64}, d::PolaritonModel) where {D} = sqrt(d.ωE^2 + (d.vs * norm(q))^2)

"""
    band_structure(disp, k)

Returns an `Eigen` object for the Hamiltonian matrix at momentum `k`.
"""
function band_structure(disp::ElectronicDispersion{D}, k::SVector{D,Float64}) where {D}
    H = ε(k, disp)
    return eigen(H)
end