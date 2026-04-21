# src/Models/spinor.jl

"""
    SpinorDispersion{D,M} <: ElectronicDispersion{D}

Opt-in wrapper that promotes an existing bare electronic dispersion into a
spin-degenerate basis. The wrapped model keeps its original orbital structure,
while `ε(k)` is lifted to a block-diagonal Hamiltonian with explicit spin-up and
spin-down sectors.
"""
struct SpinorDispersion{D,M<:ElectronicDispersion{D}} <: ElectronicDispersion{D}
    bare::M
end
Base.show(io::IO, m::SpinorDispersion{D}) where {D} = print(io, "SpinorDispersion (", D, "D, bare=", m.bare, ")")

SpinorDispersion(model::SpinorDispersion) = model
