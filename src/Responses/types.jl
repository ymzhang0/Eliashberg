# Responses/types.jl

# Auxiliary field types
abstract type AuxiliaryField end

# Propagator types
abstract type Propagator end
abstract type PhononPropagator <: Propagator end
abstract type ElectronPropagator <: Propagator end
abstract type GorkovPropagator <: Propagator end

# Many-body types
abstract type SelfEnergy end

abstract type Smearing end
abstract type Polarization end
abstract type SpectralFunction end
abstract type ElectronSpectralFunction <: SpectralFunction end
abstract type PhononSpectralFunction <: SpectralFunction end
abstract type GapFunction end
