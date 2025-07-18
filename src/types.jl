abstract type Dispersion end
abstract type ElectronicDispersion <: Dispersion end
abstract type PhononDispersion     <: Dispersion end

abstract type Smearing end

abstract type Interaction end
abstract type CoulombInteraction <: Interaction end
abstract type ElectronPhononInteraction <: Interaction end
abstract type ScreenedInteraction <: Interaction end
abstract type Polarization end 

abstract type Propagator end

abstract type PhononPropagator <: Propagator end
abstract type ElectronPropagator <: Propagator end
abstract type GorkovPropagator <: Propagator end

abstract type SelfEnergy end


abstract type SpectralFunction end
abstract type ElectronSpectralFunction <: SpectralFunction end
abstract type PhononSpectralFunction <: SpectralFunction end

abstract type GapFunction end

