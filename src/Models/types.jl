# Models/types.jl

abstract type PhysicalModel end

# Dispersion types
abstract type Dispersion{D} <: PhysicalModel end

# Electronic Dispersions
abstract type ElectronicDispersion{D} <: Dispersion{D} end

# ---------------------------------------------------------
# Phonon Dispersions
# ---------------------------------------------------------
abstract type PhononDispersion{D} <: Dispersion{D} end


# ---------------------------------------------------------
# Phonon Dispersions
# ---------------------------------------------------------
abstract type PhononDispersion{D} <: Dispersion{D} end


# Interaction types
abstract type Interaction end
abstract type CoulombInteraction <: Interaction end
abstract type ElectronPhononInteraction <: Interaction end
abstract type ScreenedInteraction <: Interaction end

