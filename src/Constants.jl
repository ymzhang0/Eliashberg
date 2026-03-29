# src/Constants.jl


const Å      = 1e-10 # m

const a0     = CODATA2022.BohrRadius            # m
const Ry     = CODATA2022.RydbergConstant            # J
const me     = CODATA2022.ElectronMass            # C
const e      = CODATA2022.ElementaryCharge            # C
const ε0     = CODATA2022.VacuumElectricPermittivity           # F/m
const h      = CODATA2022.PlanckConstant            # J·s
const ħ      = CODATA2022.ReducedPlanckConstant            # J·s
const kB     = CODATA2022.BoltzmannConstant           # J/K
const c      = CODATA2022.SpeedOfLightInVacuum            # m/s

const Ry2J     = h * c * Ry    # J            # J
const Ry2eV     = Ry2J / e    # J            # J
const Ha2J     = 2 * Ry2J
const Ha2eV     = Ha2J / e    # J            # J

const kB2meV    = kB / e * 1e3    # meV/K
const kB2eV    = kB / e    # eV/K
const kB2Ha    = kB / Ha2J    # Ha/K

const A2Bohr    = a0 / Å    # Bohr^{-1}
