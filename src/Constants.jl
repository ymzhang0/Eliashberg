module Constants

using Unitful
using PhysicalConstants.CODATA2022

const Å      = 1e-10 # m

const a0     = CODATA2022.BohrRadius            # 单位: m
const Ry     = CODATA2022.RydbergConstant            # 单位: J
const me     = CODATA2022.ElectronMass            # 单位: C
const e      = CODATA2022.ElementaryCharge            # 单位: C
const ε0     = CODATA2022.VacuumElectricPermittivity           # 单位: F/m
const h      = CODATA2022.PlanckConstant            # 单位: J·s
const ħ      = CODATA2022.ReducedPlanckConstant            # 单位: J·s
const kB     = CODATA2022.BoltzmannConstant           # 单位: J/K
const c      = CODATA2022.SpeedOfLightInVacuum            # 单位: m/s

const Ry2J     = h * c * Ry    # 单位: J            # 单位: J
const Ry2eV     = Ry2J / e    # 单位: J            # 单位: J
const Ha2J     = 2 * Ry2J
const Ha2eV     = Ha2J / e    # 单位: J            # 单位: J

const kB2meV    = kB / e * 1e3    # 单位: meV/K
const kB2eV    = kB / e    # 单位: eV/K
const kB2Ha    = kB / Ha2J    # 单位: Ha/K



const A2Bohr    = a0 / Å    # 单位: Bohr^{-1}
# === 常用单位 ===
end
