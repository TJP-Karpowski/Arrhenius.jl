# Mechanism Support

Arrhenius should advertise a mechanism feature only when construction, source
terms, Jacobian behavior, integration, and final-state validation are covered by
tests or benchmarks. The table below summarizes the current validated surface.

| Mechanism or fixture | Parser | CPU RHS | Equilibrium | Ignition tests | Analytical Jacobian | Packed Metal RHS | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mechanism/h2o2.yaml` | yes | yes | TP, HP | yes | yes | yes | Packed Metal path covers third-body, falloff, Troe, and reversible reactions without pressure-dependent tables. |
| `mechanism/h2_li_2004.yaml` | yes | yes | TP, HP | yes | partial | no | H2-air equilibrium is validated from phi 0.5 to 2.0 and 0.1 to 100 bar. |
| `mechanism/h2_12s.yaml` (`gas`) | yes | yes | not separately validated | no | no | no | Published Boivin 9-species/12-step skeletal H2 mechanism; rates and counts have regression coverage. |
| `mechanism/h2_12s.yaml` (`boivin-qss`) | yes | yes, custom QSS | not applicable | flame integration | ForwardDiff local Jacobian | no | Published three-step flame model with dynamic H2/H/O2/H2O/HO2/N2 and algebraic O/OH/H2O2. |
| `mechanism/h2_12s.yaml` (`boivin-four-step`) | yes | yes, custom QSS | not applicable | flame integration | ForwardDiff local Jacobian | no | Published 2013 four-step model with dynamic H2/H/O2/H2O/HO2/H2O2/N2 and algebraic O/OH. |
| `mechanism/h2_12s.yaml` (`boivin-h-only`) | yes | yes, custom QSS | not applicable | flame baseline | ForwardDiff local Jacobian | no | Experimental two-step descendant of the 2011 model: dynamic H2/H/O2/H2O/N2, with O/OH/HO2/H2O2 algebraic. This is not a separately published Boivin mechanism. |
| `mechanism/v_h2_air.yaml` | yes | yes | not applicable | flame baseline | yes | no | Published EM2C V-H2-Air v1.0.0 virtual mechanism (CC-BY-4.0): four reactions, four physical species, and virtual species P1/I1. CPU RHS is regression-tested against Cantera 3.2. |
| `mechanism/gri30.yaml` | yes | yes | TP, HP | yes | yes | no | CH4-air equilibrium is validated from phi 0.5 to 2.0 and 0.1 to 100 bar. |
| `mechanism/nh3_stagni_2025.yaml` | yes | yes | not separately validated | Cantera 3.2 profiles | not separately validated | no | Pure NH3/air at 20 bar and a 50% H2 fuel blend at 10 bar are validated at 1000 K against committed full-composition ignition profiles. |
| `test/mechanisms/cantera/air.yaml` | yes | yes | no | no | generated smoke | no | Parser and generated-Jacobian fixture copied from Cantera top-level data. |
| `test/mechanisms/cantera/gri30_highT.yaml` | yes | yes | no | no | no | no | Cantera-format parser fixture. |
| `test/mechanisms/cantera/SiF4_NH3_mec.yaml` | yes | finite-state fixture | no | no | no | no | Ammonia-containing Cantera fixture using the `gas` phase. |
| `test/mechanisms/cantera/nDodecane_Reitz.yaml` | yes | finite-state fixture | no | no | no | no | Long-hydrocarbon Cantera fixture using the `nDodecane_IG` phase. |

## Feature Status

| Feature | Status |
| --- | --- |
| Cantera-style YAML gas phases | Supported for the tested ideal-gas feature set. |
| NASA polynomial thermo | Supported for ideal-gas workflows. |
| Reversible Arrhenius reactions | Supported. |
| Three-body reactions | Supported on CPU and packed no-pressure-dependent Metal path. |
| Falloff and Troe reactions | Supported on CPU and packed no-pressure-dependent Metal path. |
| Pressure-dependent Arrhenius tables | Supported on CPU RHS and analytical Jacobian. |
| Cantera `linear-Burke` reactions | Supported on CPU RHS and analytical Jacobian for the tested reference-table plus collider-efficiency form. |
| Packed Metal pressure-dependent rates | Not implemented yet. |
| Multiphase, surface, and fixed-volume equilibrium | Not implemented yet. |
| Mixture-averaged transport parsing | Supported for tested mechanisms; transport solving is not the main validated path. |
| Explicit QSS chemistry | The published Boivin three- and four-step models are supported. `BoivinHOnlyReactor` adds an experimental two-step H-only representation by imposing QSS on the third global mode of the published three-step model. |

## Equilibrium validation matrix

The stored Cantera 3.2 reference suite compares complete species compositions,
not only selected major products. H2-air and CH4-air cover `phi` values 0.5,
0.75, 1.0, 1.25, 1.5, and 2.0 at 0.1, 1, 10, and 100 bar. TP cases use
2500 K and HP cases start at 1000 K. Tests also verify element conservation,
fixed pressure, convergence residuals, and enthalpy conservation for HP.

## Adding Coverage

For a new mechanism, add the smallest useful hardening tests first:

1. mechanism construction
2. finite RHS at representative states
3. finite Jacobian if Jacobian support is claimed
4. integration with `retcode == Success`
5. final temperature and major-species comparison against Cantera or another trusted reference

Mechanisms copied from Cantera should stay restricted to files distributed under
Cantera's package license unless the upstream mechanism license is explicitly
compatible with redistribution.
