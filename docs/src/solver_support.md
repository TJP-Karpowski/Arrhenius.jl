# Solver and Mechanism Support

Arrhenius currently treats solver support as a validated matrix, not as a
single universal recommendation. Combustion mechanisms are stiff, and solver
trial states can temporarily leave the physically valid domain even when the
accepted solution is valid.

## Current solver picture

- `CVODE_BDF` is the strongest baseline for larger chemistry such as `gri30`.
  In the current benchmarks it is the path to beat for GRI30 ignition.
- Native OrdinaryDiffEq stiff solvers are viable for small hydrogen chemistry
  when they are paired with domain guards and, for `h2_li_2004`, tighter step
  control.
- `h2_li_2004` is deliberately tested as a hydrogen robustness case. Native
  stiff solvers are expected to match a trusted final state; `CVODE_BDF` is
  currently expected to fail this case.
- Failed or non-successful solver runs must not be interpreted as performance
  wins. Benchmarks and tests require `retcode == Success` and final-state
  validation for supported combinations.

## Mechanism coverage

The core test suite covers bundled mechanisms including `h2o2.yaml`,
`h2_li_2004.yaml`, and `gri30.yaml`. It also includes copied Cantera 3.2.0
fixtures under `test/mechanisms/cantera`:

- `air.yaml`
- `gri30_highT.yaml`
- `SiF4_NH3_mec.yaml`, using the `gas` phase
- `nDodecane_Reitz.yaml`, using the `nDodecane_IG` ideal-gas phase

These fixtures harden Cantera-format parser support and the ideal-gas gas-phase
RHS path. The compact `air.yaml` fixture also exercises the generated Jacobian
path, while bundled mechanisms continue to cover larger analytical and generated
Jacobian behavior. The fixtures do not imply support for every Cantera YAML
feature; multiphase mechanisms, surface phases, transport-only files, and
mechanisms without gas reactions remain outside the validated surface unless
separate tests are added.

The fixture set is restricted to Cantera top-level data files distributed under
Cantera's BSD-3-Clause package license. Cantera `example_data` mechanisms are
not vendored by default because their README says Cantera is not granting a
license for those externally authored mechanisms.

Pressure-dependent Arrhenius tables are supported in the reactor RHS by
log-interpolating the Arrhenius rate in log pressure, using the instantaneous
ideal-gas mixture pressure from the concentration state. Cantera `linear-Burke`
reactions are supported for the reference `M` pressure-dependent Arrhenius table
plus collider-specific Arrhenius efficiency functions, using
`P_eff = P * sum(X_i * epsilon_i(T))`.

The analytical isobaric Jacobian supports these pressure-dependent rates,
including concentration and temperature derivatives of `P_eff`. Generated
analytical Jacobian workspaces emit mechanism-specialized kernels for ordinary
mechanisms and small pressure-dependent mechanisms, including PLOG and
`linear-Burke` effective-pressure derivatives. Large pressure-dependent
mechanisms use the schedule-based analytical kernel through the same workspace
API to avoid pathological generated-code compile latency.

## Release rule

A mechanism/solver pair should be documented as supported only after it has:

- mechanism construction tests
- finite RHS tests at representative states
- finite Jacobian tests when Jacobian support is advertised
- integration tests with `retcode == Success`
- final temperature and major-species validation against a trusted reference
