# Get Started

## Work with a gas state

Most users should begin with `GasState`. It keeps temperature, pressure,
composition, and derived thermodynamic quantities consistent, so you do not
need to perform ideal-gas conversions manually.

From the repository root:

```julia
using Arrhenius

gas = CreateSolution("mechanism/gri30.yaml")
state = GasState(gas;
    T = 1000.0,
    p = one_atm,
    X = "CH4:1,O2:2,N2:7.52",
)
```

The composition can be a `species:amount` string, a dictionary, a collection
of pairs, or a vector in mechanism species order. Amounts supplied through
`X` or `set_TPX!` are normalized as mole fractions.

### Inspect the state

```julia
T = temperature(state)
p = pressure(state)
X = mole_fractions(state)
Y = mass_fractions(state)
C = concentrations(state)

rho = density(state)
mw = mean_molecular_weight(state)
cp = cp_mass(state)
h = enthalpy_mass(state)
s = entropy_mass(state)
g = gibbs_mass(state)
```

Species properties are available separately when needed:

```julia
cp_species = species_cp_mole(state)
h_species = species_enthalpies_mole(state)
```

Use `species_index` to map a name to mechanism order:

```julia
co2 = species_index(gas, "CO2")
X_co2 = mole_fractions(state)[co2]
```

### Change the state

Use `set_TPX!` for mole-based input and `set_TPY!` for mass-based input:

```julia
set_TPX!(state, 1200.0, 2one_atm, Dict("H2" => 2, "O2" => 1, "N2" => 3.76))
set_TPY!(state, 1200.0, 2one_atm, Dict("H2" => 0.05, "O2" => 0.22, "N2" => 0.73))
```

`set_TP!` changes temperature and pressure while retaining composition.

### Chemistry and equilibrium

Evaluate molar production rates at the current state:

```julia
wdot = net_production_rates(state)
```

Equilibrate in place at fixed temperature and pressure:

```julia
set_TPX!(state, 2500.0, one_atm, "CH4:1,O2:2,N2:7.52")
result = equilibrate!(state, :TP)
```

For an adiabatic, fixed-pressure equilibrium calculation, use `:HP`:

```julia
set_TPX!(state, 300.0, one_atm, "CH4:1,O2:2,N2:7.52")
result = equilibrate!(state, :HP)
adiabatic_temperature = temperature(state)
```

See [Ideal-gas Equilibrium](@ref) for parameter sweeps, workspaces, validated
ranges, and limitations.

## Mechanism information

`Solution` stores the parsed mechanism and can be inspected directly:

```julia
gas.n_species
gas.n_reactions
gas.species_names
gas.MW
```

Arrhenius parses tested ideal-gas phases from Cantera-style YAML, including
NASA thermo, reaction-rate data, and mixture-averaged transport input. A file
with multiple phases can be selected using the `phase` keyword.

## Low-level array API

The array functions are useful when implementing custom kernels or coupling
Arrhenius to an existing solver. New users generally do not need them.

Given normalized mass fractions `Y`, temperature `T`, and pressure `P`:

```julia
using LinearAlgebra

mean_MW = inv(dot(Y, inv.(gas.MW)))
rho = P / (R * T) * mean_MW
X = Y2X(gas, Y, mean_MW)
C = Y2C(gas, Y, rho)
```

Low-level thermo and production-rate functions require the caller to keep all
arrays, units, and cached properties consistent. Prefer `GasState` unless that
control is specifically required. Reactor users should normally use
`isobaricReactor(gas)` rather than reconstructing its RHS manually.

Only single-phase ideal-gas TP and HP equilibrium are supported. Multiphase,
surface, fixed-volume, UV, and SV equilibrium are not implemented.
