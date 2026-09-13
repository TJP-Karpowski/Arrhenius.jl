# Examples

The repository includes runnable examples under `examples/`. They are intended
to show the supported public workflow, not internal benchmark harnesses.

Run them from the repository root:

```sh
julia --project=. examples/thermo_equilibrium.jl
julia --project=. examples/adiabatic_flame_temperature.jl
julia --project=. examples/batched_reactors.jl
```

## Thermodynamic State and Equilibrium

`examples/thermo_equilibrium.jl` demonstrates the `GasState` interface:

- create a gas from a Cantera-style YAML mechanism
- set temperature, pressure, and composition with `set_TPX!`
- query density, molecular weight, heat capacity, enthalpy, and entropy
- compute net production rates directly from the state
- solve ideal-gas `:TP` and `:HP` equilibrium

The core pattern is:

```julia
gas = CreateSolution("mechanism/h2o2.yaml")
state = GasState(gas)

set_TPX!(state, 1000.0, one_atm, "H2:2,O2:1,AR:3.76")

rho = density(state)
cp = cp_mass(state)
wdot = net_production_rates(state)

tp_result = equilibrate!(state, :TP)
```

## Adiabatic Flame-temperature Sweep

`examples/adiabatic_flame_temperature.jl` computes HP equilibrium for
methane-air from `phi = 0.5` to `2.0` at 1, 10, and 30 bar. Reactants enter at
300 K. The example prints a table and returns:

```julia
(; phis, pressures_bar, flame_temperatures)
```

Rows of `flame_temperatures` correspond to equivalence ratios and columns to
pressures. The example reuses one `GasState` and one `EquilibriumWorkspace`
because all calculations use the same mechanism. A workspace is internal
scratch storage for reducing allocations in repeated sequential solves; it is
not batching and does not hold a second thermodynamic state.

## Batched Reactor RHS

`examples/batched_reactors.jl` demonstrates evaluating many independent
constant-pressure reactor RHS states with a structure-of-arrays layout. Species
fields use shape `(n_species, n_reactors)` and scalar reactor fields use shape
`(n_reactors,)`.

```julia
gas = CreateSolution("mechanism/h2o2.yaml")
n = 4

batch = BatchedIdealGasReactor(gas, n)
dYdt = similar(Y)
dTdt = similar(T)

batched_isobaric_rhs!(dYdt, dTdt, batch, Y, T, p)
```

When `KernelAbstractions` and `Metal` are available, the same example also runs
the experimental packed Metal path for h2o2-style mechanisms:

```julia
packed = NoPdepPackedBatchedMechanism(gas, n; array = Metal.MtlArray, real_type = Float32)
batched_isobaric_rhs!(dYdt_d, dTdt_d, packed, Y_d, T_d, p_d, Metal.MetalBackend())
```

This GPU path evaluates RHS values only. It is not yet a full GPU ODE
integration workflow.
