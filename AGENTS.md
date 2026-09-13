# Arrhenius.jl agent guide

## What this repository does

Arrhenius.jl is a Julia ideal-gas chemical-kinetics package. The active public
surface covers Cantera-style YAML parsing, NASA thermodynamics, CPU reaction
rates, gas-state properties, TP/HP equilibrium, isobaric reactor RHS functions,
analytical Jacobians, and batched RHS evaluation. GPU support is experimental.

## Start here

- `src/Arrhenius.jl`: active module includes; files not included here are not
  automatically part of the package.
- `src/Solution.jl` and `src/parser.jl`: mechanism construction and YAML input.
- `src/State.jl`: `GasState`, thermodynamic properties, and TP/HP equilibrium.
- `src/ODE_functions.jl`: isobaric reactor RHS.
- `src/Jacobian.jl`: analytical and generated Jacobian workspaces.
- `src/BatchedReactor.jl`: CPU and extension-facing batched RHS API.
- `test/runtests.jl`: authoritative support and regression matrix.
- `test/equilibrium/`: reproducible Cantera 3.2 equilibrium references.
- `docs/src/mechanism_support.md`: concise advertised feature boundary.

## Validated equilibrium scope

Only single-phase ideal-gas `:TP` and `:HP` modes are supported. The stored
Cantera matrix covers H2-air (`h2_li_2004.yaml`) and CH4-air (`gri30.yaml`) at
phi 0.5, 0.75, 1.0, 1.25, 1.5, and 2.0 and at 0.1, 1, 10, and 100 bar. TP
references use 2500 K; HP references start at 1000 K. UV, SV, fixed-volume,
surface, and multiphase equilibrium are not implemented.

## Common commands

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=docs docs/make.jl
/path/to/python-with-cantera test/equilibrium/generate_cantera_references.py
```

The reference generator overwrites the TOML file deterministically and needs
Cantera 3.2. Review reference diffs before accepting them.

## Change discipline

- Preserve unrelated changes: this repository is often worked on with a dirty
  tree.
- Add parser/RHS/Jacobian/integration coverage before advertising a new
  mechanism feature.
- Equilibrium changes must check convergence, element conservation, full
  composition, and HP enthalpy conservation against stored references.
- A solver run is supported only when its return code and final state are both
  validated; a fast failed run is not a benchmark result.
- Do not claim general Cantera compatibility, multiphase support, or full GPU
  ODE integration.
