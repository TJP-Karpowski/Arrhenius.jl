# [Arrhenius.jl](https://github.com/DENG-MIT/Arrhenius.jl)
*Julia-native ideal-gas chemical kinetics and reactor kernels.*

![logo.png](https://user-images.githubusercontent.com/8445647/118408755-e0cd6f80-b654-11eb-8c36-80188e4703ce.png)

## Current capabilities

Arrhenius.jl currently provides:

- Cantera-style YAML parsing for tested ideal-gas mechanisms
- NASA-polynomial thermodynamics and a mutable `GasState` interface
- CPU reaction rates and homogeneous isobaric reactor RHS functions
- single-phase ideal-gas TP and HP equilibrium
- analytical and mechanism-specialized Jacobian workspaces
- batched CPU RHS evaluation and an experimental Metal extension

The equilibrium suite compares full H2-air and CH4-air compositions with
Cantera 3.2 over equivalence ratios 0.5--2.0 and pressures 0.1--100 bar. See
[Ideal-gas Equilibrium](@ref) and [Mechanism Support](@ref) for the precise
validated boundary. Arrhenius does not currently provide multiphase or flame
solvers and does not claim general Cantera API parity.

## Outline

The documentation is organized as:

1. How-to examples
2. Explanatory guide
3. Reference docstrings for the public interface are available in the API documentation section.

Developers should keep this page's section lists synchronized with
`docs/make.jl`.

## Basics

```@contents
Pages = ["install.md", "get_started.md", "equilibrium.md", "examples.md", "concepts.md"]
```

## Tutorials

```@contents
Pages = ["tutorial.md"]
```

## Reference

```@contents
Pages = ["open_source_roadmap.md", "solver_support.md", "mechanism_support.md", "jacobian_performance_plan.md", "faq.md", "api.md"]
```
