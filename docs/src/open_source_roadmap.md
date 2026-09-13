# Open Source Publication Roadmap

This roadmap tracks the work needed to make Arrhenius.jl useful, credible, and maintainable as an open-source Julia chemistry package. The goal is not to claim feature parity with Cantera or pyJac immediately. The first public target should be a clear, well-tested Julia-native chemistry and reactor kernel for mechanism parsing, source-term evaluation, analytical Jacobians, and SciML-based ignition solves.

## Publication Goals

1. Provide a reliable Julia package that users can install, run, test, and cite.
2. Make the supported feature set explicit: ideal-gas mechanisms, thermodynamic state queries, homogeneous reactors, source terms, analytical/generated Jacobians, and benchmarked ignition workflows.
3. Integrate naturally with the Julia ecosystem through SciML, Documenter, Julia package registration, and reproducible benchmark artifacts.
4. Keep experimental work visible but opt-in, especially generated kernels, ADER experiments, GPU work, and large-mechanism solver strategy.

## Release Positioning

Arrhenius.jl should be presented as:

- A Julia-native combustion chemistry toolkit focused on differentiability, generated kernels, and SciML integration.
- A research and development platform for fast source-term/Jacobian evaluation and differentiable chemistry.
- A complement to Cantera, not a drop-in replacement.
- Not yet a general production chemistry suite for all mechanisms, transport models, reactor networks, or flame solvers.

The public README and docs should state this clearly. Users should know which paths are stable, which are experimental, and which comparisons are still in progress.

## Phase 0: Repository Hygiene

Target: before announcing wider testing.

- Add or update top-level `README.md` with:
  - installation instructions
  - minimal ignition example
  - supported mechanism features
  - current limitations
  - citation/publication references
  - benchmark caveats
- Add `CONTRIBUTING.md` with:
  - test commands
  - benchmark commands
  - code style notes
  - expectations for new mechanisms and solver support
- Add `CITATION.cff`.
- Add `NEWS.md` or `CHANGELOG.md`.
- Add issue templates for bug reports, mechanism parser failures, performance regressions, and feature requests.
- Audit license compatibility for bundled mechanisms and example assets.
- Decide whether benchmark result files should be committed, ignored, or curated under a small `benchmarks/results/reference` set.

Exit criteria:

- A new user can clone the repo, run tests, build docs, and run one benchmark from documented commands.
- The repository has clear contribution and citation metadata.

## Phase 1: Stable Public API

Target: first usable public release candidate.

- Define a small supported API surface:
  - `CreateSolution`
  - `isobaricReactor`
  - `ODEFunction` integration pattern
  - `wdot_func!`
  - `jacobian!`
  - `AnalyticalIsobaricJacobianWorkspace`
  - `GeneratedAnalyticalIsobaricJacobianWorkspace`
  - `isobaric_jacobian_prototype`
- Mark internal APIs as internal by convention or documentation.
- Add docstrings for all public API entry points.
- Add examples that avoid relying on unexported internals unless the page is explicitly developer-facing.
- Add a compatibility policy:
  - public API changes require a changelog entry
  - experimental APIs can change between minor versions
  - generated kernels remain opt-in until compile-time and code-size behavior are characterized

Exit criteria:

- Docs identify stable, experimental, and internal interfaces.
- Tests cover each stable entry point with at least one small mechanism.

## Phase 2: Correctness and Mechanism Coverage

Target: trust before speed.

- Maintain a mechanism support matrix in docs:
  - parsed successfully
  - RHS finite at representative state
  - ignition solve supported
  - analytical Jacobian supported
  - generated Jacobian supported
  - known unsupported features
- Keep final-state validation in ignition benchmarks:
  - require `retcode == Success`
  - compare final temperature against a reference
  - compare major species mass fractions against a reference
  - show failed rows rather than hiding them
- Add small integration tests for supported solver/case combinations.
- Add parser regression tests for:
  - fractional orders
  - duplicate reactions
  - third-body efficiencies
  - falloff reactions
  - Troe reactions
  - reversible reactions
  - unit handling
- Add reference checks against trusted tools where practical:
  - Cantera for equilibrium/source terms on selected mechanisms
  - SciML/CVODE reference solves for ignition cases

Exit criteria:

- Public docs say exactly what works.
- Failed or experimental mechanisms are not presented as supported.
- Regression tests protect current hydrogen and GRI30 behavior.

## Phase 2A: Solution State and Equilibrium Usability

Target: make Arrhenius useful before a user ever builds a reactor.

The current `Solution` object stores mechanism data, and `isobaricReactor` stores an integration workspace. That is enough for solver internals, but it is not enough for normal chemistry workflows. Users should be able to load a mechanism, set a thermodynamic state, ask for properties, and compute equilibrium without writing reactor code.

Add a user-facing state layer, tentatively:

- `ThermoState` or `GasState`
  - owns a `Solution`
  - stores `T`, `p`, `X`, `Y`, mean molecular weight, density, and cached thermodynamic work arrays
  - separates user state handling from reactor integration workspaces
- State setters:
  - `set_TPX!(state, T, p, X)`
  - `set_TPY!(state, T, p, Y)`
  - `set_TP!(state, T, p)`
  - composition setters by vector, dictionary, or composition string
- State getters:
  - `temperature(state)`
  - `pressure(state)`
  - `mole_fractions(state)`
  - `mass_fractions(state)`
  - `density(state)`
  - `mean_molecular_weight(state)`
- Thermodynamic queries:
  - species `cp`, `h`, `s`, `g`, `u`
  - mixture `cp`, `h`, `s`, `g`, `u`
  - molar and mass-specific variants
  - dimensionless variants where useful for kinetics/Jacobians
- Source-term queries:
  - `net_production_rates!(wdot, state)`
  - `creation_rates!`, `destruction_rates!`, and reaction rates later if needed

Equilibrium should be a release-track feature, not a distant add-on. Suggested stages:

1. `equilibrate!(state, :TP)` for ideal gases.
   - Minimize total Gibbs free energy at fixed `T`, `p`, and elemental abundances.
   - Use element conservation from `gas.ele_matrix`.
   - Enforce nonnegative species moles.
   - Start with a robust CPU implementation over a fast one.
2. Add tests against known equilibrium compositions.
   - Simple H2/O2/H2O systems.
   - CO/O2/CO2 systems.
   - GRI30 methane-air at selected temperatures.
   - Compare to Cantera for reference values where available.
   - Keep stored references reproducible with `test/equilibrium/generate_cantera_references.py`.
   - Stored Cantera 3.2 references cover TP and HP H2-air and CH4-air across
     `phi = 0.5:2.0` (six points) and pressures from 0.1 to 100 bar, with full
     species-composition comparisons.
   - Track direct default and workspace-backed equilibrium performance with `benchmarks/equilibrium.jl`.
3. Add `equilibrate!(state, :HP)` after `:TP` is reliable.
   - `:HP` can wrap `:TP` inside a scalar temperature solve.
   - Temperature bounds and phase validity must be explicit.
   - Initial `:HP` support is implemented for ideal gases and validated on small hydrogen mechanisms.
4. Add `:UV` only after `:TP` and `:HP` are reliable.
5. Expose non-mutating helpers:
   - `equilibrate(state, :TP)` returning a copy
   - `equilibrium_composition(gas; T, p, X, mode=:TP)`

Important design constraints:

- The state API should be ergonomic and stable before package registration.
- Reactor workspaces should be able to reuse `GasState` internals, but the user-facing state object should not require constructing an ODE problem.
- Equilibrium must report convergence status and residuals. Silent failure is worse than no equilibrium solver.
- The first equilibrium implementation should prioritize correctness and clear errors over speed.

Exit criteria:

- A user can load a mechanism, set `TPX`, query thermodynamic properties, and compute `TP`/`HP` equilibrium in a documented example.
- Equilibrium tests compare against at least one trusted reference for small mechanisms.
- Docs clearly state which equilibrium modes are supported.

## Phase 3: Performance Baselines

Target: make performance claims defensible.

- Split benchmarks into three layers:
  - source-term throughput
  - Jacobian throughput
  - full ignition integration
- Report benchmark environment:
  - Julia version
  - CPU
  - BLAS backend
  - solver versions
  - tolerances
  - mechanism sizes
- Keep a curated benchmark summary in docs rather than committing every local run.
- Benchmark against useful baselines:
  - generic Arrhenius RHS
  - analytical Jacobian
  - generated Jacobian
  - CVODE_BDF
  - Cantera, when the comparison can be made fairly
  - pyJac-style generated kernels only for pure source/Jacobian throughput or batched reactor workloads
- Avoid broad claims until comparisons are apples-to-apples:
  - same mechanism
  - same thermodynamic model
  - same tolerances
  - same final-state validation
  - warmup/compile time reported separately from solve time

Exit criteria:

- Benchmark markdown explains which rows are valid, failed, or experimental.
- Performance claims are scoped to the measured workload.

## Phase 4: Julia Ecosystem Integration

Target: make Arrhenius.jl easy to use from normal Julia workflows.

- Register the package in General once API and tests are stable enough.
- Add GitHub Actions:
  - tests on Linux/macOS
  - supported Julia versions
  - documentation build
  - formatting or linting if a formatter is adopted
- Deploy docs with Documenter.
- Add badges:
  - CI
  - docs
  - license
  - version/registry after registration
- Ensure SciML integration is documented:
  - constructing `ODEProblem`
  - choosing `ODEFunction` with or without Jacobian
  - using CVODE_BDF from Sundials
  - using native OrdinaryDiffEq solvers with domain guards
- Ensure state/equilibrium usage is documented independently of SciML:
  - loading a mechanism
  - setting `TPX`/`TPY`
  - querying properties
  - computing `TP` equilibrium
- Consider light integrations after core stability:
  - ChainRules or DifferentiationInterface examples
  - SciMLSensitivity examples
  - ModelingToolkit examples only if they add real value

Exit criteria:

- Users can `Pkg.add("Arrhenius")` after registration.
- Documentation builds and deploys automatically.
- CI catches parser, Jacobian, and ignition regressions.

## Phase 5: Robust Solver Strategy

Target: move from fast kernels to reliable integration.

- Keep CVODE_BDF as the large-mechanism baseline until native solver robustness improves.
- Continue native solver support only for validated combinations.
- Improve nonphysical trial-state handling:
  - domain guards for negative species and temperature
  - safe rate/log evaluation for rejected stages
  - clear failure reporting when a solver returns non-success
- Investigate solver strategies for larger mechanisms:
  - sparse linear algebra
  - preconditioning
  - solver-specific step caps or initial step policies
  - positivity-preserving formulations
  - transformed state variables if needed
- Keep solver policies mechanism-specific when needed. A single universal solver setting is unlikely to be optimal.

Exit criteria:

- Supported solver/case combinations are encoded in tests.
- Native solvers do not fail from avoidable chemistry-domain exceptions.
- Large-mechanism docs recommend the currently validated path rather than the fastest failed path.
- Cantera-format fixture coverage includes representative externally maintained mechanisms for parser, RHS, and Jacobian regressions.

## Phase 6: Differentiability and Research Workflows

Target: lean into what makes the package valuable in Julia.

- Provide examples for:
  - differentiating ignition delay or final state
  - parameter sensitivities for Arrhenius coefficients
  - mechanism tuning with SciML tools
  - generated analytical Jacobian usage
- Decide which AD paths are supported:
  - ForwardDiff through RHS
  - analytical Jacobians for stiff solves
  - generated kernels for performance
  - reverse-mode sensitivity examples when stable
- Keep Hessian or higher-derivative work experimental until there is a specific integration or optimization workload that benefits from it.

Exit criteria:

- Differentiability examples run in CI or a documented tutorial environment.
- Users can reproduce one meaningful parameter-sensitivity workflow.

## Phase 7: Larger Scope After First Release

Target: avoid overcommitting before the core is reliable.

Potential later tracks:

- Batched reactor execution for pyJac-like workloads.
- GPU-compatible generated kernels.
- More complete transport support.
- Reactor networks.
- Flame solver interfaces.
- Mechanism reduction workflows.
- ADER or higher-order local integration experiments.

These should remain separate milestones, not blockers for a first public release.

## Suggested First Public Release Scope

Version target: `v0.2.0` or similar pre-1.0 release.

Include:

- Ideal-gas mechanism parsing for the documented supported mechanisms.
- Thermodynamic state object for setting `TPX`/`TPY` and querying common properties.
- Validated single-phase ideal-gas `TP` and `HP` equilibrium solvers.
- Homogeneous isobaric reactor integration.
- Generic, analytical, and generated Jacobian workspaces.
- Hydrogen ignition examples.
- GRI30 ignition example using the validated solver path.
- Correctness-validated benchmark reports.
- Documented limitations.

Exclude or mark experimental:

- General flame simulations.
- GPU execution.
- `UV`, `SV`, fixed-volume, surface, and multiphase equilibrium until implemented and tested.
- Hessian-based integration.
- Claims of pyJac or Cantera parity.
- Unsupported mechanism features without regression tests.

## Immediate Checklist

1. Update README positioning and install/run examples.
2. Add contribution, citation, and changelog files.
3. Add CI for tests and docs.
4. Add a documented benchmark command and curated benchmark summary.
5. Document the current solver support matrix.
6. Maintain the user-facing thermodynamic state API and examples.
7. Maintain TP/HP equilibrium reference coverage and performance benchmarks.
8. Deploy the built documentation.
9. Open issues for each Phase 0, Phase 1, and Phase 2A item.
10. Cut a tagged pre-release once CI, docs, state API, and support matrix are green.
