#!/usr/bin/env julia

using Pkg
Pkg.instantiate()

using ADTypes
using Arrhenius
using Dates
using LinearAlgebra
using OrdinaryDiffEq
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqSDIRK
using Printf
using Statistics
using Sundials

const RELTOL = 1.0e-6
const ABSTOL = 1.0e-9
const REFERENCE_RELTOL = 1.0e-9
const REFERENCE_ABSTOL = 1.0e-12
const FINAL_T_ATOL = 1.0e-3
const FINAL_Y_ATOL = 1.0e-6
const MAJOR_SPECIES_THRESHOLD = 1.0e-6

const CASES = [
    (
        name = "h2o2_hydrogen_air",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml"),
        temperature = 1000.0,
        pressure = Arrhenius.one_atm,
        t_end = 1.0e-3,
        composition = ("H2" => 2.0, "O2" => 1.0),
    ),
    (
        name = "h2_li_2004_hydrogen_air",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "h2_li_2004.yaml"),
        temperature = 1000.0,
        pressure = Arrhenius.one_atm,
        t_end = 1.0e-3,
        composition = ("H2" => 2.0, "O2" => 1.0, "N2" => 3.76),
        solve_kwargs = (dtmax = 1.0e-6,),
    ),
    (
        name = "gri30_methane_air",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "gri30.yaml"),
        temperature = 1500.0,
        pressure = 10.0 * Arrhenius.one_atm,
        t_end = 2.0e-3,
        composition = ("CH4" => 1.0, "O2" => 2.0, "N2" => 7.52),
    ),
]

function invalid_reactor_state(u, p, t)
    return u[end] <= 0 || any(x -> x < -1.0e-14, @view(u[1:end-1]))
end

const ROSENBROCK_JACOBIAN_VARIANTS = [
    (
        label = "Rosenbrock23(autodiff=AutoFiniteDiff())",
        jacobian = :none,
        algorithm = () -> Rosenbrock23(autodiff=AutoFiniteDiff()),
        solve_kwargs = (isoutofdomain = invalid_reactor_state,),
    ),
    (
        label = "Rosenbrock23(autodiff=AutoFiniteDiff(), jac=ForwardDiff)",
        jacobian = :forwarddiff,
        algorithm = () -> Rosenbrock23(autodiff=AutoFiniteDiff()),
        solve_kwargs = (isoutofdomain = invalid_reactor_state,),
    ),
    (
        label = "Rosenbrock23(autodiff=AutoFiniteDiff(), jac=Analytical)",
        jacobian = :analytical,
        algorithm = () -> Rosenbrock23(autodiff=AutoFiniteDiff()),
        solve_kwargs = (isoutofdomain = invalid_reactor_state,),
    ),
    (
        label = "Rosenbrock23(autodiff=AutoFiniteDiff(), jac=Generated)",
        jacobian = :generated,
        algorithm = () -> Rosenbrock23(autodiff=AutoFiniteDiff()),
        solve_kwargs = (isoutofdomain = invalid_reactor_state,),
    ),
    (
        label = "Rosenbrock23(autodiff=AutoFiniteDiff(), jac=AnalyticalSparse)",
        jacobian = :analytical_sparse,
        algorithm = () -> Rosenbrock23(autodiff=AutoFiniteDiff()),
        solve_kwargs = (isoutofdomain = invalid_reactor_state,),
    ),
]

const GENERATED_SOLVER_MATRIX = [
    (
        label = "Rodas5P(autodiff=AutoFiniteDiff(), jac=Generated)",
        jacobian = :generated,
        algorithm = () -> Rodas5P(autodiff=AutoFiniteDiff()),
        solve_kwargs = (isoutofdomain = invalid_reactor_state,),
    ),
    (
        label = "Rodas4P(autodiff=AutoFiniteDiff(), jac=Generated)",
        jacobian = :generated,
        algorithm = () -> Rodas4P(autodiff=AutoFiniteDiff()),
        solve_kwargs = (isoutofdomain = invalid_reactor_state,),
    ),
    (
        label = "KenCarp4(autodiff=AutoFiniteDiff(), jac=Generated)",
        jacobian = :generated,
        algorithm = () -> KenCarp4(autodiff=AutoFiniteDiff()),
        solve_kwargs = (isoutofdomain = invalid_reactor_state,),
    ),
    (
        label = "TRBDF2(autodiff=AutoFiniteDiff(), jac=Generated)",
        jacobian = :generated,
        algorithm = () -> TRBDF2(autodiff=AutoFiniteDiff()),
        solve_kwargs = (isoutofdomain = invalid_reactor_state,),
    ),
    (
        label = "CVODE_BDF(jac=Generated)",
        jacobian = :generated,
        algorithm = () -> CVODE_BDF(),
        solve_kwargs = (alg_hints = [:stiff],),
    ),
    (
        label = "CVODE_BDF(jac=Generated,rhs=Generated)",
        jacobian = :generated_rhs,
        algorithm = () -> CVODE_BDF(),
        solve_kwargs = (alg_hints = [:stiff],),
    ),
]

function mass_fractions(gas, composition)
    X = zeros(gas.n_species)
    for (species, amount) in composition
        idx = species_index(gas, species)
        idx === nothing && error("Species '$species' is not present in mechanism")
        X[idx] = amount
    end
    X ./= sum(X)
    return X2Y(gas, X, dot(X, gas.MW))
end

function sample!(f; samples::Int)
    times_ns = Vector{Int64}(undef, samples)
    allocations = Vector{Int64}(undef, samples)
    for i in 1:samples
        GC.gc(false)
        allocations[i] = @allocated f()
        t0 = time_ns()
        f()
        times_ns[i] = time_ns() - t0
    end
    return (
        min_ns = minimum(times_ns),
        median_ns = median(times_ns),
        mean_ns = mean(times_ns),
        max_ns = maximum(times_ns),
        alloc_bytes = median(allocations),
    )
end

function stat_value(stats, name::Symbol)
    return name in propertynames(stats) ? getproperty(stats, name) : 0
end

function solver_stats(sol)
    stats = sol.stats
    return (
        retcode = string(sol.retcode),
        nf = stat_value(stats, :nf),
        njacs = stat_value(stats, :njacs),
        nsolve = stat_value(stats, :nsolve),
        nw = stat_value(stats, :nw),
        naccept = stat_value(stats, :naccept),
        nreject = stat_value(stats, :nreject),
    )
end

function failed_stats(err)
    return (
        retcode = "Error: $(nameof(typeof(err)))",
        validation = "failed",
        final_T_abs_err = -1.0,
        final_Y_major_linf_err = -1.0,
        nf = 0,
        njacs = 0,
        nsolve = 0,
        nw = 0,
        naccept = 0,
        nreject = 0,
        min_ns = -1,
        median_ns = -1,
        mean_ns = -1,
        max_ns = -1,
        alloc_bytes = -1,
    )
end

function is_successful(sol)
    return string(sol.retcode) == "Success"
end

function failed_timing_stats()
    return (
        min_ns = -1,
        median_ns = -1,
        mean_ns = -1,
        max_ns = -1,
        alloc_bytes = -1,
    )
end

function compact_error(err)
    msg = replace(sprint(showerror, err), '\n' => ' ')
    length(msg) > 220 && (msg = first(msg, 220) * "...")
    return "$(nameof(typeof(err))): $msg"
end

function case_solve_kwargs(case)
    return :solve_kwargs in propertynames(case) ? case.solve_kwargs : NamedTuple()
end

function build_problem(case; jacobian = :none)
    gas = CreateSolution(case.mechanism)
    state = isobaricReactor(gas)
    Y0 = mass_fractions(gas, case.composition)
    u0 = vcat(Y0, case.temperature)
    tspan = (0.0, case.t_end)
    if jacobian == :forwarddiff
        jac = IsobaricJacobianWorkspace(gas)
        f = ODEFunction(state; jac = jac)
        return gas, ODEProblem(f, u0, tspan, (case.pressure,))
    elseif jacobian == :analytical
        jac = AnalyticalIsobaricJacobianWorkspace(gas)
        f = ODEFunction(state; jac = jac)
        return gas, ODEProblem(f, u0, tspan, (case.pressure,))
    elseif jacobian == :generated
        jac = GeneratedAnalyticalIsobaricJacobianWorkspace(gas)
        f = ODEFunction(state; jac = jac)
        return gas, ODEProblem(f, u0, tspan, (case.pressure,))
    elseif jacobian == :generated_rhs
        state = GeneratedIsobaricReactor(gas)
        jac = GeneratedAnalyticalIsobaricJacobianWorkspace(gas)
        f = ODEFunction(state; jac = jac)
        return gas, ODEProblem(f, u0, tspan, (case.pressure,))
    elseif jacobian == :analytical_sparse
        jac = AnalyticalIsobaricJacobianWorkspace(gas)
        prototype = isobaric_jacobian_prototype(gas)
        f = ODEFunction(state; jac = jac, jac_prototype = prototype)
        return gas, ODEProblem(f, u0, tspan, (case.pressure,))
    elseif jacobian != :none
        error("Unknown Jacobian variant: $jacobian")
    end
    return gas, ODEProblem(state, u0, tspan, (case.pressure,))
end

function build_reference(case)
    gas, prob = build_problem(case; jacobian = :none)
    algorithm, solve_kwargs = case.name == "h2_li_2004_hydrogen_air" ?
        (Rosenbrock23(autodiff=AutoFiniteDiff()), NamedTuple()) :
        (CVODE_BDF(), (alg_hints = [:stiff],))
    sol = solve(prob, algorithm;
        reltol = REFERENCE_RELTOL,
        abstol = REFERENCE_ABSTOL,
        save_everystep = false,
        solve_kwargs...,
    )
    !is_successful(sol) && error("Reference solve failed for $(case.name): $(sol.retcode)")
    major_species = [
        i for i in 1:gas.n_species
        if abs(prob.u0[i]) >= MAJOR_SPECIES_THRESHOLD || abs(sol.u[end][i]) >= MAJOR_SPECIES_THRESHOLD
    ]
    return (u = sol.u[end], major_species = major_species)
end

function final_state_validation(sol, reference)
    if !is_successful(sol)
        return (
            validation = "failed",
            final_T_abs_err = -1.0,
            final_Y_major_linf_err = -1.0,
        )
    end

    candidate_u = sol.u[end]
    reference_u = reference.u
    final_T_abs_err = abs(candidate_u[end] - reference_u[end])
    final_Y_major_linf_err = maximum(abs(candidate_u[i] - reference_u[i]) for i in reference.major_species)
    validation = final_T_abs_err <= FINAL_T_ATOL && final_Y_major_linf_err <= FINAL_Y_ATOL ? "pass" : "fail"
    return (
        validation = validation,
        final_T_abs_err = final_T_abs_err,
        final_Y_major_linf_err = final_Y_major_linf_err,
    )
end

function benchmark_problem(gas, prob, case, label, algorithm, reference; samples, solve_kwargs = NamedTuple())

    # One warmup solve compiles the solver path and checks that the case is valid.
    warmup_sol = try
        solve(prob, algorithm();
            reltol = RELTOL,
            abstol = ABSTOL,
            save_everystep = false,
            solve_kwargs...,
        )
    catch err
        @warn "Solver failed during warmup" case = case.name solver = label error = compact_error(err)
        return (
            mechanism = case.name,
            species = gas.n_species,
            reactions = gas.n_reactions,
            solver = label,
            t_end = case.t_end,
            reltol = RELTOL,
            abstol = ABSTOL,
            failed_stats(err)...,
        )
    end
    stats = solver_stats(warmup_sol)
    validation = final_state_validation(warmup_sol, reference)
    if !is_successful(warmup_sol)
        @warn "Solver returned non-success retcode" case = case.name solver = label retcode = warmup_sol.retcode
        return (
            mechanism = case.name,
            species = gas.n_species,
            reactions = gas.n_reactions,
            solver = label,
            t_end = case.t_end,
            reltol = RELTOL,
            abstol = ABSTOL,
            stats...,
            validation...,
            failed_timing_stats()...,
        )
    elseif validation.validation != "pass"
        @warn "Solver failed final-state validation" case = case.name solver = label final_T_abs_err = validation.final_T_abs_err final_Y_major_linf_err = validation.final_Y_major_linf_err
        return (
            mechanism = case.name,
            species = gas.n_species,
            reactions = gas.n_reactions,
            solver = label,
            t_end = case.t_end,
            reltol = RELTOL,
            abstol = ABSTOL,
            stats...,
            validation...,
            failed_timing_stats()...,
        )
    end

    timing_stats = try
        sample!(samples=samples) do
            sol = solve(prob, algorithm();
                reltol = RELTOL,
                abstol = ABSTOL,
                save_everystep = false,
                solve_kwargs...,
            )
            !is_successful(sol) && error("timed solve returned $(sol.retcode)")
            return sol
        end
    catch err
        @warn "Solver failed during timing" case = case.name solver = label error = compact_error(err)
        failed_timing_stats()
    end

    return (
        mechanism = case.name,
        species = gas.n_species,
        reactions = gas.n_reactions,
        solver = label,
        t_end = case.t_end,
        reltol = RELTOL,
        abstol = ABSTOL,
        stats...,
        validation...,
        timing_stats...,
    )
end

function benchmark_case(case; samples)
    rows = NamedTuple[]
    reference = build_reference(case)
    base_solve_kwargs = case_solve_kwargs(case)

    for spec in ROSENBROCK_JACOBIAN_VARIANTS
        gas, prob = build_problem(case; jacobian = spec.jacobian)
        push!(rows, benchmark_problem(
            gas,
            prob,
            case,
            spec.label,
            spec.algorithm,
            reference;
            samples = samples,
            solve_kwargs = merge(base_solve_kwargs, spec.solve_kwargs),
        ))
    end

    for spec in GENERATED_SOLVER_MATRIX
        gas, prob = build_problem(case; jacobian = spec.jacobian)
        push!(rows, benchmark_problem(
            gas,
            prob,
            case,
            spec.label,
            spec.algorithm,
            reference;
            samples = samples,
            solve_kwargs = merge(base_solve_kwargs, spec.solve_kwargs),
        ))
    end

    return rows
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "timestamp,julia_version,mechanism,species,reactions,solver,t_end,reltol,abstol,retcode,validation,final_T_abs_err,final_Y_major_linf_err,nf,njacs,nsolve,nw,naccept,nreject,min_ns,median_ns,mean_ns,max_ns,alloc_bytes")
        timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")
        for row in rows
            println(io, join((
                timestamp,
                VERSION,
                row.mechanism,
                row.species,
                row.reactions,
                row.solver,
                row.t_end,
                row.reltol,
                row.abstol,
                row.retcode,
                row.validation,
                row.final_T_abs_err,
                row.final_Y_major_linf_err,
                row.nf,
                row.njacs,
                row.nsolve,
                row.nw,
                row.naccept,
                row.nreject,
                round(Int, row.min_ns),
                round(Int, row.median_ns),
                round(Int, row.mean_ns),
                round(Int, row.max_ns),
                round(Int, row.alloc_bytes),
            ), ","))
        end
    end
end

function write_markdown(path, rows; samples)
    open(path, "w") do io
        println(io, "# Arrhenius Ignition Baseline")
        println(io)
        println(io, "- Timestamp: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
        println(io, "- Julia: $(VERSION)")
        println(io, "- Samples: $samples")
        println(io, "- Solver variants: Rosenbrock23 Jacobian comparison plus generated-Jacobian stiff solver matrix")
        println(io, "- Tolerances: `reltol=$RELTOL`, `abstol=$ABSTOL`")
        println(io, "- Reference tolerances: `reltol=$REFERENCE_RELTOL`, `abstol=$REFERENCE_ABSTOL`")
        println(io, "- Validation: warmup `retcode == Success`, final `T` absolute error <= $FINAL_T_ATOL, and final major-species mass-fraction Linf error <= $FINAL_Y_ATOL")
        println(io, "- Major species threshold: initial or reference final mass fraction >= $MAJOR_SPECIES_THRESHOLD")
        println(io, "- Case step caps: `h2_li_2004_hydrogen_air dtmax=1.0e-6`")
        println(io, "- Save policy: `save_everystep=false`")
        println(io)
        println(io, "| Case | Solver | Species | Reactions | t_end | Median solve | Median alloc |")
        println(io, "|---|---|---:|---:|---:|---:|---:|")
        for row in rows
            time_text = row.median_ns < 0 ? "failed" : @sprintf("%.3f ms", row.median_ns / 1_000_000)
            alloc_text = row.alloc_bytes < 0 ? "failed" : string(round(Int, row.alloc_bytes), " B")
            @printf(io, "| %s | `%s` | %d | %d | %.3e | %s | %s |\n",
                row.mechanism,
                row.solver,
                row.species,
                row.reactions,
                row.t_end,
                time_text,
                alloc_text,
            )
        end
        println(io)
        println(io, "| Case | Solver | validation | final T abs err | final major Y Linf err |")
        println(io, "|---|---|---|---:|---:|")
        for row in rows
            T_err_text = row.final_T_abs_err < 0 ? "failed" : @sprintf("%.3e", row.final_T_abs_err)
            Y_err_text = row.final_Y_major_linf_err < 0 ? "failed" : @sprintf("%.3e", row.final_Y_major_linf_err)
            @printf(io, "| %s | `%s` | `%s` | %s | %s |\n",
                row.mechanism,
                row.solver,
                row.validation,
                T_err_text,
                Y_err_text,
            )
        end
        println(io)
        println(io, "| Case | Solver | retcode | nf | njacs | nsolve | nw | naccept | nreject |")
        println(io, "|---|---|---|---:|---:|---:|---:|---:|---:|")
        for row in rows
            @printf(io, "| %s | `%s` | `%s` | %d | %d | %d | %d | %d | %d |\n",
                row.mechanism,
                row.solver,
                row.retcode,
                row.nf,
                row.njacs,
                row.nsolve,
                row.nw,
                row.naccept,
                row.nreject,
            )
        end
    end
end

function main()
    samples = parse(Int, get(ENV, "ARRHENIUS_IGNITION_SAMPLES", "5"))

    rows = NamedTuple[]
    for case in CASES
        println("Benchmarking $(case.name)")
        append!(rows, benchmark_case(case; samples=samples))
    end

    mkpath(joinpath(@__DIR__, "results"))
    stamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    csv_path = joinpath(@__DIR__, "results", "ignition_$stamp.csv")
    md_path = joinpath(@__DIR__, "results", "ignition_$stamp.md")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows; samples=samples)
    println("Wrote $csv_path")
    println("Wrote $md_path")
end

main()
