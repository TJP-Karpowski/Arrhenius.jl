#!/usr/bin/env julia

using Arrhenius
using KernelAbstractions
using LinearAlgebra
using Metal
using Printf

function uniform_mass_fractions(gas, ::Type{T}) where {T}
    Y = fill(T(1 / gas.n_species), gas.n_species)
    return Y ./ sum(Y)
end

function hydrogen_mass_fractions(gas, ::Type{T}, composition) where {T}
    X = zeros(T, gas.n_species)
    for (species, amount) in composition
        idx = species_index(gas, species)
        idx === nothing && error("Species '$species' is not present in $(gas.species_names)")
        X[idx] = T(amount)
    end
    X ./= sum(X)
    return T.(X2Y(gas, Float64.(X), dot(Float64.(X), gas.MW)))
end

function compare_cpu_metal(gas, Y, T, p; rtol = 1.0f-4, atol_Y = 1.0f-5, atol_T = 1.0f-4, packed = false)
    n_reactors = size(Y, 2)
    dY_cpu = similar(Y)
    dT_cpu = similar(T)
    cpu_batch = BatchedIdealGasReactor(gas, n_reactors; real_type = Float32)
    batched_isobaric_rhs!(dY_cpu, dT_cpu, cpu_batch, Y, T, p)

    Y_d = Metal.MtlArray(Y)
    T_d = Metal.MtlArray(T)
    p_d = Metal.MtlArray(p)
    dY_d = similar(Y_d)
    dT_d = similar(T_d)
    metal_batch = packed ?
                  NoPdepPackedBatchedMechanism(gas, n_reactors; array = Metal.MtlArray, real_type = Float32) :
                  BatchedIdealGasReactor(gas, n_reactors; array = Metal.MtlArray, real_type = Float32)
    batched_isobaric_rhs!(dY_d, dT_d, metal_batch, Y_d, T_d, p_d, Metal.MetalBackend())

    dY_metal = Array(dY_d)
    dT_metal = Array(dT_d)
    Y_err = maximum(abs.(dY_metal .- dY_cpu))
    T_err = maximum(abs.(dT_metal .- dT_cpu))
    Y_ok = isapprox(dY_metal, dY_cpu; rtol = rtol, atol = atol_Y)
    T_ok = isapprox(dT_metal, dT_cpu; rtol = rtol, atol = atol_T)
    return (; Y_ok, T_ok, Y_err, T_err)
end

function run_simple_case()
    gas = CreateSolution(joinpath(@__DIR__, "..", "mechanism", "2S_CH4_CM2.yaml"))
    n_reactors = 8
    Y = repeat(uniform_mass_fractions(gas, Float32), 1, n_reactors)
    T = Float32.(range(900, 1500; length = n_reactors))
    p = fill(Float32(one_atm), n_reactors)
    result = compare_cpu_metal(gas, Y, T, p)
    @printf(
        "simple 2S_CH4_CM2 Metal smoke: Y_ok=%s T_ok=%s max_Y_err=%.3e max_T_err=%.3e\n",
        result.Y_ok,
        result.T_ok,
        result.Y_err,
        result.T_err,
    )
    result.Y_ok && result.T_ok || error("Metal simple batched RHS disagrees with CPU reference")
end

function run_h2o2_probe()
    gas = CreateSolution(joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml"))
    n_reactors = 4
    compositions = [
        ("H2" => 2.0, "O2" => 1.0, "AR" => 3.76),
        ("H2" => 1.0, "O2" => 1.0, "AR" => 2.0),
        ("H2" => 0.5, "O2" => 1.0, "AR" => 1.0),
        ("H2" => 1.5, "O2" => 0.5, "AR" => 2.0),
    ]
    Y = zeros(Float32, gas.n_species, n_reactors)
    for reactor in 1:n_reactors
        Y[:, reactor] .= hydrogen_mass_fractions(gas, Float32, compositions[reactor])
    end
    T = Float32[950, 1100, 1300, 1000]
    p = Float32[one_atm, 2one_atm, 0.5one_atm, one_atm]
    result = compare_cpu_metal(gas, Y, T, p; packed = true, rtol = 5.0f-4, atol_Y = 1.0f-5, atol_T = 1.0f-3)
    @printf(
        "h2o2 packed Metal smoke: Y_ok=%s T_ok=%s max_Y_err=%.3e max_T_err=%.3e\n",
        result.Y_ok,
        result.T_ok,
        result.Y_err,
        result.T_err,
    )
    result.Y_ok && result.T_ok || error("Metal h2o2 packed batched RHS disagrees with CPU reference")
end

run_simple_case()
run_h2o2_probe()
