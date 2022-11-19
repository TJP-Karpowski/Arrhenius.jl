using Arrhenius
using ForwardDiff
using LinearAlgebra
using Test
using DifferentialEquations
using ForwardDiff
#using DiffEqSensitivity
using Sundials
using Plots
using DelimitedFiles

plotting =true
@testset  verbose=true "ignition Testcase"  begin
    include("ignition/ignition.jl")
    @test ignition_test(plotting=plotting)
end