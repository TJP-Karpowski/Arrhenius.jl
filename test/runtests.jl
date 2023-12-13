using Arrhenius
using ForwardDiff
using LinearAlgebra
using Test
using OrdinaryDiffEq
using ForwardDiff
#using DiffEqSensitivity
using Sundials
using Plots
using DelimitedFiles
using DataFrames
using CSV
using LaTeXStrings

plotting =true
#@testset  verbose=true "ignition Testcase"  begin
#    include("ignition/ignition.jl")
#    @test ignition_test(plotting=plotting)
#end
@testset  verbose=true "transport Testcase"  begin
    include("transport/transport.jl")
    @test transport_test(plotting=plotting)
end