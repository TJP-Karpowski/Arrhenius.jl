function isobaric!(du, u, p, t, gas)
    T = u[end]
    pressure =p[1]
    Y = @view(u[1:gas.n_species])
    mean_MW = 1.0 / dot(Y, 1 ./ gas.MW)
    ρ_mass = pressure / R / T * mean_MW
    X = Y2X(gas, Y, mean_MW)
    C = Y2C(gas, Y, ρ_mass)
    cp_mass =  Arrhenius.cal_cpmass_mean(gas,T,pressure,X)
    h_RT = Arrhenius.cal_h_RT(gas, T, pressure, X)
    S0_R = Arrhenius.cal_s0_R(gas, T, pressure, X)
    #wY = du[1:end-1]
    #@timeit tmr "new" 
    calc_omegaY!( @view(du[1:gas.n_species]),gas, T, C, S0_R, h_RT,ρ_mass) 
    du[1:gas.n_species] =du[1:gas.n_species] .* gas.MW ./ ρ_mass
    #@timeit tmr "old" wdot_new = wdot_func(gas.kinetics, T, C, S0_R * R, h_RT * R * T)
    #du[1:end-1] = wdot_new / ρ_mass .* gas.MW
    du[end] = -dot(h_RT * R * T ./ gas.MW, du[1:gas.n_species]) / cp_mass
    return nothing
end

function isochoric!(du, u, p, t, gas)
    T = u[end-1]
    pressure = u[end]
    Y = @view(u[1:gas.n_species])
    mean_MW = 1.0 / dot(Y, 1 ./ gas.MW)
    ρ_mass = pressure / R / T * mean_MW
    X = Y2X(gas, Y, mean_MW)
    C = Y2C(gas, Y, ρ_mass)
    cv_mass =  Arrhenius.cal_cvmass_mean(gas,T,pressure,X)
    h_RT = Arrhenius.cal_h_RT(gas, T, pressure, X)
    S0_R = Arrhenius.cal_s0_R(gas, T, pressure, X)
    #wY = du[1:end-1]
    #@timeit tmr "new" 
    calc_omegaY!( @view(du[1:gas.n_species]),gas, T, C, S0_R, h_RT,ρ_mass) 
    du[1:gas.n_species] =du[1:gas.n_species] .* gas.MW ./ ρ_mass
    du[end-1] = -dot(h_RT * R * T ./ gas.MW, du[1:gas.n_species]) / cv_mass # dU = cv dT = Q_dot => Q_dot/cv = dT
    du[end] = pressure * du[end-1] / T # following p2/p1 =T2/T1 from isochoric heat addition
    return nothing
end

export isobaric!, isochoric!