using Arrhenius, Test, LinearAlgebra
@testset "NASA intervals, generic/reactor agreement, and autodiff" begin
    g=CreateSolution(joinpath(dirname(pathof(Arrhenius)),"..","mechanism","h2_li_2004.yaml"))
    x=fill(1/g.n_species,g.n_species)
    y=Arrhenius.X2Y(g,x,dot(x,g.MW))
    reactor=isobaricReactor(g)
    for t in [350.,999.,1000.,1001.,1400.,1709.,1710.,1711.,2000.,3000.]
        h=cal_h_RT(g,t,101325.,x);cp=cal_cp_R(g,t,101325.,x);s=cal_s0_R(g,t,101325.,x)
        for i in 1:g.n_species
            a=(t<=g.thermo.Trange[i,2] ? g.thermo.nasa_low : g.thermo.nasa_high)[i,:]
            @test h[i] ≈ a[1]+a[2]*t/2+a[3]*t^2/3+a[4]*t^3/4+a[5]*t^4/5+a[6]/t
            @test cp[i] ≈ a[1]+a[2]*t+a[3]*t^2+a[4]*t^3+a[5]*t^4
            @test s[i] ≈ a[1]*log(t)+a[2]*t+a[3]*t^2/2+a[4]*t^3/3+a[5]*t^4/4+a[7]
        end
        Arrhenius.updateReactor!(reactor,y,t,101325.)
        @test reactor.h_RT ≈ h
        @test reactor.cp_R ≈ cp
        @test reactor.s0_R ≈ s
        if !(t in [1000.,1710.])
            dh=Arrhenius.ForwardDiff.jacobian(v->cal_h_RT(g,v[1],101325.,x),[t])[:,1]
            ds=Arrhenius.ForwardDiff.jacobian(v->cal_s0_R(g,v[1],101325.,x),[t])[:,1]
            @test dh ≈ (cp-h)/t rtol=1e-10 atol=1e-12
            @test ds ≈ cp/t rtol=1e-10 atol=1e-12
        end
    end
    jac=GeneratedAnalyticalIsobaricJacobianWorkspace(g)
    for t in [900.,1400.,2000.]
        u=vcat(y,t);p=[101325.];J=zeros(10,10);jac(J,u,p,0.)
        fd=similar(J);plus=zeros(10);minus=zeros(10)
        for i in 1:10
            delta=i==10 ? 1e-3 : 1e-7
            a=copy(u);b=copy(u);a[i]+=delta;b[i]-=delta
            reactor(plus,a,p,0.);reactor(minus,b,p,0.)
            fd[:,i]=(plus-minus)/(2delta)
        end
        @test norm(J-fd)/norm(fd)<1e-6
    end
end
