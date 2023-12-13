
function transport_test(;plotting=false)
    gas = CreateSolution("../mechanism/h2_li_2004.yaml")
    df = CSV.read("transport/1D_flame_scalars.csv",DataFrame)
    Temps = df[!,"T"]|>Array
    p = df[!,"p"]|>Array
    Sp = df[!,gas.species_names]|> Array |> transpose |>Array
    DArr = zeros(size(Sp))
    lambda =zeros(size(Sp,2))
    mu = zeros(size(Sp,2))
    Xsp = Y2X.(Ref(gas),eachcol(Sp))
    meanMW = map(x->dot(x,gas.MW),Xsp)
    for i in 1:size(DArr)[2]
        mu[i], lambda[i],D = Arrhenius.mix_trans(gas, p[i], Temps[i], Xsp[i], meanMW[i])
        DArr[:,i] = @view D[:,1]
    end
    DArr = DArr' |> Array 
    Dcantera =   df[!,"Dmix_" .* gas.species_names]|> Array
    #plotting
    if plotting
        colors = palette(:tab10)
        ls = plot(xlabel=L"x\,[\mathrm{m}]",ylabel=L"D_i\,[\mathrm{m/s^2}]",yscale=:log10)
        for i in 1:gas.n_species
            scatter!(df.x*100.0,Dcantera[:,i],label=gas.species_names[i], color=colors[i],markersize=2,markeralpha=0.4,markerstrokealpha = 0.1,markerstrokewidth = 0.5)
            plot!(df.x*100.0,DArr[:,i],label="",color=colors[i])
        end
        scatter!(df.x*100.0,df.lambda,label=L"\lambda", color=colors[10],markersize=2,markeralpha=0.4,markerstrokealpha = 0.1,markerstrokewidth = 0.5)
        plot!(df.x*100.0,lambda,label="",color=colors[10])
        scatter!(df.x*100.0,df[!,"trans:mu"],label=L"\mu", color="black",markersize=2,markeralpha=0.4,markerstrokealpha = 0.1,markerstrokewidth = 0.5)
        plot!(df.x*100.0,mu,label="",color="black")
        savefig(ls,"outputs/transport.pdf")

        colors = palette(:tab10)
        ls = plot(xlabel=L"x\,[\mathrm{m}]",ylabel=L"D_{Arr}-D_{cantera}\,[\mathrm{m/s^2}]")#,yscale=:log10)
        for i in 1:gas.n_species
            plot!(df.x*100.0,DArr[:,i]-Dcantera[:,i],label=gas.species_names[i],color=colors[i])
        end
        plot!(df.x*100.0,lambda-df.lambda,color=colors[10],label=L"\lambda")
        plot!(df.x*100.0,mu-df[!,"trans:mu"],color="black",label=L"\mu")
        savefig(ls,"outputs/transport-error.pdf")
    end
    return isapprox(DArr,Dcantera,atol=0,rtol=1e-3) & 
            isapprox(lambda,df.lambda,atol=0,rtol=1e-3) & 
            isapprox(mu,df[!,"trans:mu"],atol=0,rtol=1e-3)
end
transport_test(plotting=true)