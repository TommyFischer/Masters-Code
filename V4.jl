# 1/4/23 Spectral expansion is working, now just writing a clean version that can be cuda or normal using a single command + will tidy up

# Last Edit:  4:30pm 19th April

using PlotlyJS,
    Plots.PlotMeasures,
    QuantumFluidSpectra,
    #SparseArrays,
    #StaticArrays,
    LinearAlgebra,
    DifferentialEquations,
    FFTW,
    LaTeXStrings,
    Plots,
    #WAV,
    JLD2,
    #Makie, 
    #GLMakie,
    #CodecZlib,
    BenchmarkTools,
    #RecursiveArrayTools,
    CUDA

@fastmath hypot(a,b,c) = sqrt(a^2 + b^2 + c^2)
Threads.nthreads()

begin # Functions for setting up and running simulations

    function number(ψ)
        return sum(abs2.(ψ))*dr
    end

    function kfunc!(dψ,ψ) # Kinetic Energy term (no rotating frame)
        dψ .= 0.5*(ifft(k2.*fft(ψ)))
        return nothing
    end

    function GPE!(dψ,ψ,var,t) # GPE Equation 
        kfunc_opt!(dψ,ψ)
        @. dψ = -(im + γ)*(0.5*dψ + (V_0 + G*abs2(ψ) - 1)*ψ)
    end

    function NDVPE!(dψ,ψ,var,t) # GPE Equation 
        kfunc_opt!(dψ,ψ)
        @. dψ = -im*(0.5*dψ + (V_0 +  $V(t) + G*abs2(ψ))*ψ)
    end

    function VPE!(dψ,ψ,var,t) # GPE Equation 
        kfunc_opt!(dψ,ψ)
        @. dψ = -(im + γ)*(0.5*dψ + (V_0 +  $V(t) + G*abs2(ψ) - 1)*ψ)
    end

    function kfunc_opt!(dψ,ψ)
        mul!(dψ,Pf,ψ)
        dψ .*= k2
        Pi!*dψ
        return nothing
    end
end

begin # Functions for making plots

    Ek(ψ) = 0.5*ξ^3*ψ0^2 * dr*sum(abs2.(ifft(im*ksum.*fft(ψ)))) # E_Kinetic / μ
    Ep(ψ,V) = ξ^3*ψ0^2 * dr*sum((Array(V_0) .+ V).*abs2.(ψ)) |> real # E_Potential / μ
    Ei(ψ) = 0.5*ξ^3*ψ0^2 *G*dr*sum(abs2.(ψ).^2) # E_Interaction / μ

    Ekx(ψ) = 0.5*ξ^3*ψ0^2 * dr*sum(abs2.(ifft(im*Array(kx).*fft(ψ)))) # x-direction E_k
    Eky(ψ) = 0.5*ξ^3*ψ0^2 * dr*sum(abs2.(ifft(im*Array(ky).*fft(ψ)))) # y-direction E_k
    Ekz(ψ) = 0.5*ξ^3*ψ0^2 * dr*sum(abs2.(ifft(im*Array(kz).*fft(ψ)))) # z-direction E_k

    function gradsquared(ψ) # Gradient Squared of ψ
        ϕ = fft(ψ)
        out = complex(zeros(M,M,Mz))
    
        for j in (kx,ky',reshape(kz,(1,1,Mz)))
            out .+= abs2.(ifft(im*j.*ϕ))/2
        end
        return real.(out)
    end;     

    function E_Kin(sol) # Kinetic energy over time
        n = length(sol[1,1,1,:])
        E_kin = ones(n)
    
        for i in 1:n
            E_kin[i] = sum(gradsquared(sol[:,:,:,i]))
        end
        return E_kin
    end

    function E_Pot(sol) # Potential Energy over time, V needs to be the same
        n = length(sol[1,1,1,:])
        E_Pot = ones(n)
    
        for i in 1:n
            E_Pot[i] = sum(V_0.*abs2.(sol[:,:,:,i]))
        end
        return E_Pot
    end

    function E_Int(sol) # Interaction Energy over time
        n = length(sol[1,1,1,:])
        E_Int = ones(n)
    
        for i in 1:n
            E_Int[i] = G*sum((abs2.(sol[:,:,:,i])).^2)
        end
        return E_Int
    end

    function Volume1(res,t,isomax=false,opacity=0.5,surface_count=10) # 3D volume plot of solution at t
        vals = res[:,:,:,t];
        X, Y, Z = mgrid(x,y,z);
    
        p = PlotlyJS.volume(
            x=X[:],
            y=Y[:],
            z=Z[:],
            value=vals[:],
            opacity=opacity,
            isomin=0,
            isomax=isomax,
            surface_count=surface_count,
            caps=attr(x_show=false, y_show=false,z_show=false),
            colorscale=:YlOrRd,
            reversescale = true
            )
    
        data = [p];
    
        layout = Layout(title = "t = (sol.t[t])",
                width = 500,
                height = 500, 
                scene_camera=attr(eye=attr(x=0, y=2, z=1)))
    
        pl = PlotlyJS.plot(data,layout)
    
    end

    function MakieVolume(sol,t,alpha=0.12,iso=[0.15],axis=true,col=:oxy)
        if typeof(sol[1]) in (Array{ComplexF64, 3},Array{ComplexF32, 3}) # Checking if input solution is already squared or not
            density_Scaled = abs2.(sol[:,:,:,t])
        else
            density_Scaled = copy(sol[:,:,:,t])
        end
    
        for i in 1:length(sol[1,1,1,:]) # Scaling to 1 to make isovalues easier
            density_Scaled /= maximum(density_Scaled)
        end
    
        Makie.contour(density_Scaled, # Making the plot
        alpha=alpha,
        levels=iso,
        colormap=col,
        #show_axis=axis
        )
    end

    function MakieVolume!(sol,t,alpha=0.12,iso=[0.15],axis=true,col=:oxy)
        if typeof(sol[1]) == Array{ComplexF64, 3} # Checking if input solution is already squared or not
            density_Scaled = abs2.(sol[:,:,:,t])
        else
            density_Scaled = copy(sol[:,:,:,t])
        end
    
        for i in 1:length(sol[1,1,1,:]) # Scaling to 1 to make isovalues easier
            density_Scaled /= maximum(density_Scaled)
        end
    
        Makie.contour!(density_Scaled, # Making the plot
        alpha=alpha,
        levels=iso,
        colormap=col,
        show_axis=axis)
    end

    function MakieGif(sol,title,framerate, alph=0.12, iso=[0.15], axis=true, col=:oxy, )
        if typeof(sol[1]) == Array{ComplexF64, 3} # Checking if input solution is already squared or not
            density_Scaled = abs2.(sol)
        else
            density_Scaled = copy(sol)
        end
    
        for i in 1:length(density_Scaled[1,1,1,:]) # Scaling to 1 to make isovalues easier
            density_Scaled[:,:,:,i] /= maximum(density_Scaled[:,:,:,i])
        end
    
        saveat = joinpath("Gifs",title)
        tindex = Observable(1)
        scene = Makie.contour(Makie.lift(i -> density_Scaled[i],tindex),
        alpha=alph,
        levels=iso,
        colormap=col,
        show_axis=axis)
    
        Makie.record(scene,saveat,1:length(sol.t),framerate = framerate) do i 
            tindex[] = i
        end
    end
end

#-------------------------- Leshgo ------------------------------------------------

begin # Adjustable Parameters and constants 

    ħ = 1.05457182e-34
    m = 87*1.66e-27 
    a_s = 5.8e-9 
    k_B = 1.380649e-23
    μ = 1e-9 * k_B #ħ # Smaller μ, bigger norm
    g = 4π*ħ^2*a_s/m
    N = 4e5 # Bigger N smaller norm

    trap = "box" # "box", "cyl" (cylinder), or "harm" (harmonic)

    if trap == "harm"
        ω_x = .5 # Harmonic Trapping Frequencies, if using harmonic trap
        ω_y = .5
        ω_z = .1

        Rx = sqrt(2)/ω_x
        Ry = sqrt(2)/ω_y
        Rz = sqrt(2)/ω_z
    end

    ξ = ħ/sqrt(m*μ)
    ψ0 = sqrt(μ/g) #sqrt(N/ξ^3) 
    τ = ħ/μ
    G = 1 #4π*N*a_s/ξ

    Lx = 25 #6*sqrt(2)/ω_x
    Ly = 20 #6*sqrt(2)/ω_y
    Lz = 15 #6*sqrt(2)/ω_z

    Mx = 128 # Grid sizes
    My = 128
    Mz = 128

    A_V = 15 # Trap height
    n_V = 24 # Trap Power (pretty much always 24)
    L_V = 6
    L_P = 10 # no. of healing lengths for V to drop to 0.01A_V (amount of padding)
    use_cuda = CUDA.functional()
end

begin # Arrays

    x = LinRange(-Lx/2 - (L_P + 2L_V),Lx/2 + (L_P + 2L_V),Mx) |> collect
    y = LinRange(-Ly/2 - (L_P + 2L_V),Ly/2 + (L_P + 2L_V),My)' |> collect
    z = LinRange(-Lz/2 - (L_P + 2L_V),Lz/2 + (L_P + 2L_V),Mz)
    z = reshape(z,(1,1,Mz)) |> collect

    dx = x[2] - x[1]
    dy = y[2] - y[1]
    dz = z[2] - z[1]

    kx = fftfreq(Mx,2π/dx) |> collect
    ky = fftfreq(My,2π/dy)' |> collect
    kz = reshape(fftfreq(Mz,2π/dz),(1,1,Mz)) |> collect

    dkx = kx[2] - kx[1]
    dky = ky[2] - ky[1]
    dkz = kz[2] - kz[1]

    k2 =  kx.^2 .+ ky.^2 .+ kz.^2 # 3D wave vector
    dr = dx*dy*dz
    ksum = kx .+ ky .+ kz

    Vbox = (trap == "box")
    Vcyl = (trap == "cyl")
    Vharm = (trap == "harm") 
        
end;

if Vbox
    V_0 = zeros(Mx,My,Mz)
    Vboundary(x) = A_V*cos(x/λ)^n_V
    λ = L_V/acos(0.01^(1/n_V))

    for i in 1:Mx, j in 1:My, k in 1:Mz
        if (abs(x[i]) > 0.5*Lx + L_V) || (abs(y[j]) > 0.5*Ly + L_V) || (abs(z[k]) > 0.5*Lz + L_V) # V = A_V at edges
            V_0[i,j,k] = A_V + 0.05*(max(0,abs(x[i]) - (0.5*Lx + L_V)) + max(0,abs(y[j]) - (0.5*Ly + L_V)) + max(0,abs(z[k]) - (0.5*Lz + L_V)))
        else
            lx = L_V - max(0.0,abs(x[i]) - 0.5*Lx) # Finding the distance from the centre in each direction, 
            ly = L_V - max(0.0,abs(y[j]) - 0.5*Ly) # discarding if small
            lz = L_V - max(0.0,abs(z[k]) - 0.5*Lz)
        
            V_0[i,j,k] = Vboundary(min(lx,ly,lz))
        end
    end
end;

if Vcyl # Cylinder Trap Potential

    V_0 = zeros(Mx,My,Mz)
    Vboundary(x) = A_V*cos(x/λ)^n_V

    λ = L_V/acos(0.01^(1/n_V))
    
    for i in 1:Mx, j in 1:My, k in 1:Mz
        l_z = min(2*L_V,Lz/2 - abs(z[k]))
        l_r = min(2*L_V,sqrt(Lx*Ly)/2 - hypot(x[i]*sqrt(Ly/Lx),y[j]*sqrt(Lx/Ly)))

        l = map(Vboundary,(l_z,l_r))

        V_0[i,j,k] = hypot(l[1],l[2])
    end
end;

if Vharm
    V_0 = 0.5*[(ω_x*i)^2 + (ω_y*j)^2 + (ω_z*k)^2 for i in x, j in reshape(y,My), k in reshape(z,Mz)]
    ψ_gauss = [exp(-0.5*(ω_x*i^2 + ω_y*j^2 + ω_z*k^2)) for i in x, j in reshape(y,My), k in reshape(z,Mz)]  .|> ComplexF32;
end;

Plots.heatmap(x,reshape(y,My),(V_0[:,:,64]'),aspectratio=1,clims=(0,1.2*A_V),xlabel=(L"x/\xi"),ylabel=(L"y/\xi"))
Plots.heatmap(x,reshape(z,Mz),(V_0[:,64,:]'),aspectratio=1,clims=(0,1.2*A_V),xlabel=(L"x/\xi"),ylabel=(L"z/\xi"))

Plots.plot(x,V_0[:,64,64]./A_V,xlabel = L"x",lw=2,ylabel=L"V/A_V",label = false)
vline!([-0.5*Lx,0.5*Lx],alpha = 0.8,label = L"±0.5*Lx")
vline!([-0.5*Lx - L_V,0.5*Lx + L_V],alpha = 0.8,label = L"±(0.5Lx + L_V)")

ψ_TF = 1/sqrt(N)*[max(0,1-V_0[i,j,k]) for i in 1:Mx, j in 1:My, k in 1:Mz] |> complex;
ψ_rand = (randn(Mx,My,Mz) + im*randn(Mx,My,Mz));
ψ_ones = ones(Mx,My,Mz) |> complex;

if use_cuda # Transforming arrays
    x = x |> cu
    y = y |> cu
    z = z |> cu
    V_0 = V_0 |> cu
    kx = kx |> cu
    ky = ky |> cu
    kz = kz |> cu
    k2 = k2 |> cu
    ψ_rand = ψ_rand |> cu
    ψ_ones = ψ_ones |> cu
    ψ_TF = ψ_TF |> cu
end;

if use_cuda # For some reason FFTW.MEASURE doesn't work for cuda arrays
    const Pf = Float32(dr/(2π)^1.5)*plan_fft(copy(ψ_rand));
    const Pi! = Float32(Mx*My*Mz*dkx*dky*dkz/(2π)^1.5)*plan_ifft!(copy(ψ_rand));
else
    const Pf = dr/(2π)^1.5*plan_fft(copy(ψ_rand),flags=FFTW.MEASURE);
    const Pi! = Mx*My*Mz*dkx*dky*dkz/(2π)^1.5*plan_ifft!(copy(ψ_rand),flags=FFTW.MEASURE);
end

CUDA.memory_status()

#-------------------------- Finding Ground State -----------------------------------------

begin
    γ = 1
    tspan = LinRange(0.0,20,2); 

    prob = ODEProblem(GPE!,ψ_ones,(tspan[1],tspan[end]))    
    @time sol = solve(prob,saveat=tspan)
end;

res = Array(sol);
ψ_GS = res[:,:,:,end]; #sol[:,:,:,end];

@save "GS" ψ_GS
@load "GS" ψ_GS

Norm = [number(res[:,:,:,i]) for i in 1:2];
Plots.plot(Norm,ylims=(0,50*Norm[end]))

Plots.heatmap(x,reshape(y,My),abs2.(res[:,:,64,2]'),clims=(0,2),aspectratio=1,xlabel=(L"x/\xi"),ylabel=(L"y/\xi"),right_margin=8mm)
vline!([-0.5*Lx,0.5*Lx],label = "±Lx/2",width=2,alpha=0.3)
hline!([-0.5*Ly,0.5*Ly],label = "±Ly/2",width=2,alpha=0.3)

Plots.heatmap(x,reshape(z,Mz),abs2.(res[:,64,:,2]'),clims=(0,2),aspectratio=1,xlabel=(L"x/\xi"),ylabel=(L"z/\xi"),right_margin=8mm)
vline!([-0.5*Lx,0.5*Lx],label = "±Lx/2",width=2,alpha=0.3)
hline!([-0.5*Lz,0.5*Lz],label = "±Lz/2",width=2,alpha=0.3)

#-------------------------- Creating Turbulence ------------------------------------------

ΔU = .3
ω_shake = 2π * 0.03055 
shakegrid = ΔU * Array(z)./(0.5*Lz) .* ones(Mx,My,Mz) |> complex;

V(t) = sin(ω_shake*t)*shakegrid

noisegrid = randn(Mx,My,Mz) + im*randn(Mx,My,Mz)
ψ_noise = ψ_GS .+ .01*maximum(abs.(ψ_GS))*noisegrid; 
number(ψ_noise)

Plots.heatmap(abs2.(ψ_noise[:,:,64]),clims = (0,1.3))

if use_cuda
    shakegrid = shakegrid |> cu
    ψ_noise = ψ_noise |> cu
end;

CUDA.memory_status()

begin 
    γ = 5e-4
    tspan = LinRange(0,1.5/τ,250)

    prob = ODEProblem(NDVPE!,ψ_noise,(tspan[1],tspan[end]))    
    @time sol2 = solve(prob,saveat=tspan)
end;

CUDA.memory_status()

res1 = zeros(Mx,My,Mz,250) |> complex
for i in 1:250
    res1[:,:,:,i] .= sol2[:,:,:,i]#Array(sol2);
end;

tvec = Array(sol2.t);

res1[:,:,:,1] .= Array(ψ_GS);
ψ_turb = res1[:,:,:,end];

Norm1 = [number(res1[:,:,:,i]) for i in 1:150];
Plots.plot(Norm1,ylims=(0,1e4))

Plots.heatmap(x,reshape(y,My),abs.(res1[:,:,64,2]'),aspectratio=1,title="t = (tvec[4])",clims=(0,1.5),xlabel=(L"x/\xi"),ylabel=(L"y/\xi"))
Plots.heatmap(x,reshape(z,Mz),abs2.(res1[:,64,:,30]'),aspectratio=1,title="t = (tvec[4])",clims=(0,1.5),xlabel=(L"x/\xi"),ylabel=(L"z/\xi"),right_margin=8mm)

Plots.plot(abs.(res1[:,10,10,4]),ylims=(0,1),ylabel ="|ψ(x,y = -Ly, z = -Lz)|",xlabel="x",title = "t = 2/τ")

begin # Energy Plots
    E_K = [Ek(res1[:,:,:,i]) for i in eachindex(res1[1,1,1,:])];
    E_Kx = [Ekx(res1[:,:,:,i]) for i in eachindex(res1[1,1,1,:])];
    E_Ky = [Eky(res1[:,:,:,i]) for i in eachindex(res1[1,1,1,:])];
    E_Kz = [Ekz(res1[:,:,:,i]) for i in eachindex(res1[1,1,1,:])];

    #E_P = [Ep(res[:,:,:,i],V(sol2.t[i])) for i in eachindex(sol2.t)];
    E_I = [Ei(res1[:,:,:,i]) for i in eachindex(res1[1,1,1,:])];

    P = Plots.plot(tspan,E_K,lw=1.5,label=L"E_K")
    Plots.plot!(tspan,E_Kx,lw=1.5,label=L"E_{kx}",alpha=0.4)
    Plots.plot!(tspan,E_Ky,lw=1.5,label=L"E_{ky}",alpha=0.4)
    Plots.plot!(tspan,E_Kz,lw=1.5,label=L"E_{kz}",alpha=0.4)
    #Plots.plot!(sol2.t,E_P,lw=1.5,label=L"E_p")
    Plots.plot!(tspan,E_I,lw=1.5,label=L"E_i")
end

begin
    Einc = zeros(length(tspan))
    Ecom = zeros(length(tspan))
    X = map(Array,(x,reshape(y,My),reshape(z,Mz)));
    K = map(Array,(kx,reshape(ky,My),reshape(kz,My)));

    for i in 1:length(tspan)
        psi = Psi(ComplexF64.(res1[:,:,:,1]),X,K);
        #_, Einc[i], Ecom[i] 
        XX = energydecomp(psi)
    end

    Plots.plot(tspan,Einc,lw=1.5,label=L"E_i")
    Plots.plot!(tspan,Ecom,lw=1.5,label=L"E_c")
end

@save "turb200" res1
@load "turb200" res1
#Plots.savefig(P,"Energy")

#-------------------------- Relaxation ---------------------------------------------------

if use_cuda
    ψ_turb = ψ_turb |> cu
end;

CUDA.memory_status()

begin 
    γ = 5e-4
    tspan = LinRange(0,2.0/τ,5)

    prob = ODEProblem(GPE!,ψ_turb,(tspan[1],tspan[end]))    
    @time sol3 = solve(prob,saveat=tspan,abstol=1e-6,reltol=1e-4)
end;

CUDA.memory_status()
res2 = (Array(sol3));
tvec2 = Array(sol3.t);

Norm2 = [number(Array(res2[:,:,:,i])) for i in 1:4];
Plots.plot(Norm2,ylims=(0,1e4))

begin
    t1 = 5
    #Plots.heatmap(x,reshape(y,My),abs2.(res2[:,:,100,t1]'),clims=(0,1.2),aspectratio=1,xlabel=(L"x/\xi"),ylabel=(L"y/\xi"),title="t=$(tvec2[t1])")
    Plots.heatmap(x,reshape(z,Mz),abs2.(res2[:,100,:,t1]'),clims=(0,1.2),aspectratio=1,xlabel=(L"x/\xi"),ylabel=(L"z/\xi"),title="t=$(tvec2[t1])")
end

begin # Energy plots
    E_K = [Ek(res2[:,:,:,i]) for i in eachindex(tspan)];
    E_Kx = [Ekx(res2[:,:,:,i]) for i in eachindex(tspan)];
    E_Ky = [Eky(res2[:,:,:,i]) for i in eachindex(tspan)];
    E_Kz = [Ekz(res2[:,:,:,i]) for i in eachindex(tspan)];

    E_P = [Ep(res2[:,:,:,i],zeros(Mx,My,Mz)) for i in eachindex(tspan)];
    E_I = [Ei(res2[:,:,:,i]) for i in eachindex(tspan)];

    P = Plots.plot(tspan,E_K,lw=1.5,label=L"E_K")
    Plots.plot!(tspan,E_Kx,lw=1.5,label=L"E_{kx}",alpha=0.4)
    Plots.plot!(tspan,E_Ky,lw=1.5,label=L"E_{ky}",alpha=0.4)
    Plots.plot!(tspan,E_Kz,lw=1.5,label=L"E_{kz}",alpha=0.4)
    Plots.plot!(tspan,E_P,lw=1.5,label=L"E_p")
    Plots.plot!(tspan,E_I,lw=1.5,label=L"E_i")
end

@save "relax200" res2
@load "relax200" res2

#-------------------------- Expansion -------------------------------------------------

begin # Expansion Functions

    function initialise(ψ)
        global Na = number(ψ) |> Float32
        ϕi = ψ ./ sqrt(Na)

        Na *= G
        
        if use_cuda
            ϕi = cu(ϕi)
        end

        global ax2 = dr*sum(@. x^2*abs2(ϕi))
        global ay2 = dr*sum(@. y^2*abs2(ϕi))
        global az2 = dr*sum(@. z^2*abs2(ϕi))

        σi = real.(0.5*im*dr.*CurrentDensity(ϕi)./(ax2,ay2,az2))
        @. ϕi *= exp(-0.5*im*(σi[1]*x^2 + σi[2]*y^2 + σi[3]*z^2))

        if use_cuda
            ϕi = ϕi |> cu
            σi = σi |> cu
            global Pfx = Float32(dx/sqrt(2π))*plan_fft(copy(ϕi),1)
            global Pfy = Pf # Cannot do cuda fft along second dimension, have to do full transform :( 
            global Pfz = Float32(dz/sqrt(2π))*plan_fft(copy(ϕi),3)

            global Pix! = Float32(Mx*dkx/sqrt(2π))*plan_ifft!(copy(ϕi),1)
            global Piy! = Pi!
            global Piz! = Float32(Mz*dkz/sqrt(2π))*plan_ifft!(copy(ϕi),3)
        else
            global Pfx = dx/sqrt(2π)*plan_fft(copy(ϕi),1)
            global Pfy = dy/sqrt(2π)*plan_fft(copy(ϕi),2)
            global Pfz = dz/sqrt(2π)*plan_fft(copy(ϕi),3)

            global Pix! = Mx*dkx/sqrt(2π)*plan_ifft!(copy(ϕi),1)
            global Piy! = My*dky/sqrt(2π)*plan_ifft!(copy(ϕi),2)
            global Piz! = Mz*dkz/sqrt(2π)*plan_ifft!(copy(ϕi),3)
        end

        ϕ_initial =  ArrayPartition(ϕi,[1,1,1,σi[1],σi[2],σi[3]])
        return ϕ_initial
    end

    function extractinfo(sol)
        λx = [sol[:,i].x[2][1] for i in eachindex(sol.t)]
        λy = [sol[:,i].x[2][2] for i in eachindex(sol.t)]
        λz = [sol[:,i].x[2][3] for i in eachindex(sol.t)]
        
        σx = [sol[:,i].x[2][4] for i in eachindex(sol.t)]
        σy = [sol[:,i].x[2][5] for i in eachindex(sol.t)]
        σz = [sol[:,i].x[2][6] for i in eachindex(sol.t)]
    
        ax = @. sqrt(ax2*λx^2)
        ay = @. sqrt(ay2*λy^2)
        az = @. sqrt(az2*λz^2)
    
        ϕ = [sol[:,i].x[1] for i in eachindex(sol.t)]
        res = [Array(ϕ[i]) for i in eachindex(sol.t)];

        return res,ϕ,λx,λy,λz,σx,σy,σz,ax,ay,az
    end

    function CurrentDensity(ψ)
        ϕ = fft(ψ)
        ϕim = fft(conj.(ψ))
        
        σx = x.*(ψ.*ifft(im*kx.*ϕim) - conj(ψ).*ifft(im*kx.*ϕ)) |> sum
        σy = y.*(ψ.*ifft(im*ky.*ϕim) - conj(ψ).*ifft(im*ky.*ϕ)) |> sum
        σz = z.*(ψ.*ifft(im*kz.*ϕim) - conj(ψ).*ifft(im*kz.*ϕ)) |> sum

        return [σx,σy,σz]
    end

    function kfunc2!(dϕ,ϕ,λ)
        mul!(dϕ,Pf,ϕ)
        dϕ .*= (k[1]./λ[1]).^2 .+ (k[2]./λ[2]).^2 .+ (k[3]./λ[3]).^2
        Pi!*dϕ
    end

    function ρ2(λ,σ)
        λx,λy,λz = λ |> real
        σx,σy,σz = σ |> real
        return  @. x.^2*λx*σx + y^2*λy*σy + z^2*λz*σz
    end

    function firstOrder!(dϕ,ϕ,dσ,λ,i)  
        mul!(dϕ,PfArray[i],ϕ)
        dϕ .*= im*k[i]
        dσ[i] = λ[i]^(-2) * sum(abs2.(PiArray[i]*dϕ))
    end

    function spec_expansion_opt!(du,u,p,t)
        ϕ = u.x[1]
        λ = u.x[2][1:3]
        σ = u.x[2][4:6]

        dϕ = du.x[1]
        du.x[2][1:3] .= u.x[2][4:6]
        dσ = du.x[2][4:6]

        λ̄³ = prod(λ) 

        # dσ/dt

        firstOrder!(dϕ,ϕ,dσ,λ,1) # Computing 3rd term in dσ equations
        firstOrder!(dϕ,ϕ,dσ,λ,2)
        firstOrder!(dϕ,ϕ,dσ,λ,3)

        dσ .-= map(i -> sum(abs2.(ϕ).*i),(xdV,ydV,zdV)) # Computing second terms
        dσ .+= (0.5*Na/λ̄³)*sum(abs2.(ϕ).^2) # Computing first terms
        dσ .*= dr ./ (λ[1]*ax2,λ[2]*ay2,λ[3]*az2) # Coefficients

        # dϕ/dt
        
        kfunc2!(dϕ,ϕ,λ) # Kinetic Term
        dϕ .= -im*(0.5.*dϕ .+ (V_0 .+ (Na/λ̄³)*abs2.(ϕ) .+ 0.5*ρ2(λ,dσ)).*ϕ)
        
        du.x[1] .= dϕ
        du.x[2][4:6] .= dσ
    end 

    function spec_expansion_noint!(du,u,p,t)
        ϕ = u.x[1]
        λ = u.x[2][1:3]
        σ = u.x[2][4:6]

        #println(typeof(ϕ))
        dϕ = du.x[1]
        du.x[2][1:3] .= u.x[2][4:6]
        dσ = du.x[2][4:6]

        λ̄³ = prod(λ) 

        # dσ/dt

        firstOrder!(dϕ,ϕ,dσ,λ,1) # Computing 3rd term in dσ equations
        firstOrder!(dϕ,ϕ,dσ,λ,2)
        firstOrder!(dϕ,ϕ,dσ,λ,3)

        dσ .-= map(i -> sum(abs2.(ϕ).*i),(xdV,ydV,zdV)) # Computing second terms
        #dσ .+= (0.5*Na/λ̄³)*sum(abs2.(ϕ).^2) # Computing first terms
        dσ .*= dr ./ (λ[1]*ax2,λ[2]*ay2,λ[3]*az2) # Coefficients

        # dϕ/dt
        
        kfunc2!(dϕ,ϕ,λ) # Kinetic Term
        dϕ .= -im*(0.5.*dϕ .+ (V_0 .+ 0.5*ρ2(λ,dσ)).*ϕ)
        
        du.x[1] .= dϕ
        du.x[2][4:6] .= dσ
    end 

end

begin
    ψ_0 = res2[:,:,:,2]
    ϕ_initial = initialise(ψ_0)

    PfArray = [Pfx, Pfy, Pfz]
    PiArray = [Pix!,Piy!,Piz!]
    k = [kx,ky,kz]

    V_0 = zeros(Mx,My,Mz)

    if use_cuda
        V_0 = V_0 |> cu
    end

    xdV = x.*ifft(im*kx.*fft(V_0)) .|> real
    ydV = y.*ifft(im*ky.*fft(V_0)) .|> real
    zdV = z.*ifft(im*kz.*fft(V_0)) .|> real

    if use_cuda
        xdV = xdV |> cu
        ydV = ydV |> cu
        zdV = zdV |> cu
    end

end;

t = LinRange(0,.01,2);

CUDA.memory_status()

probs = ODEProblem(spec_expansion_opt!,ϕ_initial,(t[1],t[end]));
@time solt2 = solve(probs,saveat=t,abstol=1e-6,reltol=1e-5);

res,ϕ,λx,λy,λz,σx,σy,σz,ax,ay,az = extractinfo(solt2);

Norm = [number(res[i]) for i in 1:4]
Plots.plot(Norm,ylims = (0,1.5*maximum(Norm)))

Plots.plot(λx,label="λx")
Plots.plot!(λy,label="λx")
Plots.plot!(λz,label="λz")

begin
    Plots.plot(t,ay ./ ax,ylims=(0.7,1.8),c=:red,lw=2,label="ay/ax")
    Plots.plot!(t,ay ./ az,c=:blue,lw=2,label="ay/az")
    Plots.plot!(t,ax ./ az,c=:green,lw=2,label="ax/az")

    #hline!(sqrt.([ax2/ay2]),linestyle=:dash,alpha=0.6,c=:red)
    #hline!(sqrt.([az2/ay2]),linestyle=:dash,alpha=0.6,c=:blue)
    #hline!(sqrt.([az2/ax2]),linestyle=:dash,alpha=0.6,c=:green)
end

ax[1]
ay[1]
az[1]

ress = [abs2.(res[i]) for i in 1:length(res)];

for i in 1:4
    begin
        t1 = i
        p = Plots.heatmap(x*λx[t1],reshape(y,My)*λy[t1],ress[t1][:,:,100]',aspectratio=1,clims=(0,1e-3),c=:thermal,ylabel = L"y/\xi" )
        #p = Plots.heatmap(x*λx[t1],reshape(z,Mz)*λz[t1],ress[t1][:,100,:]',aspectratio=1,clims=(0,1e-3),c=:thermal,ylabel=L"z/\xi")
        Plots.xlabel!(L"x/\xi")
        display(p)
    end
end

vline!([200])
hline!([200])

fftpsi = log.(abs2.(fftshift(fft(sol[:,:,:,end]))));
Plots.heatmap(fftshift(kx),fftshift(reshape(ky,My)),fftpsi[:,:,64]',aspectratio=1,c=:thermal,ylabel=L"y/\xi")
Plots.heatmap(fftshift(kx),fftshift(reshape(kz,Mz)),fftpsi[:,64,:]',aspectratio=1,c=:thermal,ylabel=L"z/\xi")

begin # Castin-Dum tings
    t1 = 1
    dt = t[2] - t[1]

    m = ω_y^2/(λy[t1]^2*λx[t1]*λz[t1])

    pl = Plots.plot(t,σx,ylims=(0.0,.6),label="σx",xlabel=(L"t/$\tau$"),lw=2,alpha=.5,legend=:bottomright)
    Plots.plot!(t,σy,ylims=(0.0,.6),label="σy",lw=2,alpha=.5)
    Plots.plot!(t,σz,ylims=(0.0,.6),label="σz",lw=2,alpha=.5)

    Plots.plot!(x->m*x + σy[t1] - m*dt*(t1-1),style=:dash,label = "Castin-Dum predicted slope")
    vline!([t[t1]],alpha=0.5,label=false)
    hline!([σy[t1]],alpha=0.5,label=false)
    hline!([ω_x],label="ωx")

    display(pl)
end

#-------------------------- Spectra -----------------------------------------------------

# res = GS
# res1 = turb
# res2 = relax

X = map(Array,(x,reshape(y,My),reshape(z,Mz)));
K = map(Array,(kx,reshape(ky,My),reshape(kz,My)));
ψ = ComplexF64.(res2[:,:,:,1]);

psi = Psi(ψ,X,K);
k = log10range(0.1,10^2,100)
E_i = incompressible_spectrum(k,psi);
E_c = compressible_spectrum(k,psi);
E_q = qpressure_spectrum(k,psi);

begin # Plots
    P = Plots.plot(k,E_q,axis=:log,ylims=(1e2,1e5),label=false,lw=2,legend=:bottomright,alpha=0.5,title=L"E_{QP}")# / E_{incompressible}")
    #Plots.plot!(x->(1.2e4)*x^-3,[x for x in k[50:70]],label=false,alpha=1,lw=.5)
    #Plots.plot!(x->(2.2e2)*x^1.1,[x for x in k[7:55]],label=false,alpha=1,lw=.5)

    k_Lx = 2π/(Lx)# Size of the System
    k_Ly = 2π/(Ly)# Size of the System
    k_Lz = 2π/(Lz)# Size of the System
    #k_l = 2π/(Lx - 2*L_V) # Size of the condensate accounting for the box trap
    k_π = π#2π/14
    k_ξ = 2π# Healing length
    k_dr = 2π/dr^(1/3) # Geometric mean of resolution
    k_dx = 2π/hypot(dx,dx,dx)
    
    vline!([k_Lx], label = L"$k_{Lx}$",linestyle=:dash,alpha=0.5)
    vline!([k_Ly], label = L"$k_{Ly}$",linestyle=:dash,alpha=0.5)
    vline!([k_Lz], label = L"$k_{Lz}$",linestyle=:dash,alpha=0.5)
    #vline!([k_l], label = L"$k_l$",linestyle=:dash,alpha=0.5)
    vline!([k_π], label = L"$π$",linestyle=:dash,alpha=0.5)
    vline!([k_ξ], label = L"$k_\xi$",linestyle=:dash,alpha=0.5)
    vline!([k_dr], label = L"$k_{dr}$",linestyle=:dash,alpha=0.5)
    vline!([k_dx], label = L"$k_{lol}$",linestyle=:dash,alpha=0.5)
end

