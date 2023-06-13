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
    CUDA,
    Adapt

include("V5.jl")

#------------------------------- Constants + Parameters -----------------------------

ħ = 1.05457182e-34
m = 87*1.66e-27 
a_s = 5.8e-9 
k_B = 1.380649e-23

μ = 1e-9 * k_B #ħ # Smaller μ, bigger norm
g = 4π*ħ^2*a_s/m
ξ = ħ/sqrt(m*μ)
ψ0 = sqrt(μ/g) #sqrt(N/ξ^3) 
τ = ħ/μ

const L = (25,25,25)     # Condensate size
const M = (200,200,200)  # System Grid

A_V = 15    # Trap height
n_V = 24    # Trap Power (pretty much always 24)
L_V = 10    # No. of healing lengths for V to drop from A_V to 0.01A_V 
L_P = 15    # Amount of padding outside trap (for expansion)

L_T = L .+ 2*(L_P + L_V)  # Total grid size
use_cuda = CUDA.functional()
numtype = Float32

#------------------------------- Making Arrays and potentials ----------------------

X,K,k2 = MakeArrays(L_T,M);
dX = L_T ./ M  #map(x -> diff(x)[1],X) # Ask Ashton about this
dK = @. 2π / (dX*M)

V_0 = BoxTrap(X,L,M,L_V,A_V,n_V) |> cu;
ψ_rand = randn(M) |> complex |> cu;

const Pf = prod(@. dX / sqrt(2π)) * plan_fft(copy(ψ_rand));
const Pi! = prod(@. M * dK / sqrt(2π)) * plan_ifft!(copy(ψ_rand));

#------------------------------- Finding Ground State ------------------------------

γ = 1
tspan = LinRange(0.0,30,20); 
res_GS = []
GPU_Solve!(res_GS,GPE!,ψ_rand,tspan,plot_progress=true, print_progress=true,abstol=1e-8,reltol=1e-5,alg=ParsaniKetchesonDeconinck3S32());
CUDA.memory_status()

#prob = ODEProblem(GPE!,ψ_rand,(tspan[1],tspan[end]));
#sol = solve(prob,abstol = 1e-8, reltol = 1e-5);

#Plots.plot(diff(sol.t))

Norm = [number(i) for i in res_GS];
Plots.plot(Norm,ylims=(0,2*Norm[end]))

Plots.heatmap(X[1],X[3],abs.(res_GS[20][:,100,:]'),clims=(0,1),aspectratio=1,xlabel=(L"x/\xi"),ylabel=(L"z/\xi"),right_margin=8mm,cbar=false)
xlims!(-40,40)
vline!([-0.5*L[1],0.5*L[1]],label = L"± 0.5 Lx",width=2,alpha=0.3)
hline!([-0.5*L[3],0.5*L[3]],label = L"± 0.5 Lz",width=2,alpha=0.3)

ψ_GS = res_GS[end] |> cu;

#------------------------------- Creating Turbulence ------------------------------

Shake_Grad = 0.1            # Gradient of shake 
ω_shake = 2π * 0.03055      # Frequency of shake 
shakegrid = Shake_Grad * Array(z) .* ones(Mx,My,Mz) |> complex |> cu;  

V(t) = sin(ω_shake*t)*shakegrid

Plots.plot(X[3],Shake_Grad*X[3])

γ = 0
tspan = LinRange(0,2.0/τ,40)
res_turb = []
GPU_Solve!(res_turb,NDVPE!,ψ_GS,tspan,reltol=1e-5,abstol = 1e-8, plot_progress=true, print_progress=true,alg=Tsit5());
CUDA.memory_status()

Norm1 = [number(i) for i in res_turb];
Plots.plot(Norm1,ylims=(0,2e4))

Plots.heatmap(x,reshape(z,Mz),abs2.(res_turb[end][:,128,:]'),aspectratio=1,
        size = (600,600),
        #title="t = (tvec[4])",
        clims=(0,1.5),
        xlabel=(L"x/\xi"),
        ylabel=(L"z/\xi"),
        label=(L"z/\xi"),
        right_margin=8mm,
        c=:thermal,
        cbar=false,
        xlims=(-40,40),
        legendfontsize=12,
        labelfontsize=15,
        ylims=(-35,35))
vline!([-0.5*Lx,0.5*Lx],label = L"± 0.5Lx",width=2,alpha=0.3)
hline!([-0.5*Lz,0.5*Lz],label = L"± 0.5 Lz",width=2,alpha=0.3)

ψ_turb = res_turb[end] |> cu;

#------------------------------- Relxaxation --------------------------------------

γ = 0
tspan = LinRange(0,2.0/τ,40)
res_relax = []
GPU_Solve!(res_relax,GPE!,cu(res_relax[end]),tspan2,reltol=1e-5,abstol = 1e-8, plot_progress=true, print_progress=true,alg=Tsit5());
CUDA.memory_status()

Norm2 = [number(Array(i)) for i in res_relax];
Plots.plot(Norm2,ylims=(0,2e4))

#------------------------------- Relxaxation --------------------------------------

ψ_0 = res_turb[end]
ϕ_initial = initialise(ψ_0)

PfArray = [Pfx, Pfy, Pfz]
PiArray = [Pix!,Piy!,Piz!]

V_0 = zeros(M) |> cu

xdV = x.*ifft(im*kx.*fft(V_0)) .|> real |> cu
ydV = y.*ifft(im*ky.*fft(V_0)) .|> real |> cu
zdV = z.*ifft(im*kz.*fft(V_0)) .|> real |> cu

tspan = LinRange(0,20,20);
res_expand = []; 
GPU_Solve!(res_expand,spec_expansion_opt!,ϕ_initial,tspan,alg=Tsit5());

res,λx,λy,λz,σx,σy,σz,ax,ay,az = extractinfo(res_expand);

Norm = [number(i) for i in res]
Plots.plot(Norm,ylims = (0,1.5*maximum(Norm)))

Plots.plot(λx,label="λx")
Plots.plot!(λy,label="λx")
Plots.plot!(λz,label="λz")

Plots.plot(ay ./ ax,ylims=(0.7,1.8),c=:red,lw=2,label="ay/ax")
Plots.plot!(ay ./ az,c=:blue,lw=2,label="ay/az")
Plots.plot!(ax ./ az,c=:green,lw=2,label="ax/az")

#-------------------------- Spectra/Analysis-----------------------------------------------------

ψ = ComplexF64.(res_turb[end]);
psi = Psi(ψ,X,K);
k = log10range(0.1,10^2,100)
E_i = incompressible_spectrum(k,psi);
E_c = compressible_spectrum(k,psi);
E_q = qpressure_spectrum(k,psi);


begin # Spectra Plots
    P = Plots.plot(k,E_i,axis=:log,ylims=(1e2,5e4),xlims=(.1,15),
        label="Incompressible",
        lw=2,
        legend=:bottomright,
        alpha=0.8,
        framestyle=:box,
        xlabel=(L"k\xi")
    )
    Plots.plot!(k,E_c,lw = 2,alpha=0.8,label="Compressible")
    Plots.plot!(k,E_q,lw=2,alpha=0.8, label = "Quantum Pressure")


    Plots.plot!(x->(1.213e6)*x^-3,[x for x in k[50:70]],label=false,alpha=1,lw=.5)
    #Plots.plot!(x->(2.2e2)*x^1.1,[x for x in k[7:55]],label=false,alpha=1,lw=.5)

    k_Lx = 2π/(Lx)# Size of the System
    k_Ly = 2π/(Ly)# Size of the System
    k_Lz = 2π/(Lz)# Size of the System
    #k_l = 2π/(Lx - 2*L_V) # Size of the condensate accounting for the box trap
    k_π = π#2π/14
    k_ξ = 2π# Healing length
    k_dr = 2π/dr^(1/3) # Geometric mean of resolution
    k_dx = 2π/hypot(dx,dx,dx)
    
    vline!([k_Lx], label = false,linestyle=:dash,alpha=0.5,c=:black)
    vline!([k_Ly], label = false,linestyle=:dash,alpha=0.5,c=:black)
    vline!([k_Lz], label = false,linestyle=:dash,alpha=0.5,c=:black)
    #vline!([k_l], label = L"$k_l$",linestyle=:dash,alpha=0.5)
    #vline!([k_π], label = L"$π$",linestyle=:dash,alpha=0.5)
    vline!([k_ξ], label = false,linestyle=:dash,alpha=0.5,c=:black)
    #vline!([k_dr], label = L"$k_{dr}$",linestyle=:dash,alpha=0.5)
    vline!([k_dx], label = false,linestyle=:dash,alpha=0.5,c=:black)
end

res = cat(res_turb,res_relax,dims=1); # Energy Plots
length(res)
typeof(res)

tspan = LinRange(0,2.0/τ,40);

E_K = [Ek(i) for i in res];
E_Kx = [Ekx(i) for i in res];
E_Ky = [Eky(i) for i in res];
E_Kz = [Ekz(i) for i in res];
E_P = [0. for i in 1:80];
E_P[1:40] = [Ep(i,V(tspan[t])) for (t,i) in enumerate(res[1:40])];
E_P[41:80] = [Ep(i,zeros(Mx,My,Mz)) for (t,i) in enumerate(res[41:80])];
#E_P = [Ep(i,V(tspan[t])) for (t,i) in enumerate(res)];
#E_P = [Ep(i,zeros(Mx,My,Mz)) for (t,i) in enumerate(res)];
E_I = [Ei(i) for i in res];

tspan = LinRange(0,4.0,80)

P = Plots.plot(tspan,E_K,lw=1.5,label=L"E_K",size=(700,400))
Plots.plot!(tspan,E_Kx,lw=1.5,label=L"E_{kx}",alpha=0.4)
Plots.plot!(tspan,E_Ky,lw=1.5,label=L"E_{ky}",alpha=0.4)
Plots.plot!(tspan,E_Kz,lw=1.5,label=L"E_{kz}",alpha=0.4)
Plots.plot!(tspan,E_P,lw=1.5,label=L"E_p")
Plots.plot!(tspan,E_I,lw=1.5,label=L"E_i")
Plots.plot!(tspan,(E_K .+ E_P .+ E_I),label=L"E_T")
