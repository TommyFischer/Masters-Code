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
    Adapt,
    DrWatson

include("V5.jl")

@quickactivate "Watson" # <- project name                 

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
const M = (150,150,150)  # System Grid

A_V = 15    # Trap height
n_V = 24    # Trap Power (pretty much always 24)
L_V = 5    # No. of healing lengths for V to drop from A_V to 0.01A_V 
L_P = 5    # Amount of padding outside trap (for expansion)

L_T = L .+ 2*(L_P + L_V)  # Total grid size
use_cuda = CUDA.functional()
numtype = Float32

#------------------------------- Making Arrays and potentials ----------------------

X,K,k2 = MakeArrays(L_T,M);
dX = L_T ./ M  #map(x -> diff(x)[1],X) # Ask Ashton about this
dK = @. 2π / (dX*M)
ksum = Array(K[1]) .+ reshape(Array(K[2]),(1,M[2])) .+ reshape(Array(K[3]),(1,1,M[3]));

V_0 = BoxTrap(X,L,M,L_V,A_V,n_V) |> cu;
ψ_rand = randn(M) .+ im*randn(M) .|> abs |> complex |> cu;

const Pf = prod(@. dX / sqrt(2π)) * plan_fft(copy(ψ_rand));
const Pi! = prod(@. M * dK / sqrt(2π)) * plan_ifft!(copy(ψ_rand));

#------------------------------- Finding Ground State ------------------------------

function save_func(res)
    return res[end]
end

GSparams = Dict(
    "ψ" => ψ_rand,
    "γ" => [1],
    "tf" => [20],
    "Nt" => 5
) |> dict_list;

SOLS_GS = []

for (i,d) in enumerate(GSparams)
    println("sim $i / $(length(GSparams))")
    @unpack ψ, γ, tf, Nt = d
    res = Watson_sim(ψ,LinRange(0,tf,Nt),"GS",γ)
    push!(SOLS_GS,res)
end

ψ_GS = SOLS_GS[end] |> cu;
Plots.heatmap(abs2.(ψ_GS)[:,75,:],aspectratio=1)

#------------------------------- Creating Turbulence ------------------------------

function save_func(res)
    #wsave(datadir("simulations",savename(d,"jld2")),f)
    return res
end

Shake_params = Dict(
    "ψ" => ψ_GS,
    "γ" => [0],
    "tf" => [2.0/τ],
    "Nt" => 10,
    "Shake_Grad" => [0.12, 0.15, 0.2]
) |> dict_list;

SOLS_TURB = []

for (i,d) in enumerate(Shake_params)
    println("sim $i / $(length(Shake_params))")
    @unpack ψ, γ, tf, Nt, Shake_Grad = d
    
    global shakegrid = Shake_Grad * Array(X[3]) .* ones(M) |> complex |> cu;  
    global ω_shake = 2π * 0.03055      
    global V(t) = sin(ω_shake*t)*shakegrid

    res = Watson_sim(ψ,LinRange(0,tf,Nt),"SHAKE",γ)
    push!(SOLS_TURB,res)
end

ψ_turb = SOLS_TURB[1][1] |> cu;

typeof(SOLS_TURB[6][10])

function kdensity(k,psi::Psi{3})  
    @unpack ψ,X,K = psi; 
	C = auto_correlate(ψ,X,K)
    return sinc_reduce(k,X...,C)
end


nplots = []
k = log10range(0.1,20,100)   

xx = (X[1],X[2],X[3]) .|> Array
kk = (K[1],K[2],K[3]) .|> Array

for sol in SOLS_TURB[7:end]
    for i in 1:10
        ψ = sol[i]; 
        ψ /= sqrt(number(ψ)); 
        psi = Psi(ψ,xx,kk)
        nk = kdensity(k,psi)
        push!(nplots,nk)
    end
end
2π/L[3]
2π/(dX[1])

for i in 10:10
    P = Plots.plot(k,nplots[i].*k.^-2,axis=:log,ylims=(1e-6,1e4),label = "0.01")
    vline!([4π/L[3],2π/hypot(dX[1],dX[2],dX[3])],label=false)
    display(P)
end
#------------------------------- Expansion ------------------------------

grads = ["0.01", "0.02", "0.04", "0.06", "0.08", "0.1", "0.12", "0.15", "0.2"]
alpha = log10range(0.1,1,9)

Plots.plot(k,nplots[1].*k.^-2,axis=:log,ylims=(1e-6,1e4),label = grads[1],alpha=0.5,c = :red)
for i in 1:9
    display(Plots.plot!(k,nplots[i*10].*k.^-2,axis=:log,ylims=(1e-6,1e4),label = grads[i],alpha=alpha[i],c = :blue))
end

function save_func(res)
    return res
end

Expand_params = Dict(
    "ψ" => [initialise(ψ_GS),initialise(SOLS_TURB[1]),initialise(SOLS_TURB[2]),initialise(SOL_TURB[3])],
    "tf" => [30],
    "Nt" => 30,
) |> dict_list;

SOLS_EXPAND = []

PfArray = [Pfx, Pfy, Pfz]
PiArray = [Pix!,Piy!,Piz!]

V_0 = zeros(M) |> cu;

xdV = X[1].*ifft(im*K[1].*fft(V_0)) .|> real |> cu;
ydV = X[2].*ifft(im*K[2].*fft(V_0)) .|> real |> cu;
zdV = X[3].*ifft(im*K[3].*fft(V_0)) .|> real |> cu;

for (i,d) in enumerate(Expand_params)
    println("sim $i / $(length(Expand_params))")
    @unpack ψ, tf, Nt = d
    
    res = Watson_sim(ψ,LinRange(0,tf,Nt),"EXPAND",0)
    push!(SOLS_EXPAND,res)
end

res,λx,λy,λz,σx,σy,σz,ax,ay,az = extractinfo(SOLS_EXPAND[3]);

for i in 1:30
    t1 = i
    display(Plots.heatmap(Array(X[1]).*λx[t1],Array(X[2]).*λz[t1],abs2.(res[t1][:,75,:]'),aspectratio=1))
end

Plots.plot(λx); Plots.plot!(λx); Plots.plot!(λx)
