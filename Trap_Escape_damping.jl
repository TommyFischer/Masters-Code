using LinearAlgebra,
    DifferentialEquations,
    FFTW,
    JLD2,
    BenchmarkTools,
    CUDA,
    DrWatson

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

const L = (40,30,20)     # Condensate size
const M = (300,300,300)  # System Grid

A_V = 30    # Trap height
n_V = 24    # Trap Power (pretty much always 24)
L_V = 3     # No. of healing lengths for V to drop from A_V to 0.01A_V 
L_P = 6    # Amount of padding outside trap (for expansion)

L_T = L .+ 2*(L_P + L_V)  # Total grid size
use_cuda = CUDA.functional()
numtype = Float32

    #------------------------------- Making Arrays and potentials ----------------------

X,K,k2 = MakeArrays(L_T,M);
dX = map(x -> diff(x)[1],X)  #L_T ./ M 
dK = map(x -> diff(x)[1],K) #@. 2π / (dX*M)
ksum = Array(K[1]) .+ reshape(Array(K[2]),(1,M[2])) .+ reshape(Array(K[3]),(1,1,M[3]));

V_0 = BoxTrap(X,L,M,L_V,A_V,n_V) |> cu;
ψ_rand = randn(M) .+ im*randn(M) .|> abs |> complex |> cu;

const Pf = prod(@. dX / sqrt(2π)) * plan_fft(copy(ψ_rand));
const Pi! = prod(@. M * dK / sqrt(2π)) * plan_ifft!(copy(ψ_rand));

    #------------------------------- Finding Ground State ------------------------------

GSparams = Dict(
    "title" => "GS $M, $L",
    "ψ" => ψ_rand,
    "γ" => [1],
    "tf" => [5],
    "Nt" => 2
) |> dict_list;

for (i,d) in enumerate(GSparams)
    @unpack ψ, γ, tf, Nt = d
    res = []
    GPU_Solve!(res,GPE!,ψ_rand,LinRange(0,tf,Nt),γ,alg=Tsit5(),plot_progress=false)
    global ψ_GS = res[end] |> cu
end

    #------------------------------- Creating Turbulence ------------------------------

δ = 3
V_D = 2.5

V_damp = -im * V_D * [(abs(i) > 0.5*L[1] + L_V + δ) || (abs(j) > 0.5*L[2] + L_V + δ) || (abs(k) > 0.5*L[3] + L_V + δ) ? 1 : 0 for i in Array(X[1]), j in Array(X[2]), k in Array(X[3])] |> cu;

function Escape_VPE!(dψ,ψ,var,t)
    #GC.gc([full=false])
    #CUDA.memory_status()
    kfunc_opt!(dψ,ψ)
    @. dψ = -im*(0.5*dψ + (V_0 + $V(t) + V_damp + abs2(ψ) - 1)*ψ)
end

Shake_params = Dict(
    "title" => "EscapeTurb $M, $L",
    "ψ" => ψ_GS,
    "γ" => [0],
    "tf" => [6.0/τ],
    "Nt" => 100,
    "Shake_Grad" => [0.03,0.05]
) |> dict_list;   

function save_func(res,d)
    wsave("/nesi/nobackup/uoo03837/results3/" * savename(d,"jld2"),
    Dict("res" => res,
        "X" => Array.(X),
        "K" => Array.(K),
        "dX" => dX,
        "dK" => dK,
        "V" => Array(V_0),
        "M" => M,
        "L" => L
    ))
    #global ψ_turb = res[end] |> cu
end;

for (i,d) in enumerate(Shake_params)
    @unpack ψ, γ, tf, Nt, Shake_Grad = d
    
    global shakegrid = Shake_Grad * reshape(Array(X[3]),(1,1,M[3])) .* ones(M) |> complex |> cu;  
    global ω_shake = 2π * 0.03055      
    global V(t) = sin(ω_shake*t)*shakegrid

    res = []
    GPU_Solve!(res,Escape_VPE!,ψ_GS,LinRange(0,tf,Nt),γ,alg=Tsit5(),plot_progress=false)
    save_func(res,d)
    println("sublime")
    #global shakesol = res
    #global ψ_turb = res[end] |> cu
end

    #------------------------------- Relaxation ------------------------------

    #Relax_params = Dict(
    #"title" => "Relax $M, $L",
    #"ψ" => ψ_turb,
    #"γ" => [0],
    #"tf" => [2.0/τ],
    #"Nt" => 100,
#) |> dict_list;   

    function save_func(res,d)
        wsave("/nesi/nobackup/uoo03837/results3/" * savename(d,"jld2"),
        Dict("res" => res,
            "X" => Array.(X),
            "K" => Array.(K),
            "dX" => dX,
            "dK" => dK,
            "V" => Array(V_0),
            "M" => M,
            "L" => L
        ))
        ψ_turb = res[end] |> cu
    end;

    function Escape_GPE!(dψ,ψ,var,t)
        #GC.gc([full=false])
        #CUDA.memory_status()
        kfunc_opt!(dψ,ψ)
        @. dψ = -im*(0.5*dψ + (V_0 + V_damp + abs2(ψ) - 1)*ψ)
    end

    #for (i,d) in enumerate(Relax_params)
    #    @unpack ψ, γ, tf, Nt = d
    #        
    #    res = []
    #    GPU_Solve!(res,Escape_GPE!,ψ,LinRange(0,tf,Nt),γ,alg=Tsit5(),plot_progress=false)
    #    save_func(res,d)
    #end


