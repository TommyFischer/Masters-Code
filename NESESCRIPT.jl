using LinearAlgebra,
    DifferentialEquations,
    FFTW,
    JLD2,
    BenchmarkTools,
    CUDA,
    Adapt,
    DrWatson

println("Packages loaded")
include("V5.jl")

#@quickactivate "Watson" # <- project name                 

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
const M = (100,100,100)  # System Grid

A_V = 15    # Trap height
n_V = 24    # Trap Power (pretty much always 24)
L_V = 5    # No. of healing lengths for V to drop from A_V to 0.01A_V 
L_P = 5    # Amount of padding outside trap (for expansion)

L_T = L .+ 2*(L_P + L_V)  # Total grid size
use_cuda = CUDA.functional()
numtype = Float32

println("Parameters Defined")

#------------------------------- Making Arrays and potentials ----------------------

X,K,k2 = MakeArrays(L_T,M);
dX = L_T ./ M  #map(x -> diff(x)[1],X) # Ask Ashton about this
dK = @. 2π / (dX*M)
ksum = Array(K[1]) .+ reshape(Array(K[2]),(1,M[2])) .+ reshape(Array(K[3]),(1,1,M[3]));

V_0 = BoxTrap(X,L,M,L_V,A_V,n_V) |> cu;
ψ_rand = randn(M) .+ im*randn(M) .|> abs |> complex |> cu;

const Pf = prod(@. dX / sqrt(2π)) * plan_fft(copy(ψ_rand));
const Pi! = prod(@. M * dK / sqrt(2π)) * plan_ifft!(copy(ψ_rand));

println("Arrays Defined")

#------------------------------- Finding Ground State ------------------------------

function save_func(res,d)
    wsave("/home/fisto108/Masters-Code/results/" * savename(d,"jld2"),Dict("res" => res))
    #push!(SOLS_GS,res)  
end

GSparams = Dict(
    "title" => "GS $M, $L",
    "ψ" => ψ_rand,
    "γ" => [1],
    "tf" => [20],
    "Nt" => 5
) |> dict_list;

SOLS_GS = []

for (i,d) in enumerate(GSparams)
    tstart = time()
    println("GS sim $i / $(length(GSparams))")
    @unpack ψ, γ, tf, Nt = d
    res = []
    GPU_Solve!(res,GPE!,ψ_rand,LinRange(0,tf,Nt),γ,alg=Tsit5(),plot_progress=false)
    save_func(res,d)
    println(time() - tstart)
end

println("All done baby")

