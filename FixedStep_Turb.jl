using LinearAlgebra,
FFTW,
JLD2,
BenchmarkTools,
CUDA,
Adapt,
DrWatson,
Parameters

#-------- Setup -----------

#using Hwloc, CpuId
#(cpuvendor() == :Intel && FFTW.get_provider() == "fftw") ? (FFTW.set_provider!("fftw"); exit()) : #(FFTW.forget_wisdom(); FFTW.set_num_threads(num_physical_cores()))

include("V5.jl")

@consts begin # Physical Constants
    ħ = 1.05457182e-34
    m = 87*1.66e-27 
    a_s = 5.8e-9 
    k_B = 1.380649e-23

    μ = 1e-9 * k_B #ħ # Smaller μ, bigger norm
    g = 4π*ħ^2*a_s/m
    ξ = ħ/sqrt(m*μ)
    ψ0 = sqrt(μ/g) #sqrt(N/ξ^3) 
    τ = ħ/μ
end

@consts begin # Numerical Constants
    Δt = 1e-3       # Timestep, #2.5e-5
    L = (40,30,20)     # Condensate size
    M = (256,256,256)  # System Grid

    A_V = 30    # Trap height
    n_V = 24    # Trap Power (pretty much always 24)
    L_V = 3     # No. of healing lengths for V to drop from A_V to 0.01A_V 
    L_P = 7     # Amount of padding outside trap (for expansion)

    L_T = L .+ 2*(L_P + L_V)  # Total grid size
    use_cuda = CUDA.functional()
    numtype = ComplexF64
end

ψ_rand = adapt(CuArray,load("/nesi/nobackup/uoo03837/Final_res/noisetest/ψ_t=0.0")["psi"])
#ψ_rand = adapt(CuArray,load("/home/fisto108/ψ_01_GS")["psi"])
#ψ_rand = adapt(CuArray,randn(M) .+ im*randn(M)  .|> abs |> complex); # Initial State

begin # Arrays
    X,K,k2 = MakeArrays(L_T,M,use_cuda = false); # need to change V5 to allow kwarg
    const dX = map(x -> diff(x)[1],Array.(X))  
    const dK = map(x -> diff(x)[1],Array.(K)) 
    V_0 = BoxTrap(X,L,M,L_V,A_V,n_V, use_cuda = false) # Same here

    const Pf = prod(@. dX / sqrt(2π)) * plan_fft(ψ_rand)
    const Pf! = prod(@. dX / sqrt(2π)) * plan_fft!(ψ_rand)
    const Pi! = prod(@. M * dK / sqrt(2π)) * plan_ifft!(ψ_rand)
end;

#δ = 3
#V_D = 2.5
#V_damp = -im * V_D * [(abs(i) > 0.5*L[1] + L_V + δ) || (abs(j) > 0.5*L[2] + L_V + δ) || (abs(k) > 0.5*L[3] + L_V + δ) ? 1 : 0 for i in Array(X[1]), j in Array(X[2]), k in Array(X[3])]; 

const γ = 0
#const Var = adapt(tgamma + i something)

const U = adapt(CuArray,@. exp(-(im + γ)*k2*Δt/4));
const V_static = adapt(CuArray, @. V_0 - 1);#+ V_damp); # V_Trap + V_TrapDamping - μ
const ψI = deepcopy(ψ_rand);
const ψK = deepcopy(ψ_rand);

## Getting Shake_Grad
Uvals = readdir("/home/fisto108/Uvals/")
const Shake_Grad = parse(Float64,Uvals[1])
rm("/home/fisto108/Uvals/$(Uvals[1])")
touch("/home/fisto108/GPUJOBS/Shake_Grad=$Shake_Grad")

const ω_shake = 2π * 4τ # 2π * 4 Hz in dimensionless time units
const shakegrid = adapt(CuArray, reshape(Array(X[3]),(1,1,M[3])) .* ones(M) |> complex);  
V(t) = sin(ω_shake*t)*Shake_Grad * shakegrid # Test if adding a dot here improves performance

#-------- Finding Turbulent State --------------

GSparams = Dict(
    "title" => "Doesn't matter $M, $L",
    "ψ" => ψ_rand,
    "tf" => 10/τ,
    "Ns" => 129
) |> dict_list;

for (i,d) in enumerate(GSparams)
    @unpack ψ, tf, Ns = d
    
    tsaves = LinRange(0,tf,Ns) |> collect
    Shake!(ψ,tsaves,save_to_file = "/nesi/nobackup/uoo03837/Final_res/256HamilTest/") # Add wsave to evolve function
    #@time global res = GroundState!(ψ,tsaves ,save_to_file = "/nesi/nobackup/uoo03837/Final_res/Tests/") # Add wsave to evolve function
end

#@save "/nesi/nobackup/uoo03837/ψs.jld2" res
