using LinearAlgebra,
FFTW,
JLD2,
BenchmarkTools,
CUDA,
Adapt,
DrWatson,
Parameters

#-------- Setup -----------

include("V5.jl")

@consts begin # Physical Constants
    ħ = 1.05457182e-34
    m = 87*1.66e-27 
    a_s = 5.8e-9 
    k_B = 1.380649e-23

    μ = 2e-9 * k_B #ħ # Smaller μ, bigger norm
    g = 4π*ħ^2*a_s/m
    ξ = ħ/sqrt(m*μ)
    ψ0 = sqrt(μ/g) #sqrt(N/ξ^3) 
    τ = ħ/μ
end

@consts begin # Numerical Constants
    Δt = 1e-3       # Timestep, #2.5e-5
    L = 27e-6 / ξ   # Condensate Length
    R = 16e-6 / ξ   # Condensate Radius 
    M = (256,256,256)  # System Grid

    A_V = 30    # Trap height
    n_V = 24    # Trap Power (pretty much always 24)
    L_V = 3     # No. of healing lengths for V to drop from A_V to 0.01A_V 
    L_P = 7     # Amount of padding outside trap (for expansion)

    L_T = (2R,2R,L) .+ 2*(L_P + L_V)  # Total grid size
    use_cuda = CUDA.functional()
    numtype = ComplexF64
end

ψ_rand = adapt(CuArray,load("/nesi/nobackup/uoo03837/Recreating/Navon/GS/ψ_t=130.92")["psi"])
#ψ_rand = adapt(CuArray,randn(M) .+ im*randn(M)  .|> abs |> complex); # Initial State

begin # Arrays
    X,K,k2 = MakeArrays(L_T,M,use_cuda = false); # need to change V5 to allow kwarg
    const dX = map(x -> diff(x)[1],Array.(X))  
    const dK = map(x -> diff(x)[1],Array.(K)) 
    V_0 = CylinderTrap(X,L,R,M,L_V,A_V,n_V, use_cuda = false) # Same here

    const Pf = prod(@. dX / sqrt(2π)) * plan_fft(ψ_rand)
    const Pf! = prod(@. dX / sqrt(2π)) * plan_fft!(ψ_rand)
    const Pi! = prod(@. M * dK / sqrt(2π)) * plan_ifft!(ψ_rand)
end;

δ = 3
V_D = 2.5
V_damp = -im * V_D * [(sqrt(i^2 + j^2) > R + L_V + δ) || (abs(k) > 0.5*L + L_V + δ) ? 1 : 0 for i in Array(X[1]), j in Array(X[2]), k in Array(X[3])]; 

const γ = 0
const U = adapt(CuArray,@. exp(-(im + γ)*k2*Δt/4));
const V_static = adapt(CuArray, @. V_0 - 1 + V_damp); # V_Trap + V_TrapDamping - μ
const ψI = deepcopy(ψ_rand);
const ψK = deepcopy(ψ_rand);
Shake_Grad = 1
const ω_shake = 2π * 8τ # 2π * 8Hz in dimensionless time units
const shakegrid = adapt(CuArray, reshape(Array(X[3]),(1,1,M[3])) .* ones(M) ./ L |> complex);  

#-------- Finding Turbulent State --------------

for i in 2:10
    noisegrid = 0.01*[ (sqrt(i^2 + j^2) < R)  && (abs(k) < 0.5*L) ? randn() + im*randn() : 0 + 0*im for i in X[1], j in X[2], k in X[3]]
    ψ = adapt(CuArray,load("/nesi/nobackup/uoo03837/Recreating/Navon/GS/ψ_t=130.92")["psi"] .+ noisegrid)
    shake_tsaves = LinRange(0,2/τ,20)
    relax_tsaves = LinRange(0,1.5/τ,20)

    global Shake_Grad = 1
    global V(t) = sin(ω_shake*t)*Shake_Grad * shakegrid 
    println("shakin")
    psi = Shake!(ψ,shake_tsaves)
    @save "/nesi/nobackup/uoo03837/Recreating/Navon/Shake_run_$i" psi
    psi = nothing
    GC.gc()

    global Shake_Grad = 0
    global V(t) = sin(ω_shake*t)*Shake_Grad * shakegrid 
    println("chillin out maxin relaxin all cool")
    psi = Shake!(ψ,relax_tsaves)
    @save "/nesi/nobackup/uoo03837/Recreating/Navon/Relax_run_$i" psi
    psi = nothing
    GC.gc()
end



