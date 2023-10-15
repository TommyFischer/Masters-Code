using LinearAlgebra,
FFTW,
JLD2,
BenchmarkTools,
CUDA,
Adapt,
DrWatson,
Parameters,
Plots

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

    A_V = 133/2    # Trap height
    n_V = 24    # Trap Power (pretty much always 24)
    L_V = 3     # No. of healing lengths for V to drop from A_V to 0.01A_V 
    L_P = 8     # Amount of padding outside trap (for expansion)

    L_T = (2R,2R,L) .+ 2*(L_P + L_V)  # Total grid size
    use_cuda = CUDA.functional()
    numtype = ComplexF64
end
A_V

N = 1.1e5
ψ_GS = load("/home/fisto108/ψ_t=130.92")["psi"]
ψ_GS ./= sqrt(ψ0^2 * ξ^3 * prod(dX) / N * sum(abs2.(ψ_GS))) # Enforcing N ≈ 1.2e5
sum(abs2.(ψ_GS))*prod(dX) * ψ0^2 * ξ^3 / N
ψ_rand = adapt(CuArray,ψ_GS);

begin # Arrays
    X,K,k2 = MakeArrays(L_T,M,use_cuda = false); # need to change V5 to allow kwarg
    const dX = map(x -> diff(x)[1],Array.(X))  
    const dK = map(x -> diff(x)[1],Array.(K)) 
    V_0 = CylinderTrap(X,L,R,M,L_V,A_V,n_V, use_cuda = false) # Same here

    const Pf = prod(@. dX / sqrt(2π)) * plan_fft(ψ_rand)
    const Pf! = prod(@. dX / sqrt(2π)) * plan_fft!(ψ_rand)
    const Pi! = prod(@. M * dK / sqrt(2π)) * plan_ifft!(ψ_rand)
end;

δ = 7
V_D = 5
V_damp = -im * V_D * [(sqrt(i^2 + j^2) > R + L_V + δ) || (abs(k) > 0.5*L + L_V + δ) ? 1 : 0 for i in Array(X[1]), j in Array(X[2]), k in Array(X[3])]; 

const γ = 0
const U = adapt(CuArray,@. exp(-(im + γ)*k2*Δt/4));
const V_static = adapt(CuArray, @. V_0 + V_damp); # V_Trap + V_TrapDamping
const ψI = deepcopy(ψ_rand);
const ψK = deepcopy(ψ_rand);
const Shake_Grad = 1.25 / L # F_0 = 2.5nK * kB
const ω_shake = 2π * 9τ # 2π * 8Hz in dimensionless time units
const shakegrid = adapt(CuArray, reshape(Array(X[3]),(1,1,M[3])) .* ones(M) |> complex);  
V(t) = sin(ω_shake*t)*Shake_Grad * shakegrid 

CUDA.memory_status()
Sys.free_memory() / 2^20 / 10^3 # 47  GB CPU Memory 
#GC.gc()

shake_tsaves = LinRange(0,6.0/τ,40)
psi = Shake!(ψ_rand,shake_tsaves)

@save "Trap_Escape_UD=133kB" psi

norm = [sum(abs2.(i)) * prod(dX) * ξ^3 * ψ0^2 for i in psi]
plot(shake_tsaves.*τ,norm[1] .- norm, ylims = (0,7e4))
hline!([4.5e4])

yvals = [2e4, 3e4, 5e4, 5.5e4, 6.5e4, 7e4]
xvals = [1.5, 2, 3, 4, 5, 6]

scatter!(xvals,yvals)

heatmap(abs2.(psi[6][:,:,128]),aspectratio=1)



