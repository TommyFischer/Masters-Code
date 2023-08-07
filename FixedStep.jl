using LinearAlgebra,
    FFTW,
    JLD2,
    BenchmarkTools,
    CUDA,
    DrWatson

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
    L = (40,30,20)     # Condensate size
    M = (300,300,300)  # System Grid

    A_V = 30    # Trap height
    n_V = 24    # Trap Power (pretty much always 24)
    L_V = 3     # No. of healing lengths for V to drop from A_V to 0.01A_V 
    L_P = 6     # Amount of padding outside trap (for expansion)

    L_T = L .+ 2*(L_P + L_V)  # Total grid size
    use_cuda = CUDA.functional()
    numtype = Float32
end

@consts begin # Arrays
    X,K,k2 = MakeArrays(L_T,M);
const dX = map(x -> diff(x)[1],Array.(X))  #L_T ./ M 
const dK = map(x -> diff(x)[1],Array.(K)) #@. 2π / (dX*M)
const ksum = Array(K[1]) .+ reshape(Array(K[2]),(1,M[2])) .+ reshape(Array(K[3]),(1,1,M[3]));

const V_0 = BoxTrap(X,L,M,L_V,A_V,n_V) |> cu;
ψ_rand = randn(M) .+ im*randn(M) .|> abs |> complex |> cu;

const Pf = prod(@. dX / sqrt(2π)) * plan_fft(copy(ψ_rand));
const Pi! = prod(@. M * dK / sqrt(2π)) * plan_ifft!(copy(ψ_rand));

