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
    M = (64,64,64)  # System Grid

    A_V = 30    # Trap height
    n_V = 24    # Trap Power (pretty much always 24)
    L_V = 3     # No. of healing lengths for V to drop from A_V to 0.01A_V 
    L_P = 6     # Amount of padding outside trap (for expansion)

    L_T = L .+ 2*(L_P + L_V)  # Total grid size
    use_cuda = CUDA.functional()
    numtype = ComplexF64
end

#ψ_rand = adapt(CuArray,load("/nesi/nobackup/uoo03837/Final_res/Tests/ψ_noisyGS")["psi"])
ψ_rand = adapt(CuArray,randn(M) .+ im*randn(M)  .|> abs |> complex); # Initial State

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

const γ = 1
#const Var = adapt(tgamma + i something)

const U = adapt(CuArray,@. exp(-(im + γ)*k2*Δt/4));
const V_static = adapt(CuArray, @. V_0 - 1);#+ V_damp); # V_Trap + V_TrapDamping - μ
const ψI = deepcopy(ψ_rand);
const ψK = deepcopy(ψ_rand);

const Shake_Grad = 0.1 
const ω_shake = 2π * 0.03055 # could potentially be packaged with other consts on gpu
const shakegrid = adapt(CuArray, reshape(Array(X[3]),(1,1,M[3])) .* ones(M) |> complex);  
V(t) = sin(ω_shake*t)*Shake_Grad * shakegrid # Test if adding a dot here improves performance

#-------- Finding Ground State --------------
begin
    function GroundState!(ψ::CuArray{ComplexF64, 3},tsaves; save_to_file = false) # save_to_file: if a string, does not create solution object and saves solutions to file given by string. If false creates solution object and returns it
        tstart = time()

        if save_to_file == false
            ψs = [zero(Array(ψ)) for _ in 1:length(tsaves)]
            ψs[1] .= Array(ψ);
        else
            psi = Array(ψ)
            @save save_to_file*"ψ_initial.jld2" psi
        end

        t=0.
        tsteps = @. round(Int, tsaves / Δt)

        for i in 1:tsteps[end]
            rk4ip!(ψ)
            t += Δt

            if i in tsteps
                n = findall(x -> x == i, tsteps)[1]
                println("Save  $(n - 1) / $(length(tsteps) - 1) at $(round(t,digits=3)). Time taken: $(time() - tstart)")

                if save_to_file == false
                    ψs[n] .= Array(ψ);
                else
                    psi = Array(ψ)
                    @save save_to_file*"ψ_t=$(round(t,digits=3))" psi
                end
            end
        end

        if save_to_file == false
                return ψs
        end
    end

    G!(ϕ::CuArray{ComplexF64, 3},ψ::CuArray{ComplexF64, 3}) = begin 
        @. ϕ = Var * (abs2(ψ) + V_static) * ψ # Should test absorbing -(im + γ)*Δt into one gpu variable to see if there's a speedup
    end 
    
    G!(ψ::CuArray{ComplexF64, 3},t::Float64) = begin
        @. ψ *= Var * (abs2(ψ) + V_static + $V(t))
    end
    
    G!(ψ::CuArray{ComplexF64, 3}) = begin
        @. ψ *= Var * (abs2(ψ) + V_static)
    end
     
    D!(ϕ::CuArray{ComplexF64, 3},ψ::CuArray{ComplexF64, 3}) = begin 
        mul!(ϕ,Pf,ψ) 
        @. ϕ *= U  
        Pi!*ϕ 
    end
    
    D!(ψ::CuArray{ComplexF64, 3}) = begin 
        Pf!*ψ
        @. ψ *= U
        Pi!*ψ
    end
end

Var = adapt(CuArray,[-(im + γ)*Δt])
GSparams = Dict(
    "title" => "GS $M, $L",
    "ψ" => ψ_rand,
    "tf" => [50],
    "Ns" => 51
) |> dict_list;

for (i,d) in enumerate(GSparams)
    @unpack ψ, tf, Ns = d
    
    tsaves = LinRange(0,tf,Ns) |> collect
    #@time global res = Shake!(ψ,tsaves)#,save_to_file = "/nesi/nobackup/uoo03837/Final_res/Tests/") # Add wsave to evolve function
    @time global res = GroundState!(ψ,tsaves)#,save_to_file = "/nesi/nobackup/uoo03837/Final_res/Tests/") # Add wsave to evolve function
end

#@save "/nesi/nobackup/uoo03837/ψs.jld2" res

#exit()
1