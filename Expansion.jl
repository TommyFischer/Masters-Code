using JLD2, CUDA, FFTW, DifferentialEquations, LinearAlgebra, DrWatson

use_cuda = CUDA.functional()
include("V5.jl")

@load "/home/fisto108/Nt=10_Shake_Grad=0.04_tf=524.0_title=EscapeTurb (256, 256, 256), (40, 30, 20)_γ=0.jld2"

x = X[1]
y = X[2]'
z = reshape(X[3],(1,1,length(X[3])))

kx = K[1]
ky = K[2]' 
kz = reshape(K[3],(1,1,length(K[3])))

dr = prod(dX)

dx = dX[1]
dy = dX[2]
dz = dX[3]

dkx = dK[1]
dky = dK[2]
dkz = dK[3]

M = (256,256,256)
Mx = M[1]
My = M[2]
Mz = M[3]

ψ_rand = randn(M) .+ im*randn(M) .|> abs |> complex .|> ComplexF32;

begin
    ψ_0 = res[end]
    ϕ_initial = initialise(ψ_0)

    PfArray = [Pfx, Pfy, Pfz]
    PiArray = [Pix!,Piy!,Piz!]
    k = [kx,ky,kz]

    V_0 = zeros(Mx,My,Mz)

    if use_cuda
        V_0 = V_0 |> cu
    end

    xdV = @. x*$ifft(im*kx.*$fft(V_0)) |> real
    ydV = @. y*$ifft(im*ky.*$fft(V_0)) |> real
    zdV = @. z*$ifft(im*kz.*$fft(V_0)) |> real

    if use_cuda
        xdV = xdV |> cu
        ydV = ydV |> cu
        zdV = zdV |> cu
    end

end;

tspan = LinRange(0,20,10);
res_expand = []; 
GPU_Solve!(res_expand,spec_expansion_opt!,ϕ_initial,tspan,0,alg=Tsit5(),plot_progress=false);

res,λx,λy,λz,σx,σy,σz,ax,ay,az = extractinfo(res_expand);

Expansion_params = [λx,λy,λz,σx,σy,σz,ax,ay,az];

wsave("/home/fisto108/Escape_Expansion.jld2",
    Dict("res" => res_expand,
        "X" => Array.(X),
        "K" => Array.(K),
        "dX" => dX,
        "dK" => dK,
        "Expansion_params" => Expansion_params
    ))