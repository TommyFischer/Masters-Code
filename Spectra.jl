using BenchmarkTools,
SpecialFunctions,
PaddedViews,
UnPack,
TensorCast,
Tullio,
Parameters,
Plots, 
JLD2, 
VortexDistributions, 
FFTW,
CUDA,
QuantumFluidSpectra

import QuantumFluidSpectra.zeropad

include("V5.jl")

begin # Copy-Pasted from fixedstep_turb to define X,K,V etc
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

    X,K,k2 = MakeArrays(L_T,M,use_cuda = false); # need to change V5 to allow kwarg
    const dX = map(x -> diff(x)[1],Array.(X))  
    const dK = map(x -> diff(x)[1],Array.(K)) 
    V_0 = BoxTrap(X,L,M,L_V,A_V,n_V, use_cuda = false) # Same here
end;

begin # Custom QuantumFluidSpectra functions
    @fastmath hypot(a,b,c) = sqrt(a^2 + b^2 + c^2)

    function kdensity_2(k,psi::Psi{3})  
        @unpack ψ,X,K = psi; 
        C = auto_correlate_2(ψ,X,K)
        return sinc_reduce_2(k,X...,C)
    end

    function auto_correlate_2(ψ,X,K)
        n = length(X)
        DX,DK = fft_differentials(X,K)

        ϕ = zeropad(ψ)
        fft!(ϕ)
        @. ϕ *= $prod(DX)

        Threads.@threads for i in eachindex(ϕ)
            ϕ[i] = abs2.(ϕ[i])# .* conj(ϕ[i])
        end

        ifft!(ϕ)
        @. ϕ *= $prod(DK) * (2π)^(n/2)
        return ϕ
    end

    function sinc_reduce_2(k,x,y,z,C)
        dx,dy,dz = x[2]-x[1],y[2]-y[1],z[2]-z[1]
        Nx,Ny,Nz = 2*length(x),2*length(y),2*length(z)
        Lx = x[end] - x[begin] + dx
        Ly = y[end] - y[begin] + dy
        Lz = z[end] - z[begin] + dz
        xp = LinRange(-Lx,Lx,Nx+1)[1:Nx] |> fftshift
        yq = LinRange(-Ly,Ly,Ny+1)[1:Ny] |> fftshift
        zr = LinRange(-Lz,Lz,Nz+1)[1:Nz] |> fftshift
        E = zero(k)
        @tullio E[i] = real(π*sinc(k[i]*hypot(xp[p],yq[q],zr[r])/π)*C[p,q,r]) 
        @. E *= k^2*dx*dy*dz/2/pi^2  
        return E 
    end

    function incompressible_spectrum_2(k,psi::Psi{3})
        @unpack ψ,X,K = psi; 
        wx, wy, wz = velocity(psi)
        @. wx *= abs(ψ)
        @. wy *= abs(ψ)
        @. wz *= abs(ψ)

        wx,wy,wz = helmholtz_incompressible(wx,wy,wz,K...)

        C = auto_correlate_2(wx,X,K)
        wx = nothing
        @. C += $auto_correlate_2(wy,X,K)
        wy = nothing
        @. C += $auto_correlate_2(wz,X,K)
        wz = nothing
        @. C *= 0.5

        return sinc_reduce_2(k,X...,C)
    end

    function compressible_spectrum_2(k,psi::Psi{3})
        @unpack ψ,X,K = psi
        wx,wy,wz = velocity(psi)
        @. wx *= abs(ψ)
        @. wy *= abs(ψ)
        @. wz *= abs(ψ)

        wx,wy,wz = helmholtz_compressible(wx,wy,wz,K...)

        C = auto_correlate_2(wx,X,K)
        wx = nothing
        @. C += $auto_correlate_2(wy,X,K)
        wy = nothing
        @. C += $auto_correlate_2(wz,X,K)
        wz = nothing
        @. C *= 0.5

        return sinc_reduce_2(k,X...,C)
    end

    function qpressure_spectrum_2(k,psi::Psi{3})
        @unpack ψ,X,K = psi
        wx,wy,wz = gradient(abs.(ψ),K)

        C = auto_correlate_2(wx,X,K)
        wx = nothing
        @. C += $auto_correlate_2(wy,X,K)
        wy = nothing
        @. C += $auto_correlate_2(wz,X,K)
        wz = nothing
        @. C *= 0.5

        return sinc_reduce_2(k,X...,C)
    end

    function helmholtz_incompressible(wx, wy, wz, kx, ky, kz)
        wxk = fft(wx); wyk = fft(wy); wzk = fft(wz)
        @cast kw[i,j,k] := (kx[i] * wxk[i,j,k] + ky[j] * wyk[i,j,k] + kz[k] * wzk[i,j,k])/ (kx[i]^2 + ky[j]^2 + kz[k]^2)
        @cast wxkc[i,j,k] := kw[i,j,k] * kx[i]  
        @cast wykc[i,j,k] := kw[i,j,k] * ky[j] 
        @cast wzkc[i,j,k] := kw[i,j,k] * kz[k]  
        wxkc[1] = zero(wxkc[1]); wykc[1] = zero(wykc[1]); wzkc[1] = zero(wzkc[1])

        @. wxk -= wxkc
        @. wyk -= wykc
        @. wzk -= wzkc

        ifft!(wxk)
        ifft!(wyk)
        ifft!(wzk)
    
        return wxk, wyk, wzk
    end

    function helmholtz_compressible(wx, wy, wz, kx, ky, kz)
        wxk = fft(wx); wyk = fft(wy); wzk = fft(wz)
        @cast kw[i,j,k] := (kx[i] * wxk[i,j,k] + ky[j] * wyk[i,j,k] + kz[k] * wzk[i,j,k])/ (kx[i]^2 + ky[j]^2 + kz[k]^2)
        @cast wxkc[i,j,k] := kw[i,j,k] * kx[i]  
        @cast wykc[i,j,k] := kw[i,j,k] * ky[j] 
        @cast wzkc[i,j,k] := kw[i,j,k] * kz[k]  
        wxkc[1] = zero(wxkc[1]); wykc[1] = zero(wykc[1]); wzkc[1] = zero(wzkc[1])

        wxc = ifft(wxkc); wyc = ifft(wykc); wzc = ifft(wzkc)

        return wxc,wyc,wzc
    end

    function gradient(ψ,K)
        kx,ky,kz = K 
        ϕ = fft(ψ)
        ψx = ifft(im*kx.*ϕ)
        ψy = ifft(im*ky'.*ϕ)
        ψz = ifft(im*reshape(kz,1,1,length(kz)).*ϕ)
        return ψx,ψy,ψz
    end

    function incompressible_density_2(k,psi::Psi{3})
        @unpack ψ,X,K = psi 

        wx,wy,wz = velocity(psi)

        @. wx *= abs(ψ)
        @. wy *= abs(ψ)
        @. wz *= abs(ψ)

        wx,wy,wz = helmholtz_incompressible(wx,wy,wz,K...)
        
        @. wx *= exp(im*angle(ψ))
        @. wy *= exp(im*angle(ψ))
        @. wz *= exp(im*angle(ψ))

        C = auto_correlate_2(wx,X,K)
        wx = nothing
        @. C += $auto_correlate_2(wy,X,K)
        wy = nothing
        @. C += $auto_correlate_2(wz,X,K)
        wz = nothing
        @. C *= 0.5

        return sinc_reduce_2(k,X...,C)
    end

    function compressible_density_2(k,psi::Psi{3})
        @unpack ψ,X,K = psi 

        wx,wy,wz = velocity(psi)

        @. wx *= abs(ψ)
        @. wy *= abs(ψ)
        @. wz *= abs(ψ)

        wx,wy,wz = helmholtz_compressible(wx,wy,wz,K...)
        
        @. wx *= exp(im*angle(ψ))
        @. wy *= exp(im*angle(ψ))
        @. wz *= exp(im*angle(ψ))

        C = auto_correlate_2(wx,X,K)
        wx = nothing
        @. C += $auto_correlate_2(wy,X,K)
        wy = nothing
        @. C += $auto_correlate_2(wz,X,K)
        wz = nothing
        @. C *= 0.5

        return sinc_reduce_2(k,X...,C)
    end

    function qpressure_density_2(k,psi::Psi{3})
        @unpack ψ,X,K = psi
        
        wx,wy,wz = gradient(abs.(ψ),K)
        
        @. wx *= exp(im*angle(ψ))
        @. wy *= exp(im*angle(ψ))
        @. wz *= exp(im*angle(ψ))

        C = auto_correlate_2(wx,X,K)
        wx = nothing
        @. C += $auto_correlate_2(wy,X,K)
        wy = nothing
        @. C += $auto_correlate_2(wz,X,K)
        wz = nothing
        @. C *= 0.5

        return sinc_reduce_2(k,X...,C)
    end

end

k = log10range(0.1,40,300)
r = LinRange(0,40,2000)
t = round.(round.(Int, (LinRange(0,10/τ,129) |> collect) ./ 1e-3) .* (1e-3*τ), digits = 3)

psi_strings = ["/ψ_t=$i" for i in t]
load_address = "/Users/fischert/Desktop/"
title = "Shake_Grad=0.2"

nk = []     # Angle averaged momentum density

Ivcs = []   # Angle averaged Incompressible velocity correlation spectra
Cvcs = []   # Angle averaged Compressible velocity correlation spectra
QPcs = []   # Angle averaged Quantum Pressure correlation spectra

Iked = []   # Angle averaged Incompressible kinetic energy density 
Cked = []   # Angle averaged Compressible kinetic energy density
QPed = []   # Angle averaged Quantum Pressure energy density

for filename in psi_strings
    psi = load(load_address*title*filename)["psi"]
    heatmap(abs2.(psi[:,128,:]'),c=:lajolla,clims=(0,1.35)) |> display
    ψ = Psi(psi,Tuple(X),Tuple(K))
    @time push!(nk, kdensity_2(k,ψ))
end

@save "/Users/fischert/Desktop/nk_2.jld2" nk

for filename in psi_strings #### Ivcs
    psi = load(load_address*title*filename)["psi"]
    heatmap(abs2.(psi[:,128,:]'),c=cgrad(:lajolla,rev=true),clims=(0,1.5)) |> display
    ψ = Psi(psi,Tuple(X),Tuple(K))
    @time push!(Ivcs, incompressible_spectrum_2(k,ψ))
end

@save "/Users/fischert/Desktop/Ivcs_2.jld2" Ivcs

for filename in psi_strings #### Cvcs
    psi = load(load_address*title*filename)["psi"]
    heatmap(abs2.(psi[:,128,:]'),c=cgrad(:lajolla,rev=true),clims=(0,1.5)) |> display
    ψ = Psi(psi,Tuple(X),Tuple(K))
    @time push!(Cvcs, compressible_spectrum_2(k,ψ))
end

@save "/Users/fischert/Desktop/Cvcs_2.jld2" Cvcs

for filename in psi_strings #### QPcs
    psi = load(load_address*title*filename)["psi"]
    heatmap(abs2.(psi[:,128,:]'),c=cgrad(:lajolla,rev=true),clims=(0,1.5)) |> display
    ψ = Psi(psi,Tuple(X),Tuple(K))
    @time push!(QPcs, qpressure_spectrum_2(k,ψ))
end

@save "/Users/fischert/Desktop/QPcs_2.jld2" QPcs

for filename in psi_strings #### IKed
    psi = load(load_address*title*filename)["psi"]
    heatmap(abs2.(psi[:,128,:]'),c=cgrad(:lajolla,rev=true),clims=(0,1.5)) |> display
    ψ = Psi(psi,Tuple(X),Tuple(K))
    @time push!(Iked, incompressible_density_2(k,ψ))
end

@save "/Users/fischert/Desktop/Iked_2.jld2" Iked

for filename in psi_strings #### Cked
    psi = load(load_address*title*filename)["psi"]
    heatmap(abs2.(psi[:,128,:]'),c=cgrad(:lajolla,rev=true),clims=(0,1.5)) |> display
    ψ = Psi(psi,Tuple(X),Tuple(K))
    @time push!(Cked, compressible_density_2(k,ψ))
end

@save "/Users/fischert/Desktop/Cked_2.jld2" Cked

for filename in psi_strings #### QPed
    psi = load(load_address*title*filename)["psi"]
    heatmap(abs2.(psi[:,128,:]'),c=cgrad(:lajolla,rev=true),clims=(0,1.5)) |> display
    ψ = Psi(psi,Tuple(X),Tuple(K))
    @time push!(QPed, qpressure_density_2(k,ψ))
end

@save "/Users/fischert/Desktop/QPed_2.jld2" QPed


norm = []
for filename in psi_strings
    psi = load(load_address*filename)["psi"]
    push!(norm,sum(abs2.(psi)))
end

size(norm)

plot(norm,ylims=(0,1.5*norm[1]))
