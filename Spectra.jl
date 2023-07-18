begin

    using BenchmarkTools
    using SpecialFunctions
    using PaddedViews
    using UnPack
    using TensorCast

    abstract type Field end
    struct Psi{D} <: Field
        ψ::Array{Complex{Float32},D}
        X::NTuple{D}
        K::NTuple{D}
    end

    function log10range(a,b,n)
        @assert a>0
        x = LinRange(log10(a),log10(b),n)
        return @. 10^x
    end

    function zeropad(A)
        S = size(A)
        if any(isodd.(S))
            error("Array dims not divisible by 2")
        end
        nO = 2 .* S
        nI = S .÷ 2

        outer = []
        inner = []

        for no in nO
            push!(outer,(1:no))
        end

        for ni in nI
            push!(inner,(ni+1:ni+2*ni))
        end

        return PaddedView(zero(eltype(A)),A,Tuple(outer),Tuple(inner)) |> collect
    end

    function fft_differentials(X,K)
        M = length(X)
        DX = zeros(M); DK = zeros(M)
        for i ∈ eachindex(X)
            DX[i],DK[i] = dfft(X[i],K[i])
        end
        return DX,DK
    end

    function dfft(x,k)
        dx = x[2]-x[1]; dk = k[2]-k[1]
        Dx = dx/sqrt(2*pi)
        Dk = length(k)*dk/sqrt(2*pi)
        return Dx, Dk
    end

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

    function velocity(psi::Psi{3})
        @unpack ψ = psi
        rho = abs2.(ψ)
        ψx,ψy,ψz = gradient(psi)
        vx = @. imag(conj(ψ)*ψx)/rho
        vy = @. imag(conj(ψ)*ψy)/rho
        vz = @. imag(conj(ψ)*ψz)/rho
        @. vx[isnan(vx)] = zero(vx[1])
        @. vy[isnan(vy)] = zero(vy[1])
        @. vz[isnan(vz)] = zero(vz[1])
        return vx,vy,vz
    end

    function gradient(ψ,K)
        kx,ky,kz = K 
        ϕ = fft(ψ)
        ψx = ifft(im*kx.*ϕ)
        ψy = ifft(im*ky'.*ϕ)
        ψz = ifft(im*reshape(kz,1,1,length(kz)).*ϕ)
        return ψx,ψy,ψz
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
        
        wx,wy,wz = gradient(abs2.(ψ),K)
        
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

using Plots, JLD2, Parameters, VortexDistributions, FFTW, Tullio

@load "Desktop/Nt=100_Shake_Grad=0.1_tf=786.0_title=EscapeTurb (256, 256, 256), (40, 30, 20)_γ=0.jld2"

@fastmath hypot(a,b,c) = sqrt(a^2 + b^2 + c^2)
k = log10range(0.1,30,20)
times = [1,2,3,4,5,10,15,20,40,60,80,100]

nkdata = []
push!(nkdata,k)

for i in times
    psi = Psi(res[i],Tuple(X),Tuple(K))
    @time push!(nkdata,kdensity2(k,psi))
end

@save "/home/fisto108/nkdata015.jld2" nkdata

Eidata = []
Ecdata = []
EQdata = []

for i in times
    psi = Psi(res[i],Tuple(X),Tuple(K))
    @time push!(Eidata,incompressible_spectrum2(k,psi))
    @time push!(Ecdata,compressible_spectrum2(k,psi))
    @time push!(EQdata,qpressure_spectrum2(k,psi))
end

spectra015 = [k, Eidata, Ecdata, EQdata]
@save "/home/fisto108/spectra015.jld2" spectra015


ID_data = []
CD_data = []
QPD_data = []

for i in times
    psi = Psi(res[i],Tuple(X),Tuple(K))    
    @time push!(ID_data,incompressible_density2(k,psi))
    @time push!(CD_data,compressible_density2(k,psi))
    @time push!(QPD_data,qpressure_density2(k,psi))
end

density015 = [k, ID_data, CD_data, QPD_data]
@save "/home/fisto108/density015.jld2" density015

