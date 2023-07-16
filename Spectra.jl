using QuantumFluidSpectra, Plots, JLD2, Parameters, VortexDistributions, FFTW

@fastmath hypot(a,b,c) = sqrt(a^2 + b^2 + c^2)

k = log10range(0.1,30,600)

function kdensity(k,psi::Psi{3})  
    @unpack ψ,X,K = psi; 
	C = auto_correlate(ψ,X,K)
    return sinc_reduce(k,X...,C)
end

@load "/nesi/nobackup/uoo03837/results3/Nt=100_Shake_Grad=0.015_tf=786.0_title=EscapeTurb (300, 300, 300), (40, 30, 20)_γ=0.jld2"

nkdata = []
push!(nkdata,k)

for i in [1,2,3,4,5,10,20,40,60,80,100]
    psi = Psi(ComplexF64.(res[i]),(Float64.(X[1]),Float64.(X[2]),Float64.(X[3])),(Float64.(K[1]),Float64.(K[2]),Float64.(K[3])))
    @time push!(nkdata,kdensity(k,psi))
end

@save "/home/fisto108/nkdata015.jld2" nkdata

Eidata = []
Ecdata = []
EQdata = []

for i in times
    psi = Psi(ComplexF64.(res[i]),(Float64.(X[1]),Float64.(X[2]),Float64.(X[3])),(Float64.(K[1]),Float64.(K[2]),Float64.(K[3])))
    @time push!(Eidata,incompressible_spectrum(k,psi))
    @time push!(Ecdata,compressible_spectrum(k,psi))
    @time push!(EQdata,qpressure_spectrum(k,psi))
end

spectra015 = [k, Eidata, Ecdata, EQdata]

@save "/home/fisto108/spectra015.jld2" spectra015

ID_data = []
CD_data = []
QPD_data = []

for i in times
    psi = Psi(ComplexF64.(res[i]),(Float64.(X[1]),Float64.(X[2]),Float64.(X[3])),(Float64.(K[1]),Float64.(K[2]),Float64.(K[3])))

    @time push!(ID_data,incompressible_density(k,psi))
    @time push!(CD_data,compressible_density(k,psi))
    @time push!(QPD_data,qpressure_density(k,psi))
end

density015 = [k, ID_data, CD_data, QPD_data]

@save "/home/fisto108/density015.jld2" density015