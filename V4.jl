# 1/4/23 Spectral expansion is working, now just writing a clean version that can be cuda or normal using a single command + will tidy up

# Syncing is working 

using PlotlyJS,
    SparseArrays,
    StaticArrays,
    LinearAlgebra,
    DifferentialEquations,
    FFTW,
    LaTeXStrings,
    Plots,
    WAV,
    JLD2,
    Makie, 
    #GLMakie,
    CodecZlib,
    BenchmarkTools,
    RecursiveArrayTools,
    CUDA

Threads.nthreads()

begin # Functions for setting up and running simulations

    function number(ψ)
        return sum(abs2.(ψ))*dr*ξ^3*ψ0^2
    end

    function kfunc!(dψ,ψ) # Kinetic Energy term (no rotating frame)
        dψ .= 0.5*(ifft(k2.*fft(ψ)))
        return nothing
    end

    function GPE!(dψ,ψ,var,t) # GPE Equation 
        kfunc_opt!(dψ,ψ)
        @. dψ = -(im + γ)*(0.5*dψ + (V_0 + abs2(ψ) - 1)*ψ)
    end

    function VPE!(dψ,ψ,var,t) # GPE Equation 
        kfunc_opt!(dψ,ψ)
        @. dψ = -(im + γ)*(0.5*dψ + (V_0 +  $V(t) + abs2.(ψ) - 1)*ψ)
    end

    function kfunc_opt!(dψ,ψ)
        mul!(dψ,Pf,ψ)
        dψ .*= k2
        Pi!*dψ
        return nothing
    end
end

begin # Functions for making plots

    function gradsquared(ψ) # Gradient Squared of ψ
        ϕ = fft(ψ)
        out = complex(zeros(M,M,Mz))
    
        for j in (kx,ky',reshape(kz,(1,1,Mz)))
            out .+= abs2.(ifft(im*j.*ϕ))/2
        end
        return real.(out)
    end;     

    function E_Kin(sol) # Kinetic energy over time
        n = length(sol[1,1,1,:])
        E_kin = ones(n)
    
        for i in 1:n
            E_kin[i] = sum(gradsquared(sol[:,:,:,i]))
        end
        return E_kin
    end

    function E_Pot(sol) # Potential Energy over time, V needs to be the same
        n = length(sol[1,1,1,:])
        E_Pot = ones(n)
    
        for i in 1:n
            E_Pot[i] = sum(V_0.*abs2.(sol[:,:,:,i]))
        end
        return E_Pot
    end

    function E_Int(sol) # Interaction Energy over time
        n = length(sol[1,1,1,:])
        E_Int = ones(n)
    
        for i in 1:n
            E_Int[i] = G*sum((abs2.(sol[:,:,:,i])).^2)
        end
        return E_Int
    end

    function Volume1(res,t,isomax=false,opacity=0.5,surface_count=10) # 3D volume plot of solution at t
        vals = res[:,:,:,t];
        X, Y, Z = mgrid(x,y,z);
    
        p = PlotlyJS.volume(
            x=X[:],
            y=Y[:],
            z=Z[:],
            value=vals[:],
            opacity=opacity,
            isomin=0,
            isomax=isomax,
            surface_count=surface_count,
            caps=attr(x_show=false, y_show=false,z_show=false),
            colorscale=:YlOrRd,
            reversescale = true
            )
    
        data = [p];
    
        layout = Layout(title = "t = (sol.t[t])",
                width = 500,
                height = 500, 
                scene_camera=attr(eye=attr(x=0, y=2, z=1)))
    
        pl = PlotlyJS.plot(data,layout)
    
    end

    function MakieVolume(sol,t,alpha=0.12,iso=[0.15],axis=true,col=:oxy)
        if typeof(sol[1]) == Array{ComplexF64, 3} # Checking if input solution is already squared or not
            density_Scaled = abs2.(sol[:,:,:,t])
        else
            density_Scaled = copy(sol[:,:,:,t])
        end
    
        for i in 1:length(sol[1,1,1,:]) # Scaling to 1 to make isovalues easier
            density_Scaled /= maximum(density_Scaled)
        end
    
        Makie.contour(density_Scaled, # Making the plot
        alpha=alpha,
        levels=iso,
        colormap=col,
        show_axis=axis)
    end

    function MakieVolume!(sol,t,alpha=0.12,iso=[0.15],axis=true,col=:oxy)
        if typeof(sol[1]) == Array{ComplexF64, 3} # Checking if input solution is already squared or not
            density_Scaled = abs2.(sol[:,:,:,t])
        else
            density_Scaled = copy(sol[:,:,:,t])
        end
    
        for i in 1:length(sol[1,1,1,:]) # Scaling to 1 to make isovalues easier
            density_Scaled /= maximum(density_Scaled)
        end
    
        Makie.contour!(density_Scaled, # Making the plot
        alpha=alpha,
        levels=iso,
        colormap=col,
        show_axis=axis)
    end

    function MakieGif(sol,title,framerate, alph=0.12, iso=[0.15], axis=true, col=:oxy, )
        if typeof(sol[1]) == Array{ComplexF64, 3} # Checking if input solution is already squared or not
            density_Scaled = abs2.(sol)
        else
            density_Scaled = copy(sol)
        end
    
        for i in 1:length(density_Scaled[1,1,1,:]) # Scaling to 1 to make isovalues easier
            density_Scaled[:,:,:,i] /= maximum(density_Scaled[:,:,:,i])
        end
    
        saveat = joinpath("Gifs",title)
        tindex = Observable(1)
        scene = Makie.contour(Makie.lift(i -> density_Scaled[i],tindex),
        alpha=alph,
        levels=iso,
        colormap=col,
        show_axis=axis)
    
        Makie.record(scene,saveat,1:length(sol.t),framerate = framerate) do i 
            tindex[] = i
        end
    end
end


############################################ Leshgo ##########################################


begin # Adjustable Parameters and constants 

    ħ = 1.05457182e-34
    m = 87*1.66e-27 
    a_s = 5.8e-9 
    k_B = 1.380649e-23
    μ = 2e-9 * k_B
    g = 4π*ħ^2*a_s/m

    ξ = ħ/sqrt(m*μ)
    ψ0 = sqrt(μ/g)

    L = 36 # Box width
    M = 130 # Grid size

    A_V = 15 # Trap height
    n_V = 24 # Trap Power (pretty much always 24)
    L_V = 8 # no. of healing lengths for V to drop to 0.01
    use_cuda = CUDA.functional()

end

begin # Arrays

    x = LinRange(-L/2,L/2,M) |> collect
    y = x' |> collect
    z = reshape(x,(1,1,M)) |> collect

    dx = x[2] - x[1]
    dy = y[2] - y[1]
    dz = z[2] - z[1]

    kx = fftfreq(M,2π/dx) |> collect
    ky = fftfreq(M,2π/dy)' |> collect
    kz = reshape(fftfreq(M,2π/dz),(1,1,M)) |> collect

    dkx = kx[2] - kx[1]
    dky = ky[2] - ky[1]
    dkz = kz[2] - kz[1]

    k2 =  kx.^2 .+ ky.^2 .+ kz.^2 # 3D wave vector
    dr = dx*dy*dz

end

begin # Box Trap Potential

    V_0 = zeros(M,M,M)
    Vboundary(x) = A_V*cos(x/λ)^n_V

    λ = L_V/acos(0.01^(1/n_V))
    
    for i in 1:M, j in 1:M, k in 1:M
        l_x = min(2*L_V,L/2 - abs(x[i])) # Finding the distance to the edge in each dimension, 
        l_y = min(2*L_V,L/2 - abs(y[j])) # discarding if further than 2*L_V
        l_z = min(2*L_V,L/2 - abs(y[k]))

        l = map(Vboundary,(l_x,l_y,l_z))

        V_0[i,j,k] = hypot(l[1],l[2],l[3])
    end
end;

ψ_rand = (randn(M,M,M) + im*randn(M,M,M));
ψ_ones = ones(M,M,M) |> complex;

if use_cuda # Transforming arrays
    x = x |> cu
    y = y |> cu
    z = z |> cu
    V_0 = V_0 |> cu
    kx = kx |> cu
    ky = ky |> cu
    kz = kz |> cu
    k2 = k2 |> cu
    ψ_rand = ψ_rand |> cu
    ψ_ones = ψ_ones |> cu
end;

if use_cuda # For some reason FFTW.MEASURE doesn't work for cuda arrays
    const Pf = dr/(2π)^1.5*plan_fft(copy(ψ_rand));
    const Pi! = M^3*dkx*dky*dkz/(2π)^1.5*plan_ifft!(copy(ψ_rand));
else
    const Pf = dr/(2π)^1.5*plan_fft(copy(ψ_rand),flags=FFTW.MEASURE);
    const Pi! = M^3*dkx*dky*dkz/(2π)^1.5*plan_ifft!(copy(ψ_rand),flags=FFTW.MEASURE);
end

############################################ Finding Ground State ##########################################

begin 
    γ = 1
    tspan = LinRange(0.0,40.0,2); 

    prob = ODEProblem(GPE!,ψ_ones,(tspan[1],tspan[end]))    
    @time sol = solve(prob,saveat=tspan)
end;

size(sol.t)
typeof(sol)
#Plots.plot([number(res[:,:,:,i]) for i in eachindex(sol.t)])

number(sol[end])

res = abs2.(Array(sol));
Plots.heatmap(res[:,75,:,end],clims=(0,1.5),aspectratio=1)

for i in 1:2:length(sol_GS.t)
    P = Plots.heatmap(res[:,:,25,i],clims=(0,1.5))
    display(P)
end


############################################ Creating Turbulence ##########################################


ψ_GS = sol[:,:,:,end];
typeof(ψ_GS)

@save "GS" ψ_GS
@load "GS" ψ_GS

ω_shake = π/2
shakegrid = Array(z).* ones(M,M,M) |> complex;

V(t) = sin(ω_shake*t)*shakegrid
ψ_noise = ψ_GS .+ .01*(randn(M,M,M) .+ im*randn(M,M,M));

if use_cuda
    shakegrid = shakegrid |> cu
    ψ_noise = ψ_noise |> cu
end;

begin 
    γ = 0.0005
    tspan = LinRange(0,75,2)

    prob = ODEProblem(VPE!,ψ_noise,(tspan[1],tspan[end]))    
    @time sol = solve(prob,saveat=tspan)
end;

@save "sol" sol 
@load "sol" sol

size(sol)
Plots.plot([number(sol[:,:,:,i]) for i in eachindex(sol.t)],ylims=(0,5e5))

rizz = abs2.(Array(sol));
riss = angle.(Array(sol));

Plots.heatmap(x,x,rizz[:,75,:,2],clims=(0,1.5),aspectratio=1,c=:thermal)
vline!([-7.1,7.1])
Plots.heatmap(riss[:,5,:,80],clims=(0,3),aspectratio=1)

for i in 1:2:length(sol.t)
    P = Plots.heatmap(x,x,rizz[:,75,:,i],clims=(0,3),aspectratio=1)#,xlims=(-15,15),c=:thermal)
    #Plots.xlims!(-15,15)
    display(P)
    sleep(0.02)
end

############################################ Spectra ##########################################

using QuantumFluidSpectra

@fastmath hypot(a,b,c) = sqrt(a^2 + b^2 + c^2)

X = map(Array,(x,x,x));
K = map(Array,(kx,kx,kx));
ψ = ComplexF64.(Array(sol[end]));

psi = Psi(ψ,X,K);
k = log10range(0.1,10^2,100)#ln.(LinRange(-1,3,100));
E_i = incompressible_spectrum(k,psi);
#E_c = compressible_spectrum(k,psi);

k_L = 2π/(L)# Size of the System
k_l = 2π/(L - 2*L_V) # Size of the condensate accounting for the box trap
k_hmm = 0.44#2π/14
k_ξ = 2π# Healing length
k_dr = 2π/dr^(1/3) # Geometric mean of resolution
k_lol = 2π/hypot(dx,dx,dx)

#klines = [k_L,k_l,k_ξ,k_dr,k_lol];

begin
    P = Plots.plot(k,E_i,axis=:log,ylims=(0.1,10^7),label=false,lw=2,alpha=0.5)
    Plots.plot!(x->(2e3)*x^-3,[x for x in k[40:75]],label=false,alpha=1,lw=.5)
    Plots.plot!(x->(2e2)*x^0,[x for x in k[15:50]],label=false,alpha=1,lw=.5)

    #for i in klines
    #    vline!([i], label = (@Name i),linestyle=:dash,alpha=0.5)
    #end
    
    vline!([k_L], label = L"$k_L$",linestyle=:dash,alpha=0.5)
    vline!([k_l], label = L"$k_l$",linestyle=:dash,alpha=0.5)
    vline!([k_hmm], label = L"$k_{?}$",linestyle=:dash,alpha=0.5)
    vline!([k_ξ], label = L"$k_\xi$",linestyle=:dash,alpha=0.5)
    vline!([k_dr], label = L"$k_{dr}$",linestyle=:dash,alpha=0.5)
    vline!([k_lol], label = L"$k_{lol}$",linestyle=:dash,alpha=0.5)
end

fftpsi = log.(abs2.(fftshift(fft(sol[end]))));


############################################ Expansion ##########################################


begin # Expansion Functions

    function initialise(ψ)
        global Na = number(ψ)
        ϕi = ψ ./ sqrt(Na);

        global ax2 = dr*sum(@. x^2*abs2(ϕi))
        global ay2 = dr*sum(@. y^2*abs2(ϕi))
        global az2 = dr*sum(@. z^2*abs2(ϕi))

        σi = real.(0.5*im*dr.*CurrentDensity(ϕi)./(ax2,ay2,az2))
        @. ϕi *= exp(-0.5*im*(σi[1]*x^2 + σi[2]*y^2 + σi[3]*z^2))

        if use_cuda
            ϕi, σi = ϕi, σi |> cu
        end

        global Pfx = dx/sqrt(2π)*plan_fft(copy(ϕi),1)
        global Pfy = dy/sqrt(2π)*plan_fft(copy(ϕi),2)
        global Pfz = dz/sqrt(2π)*plan_fft(copy(ϕi),3)
    
        global Pix! = M*dkx/sqrt(2π)*plan_ifft!(copy(ϕi),1)
        global Piy! = M*dky/sqrt(2π)*plan_ifft!(copy(ϕi),2)
        global Piz! = M*dkz/sqrt(2π)*plan_ifft!(copy(ϕi),3)

        ϕ_initial =  ArrayPartition(ϕi,[1,1,1,σi[1],σi[2],σi[3]])
        return ϕ_initial
    end

    function extractinfo(sol)
        λx = [sol[:,i].x[2][1] for i in eachindex(t)]
        λy = [sol[:,i].x[2][2] for i in eachindex(t)]
        λz = [sol[:,i].x[2][3] for i in eachindex(t)]
        
        σx = [sol[:,i].x[2][4] for i in eachindex(t)]
        σy = [sol[:,i].x[2][5] for i in eachindex(t)]
        σz = [sol[:,i].x[2][6] for i in eachindex(t)]
    
        ax = @. sqrt(ax2*λx^2)
        ay = @. sqrt(ay2*λy^2)
        az = @. sqrt(az2*λz^2)
    
        ϕ = [sol[:,i].x[1] for i in eachindex(t)]
        res = [abs2.(ϕ) for i in eachindex(solt2.t)];

        return res,ϕ,λx,λy,λz,σx,σy,σz,ax,ay,az
    end

    function CurrentDensity(ψ)
        ϕ = fft(ψ)
        ϕim = fft(conj.(ψ))
        
        σx = x.*(ψ.*ifft(im*kx.*ϕim) - conj(ψ).*ifft(im*kx.*ϕ)) |> sum
        σy = y.*(ψ.*ifft(im*ky.*ϕim) - conj(ψ).*ifft(im*ky.*ϕ)) |> sum
        σz = z.*(ψ.*ifft(im*kz.*ϕim) - conj(ψ).*ifft(im*kz.*ϕ)) |> sum

        return [σx,σy,σz]
    end

    function kfunc2!(dϕ,ϕ,λ)
        mul!(dϕ,Pf,ϕ)
        dϕ .*= (k[1]./λ[1]).^2 .+ (k[2]./λ[2]).^2 .+ (k[3]./λ[3]).^2
        Pi!*dϕ
    end

    function ρ2(λ,σ)
        λx,λy,λz = λ |> real
        σx,σy,σz = σ |> real
        return @. x^2*λx*σx + y^2*λy*σy + z^2*λz*σz
    end

    function firstOrder!(dϕ,ϕ,dσ,λ,i)  
        mul!(dϕ,PfArray[i],ϕ)
        dϕ .*= im*k[i]
        dσ[i] = λ[i]^(-2) * sum(abs2.(PiArray[i]*dϕ))
    end

    function spec_expansion_opt!(du,u,p,t)
        ϕ = u.x[1]
        λ = u.x[2][1:3]
        σ = u.x[2][4:6]

        dϕ = du.x[1]
        du.x[2][1:3] = σ
        dσ = du.x[2][4:6]

        λ̄³ = prod(λ) 

        # dσ/dt

        firstOrder!(dϕ,ϕ,dσ,λ,1) # Computing 3rd term in dσ equations
        firstOrder!(dϕ,ϕ,dσ,λ,2)
        firstOrder!(dϕ,ϕ,dσ,λ,3)

        dσ .-= map(i -> sum(abs2.(ϕ).*i),(xdV,ydV,zdV)) # Computing second terms
        dσ .+= (0.5*Na/λ̄³)*sum(abs2.(ϕ).^2) # Computing first terms
        dσ .*= dr ./ (λ[1]*ax2,λ[2]*ay2,λ[3]*az2) # Coefficients

        # dϕ/dt
        
        kfunc2!(dϕ,ϕ,λ) # Kinetic Term
        dϕ .= -im*(0.5.*dϕ .+ (V_0 .+ (Na/λ̄³)*abs2.(ϕ) .+ 0.5*ρ2(λ,dσ)).*ϕ)
        
        du.x[1] .= dϕ
        du.x[2][4:6] .= dσ

    end 

end

@time Pfx*ψ_GS;



begin

    ψ_0 = sol[end]
    ϕ_initial = initialise(ψ_0)

    PfArray = [Pfx, Pfy, Pfz]
    PiArray = [Pix!,Piy!,Piz!]
    k = [kx,ky,kz]

    V_0 = zeros(M,M,M)

    xdV = V_0#x.*ifft(im*kx.*fft(V_0)) .|> real
    ydV = V_0#y.*ifft(im*ky.*fft(V_0)) .|> real
    zdV = V_0#z.*ifft(im*kz.*fft(V_0)) .|> real

    if use_cuda
        xdV = xdV |> cu
        ydV = ydV |> cu
        zdV = zdV |> cu
    end

end;

t = LinRange(0,10,2);

probs = ODEProblem(spec_expansion!,ϕ_initial,(t[1],t[end]));
@time solt2 = solve(probs,saveat=t,abstol=1e-6,reltol=1e-6);

res,ϕ,λx,λy,λz,σx,σy,σz,ax,ay,az = extractinfo(solt2);

Norm = [ξ^3*ψ0^2*dr*sum(res[i]) for i in eachindex(solt2.t)]

Plots.plot(σx,ylims=(0.0,.3))
Plots.plot(Norm,ylims=(0,1.5))

for i in 1:length(solt2.t)
    P = Plots.heatmap(x,x,res[i][:,25,:],aspectratio=1)#,clims=(0,2e0),c=:thermal)
    display(P)
    sleep(0.01)
end

