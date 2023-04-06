# 1/4/23 Spectral expansion is working, now just writing a clean version that can be cuda or normal using a single command + will tidy up

# Last Edit:  12am Tuesday 7th April

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

    Ek(ψ) = 0.5*ξ^3*ψ0^2 * dr*sum(abs2.(ifft(im*ksum.*fft(ψ)))) # E_Kinetic / μ
    Ep(ψ,V) = ξ^3*ψ0^2 * dr*sum((V_0 .+ V).*abs2.(ψ)) |> real # E_Potential / μ
    Ei(ψ) = 0.5*ξ^3*ψ0^2 * dr*sum(abs2.(ψ).^2) # E_Interaction / μ

    Ekx(ψ) = 0.5*ξ^3*ψ0^2 * dr*sum(abs2.(ifft(im*Array(kx).*fft(ψ)))) # x-direction E_k
    Eky(ψ) = 0.5*ξ^3*ψ0^2 * dr*sum(abs2.(ifft(im*Array(ky).*fft(ψ)))) # y-direction E_k
    Ekz(ψ) = 0.5*ξ^3*ψ0^2 * dr*sum(abs2.(ifft(im*Array(kz).*fft(ψ)))) # z-direction E_k

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
    τ = ħ/μ

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
    ksum = kx .+ ky .+ kz
end;

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
    const Pf = Float32(dr/(2π)^1.5)*plan_fft(copy(ψ_rand));
    const Pi! = Float32(M^3*dkx*dky*dkz/(2π)^1.5)*plan_ifft!(copy(ψ_rand));
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

typeof(res)
res = Array(sol);

Ei(res[:,:,:,2])*1e-4
Ek(res[:,:,:,2])*1e-4
Ep(res[:,:,:,2],Array(V_0))*1e-4

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

ΔU = 1.5
ω_shake = 2π * 0.03055 
shakegrid = ΔU * Array(z)./(L/2) .* ones(M,M,M) |> complex;

V(t) = sin(ω_shake*t)*shakegrid
ψ_noise = ψ_GS .+ .01*(randn(M,M,M) .+ im*randn(M,M,M));

if use_cuda
    shakegrid = shakegrid |> cu
    ψ_noise = ψ_noise |> cu
end;

begin 
    γ = 0.0005
    tspan = LinRange(400,500,30)

    prob = ODEProblem(VPE!,ψ,(tspan[1],tspan[end]))    
    @time sol = solve(prob,saveat=tspan)
end;

size(sol)
typeof(sol)
ψ = sol[:,:,:,end] |> cu;
typeof(ψ)

#sol1 = Array(sol); done
#sol2 = Array(sol); done
#sol3 = Array(sol); done
#sol4 = Array(sol); done 
#sol5 = Array(sol); done 

E_k = zeros(150);
E_p = zeros(150);
E_i = zeros(150);

res = map(x->abs2.(Array(x)),[sol1,sol2,sol3,sol4,sol5]);

for i in 1:5
    vec = res[i]
    for j in 1:30
        E_k[30*(i-1) + j] = Ek(vec[:,:,:,j])
        E_p[30*(i-1) + j] = Ep(vec[:,:,:,j],V(30*(i-1) + j))
        E_i[30*(i-1) + j] = Ei(vec[:,:,:,j])
    end
end


Plots.plot(LinRange(0,500,150),E_k,label=L"Ek",ylims=(0,3e5))
Plots.plot!(LinRange(0,500,150),E_p,label=L"Ep")
Plots.plot!(LinRange(0,500,150),E_i,label=L"Ei")
Plots.plot!(E_k .+ E_i .+ E_p,label=L"E_{total}")

Plots.heatmap(abs2.(sol1[:,65,:,30]),clims=(0,3),aspectratio=1)

@save "sol" sol 
@load "sol" sol

CUDA.memory_status()
size(sol)
res = Array(sol);
#Plots.plot([number(sol[:,:,:,i]) for i in eachindex(sol.t)],ylims=(0,5e5))

rizz = abs2.(Array(sol));
riss = angle.(Array(sol));

Plots.heatmap(x,x,rizz[:,65,:,,:],clims=(0,1.5),aspectratio=1,c=:greys)
vline!([-7.1,7.1])
Plots.heatmap(riss[:,5,:,80],clims=(0,3),aspectratio=1)

E_K = [Ek(res[:,:,:,i]) for i in eachindex(sol.t)];

E_Kx = [Ekx(res[:,:,:,i]) for i in eachindex(sol.t)];
E_Ky = [Eky(res[:,:,:,i]) for i in eachindex(sol.t)];
E_Kz = [Ekz(res[:,:,:,i]) for i in eachindex(sol.t)];

E_P = [Ep(res[:,:,:,i],V(sol.t[i])) for i in eachindex(sol.t)];
E_I = [Ei(res[:,:,:,i]) for i in eachindex(sol.t)];

P = Plots.plot(sol.t,E_K,lw=1.5,label=L"E_K")
Plots.plot!(sol.t,E_Kx,lw=1.5,label=L"E_{kx}",alpha=0.4)
Plots.plot!(sol.t,E_Ky,lw=1.5,label=L"E_{ky}",alpha=0.4)
Plots.plot!(sol.t,E_Kz,lw=1.5,label=L"E_{kz}",alpha=0.4)
Plots.plot!(sol.t,E_P,lw=1.5,label=L"E_p")
Plots.plot!(sol.t,E_I,lw=1.5,label=L"E_i")

Plots.savefig(P,"Energy")

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
ψ = ComplexF64.(Array(ψ_GS));

E = zeros(100,10)

for i in 2:10
    ψ = ComplexF64.(Array(sol[i]));
    psi = Psi(ψ,X,K);
    E[:,i] = incompressible_spectrum(k,psi);
end;

psi = Psi(ψ,X,K);
k = log10range(0.1,10^2,100)#ln.(LinRange(-1,3,100));
E_i = incompressible_spectrum(k,psi);
#E_c = compressible_spectrum(k,psi);
E[:,1] .= E_i

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
        global Na = number(ψ) |> Float32
        ϕi = ψ ./ sqrt(Na)
        
        if use_cuda
            ϕi = cu(ϕi)
        end

        global ax2 = dr*sum(@. x^2*abs2(ϕi))
        global ay2 = dr*sum(@. y^2*abs2(ϕi))
        global az2 = dr*sum(@. z^2*abs2(ϕi))

        σi = real.(0.5*im*dr.*CurrentDensity(ϕi)./(ax2,ay2,az2))
        @. ϕi *= exp(-0.5*im*(σi[1]*x^2 + σi[2]*y^2 + σi[3]*z^2))

        if use_cuda
            ϕi, σi = ϕi, σi |> cu
            global Pfx = Float32(dx/sqrt(2π))*plan_fft(copy(ϕi),1)
            global Pfy = Pf # Cannot do cuda fft along second dimension, have to do full transform :( 
            global Pfz = Float32(dz/sqrt(2π))*plan_fft(copy(ϕi),3)

            global Pix! = Float32(M*dkx/sqrt(2π))*plan_ifft!(copy(ϕi),1)
            global Piy! = Pi!
            global Piz! = Float32(M*dkz/sqrt(2π))*plan_ifft!(copy(ϕi),3)
        else
            global Pfx = dx/sqrt(2π)*plan_fft(copy(ϕi),1)
            global Pfy = dy/sqrt(2π)*plan_fft(copy(ϕi),2)
            global Pfz = dz/sqrt(2π)*plan_fft(copy(ϕi),3)

            global Pix! = M*dkx/sqrt(2π)*plan_ifft!(copy(ϕi),1)
            global Piy! = M*dky/sqrt(2π)*plan_ifft!(copy(ϕi),2)
            global Piz! = M*dkz/sqrt(2π)*plan_ifft!(copy(ϕi),3)
        end

        ϕ_initial =  ArrayPartition(ϕi,[1,1,1,σi[1],σi[2],σi[3]])
        return ϕ_initial
    end

    function extractinfo(sol)
        λx = [sol[:,i].x[2][1] for i in eachindex(sol.t)]
        λy = [sol[:,i].x[2][2] for i in eachindex(sol.t)]
        λz = [sol[:,i].x[2][3] for i in eachindex(sol.t)]
        
        σx = [sol[:,i].x[2][4] for i in eachindex(sol.t)]
        σy = [sol[:,i].x[2][5] for i in eachindex(sol.t)]
        σz = [sol[:,i].x[2][6] for i in eachindex(sol.t)]
    
        ax = @. sqrt(ax2*λx^2)
        ay = @. sqrt(ay2*λy^2)
        az = @. sqrt(az2*λz^2)
    
        ϕ = [sol[:,i].x[1] for i in eachindex(sol.t)]
        res = [abs2.(ϕ[i]) for i in eachindex(sol.t)];

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
        return  @. x.^2*λx*σx + y^2*λy*σy + z^2*λz*σz
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

        #println(typeof(ϕ))
        dϕ = du.x[1]
        du.x[2][1:3] .= u.x[2][4:6]
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

begin

    ψ_0 = ψ_GS#sol5[:,:,:,end]
    ϕ_initial = initialise(ψ_0)

    PfArray = [Pfx, Pfy, Pfz]
    PiArray = [Pix!,Piy!,Piz!]
    k = [kx,ky,kz]

    V_0 = zeros(M,M,M)

    xdV = V_0#x.*ifft(im*kx.*fft(V_0)) .|> real
    ydV = V_0#y.*ifft(im*ky.*fft(V_0)) .|> real
    zdV = V_0#z.*ifft(im*kz.*fft(V_0)) .|> real

    if use_cuda
        V_0 = V_0 |> cu
        xdV = xdV |> cu
        ydV = ydV |> cu
        zdV = zdV |> cu
    end

end;

t = LinRange(0,30,6);

CUDA.memory_status()

probs = ODEProblem(spec_expansion_opt!,ϕ_initial,(t[1],t[end]));
@time solt2 = solve(probs,saveat=t,abstol=1e-8,reltol=1e-5);

res,ϕ,λx,λy,λz,σx,σy,σz,ax,ay,az = extractinfo(solt2);

Norm = [ξ^3*ψ0^2*dr*sum(res[i]) for i in eachindex(solt2.t)]

Plots.plot(t,σx,ylims=(0.0,.3),label="σx",xlabel=(L"t/$\tau$"))
Plots.plot!(t,σy,ylims=(0.0,.3),label="σy")
Plots.plot!(t,σz,ylims=(0.0,.3),label="σz")



Plots.plot(Norm,ylims=(0,1.5))
Plots.heatmap(x,x,res[3][:,65,:],aspectratio=1)#,clims=(0,2e0),c=:thermal)

for i in 1:length(solt2.t)
    P = Plots.heatmap(λ   x,x,res[i][65,:,:],aspectratio=1)#,clims=(0,2e0),c=:thermal)
    display(P)
    sleep(0.05)
end


Plots.heatmap(λx[end]*x,λz[end]*x,res[end][:,:,65],aspectratio=1,clims=(0,6e-5),c=:thermal)


