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
    GLMakie,
    CodecZlib


FFTW.forget_wisdom # Still not 100% sure what this does
FFTW.set_num_threads(6) # Need to test to find optimal number

begin # Functions for setting up and running simulations

    function MakeArrays(L,M) # Makes position space and k-space arrays

        global x = LinRange(-L/2,L/2,M)
        global y = x'
        global z = reshape(x,(1,1,M))

        global dx = x[2] - x[1]
        global dy = y[2] - y[1]
        global dz = z[2] - z[1]

        global kx = fftfreq(M,2π/dx)
        global ky = fftfreq(M,2π/dy)'
        global kz = reshape(fftfreq(M,2π/dz),(1,1,M))

        global dkx = kx[2] - kx[1]
        global dky = ky[2] - ky[1]
        global dkz = kz[2] - kz[1]

        global k2 =  kx.^2 .+ ky.^2 .+ kz.^2 # 3D wave vector
        global dr = dx*dy*dz

    end;

    function kfunc!(dψ,ψ) # Kinetic Energy term (no rotating frame)
        dψ .= 0.5*(ifft(k2.*fft(ψ)))
        return nothing
    end

    function GPE!(dψ,ψ,var,t) # GPE Equation 
        kfunc_opt!(dψ,ψ)
        dψ .= @. -(im + γ)*(0.5*dψ + (V_0 + abs2(ψ) - 1)*ψ)
    end

    function VPE!(dψ,ψ,var,t) # GPE Equation 
        V1 = V(t)
        kfunc_opt!(dψ,ψ)
        dψ .= @. -(im + γ)*(0.5*dψ + (V_0 + V1 + abs2(ψ) - 1)*ψ)
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

    function Normcheck(solut,l=(0,2)) # Checks Normalisation
        n = length(solut[1,1,1,:])
        norm = ones(n)
        for i in 1:n
            norm[i] = sum(abs2.(solut[:,:,:,i]))*dx*dy*dz
        end
        Plots.plot(norm,ylims = l)
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



begin # Constants for Rb87

    ħ = 1.05457182e-34
    m = 87*1.66e-27 # Rubidium-87 mass (need to check this is correct)
    a_s = 5.8e-9 # Rubidium-87 scattering length (also check this)

end

begin # Adjustable Parameters

    μ = 30
    N = 3e5
    L = 30
    M = 100

    A_V = 15 # Trap height
    n_V = 24 # Trap Power (pretty much always 24)
    L_V = 2 # no. of healing lengths for V to drop to 0.01
    
end


ψ_rand = (randn(M,M,M) + im*randn(M,M,M));
Plots.heatmap(abs2.(ψ_rand[:,:,50]))

MakeArrays(L,M)
const Pf = dr/(2π)^1.5*plan_fft(copy(ψ_rand),flags=FFTW.MEASURE);
const Pi! = M^3*dkx*dky*dkz/(2π)^1.5*plan_ifft!(copy(ψ_rand),flags=FFTW.MEASURE);

ψ_ones = ones(M,M,M) |> complex;

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

Plots.heatmap(V_0[:,:,50])
Plots.plot(V_0[:,100,50],ylims=(0,15))

begin 
    γ = 1
    tspan = (0.0,3);
    var = [ħ,μ]; 

    prob = ODEProblem(GPE!,ψ_ones,tspan)    
    @time sol_GS = solve(prob,reltol=10e-6,saveat=0.1)
end;

length(sol_GS.t)
Normcheck(sol_GS,(0,3e4))
sum(abs2.(sol_GS[:,:,:,end]))*dr

res = abs2.(sol_GS);

Plots.heatmap(res[:,:,50,end],clims=(0,2))


for i in 1:length(sol_GS.t)
    P = Plots.heatmap(res[:,:,50,i],clims=(0,2))
    display(P)
end

ψ_GS = sol_GS[:,:,:,end];

ω_shake = π/2
shakegrid = z.* ones(M,M,M) |> complex;

V(t) = sin(ω_shake*t)*shakegrid
ψ_GS .+= .1(randn(M,M,M) .+ im*randn(M,M,M));


begin 
    γ = 0.0005
    tspan = (0.0,50);
    var = [ħ,μ]; 

    prob = ODEProblem(VPE!,ψ_GS,tspan)    
    @time sol = solve(prob,reltol=10e-6,saveat=0.1)
end;

@save "sol" sol 

size(sol)
Normcheck(sol,(0,2e4))
rizz = abs2.(sol);
riss = angle.(sol);
Plots.heatmap(rizz[:,50,:,1],clims=(0,3))


for i in 1:4:length(sol.t)
    P = Plots.heatmap(rizz[:,50,:,i],clims=(0,3),c=:thermal)
    display(P)
end


MakieVolume(sol,500,0.1,[0.2])
MakieGif(sol,"BoxTrop.gif",20,0.1,[.1])

Plots.plot(z)

cc = 66.88 + 69.86 + 54.82 + 56.45 + 54.81 + 49.92 + 47.94 + 47.94 + 46.95 + 46.46 + 47.15 + 48.93
cc / 12











l = length(sol_GS.t)
t = zeros(l)
t[1] = sol_GS.t[1]

for i in 2:l
    t[i] = sol_GS.t[i] - sol_GS.t[i-1]
end

Plots.plot(t)





xx = LinRange(0,2π,100)
λ = 0.5

ff(x) = @. cos(x/λ)^10
softwall = ff(xx)[1:10]

Plots.plot(ff(xx))
vline!([10])