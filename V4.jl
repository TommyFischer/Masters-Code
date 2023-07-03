# 1/4/23 Spectral expansion is working, now just writing a clean version that can be cuda or normal using a single command + will tidy up
# Last Edit:  4:30pm 19th April

using PlotlyJS,
    Plots.PlotMeasures,
    QuantumFluidSpectra,
    #SparseArrays,
    #StaticArrays,
    LinearAlgebra,
    DifferentialEquations,
    FFTW,
    LaTeXStrings,
    Plots,
    #WAV,
    JLD2,
    #Makie, 
    #GLMakie,
    #CodecZlib,
    BenchmarkTools,
    #RecursiveArrayTools,
    CUDA,
    Adapt

@fastmath hypot(a,b,c) = sqrt(a^2 + b^2 + c^2)
Threads.nthreads()

CUDA.memory_status()

begin # Functions for setting up and running simulations

    function number(ψ)
        return sum(abs2.(ψ))*dr
    end

    function kfunc!(dψ,ψ) # Kinetic Energy term (no rotating frame)
        dψ .= 0.5*(ifft(k2.*fft(ψ))) 
        return nothing
    end

    function NDGPE!(dψ,ψ,var,t) # GPE Equation 
        kfunc_opt!(dψ,ψ)
        @. dψ = -im(0.5*dψ + (V_0 + abs2(ψ))*ψ)
    end

    function GPE!(dψ,ψ,var,t) # GPE Equation 
        kfunc_opt!(dψ,ψ)
        @. dψ = -(im + γ)*(0.5*dψ + (V_0 + abs2(ψ) - 1)*ψ)
    end

    function NDVPE!(dψ,ψ,var,t) # GPE Equation 
        kfunc_opt!(dψ,ψ)
        @. dψ = -im*(0.5*dψ + (V_0 +  $V(t) + abs2(ψ))*ψ)
    end

    function VPE!(dψ,ψ,var,t) # GPE Equation 
        kfunc_opt!(dψ,ψ)
        @. dψ = -(im + γ)*(0.5*dψ + (V_0 +  $V(t) + abs2(ψ) - 1)*ψ)
    end

    function kfunc_opt!(dψ,ψ)
        mul!(dψ,Pf,ψ)
        dψ .*= k2
        Pi!*dψ
        return nothing
    end

    function GPU_Solve!(savearray,EQ!, ψ, tspan; reltol = 1e-5, abstol = 1e-6, plot_progress = true, print_progress = false,alg = auto)
        

        if plot_progress # Setup for plot of progress
            t_elapsed = zeros(length(tspan))
        end
            
        savepoints = tspan[2:end]  
        condition(u, t, integrator) = t ∈ savepoints
        
            function affect!(integrator)                    # Function which saves states to CPU + makes plots / printouts if required
    
                i += 1

                if typeof(ψ) in [CuArray{ComplexF32, 3, CUDA.Mem.DeviceBuffer}, CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}]
                    push!(savearray,Array(integrator.u))
                else
                    push!(savearray,ArrayPartition(Array(integrator.u.x[1]),Array(integrator.u.x[2])))
                end    
                       
                if print_progress
                    println("Save $(i - 1) / $(length(tspan) - 1) at t = ", integrator.t, " s")     
                    println("Actual time elapsed = $(time() - tprev) s")
                    println("")
                end
                
                if plot_progress
                    t_elapsed[i] = time() - tprev + t_elapsed[i - 1]
                    P = Plots.plot(tspan,t_elapsed,
                            ylims=(0,1.5*t_elapsed[i]),
                            xlabel="Simulated Time",
                            ylabel="Time Elapsed",
                            label = false,
                            lw = 2,
                            fill=true,
                            fillalpha=0.5,
                            c=:orange,
                            framestyle = :box
                            )
                
                    vline!(savepoints,
                        linestyle=:dash,
                        alpha = 0.3,
                        c=:black,
                        label = "Savepoints")

                    display(P)
                end
                
                tprev = time()
            end

        if typeof(ψ) in [CuArray{ComplexF32, 3, CUDA.Mem.DeviceBuffer}, CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}]
            push!(savearray,Array(ψ))
        else
            push!(savearray,ψ)
        end
                    
        cb = DiscreteCallback(condition, affect!)       # Callback Function 
        i = 1                                           # Counter for saving states
        tprev = time()                                  # Timer for tracking progress
            
        prob = ODEProblem(EQ!,ψ,(tspan[1],tspan[end]))   
        solve(prob, callback=cb, tstops = savepoints, save_start = false, save_everystep = false, save_end = false,abstol=abstol,reltol=reltol,alg=alg)
    end
end

begin # Functions for making plots

    function n(ψ;minbin = 0) # returns spectra and k values
        ϕ = fft(ψ) .|> abs2
        ϕk = []

        for i in 1:Mx, j in 1:My, k in 1:Mz
            push!(ϕk,(ϕ[i,j,k],hypot(kx[i],ky[j],kz[k])))
        end

        sort!(ϕk, by = x->x[2])

        q = popfirst!(ϕk)
        ϕ_out = [q[1]]
        k_out = [q[2]]

        sum1 = 0 |> complex
        count = 0
        k = 0
        
        while length(ϕk) > 0
            q = popfirst!(ϕk)
            sum1 += q[1]
            count += 1

            if q[2] != k
                push!(ϕ_out, sum1 / count)
                push!(k_out, k)

                k = q[2]
                sum1 = 0
                count = 0
            end
        end

        if minbin == 0
            return k_out, ϕ_out  
        else

            kout = []
            ϕout = []

            for i in 1:ceil(Int,maximum(k_out)/minbin)
                for j in eachindex(k_out)
                    if k_out[j] > i*minbin && (j > 1)
                        push!(kout,sum(k_out[1:(j-1)]) / (j-1))
                        deleteat!(k_out,1:(j-1))

                        push!(ϕout,sum(ϕ_out[1:(j-1)])/(j-1))
                        deleteat!(ϕ_out,1:(j-1))
                        break
                    elseif k_out[end] < i*minbin
                        push!(kout,sum(k_out/length(k_out)))
                        push!(ϕout,sum(ϕ_out[1:(j-1)])/(j-1))
                        break
                    end
                end
            end
            return kout, ϕout
        end
        
    end

    function n2D(ψ;minbin = 0) # returns spectra and k values
        ϕ = fft(ψ) .|> abs2
        ϕk = []

        for i in 1:Mx, j in 1:My
            push!(ϕk,(ϕ[i,j],hypot(kx[i],ky[j])))
        end

        sort!(ϕk, by = x->x[2])

        q = popfirst!(ϕk)
        ϕ_out = [q[1]]
        k_out = [q[2]]

        sum1 = 0 |> complex
        count = 0
        k = 0
        
        while length(ϕk) > 0
            q = popfirst!(ϕk)
            sum1 += q[1]
            count += 1

            if q[2] != k
                push!(ϕ_out, sum1 / count)
                push!(k_out, k)

                k = q[2]
                sum1 = 0
                count = 0
            end
        end

        if minbin == 0
            return k_out, ϕ_out  
        else

            kout = []
            ϕout = []

            for i in 1:ceil(Int,maximum(k_out)/minbin)
                for j in 1:length(k_out)
                    if k_out[j] > i*minbin && (j > 1)
                        push!(kout,sum(k_out[1:(j-1)]) / (j-1))
                        deleteat!(k_out,1:(j-1))

                        push!(ϕout,sum(ϕ_out[1:(j-1)])/(j-1))
                        deleteat!(ϕ_out,1:(j-1))
                    elseif k_out[end] < i*minbin
                        push!(kout,sum(k_out/length(k_out)))
                        push!(ϕout,sum(ϕ_out[1:(j-1)])/(j-1))
                    end
                end
            end
            return kout, ϕout
        end
        
    end

    Ek(ψ) = 0.5*ξ^3*ψ0^2 * dr*sum(abs2.(ifft(im*ksum.*fft(ψ)))) # E_Kinetic / μ
    Ep(ψ,V) = ξ^3*ψ0^2 * dr*sum((Array(V_0) .+ Array(V)).*abs2.(ψ)) |> real # E_Potential / μ
    Ei(ψ) = 0.5*ξ^3*ψ0^2 * dr*sum(abs2.(ψ).^2) # E_Interaction / μ

    Ekx(ψ) = 0.5*ξ^3*ψ0^2 * dr*sum(abs2.(ifft(im*Array(kx).*fft(ψ)))) # x-direction E_k
    Eky(ψ) = 0.5*ξ^3*ψ0^2 * dr*sum(abs2.(ifft(im*Array(ky).*fft(ψ)))) # y-direction E_k
    Ekz(ψ) = 0.5*ξ^3*ψ0^2 * dr*sum(abs2.(ifft(im*Array(kz).*fft(ψ)))) # z-direction E_k

    function NormChange(sol) # Percentage change in norm 
        return 100*(number(sol[:,:,:,end]) - number(sol[:,:,:,1])) / number(sol[:,:,:,1]) |> abs
    end

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
        if typeof(sol[1]) in (Array{ComplexF64, 3},Array{ComplexF32, 3}) # Checking if input solution is already squared or not
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
        #show_axis=axis
        )
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

#-------------------------- Leshgo ------------------------------------------------

begin # Adjustable Parameters and constants 

    ħ = 1.05457182e-34
    m = 87*1.66e-27 
    a_s = 5.8e-9 
    k_B = 1.380649e-23
    μ = 1e-9 * k_B #ħ # Smaller μ, bigger norm
    g = 4π*ħ^2*a_s/m
    N = 4e5 # Bigger N smaller norm

    trap = "box" # "box", "cyl" (cylinder), or "harm" (harmonic)
    type = "F32"

    if trap == "harm"
        ω_x = 1 # Harmonic Trapping Frequencies, if using harmonic trap
        ω_y = 1
        ω_z = 1

        Rx = sqrt(2)/ω_x
        Ry = sqrt(2)/ω_y
        Rz = sqrt(2)/ω_z
    end

    ξ = ħ/sqrt(m*μ)
    ψ0 = sqrt(μ/g) #sqrt(N/ξ^3) 
    τ = ħ/μ

    Lx = 30 #*sqrt(2)/ω_x
    Ly = 25 #*sqrt(2)/ω_y
    Lz = 20 #6*sqrt(2)/ω_z

    Mx = 256 # Grid sizes
    My = Mx
    Mz = Mx

    A_V = 15    # Trap height
    n_V = 24    # Trap Power (pretty much always 24)
    L_V = 10    # No. of healing lengths for V to drop from A_V to 0.01A_V 
    L_P = 15     # Amount of padding outside trap (for expansion)

    Lx = 20#6*sqrt(2)/ω_x
    Ly = 20 #6*sqrt(2)/ω_y
    Lz = 20 #6*sqrt(2)/ω_z

    Mx = 64 # Grid sizes
    My = 64
    Mz = 64

    A_V = 60    # Trap height
    n_V = 24    # Trap Power (pretty much always 24)
    L_V = 8    # No. of healing lengths for V to drop from A_V to 0.01A_V 
    L_P = 8     # Amount of padding outside trap (for expansion)
    use_cuda = CUDA.functional()
end

begin # Arrays

    x = LinRange(-Lx/2 - (L_P + L_V),Lx/2 + (L_P + L_V),Mx + 1)[2:end] |> collect
    y = LinRange(-Ly/2 - (L_P + L_V),Ly/2 + (L_P + L_V),My + 1)[2:end]' |> collect
    z = LinRange(-Lz/2 - (L_P + L_V),Lz/2 + (L_P + L_V),Mz + 1)[2:end]
    x = LinRange(-Lx/2 - (L_P + L_V),Lx/2 + (L_P + L_V),Mx) |> collect
    y = LinRange(-Ly/2 - (L_P + L_V),Ly/2 + (L_P + L_V),My)' |> collect
    z = LinRange(-Lz/2 - (L_P + L_V),Lz/2 + (L_P + L_V),Mz)
    z = reshape(z,(1,1,Mz)) |> collect

    dx = x[2] - x[1]
    dy = y[2] - y[1]
    dz = z[2] - z[1]

    kx = fftfreq(Mx,2π/dx) |> collect
    ky = fftfreq(My,2π/dy)' |> collect
    kz = reshape(fftfreq(Mz,2π/dz),(1,1,Mz)) |> collect

    dkx = kx[2] - kx[1]
    dky = ky[2] - ky[1]
    dkz = kz[2] - kz[1]

    k2 =  kx.^2 .+ ky.^2 .+ kz.^2 # 3D wave vector
    dr = dx*dy*dz
    ksum = kx .+ ky .+ kz

    Vbox = (trap == "box")
    Vcyl = (trap == "cyl")
    Vharm = (trap == "harm") 
        
end;

if Vbox
    V_0 = zeros(Mx,My,Mz)
    Vboundary(x) = A_V*cos(x/λ)^n_V
    λ = L_V/acos(0.01^(1/n_V))

    for i in 1:Mx, j in 1:My, k in 1:Mz
        #if (abs(x[i]) > 0.5*Lx + L_V + L_P - 4) || (abs(y[j]) > 0.5*Ly + L_V + L_P - 4) || (abs(z[k]) > 0.5*Lz + L_V + L_P - 4) # V = A_V at edges
        #    V_0[i,j,k] = A_V #+ (max(0,abs(x[i]) - (0.5*Lx + L_V + L_P - 4))^4 + max(0,abs(y[j]) - (0.5*Ly + L_V + L_P - 4))^4 + max(0,abs(z[k]) - (0.5*Lz + L_V + L_P - 4))^4)
        if (abs(x[i]) > 0.5*Lx + L_V) || (abs(y[j]) > 0.5*Ly + L_V) || (abs(z[k]) > 0.5*Lz + L_V) # V = A_V at edges
            V_0[i,j,k] = A_V #+ 0.5*(max(0,abs(x[i]) - (0.5*Lx + L_V))^2 + max(0,abs(y[j]) - (0.5*Ly + L_V))^2 + max(0,abs(z[k]) - (0.5*Lz + L_V))^2)
            V_0[i,j,k] = A_V #+ 0.5*(max(0,abs(x[i]) - (0.5*Lx + L_V)) + max(0,abs(y[j]) - (0.5*Ly + L_V)) + max(0,abs(z[k]) - (0.5*Lz + L_V)))
        else
            lx = L_V + π*λ/4 - max(0.0,abs(x[i]) - (0.5*Lx - π*λ/4)) # Finding the distance from the centre in each direction, 
            ly = L_V + π*λ/4 - max(0.0,abs(y[j]) - (0.5*Ly - π*λ/4)) # discarding if small
            lz = L_V + π*λ/4 - max(0.0,abs(z[k]) - (0.5*Lz - π*λ/4))
        
            #V_0[i,j,k] = Vboundary(min(lx,ly,lz))
            V_0[i,j,k] = hypot(Vboundary(lx),Vboundary(ly),Vboundary(lz))
            #V_0[i,j,k] = Vboundary(smin(smin(lx,ly,V_k),lz,V_k))
            #V_0[i,j,k] = (Vboundary(lx),Vboundary(ly),V_k),Vboundary(lz),V_k)
            if V_0[i,j,k] > A_V
                V_0[i,j,k] = A_V
            end
        end
    end
end;

if Vcyl # Cylinder Trap Potential

    V_0 = zeros(Mx,My,Mz)
    Vboundary(x) = A_V*cos(x/λ)^n_V

    λ = L_V/acos(0.01^(1/n_V))
    
    for i in 1:Mx, j in 1:My, k in 1:Mz
        l_z = min(2*L_V,Lz/2 - abs(z[k]))
        l_r = min(2*L_V,sqrt(Lx*Ly)/2 - hypot(x[i]*sqrt(Ly/Lx),y[j]*sqrt(Lx/Ly)))

        l = map(Vboundary,(l_z,l_r))

        V_0[i,j,k] = hypot(l[1],l[2])
    end
end;

if Vharm
    V_0 = 0.5*.04*[(ω_x*i)^2 + (ω_y*j)^2 + (ω_z*k)^2 for i in x, j in reshape(y,My), k in reshape(z,Mz)]
    ψ_gauss = [exp(-0.5*(ω_x*i^2 + ω_y*j^2 + ω_z*k^2)) for i in x, j in reshape(y,My), k in reshape(z,Mz)]  #.|> ComplexF32;
end;

Plots.heatmap(x,reshape(y,My),(V_0[:,:,128]'),aspectratio=1,clims=(0,1.2*A_V),xlabel=(L"x/\xi"),ylabel=(L"y/\xi"))
Plots.heatmap(x,reshape(z,Mz),(V_0[:,64,:]'),aspectratio=1,clims=(0,1.2*A_V),xlabel=(L"x/\xi"),ylabel=(L"z/\xi"))

Plots.plot(x,V_0[:,128,128],xlabel = L"x",lw=2,ylabel=L"V/A_V",label = false)
Plots.heatmap(x,reshape(y,My),(V_0[:,:,32]'),aspectratio=1,clims=(0,1.2*A_V),xlabel=(L"x/\xi"),ylabel=(L"y/\xi"))
Plots.heatmap(x,reshape(z,Mz),(V_0[:,32,:]'),aspectratio=1,clims=(0,1.2*A_V),xlabel=(L"x/\xi"),ylabel=(L"z/\xi"))

Plots.plot(x,V_0[:,32,32],xlabel = L"x",lw=2,ylabel=L"V/A_V",label = false)
#xlims!(-0.5*Lx - L_V - 1, -0.5*Lx - L_V + 1)
#ylims!(58,62)
vline!([-0.5*Lx,0.5*Lx],alpha = 0.8,label = L"±0.5*Lx")
vline!([-0.5*Lx - L_V,0.5*Lx + L_V],alpha = 0.8,label = L"±(0.5Lx + L_V)")
vline!([-0.5*Lx + λ*π/4, 0.5*Lx - λ*π/4])

#ψ_TF = [max(0,1 - hypot(i,j,k)^2/R_tf^2) for i in x, j in x, k in x] |> complex;
ψ_rand = (rand(Mx,My,Mz) + im*rand(Mx,My,Mz));
#ψ_ones = ones(Mx,My,Mz) |> complex;   
ψ_TF = 1/sqrt(N)*[max(0,1-V_0[i,j,k]) for i in 1:Mx, j in 1:My, k in 1:Mz] |> complex;
ψ_rand = (rand(Mx,My,Mz) + im*rand(Mx,My,Mz));
ψ_ones = ones(Mx,My,Mz) |> complex;

if use_cuda # Transforming arrays
    if type == "F64"
        V_0 = adapt(CuArray,V_0)
        k2 = adapt(CuArray,k2)
        ψ_rand = adapt(CuArray,ψ_rand)
        #ψ_ones = adapt(CuArray,ψ_ones)
        #ψ_TF = adapt(CuArray,ψ_TF)
    elseif type == "F32"
        V_0 = V_0 |> cu
        k2 = k2 |> cu
        ψ_rand = ψ_rand |> cu
        #ψ_ones = ψ_ones |> cu
        #ψ_TF = ψ_TF |> cu
    else
        println("Invalid Type!")
    end
end;

if use_cuda 
    const Pf = (dr/(2π)^1.5) * plan_fft(copy(ψ_rand));
    const Pi! = Mx*My*Mz*dkx*dky*dkz/(2π)^1.5 * plan_ifft!(copy(ψ_rand));
else
    const Pf = dr/(2π)^1.5 * plan_fft(copy(ψ_rand),flags=FFTW.MEASURE);
    const Pi! = Mx*My*Mz*dkx*dky*dkz/(2π)^1.5 * plan_ifft!(copy(ψ_rand),flags=FFTW.MEASURE);
end

l = @layout [a ; b c ; d e]
plot(rand(10,11),layout = l)

#-------------------------- Finding Ground State -----------------------------------------

γ = 1
tspan = LinRange(0.0,5,20); 
res_GS = []
GPU_Solve!(res_GS,GPE!,ψ_rand,tspan,plot_progress=true, print_progress=true,abstol=1e-8,reltol=1e-5,alg=Tsit5());#ParsaniKetchesonDeconinck3S32());
CUDA.memory_status()

Norm = [number(i) for i in res_GS];
Plots.plot(Norm,ylims=(0,2*Norm[end]))
tees = [50]
con(u, t, integrator) = t ∈ tees
function affect!(integrator)
    cbres[:,:,:,Int(round(integrator.t))] .= Array(integrator.u)
    println("callback at t = $(time() - t0)")
end
cb = DiscreteCallback(con,affect!)

cbres = zeros(Mx,My,Mz,5) .|> ComplexF32;

begin
    γ = 1
    tspan = LinRange(0.0,40,5); 

    t0 = time()
    prob = ODEProblem(GPE!,ψ_rand,(tspan[1],tspan[end]))   
    @time prob = solve(prob,callback=cb)#saveat=tspan)#,save_everystep = false, save_start = false, save_end = false);#,callback=cb)
end;

size(sol)
res = Array(sol);
ψ_GS = res[:,:,:,end]; #sol[:,:,:,end];

Plots.heatmap(x,reshape(y,My),abs2.(res_GS[5][:,:,128]'),clims=(0,1),aspectratio=1,xlabel=(L"x/\xi"),ylabel=(L"y/\xi"),right_margin=8mm,cbar=false)
xlims!(-40,40)
vline!([-0.5*Lx,0.5*Lx],label = L"± 0.5Lx",width=2,alpha=0.3)
hline!([-0.5*Ly,0.5*Ly],label = L"± 0.5 Ly",width=2,alpha=0.3)

Plots.heatmap(x,reshape(z,Mz),abs.(res_GS[20][:,128,:]'),clims=(0,1),aspectratio=1,xlabel=(L"x/\xi"),ylabel=(L"z/\xi"),right_margin=8mm,cbar=false)
xlims!(-40,40)
vline!([-0.5*Lx,0.5*Lx],label = L"± 0.5 Lx",width=2,alpha=0.3)
hline!([-0.5*Lz,0.5*Lz],label = L"± 0.5 Lz",width=2,alpha=0.3)

Plots.plot(x,abs2.(res_GS[3][:,64,64]))#,ylims=(0.9,1.1))

ψ_GS = res_GS[20][:,:,:];

#-------------------------- Creating Turbulence ------------------------------------------

Shake_Grad = 0.1          # Gradient of shake 
ω_shake = 2π * 0.03055      # Frequency of shake 
shakegrid = Shake_Grad * Array(z) .* ones(Mx,My,Mz) |> complex;    

V(t) = sin(ω_shake*t)*shakegrid

Plots.plot(reshape(z,Mz),Shake_Grad*reshape(z,Mz))

if use_cuda
    if type == "F64"
        shakegrid = adapt(CuArray,shakegrid)
        ψ_GS = adapt(CuArray,ψ_GS)
    elseif type == "F32"
        shakegrid = shakegrid |> cu
        ψ_GS = ψ_GS |> cu
    else
        println("Invalid Type!")
    end
Norm = [number(res[:,:,:,i]) for i in 1:5];
Plots.plot(Norm,ylims=(0,15*Norm[end]))

Plots.heatmap(x,reshape(y,My),abs2.(res[:,:,64,5]'),clims=(0,2),aspectratio=1,xlabel=(L"x/\xi"),ylabel=(L"y/\xi"),right_margin=8mm)
vline!([-0.5*Lx,0.5*Lx],label = "±Lx/2",width=2,alpha=0.3)
hline!([-0.5*Ly,0.5*Ly],label = "±Ly/2",width=2,alpha=0.3)

Plots.heatmap(x,reshape(z,Mz),abs2.(res[:,64,:,5]'),clims=(0,2),aspectratio=1,xlabel=(L"x/\xi"),ylabel=(L"z/\xi"),right_margin=8mm)
vline!([-0.5*Lx,0.5*Lx],label = "±Lx/2",width=2,alpha=0.3)
hline!([-0.5*Lz,0.5*Lz],label = "±Lz/2",width=2,alpha=0.3)

#-------------------------- Creating Turbulence ------------------------------------------

ΔU = 1
ω_shake = 2π * 0.03055 
shakegrid = ΔU * Array(z)./(0.5*Lz) .* ones(Mx,My,Mz) |> complex;

V(t) = sin(ω_shake*t)*shakegrid

noisegrid = randn(Mx,My,Mz) + im*randn(Mx,My,Mz);
ψ_noise = ψ_GS;# .+ .01*maximum(abs.(ψ_GS))*noisegrid; 
number(ψ_noise)

Plots.heatmap(abs2.(ψ_noise[:,64,:]'),clims = (0,1.3),aspectratio=1)

if use_cuda
    shakegrid = shakegrid |> cu
    ψ_noise = ψ_noise |> cu
end;

CUDA.memory_status()

begin 
    γ = 5e-4
    tspan = LinRange(0,2.0/τ,5)

    prob = ODEProblem(NDVPE!,ψ_noise,(tspan[1],tspan[end]))    
    @time sol2 = solve(prob,saveat=tspan,reltol=1e-5)
end;

γ = 0
tspan = LinRange(0,2.0/τ,40)
tspan2 = LinRange(tspan[33],tspan[end],8)
#res_turb = []
GPU_Solve!(res_turb,NDVPE!,ψ_GS,tspan,reltol=1e-5,abstol = 1e-8, plot_progress=true, print_progress=true,alg=Tsit5());
GPU_Solve!(res_turb,NDVPE!,cu(res_turb[end]),tspan2,reltol=1e-5,abstol = 1e-8, plot_progress=true, print_progress=true,alg=Tsit5());
CUDA.memory_status()

length(res_turb)
deleteat!(res_relax,33)

Norm1 = [number(i) for i in res_turb];
Plots.plot(Norm1,ylims=(0,2e4))
#NormChange(res_turb)

Plots.heatmap(x,reshape(y,My),abs2.(res_turb[2][:,:,64]'),aspectratio=1,title="t = (tvec[4])",clims=(0,1),xlabel=(L"x/\xi"),ylabel=(L"y/\xi"))
RANDPLOT = Plots.heatmap(x,reshape(z,Mz),abs2.(res_turb[end][:,128,:]'),aspectratio=1,
        size = (600,600),
        #title="t = (tvec[4])",
        clims=(0,1.5),
        xlabel=(L"x/\xi"),
        ylabel=(L"z/\xi"),
        label=(L"z/\xi"),
        right_margin=8mm,
        c=:thermal,
        cbar=false,
        xlims=(-40,40),
        legendfontsize=12,
        labelfontsize=15,
        ylims=(-35,35))
vline!([-0.5*Lx,0.5*Lx],label = L"± 0.5Lx",width=2,alpha=0.3)
hline!([-0.5*Lz,0.5*Lz],label = L"± 0.5 Lz",width=2,alpha=0.3)
Plots.savefig(RANDPLOT,"Turbfig.svg")

Plots.plot(x,abs2.(res_turb[33][128,:,128]),ylims=(0,1),ylabel ="|ψ(x,y = -Ly, z = -Lz)|",xlabel="x",title = "t = 2/τ")
Plots.plot(x,abs2.(res_turb[40][64,:,64]),ylims=(0,1),ylabel ="|ψ(x,y = -Ly, z = -Lz)|",xlabel="x",title = "t = 2/τ")
Plots.heatmap(abs.(res_turb[40][:,:,50])*1e0,aspectratio=1,clims=(0,1))

ψ_turb = res_turb[end][:,:,:];
res1 = Array(sol2);
tvec = Array(sol2.t);

#res1[:,:,:,1] .= Array(ψ_GS);
ψ_turb = res1[:,:,:,end];

Norm1 = [number(res1[:,:,:,i]) for i in 1:5];
Plots.plot(Norm1,ylims=(0,1e4))

Plots.heatmap(x,reshape(y,My),abs.(res1[:,:,64,5]'),aspectratio=1,title="t = (tvec[4])",clims=(0,1.5),xlabel=(L"x/\xi"),ylabel=(L"y/\xi"))
Plots.heatmap(x,reshape(z,Mz),abs2.(res1[:,64,:,5]'),aspectratio=1,title="t = (tvec[4])",clims=(0,1.5),xlabel=(L"x/\xi"),ylabel=(L"z/\xi"),right_margin=8mm)

using VortexDistibutions

Plots.plot(x,abs.(res1[:,64,64,5]),ylims=(0,0.01),ylabel ="|ψ(x,y = -Ly, z = -Lz)|",xlabel="x",title = "t = 2/τ")
vline!([-0.5*Lx - L_V,0.5*Lx + L_V])

begin # Energy Plots
    E_K = [Ek(res1[:,:,:,i]) for i in eachindex(res1[1,1,1,:])];
    E_Kx = [Ekx(res1[:,:,:,i]) for i in eachindex(res1[1,1,1,:])];
    E_Ky = [Eky(res1[:,:,:,i]) for i in eachindex(res1[1,1,1,:])];
    E_Kz = [Ekz(res1[:,:,:,i]) for i in eachindex(res1[1,1,1,:])];

    #E_P = [Ep(res[:,:,:,i],V(sol2.t[i])) for i in eachindex(sol2.t)];
    E_I = [Ei(res1[:,:,:,i]) for i in eachindex(res1[1,1,1,:])];

    P = Plots.plot(tspan,E_K,lw=1.5,label=L"E_K")
    Plots.plot!(tspan,E_Kx,lw=1.5,label=L"E_{kx}",alpha=0.4)
    Plots.plot!(tspan,E_Ky,lw=1.5,label=L"E_{ky}",alpha=0.4)
    Plots.plot!(tspan,E_Kz,lw=1.5,label=L"E_{kz}",alpha=0.4)
    #Plots.plot!(sol2.t,E_P,lw=1.5,label=L"E_p")
    Plots.plot!(tspan,E_I,lw=1.5,label=L"E_i")
end

begin
    Einc = zeros(length(tspan))
    Ecom = zeros(length(tspan))
    X = map(Array,(x,reshape(y,My),reshape(z,Mz)));
    K = map(Array,(kx,reshape(ky,My),reshape(kz,My)));

    for i in 1:length(tspan)
        psi = Psi(ComplexF64.(res1[:,:,:,1]),X,K);
        #_, Einc[i], Ecom[i] 
        XX = energydecomp(psi)
    end

    Plots.plot(tspan,Einc,lw=1.5,label=L"E_i")
    Plots.plot!(tspan,Ecom,lw=1.5,label=L"E_c")
end

@save "turb200" res1
@load "turb200" res1
#Plots.savefig(P,"Energy")

#-------------------------- Relaxation ---------------------------------------------------

if use_cuda
    if type == "F64"
        ψ_turb = adapt(CuArray,ψ_turb)
    elseif type == "F32"
        ψ_turb = ψ_turb |> cu
    else
        println("Invalid Type!")
    end
end;

#-------------------------- Relaxation ---------------------------------------------------
γ = 0
tspan = LinRange(0,2.0/τ,40)
tspan2 = LinRange(tspan[33],tspan[end],8)
#res_relax = []
GPU_Solve!(res_relax,GPE!,cu(res_relax[end]),tspan2,reltol=1e-5,abstol = 1e-8, plot_progress=true, print_progress=true,alg=Tsit5());
CUDA.memory_status()

length(res_relax)

Norm2 = [number(Array(i)) for i in res_relax];
Plots.plot(Norm2,ylims=(0,2e4))
#NormChange(res_relax)

for i in 1:40
    t1 = i
    #P = Plots.heatmap(x,reshape(y,My),abs2.(res_relax[t1][:,:,100]'),clims=(0,1.2),aspectratio=1,xlabel=(L"x/\xi"),ylabel=(L"y/\xi"),title="t=(tvec2[t1])")
    P = Plots.heatmap(x,reshape(z,Mz),abs2.(res_relax[t1][:,128,:]'),clims=(0,1.2),aspectratio=1,xlabel=(L"x/\xi"),ylabel=(L"z/\xi"),title="t=(tvec2[t1])")
    display(P)
end

#@save "relax200" res2
#@load "relax200" res2

#-------------------------- Expansion -------------------------------------------------

begin # Expansion Functions

    function initialise(ψ)
        global x
        global y
        global z
        global kx
        global ky
        global kz

        global Na = number(ψ) |> Float32
        ϕi = ψ ./ sqrt(Na)

        Na *= G
        
        if use_cuda
            ϕi = cu(ϕi)
            x = cu(x)
            y = cu(y)
            z = cu(z)
            kx = cu(kx)
            ky = cu(ky)
            kz = cu(kz)
        end

        global ax2 = dr*sum(@. x^2*abs2(ϕi))
        global ay2 = dr*sum(@. y^2*abs2(ϕi))
        global az2 = dr*sum(@. z^2*abs2(ϕi))

        σi = real.(0.5*im*dr.*CurrentDensity(ϕi)./(ax2,ay2,az2))
        @. ϕi *= exp(-0.5*im*(σi[1]*x^2 + σi[2]*y^2 + σi[3]*z^2))

        if use_cuda
            ϕi = ϕi |> cu
            σi = σi |> cu
            global Pfx = Float32(dx/sqrt(2π))*plan_fft(copy(ϕi),1)
            global Pfy = Pf # Cannot do cuda fft along second dimension, have to do full transform :( 
            global Pfz = Float32(dz/sqrt(2π))*plan_fft(copy(ϕi),3)

            global Pix! = Float32(Mx*dkx/sqrt(2π))*plan_ifft!(copy(ϕi),1)
            global Piy! = Pi!
            global Piz! = Float32(Mz*dkz/sqrt(2π))*plan_ifft!(copy(ϕi),3)
        else
            global Pfx = dx/sqrt(2π)*plan_fft(copy(ϕi),1)
            global Pfy = dy/sqrt(2π)*plan_fft(copy(ϕi),2)
            global Pfz = dz/sqrt(2π)*plan_fft(copy(ϕi),3)

            global Pix! = Mx*dkx/sqrt(2π)*plan_ifft!(copy(ϕi),1)
            global Piy! = My*dky/sqrt(2π)*plan_ifft!(copy(ϕi),2)
            global Piz! = Mz*dkz/sqrt(2π)*plan_ifft!(copy(ϕi),3)
        end

        ϕ_initial =  ArrayPartition(ϕi,[1,1,1,σi[1],σi[2],σi[3]])
        return ϕ_initial
    end

    function extractinfo(sol)
        λx = [i.x[2][1] for i in sol]
        λy = [i.x[2][2] for i in sol]
        λz = [i.x[2][3] for i in sol]
        
        σx = [i.x[2][4] for i in sol]
        σy = [i.x[2][5] for i in sol]
        σz = [i.x[2][6] for i in sol]
    
        ax = @. sqrt(ax2*λx^2)
        ay = @. sqrt(ay2*λy^2)
        az = @. sqrt(az2*λz^2)
    
        res = [i.x[1] for i in sol]

        return res,λx,λy,λz,σx,σy,σz,ax,ay,az
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

    function spec_expansion_noint!(du,u,p,t)
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
        #dσ .+= (0.5*Na/λ̄³)*sum(abs2.(ϕ).^2) # Computing first terms
        dσ .*= dr ./ (λ[1]*ax2,λ[2]*ay2,λ[3]*az2) # Coefficients

        # dϕ/dt
        
        kfunc2!(dϕ,ϕ,λ) # Kinetic Term
        dϕ .= -im*(0.5.*dϕ .+ (V_0 .+ 0.5*ρ2(λ,dσ)).*ϕ)
        
        du.x[1] .= dϕ
        du.x[2][4:6] .= dσ
    end 

end

begin
    ψ_0 = res_turb[end]
    ϕ_initial = initialise(ψ_0)

    PfArray = [Pfx, Pfy, Pfz]
    PiArray = [Pix!,Piy!,Piz!]
    k = [kx,ky,kz]

    V_0 = zeros(Mx,My,Mz)

    if use_cuda
        V_0 = V_0 |> cu
    end

    xdV = x.*ifft(im*kx.*fft(V_0)) .|> real
    ydV = y.*ifft(im*ky.*fft(V_0)) .|> real
    zdV = z.*ifft(im*kz.*fft(V_0)) .|> real

    if use_cuda
        xdV = xdV |> cu
        ydV = ydV |> cu
        zdV = zdV |> cu
    end

end;

tspan = LinRange(0,20,20);
res_expand = []; 
GPU_Solve!(res_expand,spec_expansion_opt!,ϕ_initial,tspan,alg=Tsit5());

res,λx,λy,λz,σx,σy,σz,ax,ay,az = extractinfo(res_expand[2:end]);

Norm = [number(i) for i in res]
 Plots.plot(Norm,ylims = (0,1.5*maximum(Norm)))

Plots.plot(λx,label="λx")
Plots.plot!(λy,label="λx")
Plots.plot!(λz,label="λz")

begin
    Plots.plot(ay ./ ax,ylims=(0.7,1.8),c=:red,lw=2,label="ay/ax")
    Plots.plot!(ay ./ az,c=:blue,lw=2,label="ay/az")
    Plots.plot!(ax ./ az,c=:green,lw=2,label="ax/az")
end

for i in 1:7
    begin
        t1 = i
        #p = Plots.heatmap(x*λx[t1],reshape(y,My)*λy[t1],abs2.(res[t1][:,:,64])',aspectratio=1,clims=(0,2e-4),c=:thermal,ylabel = L"y/\xi" )
        p = Plots.heatmap(x*λx[t1],reshape(z,Mz)*λz[t1],abs2.(res[t1][:,128,:])',aspectratio=1,clims=(0,2e-4),c=:thermal,ylabel=L"z/\xi")
        Plots.xlabel!(L"x/\xi")
        display(p)
    end
end

fftpsi = log.(abs2.(fftshift(fft(sol[:,:,:,end]))));
Plots.heatmap(fftshift(kx),fftshift(reshape(ky,My)),fftpsi[:,:,64]',aspectratio=1,c=:thermal,ylabel=L"y/\xi")
Plots.heatmap(fftshift(kx),fftshift(reshape(kz,Mz)),fftpsi[:,64,:]',aspectratio=1,c=:thermal,ylabel=L"z/\xi")

begin # Castin-Dum tings
    t1 = 1
    dt = t[2] - t[1]

    m = ω_y^2/(λy[t1]^2*λx[t1]*λz[t1])

    pl = Plots.plot(t,σx,ylims=(0.0,.6),label="σx",xlabel=(L"t/$\tau$"),lw=2,alpha=.5,legend=:bottomright)
    Plots.plot!(t,σy,ylims=(0.0,.6),label="σy",lw=2,alpha=.5)
    Plots.plot!(t,σz,ylims=(0.0,.6),label="σz",lw=2,alpha=.5)

    Plots.plot!(x->m*x + σy[t1] - m*dt*(t1-1),style=:dash,label = "Castin-Dum predicted slope")
    vline!([t[t1]],alpha=0.5,label=false)
    hline!([σy[t1]],alpha=0.5,label=false)
    hline!([ω_x],label="ωx")

    display(pl)
end

#-------------------------- Plots/Spectra/Analysis-----------------------------------------------------

# res = GS
# res1 = turb
# res2 = relax

X = map(Array,(x,reshape(y,My),reshape(z,Mz)));
K = map(Array,(kx,reshape(ky,My),reshape(kz,My)));
ψ = ComplexF64.(res_turb[end]);

psi = Psi(ψ,X,K);
k = log10range(0.1,10^2,100)
E_i = incompressible_spectrum(k,psi);
E_c = compressible_spectrum(k,psi);
E_q = qpressure_spectrum(k,psi);

Plots.plot(E_c ./ E_i,xlims=(0.1,10))
Plots.savefig(P,"spectra.svg")

begin # Plots
    P = Plots.plot(k,E_i,axis=:log,ylims=(1e2,5e4),xlims=(.1,15),
        label="Incompressible",
        lw=2,
        legend=:bottomright,
        alpha=0.8,
        framestyle=:box,
        xlabel=(L"k\xi")
    )
    Plots.plot!(k,E_c,lw = 2,alpha=0.8,label="Compressible")
    Plots.plot!(k,E_q,lw=2,alpha=0.8, label = "Quantum Pressure")


    Plots.plot!(x->(1.213e6)*x^-3,[x for x in k[50:70]],label=false,alpha=1,lw=.5)
    #Plots.plot!(x->(2.2e2)*x^1.1,[x for x in k[7:55]],label=false,alpha=1,lw=.5)

    k_Lx = 2π/(Lx)# Size of the System
    k_Ly = 2π/(Ly)# Size of the System
    k_Lz = 2π/(Lz)# Size of the System
    #k_l = 2π/(Lx - 2*L_V) # Size of the condensate accounting for the box trap
    k_π = π#2π/14
    k_ξ = 2π# Healing length
    k_dr = 2π/dr^(1/3) # Geometric mean of resolution
    k_dx = 2π/hypot(dx,dx,dx)
    
    vline!([k_Lx], label = false,linestyle=:dash,alpha=0.5,c=:black)
    vline!([k_Ly], label = false,linestyle=:dash,alpha=0.5,c=:black)
    vline!([k_Lz], label = false,linestyle=:dash,alpha=0.5,c=:black)
    #vline!([k_l], label = L"$k_l$",linestyle=:dash,alpha=0.5)
    #vline!([k_π], label = L"$π$",linestyle=:dash,alpha=0.5)
    vline!([k_ξ], label = false,linestyle=:dash,alpha=0.5,c=:black)
    #vline!([k_dr], label = L"$k_{dr}$",linestyle=:dash,alpha=0.5)
    vline!([k_dx], label = false,linestyle=:dash,alpha=0.5,c=:black)
end

plot(k,nk,axis=:log)

# density of slice through z = 0
ψz = ComplexF64.(res_relax[:,:,100,end]);
ψz ./= sqrt(number(ψz));

Plots.heatmap(abs2.(ψz),aspectratio=1)

X = map(Array,(x,reshape(y,My)));
K = map(Array,(kx,reshape(ky,My)));

psi = Psi(ψz,X,K);
k = log10range(.001,1,100)

nk = density_spectrum(k,psi)
Plots.plot(k,nk)

# Energy Plots
res = cat(res_turb,res_relax,dims=1);
length(res)
typeof(res)

tspan = LinRange(0,2.0/τ,40);

E_K = [Ek(i) for i in res];
E_Kx = [Ekx(i) for i in res];
E_Ky = [Eky(i) for i in res];
E_Kz = [Ekz(i) for i in res];
E_P = [0. for i in 1:80];
E_P[1:40] = [Ep(i,V(tspan[t])) for (t,i) in enumerate(res[1:40])];
E_P[41:80] = [Ep(i,zeros(Mx,My,Mz)) for (t,i) in enumerate(res[41:80])];
#E_P = [Ep(i,V(tspan[t])) for (t,i) in enumerate(res)];
#E_P = [Ep(i,zeros(Mx,My,Mz)) for (t,i) in enumerate(res)];
E_I = [Ei(i) for i in res];

tspan = LinRange(0,4.0,80)

P = Plots.plot(tspan,E_K,lw=1.5,label=L"E_K",size=(700,400))
Plots.plot!(tspan,E_Kx,lw=1.5,label=L"E_{kx}",alpha=0.4)
Plots.plot!(tspan,E_Ky,lw=1.5,label=L"E_{ky}",alpha=0.4)
Plots.plot!(tspan,E_Kz,lw=1.5,label=L"E_{kz}",alpha=0.4)
Plots.plot!(tspan,E_P,lw=1.5,label=L"E_p")
Plots.plot!(tspan,E_I,lw=1.5,label=L"E_i")
Plots.plot!(tspan,(E_K .+ E_P .+ E_I),label=L"E_T")

vline!([tspan[40],tspan[60]],linestyle=:dash,alpha=0.5,label = false,c = :black)
xlabel!(L"t",labelfontsize=15)

Plots.savefig(P,"E.svg")

Plots.plot(abs.(res_turb[end][:,128,128]))

y = Array(y)
Volume1(res_turb,40)

E_T = E_K .+ E_P .+ E_I; # Total energy 
dE = zeros(length(tspan) - 1); # dE/dt from energy functions
dV = zeros(length(tspan) - 1);
dt = tspan[2] - tspan[1]

for i in 2:length(tspan)
    dE[i - 1] = (E_T[i] - E_T[i - 1]) / dt
    dV[i - 1] = sum(z .* abs2.(res_turb[:,:,:,i - 1]))
    dV[i - 1] *= Shake_Grad*ω_shake*cos(ω_shake * tspan[i - 1])
end

Plots.plot(dE,label = L"dU/dt",lw = 2,alpha = 0.8)
Plots.plot!(dV, label = L"\left< \partial V / \partial t \right>", lw = 2, alpha = 0.8)


function kdensity(k,psi::Psi{3})  
    @unpack ψ,X,K = psi; 
	C = auto_correlate(ψ,X,K)
    return sinc_reduce(k,X...,C)
end

nkplots = []

for i in 1:40
    k = log10range(0.1,20,100);
    ψ = res_turb[:,:,:,i];
    ψ ./= sqrt(number(ψ));

    X = map(Array,(x,reshape(y,My),reshape(z,Mz)));
    K = map(Array,(kx,reshape(ky,My),reshape(kz,Mz)));

    psi = Psi(ψ,X,K);
    @time nk = kdensity(k,psi)
    push!(nkplots,nk)
end

for i in 1:40
    P = Plots.plot(k,nkplots[i] ./ k.^2,axis=:log,ylims=(1e-2,1e3))
    vline!([2π/Lz,2π/Lx,2π/Ly])
    display(P)
end
Plots.plot!(k,nkplots[40] ./ k.^2,axis=:log)

Plots.plot!(x->.5e0*x^-2.3)
vline!([2π/Lz,2π/hypot(dx,dx,dx)])


kplots = []
nplots = []

for i in [1,2,5,10,20,40]
    ψ = res_turb[:,:,:,i]; 
    ψ /= sqrt(number(ψ)); 
    k, nk = n(ψ,minbin=0.2);
    push!(kplots,k)
    push!(nplots,nk)
end

dk = diff(kplots[6])
size(nplots[6])
sum(nplots[6][1:end-1] .* dk) 

dk2 = diff(k)
sum(nk[1:end-20] .* dk2[1:end-19]) / π


Plots.plot!(kplots[6][3:end],kplots[6][3:end].^0 .* nplots[6][3:end] .* 1.3e-4,axis=:log,xlabel=L"\xi k",label = " n(k,t = 5.0)",alpha=0.7,legend=:bottomleft,xlims = (0.4,12))
Plots.plot!(x->.5e0*x^-3) 
ylims!(1e-3,1e6)


for i in 2:6
    display(Plots.plot!(kplots[i][3:end],nplots[i][3:end],axis=:log,xlabel=L"\xi k",ylabel = "k^2.2 n(k,t = $(τ*tspan[i])))",alpha=0.4,legend=false,xlims = (0.4,12)))
end
#Plots.plot!(kplots[6][3:end],nplots[6][3:end],axis=:log,xlabel=L"\xi k",label = L"n(k,t = 5)",alpha=0.7)
hline!([2e2])
Plots.plot!(x->1.5e2*x^-2)

vline!([k[80]])





function smin(a, b, k)
    h = clamp(0.5 + 0.5*(a-b)/k, 0.0, 1.0);
    return (a*(1 -h) + b*(h)) - k*h*(1.0-h);
end;

function smax(args,α)
    return sum(x -> x*exp(α*x),args)/sum(x -> exp(α*x),args)
end







function smin(a, b, k)
    h = clamp(0.5 + 0.5*(a-b)/k, 0.0, 1.0);
    return (a*(1 -h) + b*(h)) - k*h*(1.0-h);
end;

function smax(args,α)
    return sum(x -> x*exp(α*x),args)/sum(x -> exp(α*x),args)
end

begin
    xxx = -2:0.01:2
    α = 3
    aa(x) = sin(4x)
    bb(x) = exp(-x^2) - 1
    cc(x) = 1.5x + .2

    dd(x) = smin(smin(aa(x),bb(x),α),cc(x),α),#[smin([aa.(xxx)[i],bb.(xxx)[i],cc.(xxx)[i]],α) for i in 1:length(xxx)]

    Plots.plot(aa,lw=2,alpha=0.3,linestyle=:dash)
    Plots.plot!(bb,lw=2,alpha=0.3,linestyle=:dash,)
    Plots.plot!(cc,lw=2,alpha=0.3,linestyle=:dash)
    Plots.plot!(dd,lw=2,alpha=0.7)
    xlims!(-2,2)
    ylims!(-2,2)
end

begin # Box Trap Potential

    V_0 = zeros(Mx,My,Mz) 
    Vboundary(x) = A_V*cos(x/λ)^n_V 

    λ = L_V/acos(0.01^(1/n_V))
    
    for i in 1:Mx, j in 1:My, k in 1:Mz
        l_x = min(2*L_V,Lx/2 - abs(x[i])) # Finding the distance to the edge in each dimension, 
        l_y = min(2*L_V,Ly/2 - abs(y[j])) # discarding if further than 2*L_V
        l_z = min(2*L_V,Lz/2 - abs(y[k]))

        l = map(Vboundary,(l_x,l_y,l_z))

        V_0[i,j,k] = hypot(l[1],l[2],l[3])
    end
    V_0 = cu(V_0)
end;

x̃(x) = L/π * sin((x - L/2)*π / L)
f(x) = tanh((x̃(x)^2 - (L_trap/3)^2) / (2*s^2))
U(x,y,z) = A_V * min(1,(1.5 + 0.5*(f(x) + f(y) + f(z))))

newtrap(x,y,z) = U(x - L/2, y - L/2, z - L/2)

L = Lx
L_trap = 0.9*L
s = 1.5

V_0 = [newtrap(i,j,k)   for i in x, j in x, k in x];
Plots.plot(x,V_0[64,64,:])#,xlims=(-25,-15),ylims=(-1,10))

if Vbox
    V_0 = zeros(Mx,My,Mz)
    #Vboundary(x) = A_V*cos(x/λ)^n_V
    #λ = L_V/acos(0.01^(1/n_V))

    for i in 1:Mx, j in 1:My, k in 1:Mz
        if (abs(x[i]) >= 0.5*Lx) || (abs(y[j]) >= 0.5*Ly) || (abs(z[k]) >= 0.5*Lz) # V = A_V at edges
        
            lx = max(0,abs(x[i]) - 0.5*Lx) # Finding the distance from the centre in each direction, 
            ly = max(0,abs(y[j]) - 0.5*Ly) # discarding if small
            lz = max(0,abs(z[k]) - 0.5*Lz)
        
            V_0[i,j,k] = 0.01*hypot(lx^2,ly^2,lz^2)
        end
    end
end;
