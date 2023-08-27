# Tidying up last version using Ashtons code. To-dos:
# - 3D Isosurfaces: Either look at exporting data and using another program to make them or look at different approaches using julia packages
# - DrWatson Parallel for scanning over parameters
# - Seperate functions and cases
# - stop using y and z already reshaped
# - Fully optimise GPU solve

@fastmath hypot(a,b,c) = sqrt(a^2 + b^2 + c^2)

# Functions for setting up and running simulations

function number(ψ)
    return sum(abs2.(ψ))*prod(dX)
end

function kfunc!(dψ,ψ) # Kinetic Energy term (no rotating frame)
    dψ .= 0.5*(ifft(k2.*fft(ψ))) 
    return nothing
end

function NDGPE!(dψ,ψ,var,t) # GPE Equation without damping or chemical potential
    γ = var
    kfunc_opt!(dψ,ψ)
    @. dψ = -im*(0.5*dψ + (V_0 + abs2(ψ))*ψ)
end

function GPE!(dψ,ψ,var,t) # GPE Equation 
    γ = var
    kfunc_opt!(dψ,ψ)
    @. dψ = -(im + γ)*(0.5*dψ + (V_0 + abs2(ψ) - 1)*ψ)
end

function NDVPE!(dψ,ψ,var,t) # GPE Equation 
    γ = var
    kfunc_opt!(dψ,ψ)
    @. dψ = -im*(0.5*dψ + (V_0 +  $V(t) + abs2(ψ))*ψ)
end

function VPE!(dψ,ψ,var,t) # GPE Equation 
    γ = var
    kfunc_opt!(dψ,ψ)
    @. dψ = -(im + γ)*(0.5*dψ + (V_0 +  $V(t) + abs2(ψ) - 1)*ψ)
end

function kfunc_opt!(dψ,ψ)
    mul!(dψ,Pf,ψ)
    dψ .*= k2
    Pi!*dψ
    return nothing
end

function GPU_Solve!(savearray,EQ!, ψ, tspan, γ; reltol = 1e-5, abstol = 1e-6, plot_progress = true, print_progress = false,alg = auto)
    
    if plot_progress # Setup for plot of progress
        t_elapsed = zeros(length(tspan))
    end
        
    savepoints = tspan[2:end]  
    condition(u, t, integrator) = t ∈ savepoints
    
    function affect!(integrator)                    # Function which saves states to CPU + makes plots / printouts if required

        i += 1

        if typeof(ψ) in [CuArray{ComplexF32, 3, CUDA.Mem.DeviceBuffer}, CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer},Array{ComplexF32, 3},Array{ComplexF64, 3}]
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

    if typeof(ψ) in [CuArray{ComplexF32, 3, CUDA.Mem.DeviceBuffer}, CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer},Array{ComplexF32, 3},Array{ComplexF64, 3}]
        push!(savearray,Array(ψ))
    else
        push!(savearray,ψ)
    end
                
    cb = DiscreteCallback(condition, affect!)       # Callback Function 
    i = 1                                           # Counter for saving states
    tprev = time()                                  # Timer for tracking progress
        
    prob = ODEProblem(EQ!,ψ,(tspan[1],tspan[end]),γ)   
    solve(prob, callback=cb, dt = 1e-3,tstops = savepoints, save_on = false,abstol=abstol,reltol=reltol,alg=alg)
end

function MakeArrays(L_T, M)
    X = []
    K = []

    for i in 1:3 
        x = LinRange(-L_T[i] / 2, L_T[i] / 2, M[i] + 1)[2:end] |> collect
        kx = fftfreq(M[i],2π * M[i]/L_T[i]) |> collect

        push!(X,x)
        push!(K,kx)
    end

    k2 = [i^2 + j^2 + k^2 for i in K[1], j in K[2], k in K[3]]

    if use_cuda
        if numtype == Float32
            X =  cu.(X)
            K = cu.(K)
            k2 = cu(k2)
        elseif numtype == Float64
            X = adapt.(CuArray,X)
            K = adapt.(CuArray,K)
            k2 = adapt.(CuArray,k2)
        end
    end

    return X,K,k2
end

function BoxTrap(X,L,M,L_V,A_V,n_V);
    V_0 = zeros(M)
    Vboundary(x) = A_V*cos(x/λ)^n_V
    λ = L_V/acos(0.01^(1/n_V))

    Xarray = Array.(X)

    for i in 1:M[1], j in 1:M[2], k in 1:M[3]
        if (abs(Xarray[1][i]) > 0.5*L[1] + L_V) || (abs(Xarray[2][j]) > 0.5*L[2] + L_V) || (abs(Xarray[3][k]) > 0.5*L[3] + L_V) # V = A_V at edges
            V_0[i,j,k] = A_V
        else
            lx = L_V + π*λ/4 - max(0.0,abs(Xarray[1][i]) - (0.5*L[1] - π*λ/4)) # Finding the distance from the centre in each direction, 
            ly = L_V + π*λ/4 - max(0.0,abs(Xarray[2][j]) - (0.5*L[2] - π*λ/4)) # discarding if small
            lz = L_V + π*λ/4 - max(0.0,abs(Xarray[3][k]) - (0.5*L[3] - π*λ/4))
        
            V_0[i,j,k] = hypot(Vboundary(lx),Vboundary(ly),Vboundary(lz))

            if V_0[i,j,k] > A_V
                V_0[i,j,k] = A_V
            end
        end
    end

    if use_cuda
        if numtype == Float32
            V_0 = cu(V_0)
        elseif numtype == Float64
            V_0 = adapt(CuArray,V_0)
        end
    end

    return V_0
end

function tsteps!(savearray,EQ!, ψ, tspan, reltol, abstol,alg) # Find the tsteps used for a simulation without saving states

    function EOM!(du,u,p,t)
        EQ!(du,u,p,t)
        push!(savearray,t)
    end
        
    prob = ODEProblem(EOM!,ψ,(tspan[1],tspan[end]))   
    solve(prob, save_on = false,abstol=abstol,reltol=reltol,alg=alg)
end

# Expansion Functions

function initialise(ψ)
    global x
    global y
    global z
    global kx
    global ky
    global kz

    global Na = number(ψ) |> Float32
    ϕi = ψ ./ sqrt(Na)
    
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

    res = [Array(i.x[1]) for i in sol]

    return res,λx,λy,λz,σx,σy,σz,ax,ay,az
end

function CurrentDensity(ψ)
    ϕ = fft(ψ)
    ϕim = fft(conj.(ψ))
    
    σx = sum(@. x*(ψ*$ifft(im*kx*ϕim) - conj(ψ)*$ifft(im*kx*ϕ)))
    σy = sum(@. y*(ψ*$ifft(im*ky*ϕim) - conj(ψ)*$ifft(im*ky*ϕ)))
    σz = sum(@. z*(ψ*$ifft(im*kz*ϕim) - conj(ψ)*$ifft(im*kz*ϕ)))

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
    @. dϕ *= im*k[i]
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
    dϕ .= @. -im*(0.5*dϕ + (V_0 + (Na/λ̄³)*abs2(ϕ) + 0.5*$ρ2(λ,dσ))*ϕ)
    
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


# Functions for making plots

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

#----------------- Optimised Functions ---------------------#

G!(ϕ::CuArray{ComplexF64, 3},ψ::CuArray{Complex{Int64}, 3},t::Float64) = begin 
    @. ϕ = -(im + γ)*Δt * (abs2(ψ) + V_static + V(t)) * ψ # Should test absorbing -(im + γ)*Δt into one gpu variable to see if there's a speedup
end  

G!(ϕ::CuArray{ComplexF64, 3},ψ::CuArray{ComplexF64, 3}) = begin 
    @. ϕ = -(im + γ)*Δt * (abs2(ψ) + V_static) * ψ # Should test absorbing -(im + γ)*Δt into one gpu variable to see if there's a speedup
end 

G!(ψ::CuArray{ComplexF64, 3},t::Float64) = begin
    @. ψ *= -(im + γ)*Δt * (abs2(ψ) + V_static + V(t))
end

G!(ψ::CuArray{ComplexF64, 3}) = begin
    @. ψ *= -(im + γ)*Δt * (abs2(ψ) + V_static)
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

function rk4ip!(ψ::CuArray{ComplexF64, 3}) 
    ψK .= ψ             # (B.15a) 
    D!(ψ)               # (B.15b)     
    ψI .= ψ             # (B.15c) 
    G!(ψK)            # (B.15d)       
    D!(ψK)              # (B.15e)       
    @. ψ += ψK/6        # (B.15f) 
    @. ψK = ψK/2 + ψI   # (B.15g)

    G!(ψK)            # (B.15i)      
    @. ψ += ψK/3        # (B.15j)
    @. ψK = ψK/2 + ψI   # (B.15k)
    G!(ψK)            # (B.15l)       
    @. ψ += ψK/3        # (B.15m)
    @. ψK += ψI         # (B.15n)
    D!(ψK)              # (B.15o)        
    D!(ψ)               # (B.15p)

    G!(ψK)            # (B.15r)        
    @. ψ += ψK/6       # (B.15s)
end

function rk4ip!(ψ::CuArray{ComplexF64, 3}, t::Float64) 
    ψK .= ψ             # (B.15a) 
    D!(ψ)               # (B.15b)     
    ψI .= ψ             # (B.15c) 
    G!(ψK,t)            # (B.15d)       
    D!(ψK)              # (B.15e)       
    @. ψ += ψK/6        # (B.15f) 
    @. ψK = ψK/2 + ψI   # (B.15g)

    t += Δt/2           # (B.15h)
    G!(ψK,t)            # (B.15i)      
    @. ψ += ψK/3        # (B.15j)
    @. ψK = ψK/2 + ψI   # (B.15k)
    G!(ψK,t)            # (B.15l)       
    @. ψ += ψK/3        # (B.15m)
    @. ψK += ψI         # (B.15n)
    D!(ψK)              # (B.15o)        
    D!(ψ)               # (B.15p)

    t += Δt/2           # (B.15q)
    G!(ψK,t)            # (B.15r)        
    @. ψ += ψK/6       # (B.15s)
end

function GroundState!(ψ::CuArray{ComplexF64, 3},tsaves; save_to_file = false) # save_to_file: if a string, does not create solution object and saves solutions to file given by string. If false creates solution object and returns it
    tstart = time()

    if save_to_file == false
        ψs = [zero(Array(ψ)) for _ in 1:length(tsaves)]
        ψs[1] .= Array(ψ);
    else
	    psi = Array(ψ)
	    @save save_to_file*"ψ_initial" psi
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

function Shake!(ψ::CuArray{ComplexF64, 3},tsaves)
    tstart = time()

    if save_to_file == false
        ψs = [zero(Array(ψ)) for _ in 1:length(tsaves)]
        ψs[1] .= Array(ψ);
    else
	    psi = Array(ψ)
	    @save save_to_file*"ψ_initial" psi
    end

    t=0.
    tsteps = @. round(Int, tsaves / Δt)

    for i in 1:tsteps[end]
        rk4ip!(ψ,t)
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

