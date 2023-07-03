using JLD2, CUDA, FFTW, DifferentialEquations, LinearAlgebra

include("V5.jl")

@load 

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

wsave