using PlotlyJS, Plots, VortexDistributions

X = Array.(X)

vpoints = find_vortex_points_3d(ComplexF64.(res[12]),(Float64.(X[1]),Float64.(X[2]),Float64.(X[3])),1)

function Vortices_inbox(ψ,X,L)
    X_box = []     # trimmed x,y,z arrays
    trim = [0,0,0] # number of points being trimmed off each end

    for d in 1:3 # in each dimension
        x = Array(X[d]) .|> Float64
        l = 0.5 * L[d]
        for i in length(x):-1:1
            if abs(x[i]) > l # If value is outside L_i, trim
                trim[d] += 1
                deleteat!(x,i)
            end
        end

        push!(X_box,x)
    end

    trim //= 2 # Since we're trimming on both ends

    # Making ψ smaller by the same amount
    ψ_box = [ψ[i,j,k] for i in (1 + trim[1]):(length(X[1]) - trim[1]), j in (1 + trim[2]):(length(X[2]) - trim[2]), k in (1 + trim[3]):(length(X[3]) - trim[3])] .|> ComplexF64
    
    return find_vortex_points_3d(ψ_box,(X_box[1],X_box[2],X_box[3]))
end

vpoints = Vortices_inbox(SOLS_GS[1])

for i in length(vpoints):-1:1
    v = vpoints[i]
    if (abs(v[1]) > 0.5*L[1]) || (abs(v[2]) > 0.5*L[2]) || (abs(v[3]) > 0.5*L[3])
        deleteat!(vpoints,i)
    end
end

vorts = []
antivorts = []

for i in vpoints
    if i[4] == 1
        push!(vorts,i)
    else
        push!(antivorts,i)
    end
end



vortplot = PlotlyJS.scatter3d(;x = [i[1] for i in vorts], 
                        y = [i[2] for i in vorts], 
                        z = [i[3] for i in vorts],
                        mode="markers",
                        marker = attr(color=:red, size = 4)
                        )

antivortplot = PlotlyJS.scatter3d(;x = [i[1] for i in antivorts], 
                        y = [i[2] for i in antivorts], 
                        z = [i[3] for i in antivorts],
                        mode="markers",
                        marker = attr(color=:blue, size = 4)
                        )

PlotlyJS.plot([vortplot,antivortplot])

Plots.scatter([i[1] for i in antivorts],[i[2] for i in antivorts],[i[3] for i in antivorts],markersize=1.5,camera=(20,30))
Plots.scatter!([i[1] for i in vorts],[i[2] for i in vorts],[i[3] for i in vorts],markersize=1.5,camera=(20,30))