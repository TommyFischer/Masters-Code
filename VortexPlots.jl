using PlotlyJS, Plots, VortexDistributions

X = Array.(X)

vpoints = find_vortex_points_3d(ComplexF64.(SOLS_GS[1][10]),(Float64.(X[1]),Float64.(X[2]),Float64.(X[3])),1)

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
                        marker = attr(color=:red)
                        )

antivortplot = PlotlyJS.scatter3d(;x = [i[1] for i in antivorts], 
                        y = [i[2] for i in antivorts], 
                        z = [i[3] for i in antivorts],
                        mode="markers",
                        marker = attr(color=:blue)
                        )
                        
PlotlyJS.plot([vortplot,antivortplot])