# Using Point from Meshes (imported in intentional_agent.jl)
using Meshes: viz, viz!
import CairoMakie as mke

export viz_system_single_timestep

"""Return a base Makie FigurePlotAxis object with obstacles and goals plotted.

Users must provide both as vectors.
"""
function viz_env(obcs::Vector, goals::Vector)
    mvz = viz(obcs)
    viz!(goals, pointsize=20; pointcolor = :red)

    # Currently nothing else to vizualize.
    mvz
end

"""Single timestep vizualization of the system. Currently single agent.

Users must provide the obstacles as a vector.
"""
function viz_system_single_timestep(obcs::Vector, goals::Vector, s::KAgentState, a::Symbol)
    mvz = viz_env(obcs, goals)

    viz!(Point(Tuple(s.x)), pointsize=20)
    mke.arrows!([mke.Point2f(Tuple(s.x))], [mke.Vec2f(Tuple(action_heading_assoc_kagent[a]))],
                lengthscale = 0.3)

    mvz
end

