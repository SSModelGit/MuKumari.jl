module MuKumari

using Reexport

import GeoInterface as GI
import GeometryOps as GO

export MuEnv, predict_μ, predict_env, update_μf
export KAgentState, ObjectiveLandscape

"""Type for the Mu Environment.

At construction time, the user must provide the `M` environment characteristic functions.

Currently, the implementation does not track time. This may be changed later.
Thus, the functions are internally assumed time-invariant.
To change the functions over time, the user may use the function `update_μf`.
* This is useful when the user manually controls the time-update step in simulation.
* Between each time-step update, use the `update_μf` function to optionally modify the environment.

To predict the environment state, use the associated methods `predict_μ` or `predict_env`.
"""
struct MuEnv
    M::Integer # number of characteristics tracked by the pseudo-environment
    μ_order::Vector{Symbol} # Order of characteristics for vector creation
    μf::Dict{Symbol, Function} # Association between characteristic symbol and characteristic function

    function MuEnv(M::Integer, μ_order::Vector, μf::Dict)
        @assert length(μ_order) == M
        @assert length(μf) == M
        new(M, μ_order, μf)
    end
end

predict_μ(muenv::MuEnv, μ::Symbol, X::Matrix; rounding::Integer=2) = round(muenv.μf[μ](X); digits=rounding)

predict_env(muenv::MuEnv, X::Matrix) = reshape([predict_μ(muenv, μ, X) for μ in muenv.μ_order], muenv.M)

update_μf(muenv::MuEnv, μ::Symbol, f::Function) = muenv.μf[μ] = f

struct KAgentState
    x::Matrix
    z::Vector
    hist::Vector

    KAgentState(x::Matrix, z::Vector, hist::Vector) = new(x, z, hist)
end

function Base.:(==)(kag1::KAgentState, kag2::KAgentState)
    return kag1.x==kag2.x && kag1.z==kag2.z && kag1.hist==kag2.hist
end

function Base.show(io::IO, s::KAgentState)
    println(io, "KAgent State")
    println(io, "\tAgent Location: $(s.x)")
    println(io, "\tEnvironment Observations @ location: $(s.z[end])")
    println(io, "\tAgent Location History: $(s.hist)")
end

"""A struct containing a vector of objectives. With special format.

Each element of the vector is a tuple:
* The first element of the tuple is a symbol indicating objective type
* The second element of the tuple contains the details of the objective itself

An example is:
```
(:goal, Dict(:target=>[9.5 9.5], :strength=>100., :influence=>10., :size=>0.5))
```

The objective types can be:
* Goal-attraction objective: :goal
  * The contents should be a dictionary
* Goal-avoidance objective: :obc
  * The contents should be a vector of all obstacles. Look at obstacle use for more information.
* Time horizon objective: :horz
  * The content is a single float to indicate horizon urgency (relative to strongest reward).
"""
struct ObjectiveLandscape
    objectives::Vector

    ObjectiveLandscape(objectives::Vector) = new(objectives)
end

"""Keyword-based constructor.
"""
ObjectiveLandscape(; objectives) = ObjectiveLandscape(objectives)

include("basic_objectives.jl")
include("intentional_agent.jl")
include("basic_viz.jl")

end
