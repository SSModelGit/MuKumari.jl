module MuKumari

using Reexport
using DocStringExtensions

using Match
import GeoInterface as GI
import GeometryOps as GO

export MuEnv, predict_μ, predict_env, update_μf, tangle_agent_env
export KAgentState, AbstractObjectiveLandscape, AgentObjectiveLandscape, GlobalObjectiveLandscape, tangle_agent_landscape

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

function tangle_agent_env(muenv::MuEnv, μ_list::Vector{Symbol})
  @assert μ_list ⊆ muenv.μ_order
  MuEnv(length(μ_list), μ_list, filter(p->p[1]∈μ_list, muenv.μf))
end

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

abstract type AbstractObjectiveLandscape end

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

The final component of the struct is `f_types`. This does not need user input.
* This is relevant only when constructing agent landscapes from a global context.
* It will be automatically filled in based on what feature access the user defines for the agent.
"""
struct AgentObjectiveLandscape <: AbstractObjectiveLandscape
    objectives::Vector
    f_types::Vector

    AgentObjectiveLandscape(objectives::Vector, f_types) = new(objectives, f_types)
end

"""Keyword-based constructor.
"""
AgentObjectiveLandscape(; objectives, feature_types=[]) = AgentObjectiveLandscape(objectives, feature_types)

"""Global container for all the landscape information regarding obstacles, goals, etc.

Consists of three fields: goals, obstacles, and horizons.
(The feature field is automatically constructed via the keyword constructor.)
* Each field takes an array of tuples. Each tuple consists of:
  * The feature's landscape level, ex. a surface obstacle would be tagged :surface
  * The feature information. The specific syntax is similar to the `AgentObjectiveLandscape`.

Example:
* We have two goals, one "aerial"- and one "surface"-level
* We have two obstacles that are both "subsurface"
* We have no horizons
```
GlobalObjectiveLandscape(
    [(:surface, Dict(:target=>[9.5 9.5], :strength=>100., :influence=>10., :size=>0.5)),
     (:aerial,  Dict(:target=>[7.5 7.5], :strength=>50., :influence=>10., :size=>1.5))],
    [(:sub, Dict(:poly => [(0., 0.), (0., 0.5), (0.4, 0.3), (0.5, 0.), (0., 0.)], :risk => 3., :impact => 10.)),
     (:sub, Dict(:poly => [(4., 4.5), (5., 4.5), (7., 5.), (2., 5.), (4., 4.5)],  :risk => 3., :impact => 10.))],
    []
)
```
"""
struct GlobalObjectiveLandscape <: AbstractObjectiveLandscape
    goals::Vector
    obstacles::Vector
    horizons::Vector
    feature_list::Vector

    GlobalObjectiveLandscape(goals::Vector, obstacles::Vector, horizons::Vector, feature_list::Vector) = new(goals, obstacles, horizons, feature_list)
end

"""Keyword-based constructor for the global landscape.

Automatically constructs the feature list.
"""
function GlobalObjectiveLandscape(; goals::Vector, obstacles::Vector, horizons::Vector)
    feature_set = Set()
    map(f->push!(feature_set, f[1]), Iterators.flatten([goals, obstacles, horizons]))

    GlobalObjectiveLandscape(goals, obstacles, horizons, collect(feature_set))
end

"""Helper function for tangling features from global landscape.

Pulls out the features relevant to a given agent based on the list of accessible features.
"""
tangle_by_feature_access(features::Vector, access_list::Vector) = map(f -> f[2], filter(f -> f[1] ∈ access_list, features))

"""Helper function for satisfying the specific syntax requirements of the `AgentObjectiveLandscape`.
"""
function tupleify_features(f_type::Symbol, fs::Vector)
    @match f_type begin
        :goal => map(f->(:goal, f), fs)
        :obc  => (:obc, fs)
        :horz => map(f->(:horz, f), fs)
    end
end

"""Main function to derive an agent's objective landscape from the global objective landscape.

Requires, in addition to the global landscape, a vector of symbols corresponding to accessible feature types.
"""
function tangle_agent_landscape(gobj::GlobalObjectiveLandscape, f_access::Vector)
    @assert f_access ⊆ gobj.feature_list
    (goals, obcs, horzs) = map(f->tangle_by_feature_access(f, f_access), [gobj.goals, gobj.obstacles, gobj.horizons])
    objectives = [tupleify_features(:goal, goals)..., tupleify_features(:obc, obcs), tupleify_features(:horz, horzs)...]
    AgentObjectiveLandscape(; objectives=objectives, feature_types=f_access)
end

include("basic_objectives.jl")
include("intentional_agent.jl")
include("basic_viz.jl")

end
