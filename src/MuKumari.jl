module MuKumari

###############################################
## Packages used across multiple files

### Quality-of-life packages (used throughout all files)
using Reexport
using DocStringExtensions
using Parameters: @with_kw, @with_kw_noshow
using Match
using ProgressMeter
###

### Packages used to represent geometries (used in all sub-files)
import GeoInterface as GI
import GeometryOps as GO
###

### Below are packages used exclusively in the agent definitions (`intentional_*.jl` files)
@reexport using POMDPs

using LinearAlgebra: normalize, ⋅, norm
using Distributions: Normal, MvNormal
using IterTools: partition
using POMDPTools, MCTS
using Flux: onehot
###
###############################################

export MuEnv, predict_μ, predict_env, update_μf, tangle_agent_env
export KAgentState

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

muenv_characteristics_namelist(muenv::MuEnv; base="", offset="\t") = mapreduce(x->"$(base)$(offset)\":$x\"\n", *, muenv.μ_order; init="")

function Base.show(io::IO, muenv::MuEnv)
    println(io, "Number of observable characteristics: $(muenv.M)")
    print(io, "List of characteristics:\n$(muenv_characteristics_namelist(muenv))") # new line auto-added by characteristic list function
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

state(s::KAgentState) = s.x
z(s::KAgentState) = s.z[end]
t(s::KAgentState) = length(s.hist)

include("basic_objectives.jl")
include("intentional_agent.jl")
include("intentional_agent_pomdp.jl")
include("intentional_kworld.jl")
include("basic_sim.jl")
include("basic_viz.jl")

end
