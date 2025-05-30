@reexport using POMDPs

using LinearAlgebra: normalize
using Distributions: Normal, MvNormal
using Meshes: Point, Vec, Quadrangle
using POMDPTools, MCTS

export KAgentState, blindstart_KAgentState, KAgentMDP, init_standard_KAgentMDP
export KWorld, create_empty_kworld, add_agent_to_world, get_num_agents

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

struct KAgentMDP <: POMDPs.MDP{KAgentState, Symbol}
    name::String
    start::Matrix # Grid location of starting pose of agent
    dimensions::Tuple # Dimensions of 2D grid-world
    boxworld::Quadrangle # 2D world constructed from dimensions
    obj::Function # objective function: must return two values, [Immediate reward for reaching state (KAgentState), Boolean True if Objective Accomplished]
    s::Float64 # movement speed
    w::Float64 # movement noise (in the angle, i.e. action, not in movement speed)
    menv::MuEnv # Observable environment process of the world
    v::Float64 # variance of environment observation noise process
    γ::Float64 # discount factor

    KAgentMDP(name::String, start::Matrix,
              dimensions::Tuple, boxworld::Quadrangle,
              obj::Function,
              s::Float64, w::Float64,
              menv::MuEnv, v::Float64,
              γ::Float64) = new(name, start, dimensions, boxworld, obj, s, w, menv, v, γ)
end

function KAgentMDP(;
    name::String, start::Matrix,
    dimensions::Tuple, boxworld::Quadrangle,
    obj::Function,
    s::Float64,
    w::Float64,
    menv::MuEnv,
    v::Float64,
    γ::Float64)
    return KAgentMDP(name, start, dimensions, boxworld, obj, s, w, menv, v, γ)
end

function init_standard_KAgentMDP(;
    name::String, start::Matrix,
    dimensions::Tuple, obj::Function, menv::MuEnv)
    let d=dimensions
        KAgentMDP(name, start,
                  d, Quadrangle((d[1], d[1]), (d[1], d[2]), (d[2], d[1]), (d[2], d[2])),
                  obj, 1., 0.05, menv, 0.05, 0.95)
    end
end

blindstart_KAgentState(mdp::KAgentMDP, x::Matrix) = KAgentState(x, [predict_env(mdp.menv, x)], Matrix[])

function Base.show(io::IO, mdp::KAgentMDP)
    println(io, "KAgent MDP")
    println(io, "\tLength of grid-space along the x-dimension: $(mdp.dimensions)")
    println(io, "\tObjective function: $(mdp.obj)")
    println(io, "\tEnvironment characteristics observed: $(mdp.menv.μ_order)")
end

POMDPs.isterminal(mdp::KAgentMDP, s::KAgentState) = mdp.obj(s)[2]

POMDPs.initialstate(mdp::KAgentMDP) = Deterministic(blindstart_KAgentState(mdp.start))

POMDPs.discount(mdp::KAgentMDP) = mdp.γ

"""Action space of the KAgentMDP.

Represents the eight cardinal and diagonal directions, along with "staying in the center".

Currently assumes constant motion speed.
"""
POMDPs.actions(mdp::KAgentMDP) = [:n, :ne, :e, :se, :s, :sw, :w, :nw, :c]

action_heading_assoc_kagent = Dict([(:n,  normalize([ 0,  1])),
                                    (:ne, normalize([ 1,  1])),
                                    (:e,  normalize([ 1,  0])),
                                    (:se, normalize([ 1, -1])),
                                    (:s,  normalize([ 0, -1])),
                                    (:sw, normalize([-1, -1])),
                                    (:w,  normalize([-1,  0])),
                                    (:nw, normalize([-1,  1])),
                                    (:c,  [0., 0.])])

function POMDPs.gen(mdp::KAgentMDP, s::KAgentState, a::Symbol, rng)
    real_a = reshape(round.(rand(MvNormal(action_heading_assoc_kagent[a], mdp.w)), digits=2), (1,:)) # real action factoring in noise
    xp = @. s.x + real_a * mdp.s
    if Point(Tuple(xp)) ∉ mdp.boxworld
        xp = copy(s.x)
    end
    zp = push!(copy(s.z), predict_env(mdp.menv, xp))
    hist_p = push!(copy(s.hist), s.x)
    sp = KAgentState(xp, zp, hist_p)

    # POMDP observation refers to state observation. Noise will occur in the position of the vehicle
    # for simplicity doubling noise in observation of position with noise of movement
    o_x = xp .+ reshape(round.(rand(MvNormal([0.0, 0.0], mdp.w)), digits=2), (1,:))
    #= o = KAgentState(o_x, zp, hist_p) =#
    r = mdp.obj(sp)[1]
    return (sp = sp, o = o_x, r = r)
end

struct KWorld
    dimensions::Vector # Dimensions of 2D-world
    inhabitants::Dict{Symbol, KAgentMDP} # Dictionary of agents (agent POMDPs) operating in this world
    menv::MuEnv # Environment associated with this world

    KWorld(dimensions::Vector, inhabitants::Dict, menv::MuEnv) = new(dimensions, inhabitants, menv)
end

"""Keyword constructor for a new world.

Defaults to no inhabitants (i.e., an empty dictionary.) Use `add_agent_to_world` to populate one-by-one.
"""
create_kworld(;dims::Vector, menv::MuEnv, inhabitants::Dict=Dict{String, KAgentMDP}()) = KWorld(dims, inhabitants, menv)

"""Add a single agent to the world.
"""
add_agent_to_world(kworld::KWorld, kagent::KAgentMDP) = kworld.inhabitants[kagent.name] = kagent

"""Populate world with a list of agents at once.
"""
populate_world(kworld::KWorld, kagents::Vector{KAgentMDP}) = map(kag->add_agent_to_world(kworld, kag), kagents)

"""Get total number of agents.
"""
get_num_agents(kworld::KWorld) = length(kworld.inhabitants)