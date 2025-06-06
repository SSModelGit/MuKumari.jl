@reexport using POMDPs

using LinearAlgebra: normalize, ⋅
using Distributions: Normal, MvNormal
using IterTools: partition
using POMDPTools, MCTS

import GeoInterface as GI
import GeometryOps as GO

export KAgentState, blindstart_KAgentState, pseudo_agent_placement,KAgentMDP, init_standard_KAgentMDP
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
    boxworld::GI.Polygon # 2D world constructed from dimensions
    obj::Function # objective function: must return two values, [Immediate reward for reaching state (KAgentState), Boolean True if Objective Accomplished]
    s::Float64 # movement speed
    w::Float64 # movement noise (in the angle, i.e. action, not in movement speed)
    menv::MuEnv # Observable environment process of the world
    v::Float64 # variance of environment observation noise process
    γ::Float64 # discount factor
    digits::Integer # rounding factor

    KAgentMDP(name::String, start::Matrix,
              dimensions::Tuple, boxworld::GI.Polygon,
              obj::Function,
              s::Float64, w::Float64,
              menv::MuEnv, v::Float64,
              γ::Float64, digits::Integer) = new(name, start, dimensions, boxworld, obj, s, w, menv, v, γ, digits)
end

function KAgentMDP(;
    name::String, start::Matrix,
    dimensions::Tuple, boxworld::GI.Polygon,
    obj::Function,
    s::Float64,
    w::Float64,
    menv::MuEnv,
    v::Float64,
    γ::Float64,
    digits::Integer)
    return KAgentMDP(name, start, dimensions, boxworld, obj, s, w, menv, v, γ, digits)
end

function init_standard_KAgentMDP(;
    name::String, start::Matrix,
    dimensions::Tuple, obj::Function, menv::MuEnv)
    let d=dimensions
        KAgentMDP(name=name, start=start,
                  dimensions=d,
                  boxworld=GI.Polygon([[(d[1], d[1]), (d[1], d[2]), (d[2], d[2]), (d[2], d[1]), (d[1], d[1])]]),
                  obj=obj,
                  s=1., w=0.05, menv=menv, v=0.05, γ=0.95,
                  digits=3)
    end
end

"""Start the agent at a desired location.

The agent starts unaware of the world beyond its immediate location.
"""
blindstart_KAgentState(mdp::KAgentMDP, x::Matrix) = KAgentState(x, [predict_env(mdp.menv, x)], Matrix[])

"""Shifts an agent instantaneously to any arbitrary desired location.

Note that this violates the agent dynamics.
"""
pseudo_agent_placement(s::KAgentState, x::Matrix) = KAgentState(x, copy(s.z), copy(s.hist))

function Base.show(io::IO, mdp::KAgentMDP)
    println(io, "KAgent MDP")
    println(io, "\tLength of grid-space along the x-dimension: $(mdp.dimensions)")
    println(io, "\tObjective function: $(mdp.obj)")
    println(io, "\tEnvironment characteristics observed: $(mdp.menv.μ_order)")
end

POMDPs.isterminal(mdp::KAgentMDP, s::KAgentState) = mdp.obj(s)[2]

POMDPs.initialstate(mdp::KAgentMDP) = Deterministic(blindstart_KAgentState(mdp, mdp.start))

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

function stay_in_boundary(xs::Matrix, xp::Matrix, pgon; new_partition=nothing, debug::Bool=false, tol=1e-5)
    if GO.distance(GI.Point(Tuple(xp)), pgon) > tol
        println("Offending point: ", xp, "| Stated distance: ", GO.distance(GI.Point(Tuple(xp)), pgon), " | Polygon: ", pgon)
        let boundary = GI.getexterior(pgon), movement = GI.LineString([GI.Point(Tuple(xs)), GI.Point(Tuple(xp))]), new_boundary = [], found_intersect = false
            st_v=copy(xs)
            part = isnothing(new_partition) ? partition(boundary.geom, 2, 1) : new_partition
            for line in part
                linestring = GI.LineString([GI.Point(line[1]), GI.Point(line[2])])
                intersect = GO.intersection(movement, linestring; target=GI.PointTrait())
                println("Partition: ", line,"| Line: ", linestring, "| Intersections: ", intersect)
                if !(isempty(intersect) || found_intersect)
                    found_intersect = true
                    st_v = reshape(collect(intersect[end]), (1,:))
                    end_v = reshape(collect(GI.getcoord(GI.getpoint(linestring)[2])), (1,:))
                    inter_h = normalize(end_v - st_v)
                    mvt_v = xp - st_v
                    xp = (mvt_v ⋅ inter_h) .* inter_h .+ st_v
                    if debug
                        inter_v = end_v - st_v
                        println("Found the intersection: ", st_v)
                        println("Vector of movement: ", mvt_v)
                        println("Vector of intercepted side: ", inter_v)
                        println("The rest of the movement then is: ", (mvt_v ⋅ inter_h) .* inter_h)
                        println("Final endpoint is: ", (mvt_v ⋅ inter_h) .* inter_h .+ st_v)
                    end
                else
                    push!(new_boundary, line)
                end
            end
            return stay_in_boundary(st_v, xp, pgon; new_partition=new_boundary, debug=debug)
        end
    else
        return xp
    end
end

function POMDPs.gen(mdp::KAgentMDP, s::KAgentState, a::Symbol, rng)
    real_a = reshape(round.(rand(MvNormal(action_heading_assoc_kagent[a], mdp.w)), digits=mdp.digits), (1,:)) # real action factoring in noise
    xp = @. s.x + real_a * mdp.s
    xp = stay_in_boundary(s.x, xp, mdp.boxworld)
    # isoutside = GO.distance(GI.Point(Tuple(xp)), mdp.boxworld) > 0
    # world_intersect = GO.intersection(GI.LineString([GI.Point(Tuple(s.x)), GI.Point(Tuple(xp))]), GI.getexterior(mdp.boxworld); target=GI.PointTrait())
    # xp = (!isempty(world_intersect) && isoutside) ? reshape(collect(world_intersect[end]), (1,:)) : xp
    # if GO.distance(GI.Point(Tuple(xp)), mdp.boxworld) > 0
    #     xp = copy(s.x)
    # end
    zp = push!(copy(s.z), predict_env(mdp.menv, xp))
    hist_p = push!(copy(s.hist), s.x)
    sp = KAgentState(xp, zp, hist_p)

    # POMDP observation refers to state observation. Noise will occur in the position of the vehicle
    # for simplicity doubling noise in observation of position with noise of movement
    o_x = xp .+ reshape(round.(rand(MvNormal([0.0, 0.0], mdp.w)), digits=mdp.digits), (1,:))
    
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