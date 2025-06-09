@reexport using POMDPs

using LinearAlgebra: normalize, ⋅
using Distributions: Normal, MvNormal
using IterTools: partition
using POMDPTools, MCTS

export blindstart_KAgentState, pseudo_agent_placement,KAgentMDP, init_standard_KAgentMDP
export KWorld, create_empty_kworld, add_agent_to_world, get_num_agents

struct KAgentMDP <: POMDPs.MDP{KAgentState, Symbol}
    name::String
    start::Matrix # Grid location of starting pose of agent
    dimensions::Tuple # Dimensions of 2D grid-world
    boxworld::GI.Polygon # 2D world constructed from dimensions
    obcs::Vector # 2D world obstacles (holes in the traversable region)
    world::GI.Polygon # Effectively traversable 2D world (built from boxworld and obcs)
    obj::Function # objective function: must return two values, [Immediate reward for reaching state (KAgentState), Boolean True if Objective Accomplished]
    s::Float64 # movement speed
    w::Float64 # movement noise (in the angle, i.e. action, not in movement speed)
    menv::MuEnv # Observable environment process of the world
    v::Float64 # variance of environment observation noise process
    γ::Float64 # discount factor
    digits::Integer # rounding factor

    KAgentMDP(name::String, start::Matrix,
              dimensions::Tuple, boxworld::GI.Polygon,
              obcs::Vector, world::GI.Polygon,
              obj::Function,
              s::Float64, w::Float64,
              menv::MuEnv, v::Float64,
              γ::Float64, digits::Integer) = new(name, start, dimensions, boxworld, obcs, world, obj, s, w, menv, v, γ, digits)
end

function KAgentMDP(;
    name::String, start::Matrix,
    dimensions::Tuple, boxworld::GI.Polygon,
    obcs::Vector, world::GI.Polygon,
    obj::Function,
    s::Float64,
    w::Float64,
    menv::MuEnv,
    v::Float64,
    γ::Float64,
    digits::Integer)
    return KAgentMDP(name, start, dimensions, boxworld, obcs, world, obj, s, w, menv, v, γ, digits)
end

function init_standard_KAgentMDP(;
    name::String, start::Matrix,
    dimensions::Tuple, objl::ObjectiveLandscape, menv::MuEnv, digits::Integer=3)
    let d=dimensions, boxworld=GI.Polygon([[(d[1], d[1]), (d[1], d[2]), (d[2], d[2]), (d[2], d[1]), (d[1], d[1])]]), obcs=obcs_from_landscape(objl)
        KAgentMDP(name=name, start=start,
                  dimensions=d, boxworld = boxworld, obcs=obcs,
                  world=GI.Polygon([GI.getexterior(boxworld), map(o->GI.getexterior(o), obcs)...]),
                  obj=obj_from_landscape(objl; digits=digits),
                  s=1., w=0.05, menv=menv, v=0.05, γ=0.95,
                  digits=digits)
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

function collision_check(xs::Matrix, xp::Matrix, pgon; trapped=false, debug::Bool=true, tol=1e-5, iter_count=1)
    if iter_count > 10
        return xp
    else
        if debug; println("Iter no.: ", iter_count); end
    end
    # is it outside the boundaries of the traversible world?
    if debug
        dist = GO.distance(GI.Point(Tuple(xp)), pgon)
        println("Point under question: ", xp, "| Stated distance: ", dist, " | Polygon: ", pgon);
    end
    # or did it happen to cut through a (thin) obstacle?
    movement = GI.LineString([GI.Point(Tuple(xs)), GI.Point(Tuple(xp))])
    boundary_intersect = GO.intersection(movement, GI.getexterior(pgon); target=GI.PointTrait())
    if debug; println("Boundary crossings: ", boundary_intersect); end
    if isempty(boundary_intersect)
        adj_movement = movement
        isoutside = false
    else
        # draw line to an end point just near the boundary crossing
        adj_end = xs .+ ((reshape(collect(boundary_intersect[end]), (1,:)) - xs) .* (1-5*tol))
        adj_movement = GI.LineString([GI.Point(Tuple(xs)), GI.Point(Tuple(adj_end))])
        isoutside = true
    end
    through_hole = filter(!isempty, map(GI.gethole(pgon)) do hole
        GO.intersection(adj_movement, hole; target=GI.PointTrait())
    end)
    println("Through holes are: ", through_hole)

    if isoutside || !isempty(through_hole)
        boundaries = map(geo->partition(geo.geom, 2, 1), pgon.geom) |> Iterators.flatten
        let intersects = []
            for line in boundaries
                linestring = GI.LineString(GI.Point.(collect(line)))
                intersect = GO.intersection(movement, linestring; target=GI.PointTrait())
                if !isempty(intersect)
                    interpoint = reshape(collect(intersect[end]), (1, :))
                    if debug; println("Partition: ", line,"| Line: ", linestring, "| Intersections: ", intersect); end
                    push!(intersects, Any[copy(reshape(collect(line[1]), (1,:))),
                                          copy(reshape(collect(line[2]), (1,:))),
                                          copy(interpoint), norm(xs - interpoint)])
                end
            end
            for line_data in sort(intersects, by=s->s[4])
                v_end_1, v_end_2, interp, d_to_interp = line_data
                # st_v = line_data[2]
                # end_v = reshape(collect(line[2]), (1,:))
                inter_v = v_end_1 - interp
                inter_v = iszero(norm(inter_v)) ? v_end_2 - interp : inter_v
                inter_h = normalize(inter_v)
                mvt_v = xp - interp
                shifted_v = (mvt_v ⋅ inter_h) .* inter_h
                if debug
                    println("Found the intersection: ", interp)
                    println("Vector of movement: ", mvt_v)
                    println("Vector of intercepted side: ", inter_v)
                    println("The rest of the movement then is: ", shifted_v)
                end
                if norm(mvt_v - shifted_v) > tol
                    xp = shifted_v .+ interp
                    if debug; println("Final endpoint is: ", xp); end
                    stuck = norm(interp - xs) < tol # check if we've moved or just rotated vectors
                    if stuck && trapped # corner check
                        return xs # turns out we've arrived at a (non-right) corner
                    else
                        return collision_check(interp, xp, pgon; trapped = stuck, debug=debug, iter_count=iter_count+1)
                    end
                end
            end
        end
        return xp #idk how you got here but just toss it back I guess clip through everything like a boss
    else
        return xp
    end
end

function POMDPs.gen(mdp::KAgentMDP, s::KAgentState, a::Symbol, rng)
    real_a = reshape(round.(rand(MvNormal(action_heading_assoc_kagent[a], mdp.w)), digits=mdp.digits), (1,:)) # real action factoring in noise
    xp = @. s.x + real_a * mdp.s
    xp = collision_check(s.x, xp, mdp.world)
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