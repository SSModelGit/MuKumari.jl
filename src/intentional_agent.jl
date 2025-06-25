@reexport using POMDPs

using LinearAlgebra: normalize, ⋅
using Distributions: Normal, MvNormal
using IterTools: partition
using POMDPTools, MCTS

export blindstart_KAgentState, pseudo_agent_placement,KAgentMDP, init_standard_KAgentMDP
export KWorld, create_kworld, add_agent_to_world, get_num_agents

struct KAgentMDP <: POMDPs.MDP{KAgentState, Symbol}
    name::String
    start::Matrix # Grid location of starting pose of agent
    dimensions::Tuple # Dimensions of 2D grid-world
    boxworld::GI.Polygon # 2D world constructed from dimensions
    obcs::Vector # 2D world obstacles (holes in the traversable region)
    world::GI.Polygon # Effectively traversable 2D world (built from boxworld and obcs)
    width::Float64
    obj::Function # objective function: must return two values, [Immediate reward for reaching state (KAgentState), Boolean True if Objective Accomplished]
    s::Float64 # movement speed
    w::Float64 # movement noise (in the angle, i.e. action, not in movement speed)
    menv::MuEnv # Observable environment process of the world
    v::Float64 # variance of environment observation noise process
    γ::Float64 # discount factor
    digits::Integer # rounding factor

    KAgentMDP(name::String, start::Matrix,
              dimensions::Tuple, boxworld::GI.Polygon,
              obcs::Vector, world::GI.Polygon, width::Float64,
              obj::Function,
              s::Float64, w::Float64,
              menv::MuEnv, v::Float64,
              γ::Float64, digits::Integer) = new(name, start, dimensions, boxworld, obcs, world, width, obj, s, w, menv, v, γ, digits)
end

function KAgentMDP(;
    name::String, start::Matrix,
    dimensions::Tuple, boxworld::GI.Polygon,
    obcs::Vector, world::GI.Polygon, width::Float64,
    obj::Function,
    s::Float64,
    w::Float64,
    menv::MuEnv,
    v::Float64,
    γ::Float64,
    digits::Integer)
    return KAgentMDP(name, start, dimensions, boxworld, obcs, world, width, obj, s, w, menv, v, γ, digits)
end

function init_standard_KAgentMDP(;
    name::String, start::Matrix,
    dimensions::Tuple, objl::AgentObjectiveLandscape, menv::MuEnv,
    digits::Integer=3, agent_width::Float64=0.1, agent_speed::Float64=1., ag_mvt_noise::Float64=0.05, obs_noise::Float64=0.05,
    mdp_horizon_discount::Float64=0.95)
    let d=dimensions, boxworld=GI.Polygon([[(d[1], d[1]), (d[1], d[2]), (d[2], d[2]), (d[2], d[1]), (d[1], d[1])]]), obcs=obcs_from_landscape(objl)
        KAgentMDP(name=name, start=start,
                  dimensions=d, boxworld = boxworld, obcs=obcs,
                  world=GI.Polygon([GI.getexterior(boxworld), map(o->GI.getexterior(o), obcs)...]),
                  width=agent_width,
                  obj=obj_from_landscape(objl; digits=digits),
                  s=agent_speed, w=ag_mvt_noise, menv=menv, v=obs_noise, γ=mdp_horizon_discount,
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

# Constant association between symbols (for ease of use) and movement vectors (for computation)
action_heading_assoc_kagent = Dict([(:n,  normalize([ 0,  1])),
                                    (:ne, normalize([ 1,  1])),
                                    (:e,  normalize([ 1,  0])),
                                    (:se, normalize([ 1, -1])),
                                    (:s,  normalize([ 0, -1])),
                                    (:sw, normalize([-1, -1])),
                                    (:w,  normalize([-1,  0])),
                                    (:nw, normalize([-1,  1])),
                                    (:c,  [0., 0.])])

"""Collision checking function. Stops the agent at the first possible collision.

Note that the agent will stop  at an agent's width away from the 
"""
function collision_check(xs::Matrix, xp::Matrix, pgon, width; debug::Bool=true, digits=3)
    if debug
        dist = GO.distance(GI.Point(Tuple(xp)), pgon)
        println("Point under question: ", xp, "| Stated distance: ", dist, " | Polygon: ", pgon)
    end

    # is it outside the boundaries of the traversible world?
    movement = GI.LineString([GI.Point(Tuple(xs)), GI.Point(Tuple(xp))])
    boundary_intersect = GO.intersection(movement, GI.getexterior(pgon); target=GI.PointTrait())
    if debug; println("Boundary crossings: ", boundary_intersect); end

    # or did it happen to cut through an obstacle?
    through_hole = filter(!isempty, map(GI.gethole(pgon)) do hole
        GO.intersection(movement, hole; target=GI.PointTrait())
    end) |> Iterators.flatten |> collect

    # Put it all together
    total_intersects = vcat(boundary_intersect, through_hole)
    if debug; println("Through holes are: ", through_hole); println("Concated: ", total_intersects); end
    # unique_intersects = unique(x->map(d->round(d, digits=digits), x), total_intersects)
    unique_intersects = unique(x->round.(x; digits=digits), total_intersects)
    if !isempty(unique_intersects)
        # find the closest intersection to the starting position
        intersect_info = map(unique_intersects) do isect
            isect_mat = reshape(collect(isect), (1, :))
            [isect_mat, norm(xs - isect_mat)]
        end
        intersects_by_dists = sort(intersect_info, by=x->x[2])
        if debug; println("Intersection info: ", intersects_by_dists); end

        # take closest intersection point
        nearest_collision, col_dist = intersects_by_dists[1]
        if debug; println("Nearest collision: ", nearest_collision); println("Distance to nearest collision: ", col_dist); end
        vec_reduction_frac = (col_dist - width) / col_dist
        xp = (nearest_collision .- xs) .* vec_reduction_frac .+ xs
    end

    return round.(xp; digits=digits)
end


function POMDPs.gen(mdp::KAgentMDP, s::KAgentState, a::Symbol, rng)
    # add noise to the action taken (both in direction and speed)
    real_a = reshape(round.(rand(MvNormal(action_heading_assoc_kagent[a], mdp.w)), digits=mdp.digits), (1,:)) # real action factoring in noise
    # propagate next location
    xp = @. s.x + real_a * mdp.s
    # adjust for any collision
    xp = collision_check(s.x, xp, mdp.world, mdp.width; debug=false, digits=mdp.digits)
    # make an observation vector for next location
    zp = push!(copy(s.z), predict_env(mdp.menv, xp))
    # update the next timestep's history
    hist_p = push!(copy(s.hist), s.x)
    # create state for next time step
    sp = KAgentState(xp, zp, hist_p)

    # POMDP observation refers to state observation. Noise will occur in the position of the vehicle
    # for simplicity doubling noise in observation of position with noise of movement
    o_x = xp .+ reshape(round.(rand(MvNormal([0.0, 0.0], mdp.w)), digits=mdp.digits), (1,:))

    # compute reward for reaching the next state (first output of the MDP's defined objective function)
    r = mdp.obj(sp)[1]

    # return required items for the POMDPs.gen function (next state, observation, reward)
    return (sp = sp, o = o_x, r = r)
end

struct KWorld
    solver::Union{MCTSSolver} # Untyped to allow for a broad array of possible types
    dimensions::Tuple # Dimensions of 2D-world
    inhabitants::Dict{String, KAgentMDP} # Dictionary of agents (agent POMDPs) operating in this world
    menv::MuEnv # Global environment of the world
    glob_landscape::GlobalObjectiveLandscape # Global objective landscape of the world

    KWorld(solver, dimensions::Tuple, inhabitants::Dict, menv::MuEnv, glob_landscape::GlobalObjectiveLandscape) = new(solver,
                                                                                                                      dimensions,
                                                                                                                      inhabitants,
                                                                                                                      menv, glob_landscape)
end

"""
$(SIGNATURES)

Keyword constructor for a new world.

Defaults to no inhabitants (i.e., an empty dictionary.) Use `add_agent_to_world` to populate one-by-one.
"""
create_kworld(;
              solver::Union{MCTSSolver}, dims::Tuple,
              menv::MuEnv, gobj::GlobalObjectiveLandscape,
              inhabitants::Dict=Dict{String, KAgentMDP}()) = KWorld(solver, dims, inhabitants, menv, gobj)

"""
$(SIGNATURES)

Base function to add a single agent to the world.

Separated in case there exists an already-defined agent MDP that needs to be added.
"""
add_agent_to_world(kworld::KWorld, kagent::KAgentMDP) = kworld.inhabitants[kagent.name] = kagent

"""
$(SIGNATURES)

Proper keyword-based constructor to create and add a single agent to the world.

Explicitly calls out all the arguments for the agent.
Lacks safety checks.
"""
function add_agent_to_world(;
                            kworld::KWorld,
                            name::String, start_pos::Matrix,
                            ag_flist::Vector, ag_sensor_list::Vector,
                            dimensions::Union{Tuple, Nothing}=nothing, digits::Integer=3, mdp_horizon_discount::Float64=0.95,
                            agent_width::Float64=0.1, agent_speed::Float64=1., ag_mvt_noise::Float64=0.05,
                            obs_noise::Float64=0.05)
    if isnothing(dimensions)
        dimensions = kworld.dimensions
    end
    ag_menv = tangle_agent_env(kworld.menv, ag_sensor_list)
    ag_landscape = tangle_agent_landscape(kworld.glob_landscape, ag_flist)
    agent_mdp = init_standard_KAgentMDP(name=name, start=start_pos,
                                        dimensions=dimensions, objl=ag_landscape, menv=ag_menv,
                                        digits=digits, mdp_horizon_discount=mdp_horizon_discount,
                                        agent_width=agent_width, agent_speed=agent_speed, ag_mvt_noise=ag_mvt_noise,
                                        obs_noise=obs_noise)
    add_agent_to_world(kworld, agent_mdp)
end

"""
$(SIGNATURES)

Dictionary-based constructor for creating and adding an agent to the world.

Necessarily not only keywords as multiple dispatch does not operate on keyword args.
"""
function add_agent_to_world(kworld::KWorld, agent_params::Dict; add_safely::Bool=true)
    if add_safely
        param_list = keys(agent_params)
        print("Checking all required components exist...")
        @assert :name ∈ param_list "No :name specified!"
        @assert :start ∈ param_list "No :start position specified!"
        @assert :flist ∈ param_list "No feature list (:flist) specified!"
        @assert :elist ∈ param_list "No observable environment characteristics (:elist) specified!"
        print(" ok.\nChecking parameter definitions obey global world definition...")
        @assert agent_params[:flist] ⊆ kworld.glob_landscape.feature_list "Agent feature list exceeds globally captured features!"
        @assert agent_params[:elist] ⊆ kworld.menv.μ_order "Agent can observe environment characteristics not captured in the global environment!"
        if :dims ∈ param_list
            @assert agent_params[:dims][1] ≥ kworld.dimensions[1] "Agent is operating beyond the lower global bounds!"
            @assert agent_params[:dims][2] ≤ kworld.dimensions[2] "Agent is operating beyond the upper global bounds!"
        end
        println(" ok.")
    end
    dims=get(agent_params, :dims, nothing)
    digits=get(agent_params, :digits, 3)
    mdp_horizon_discount=get(agent_params, :γ, 0.95)
    agent_width=get(agent_params, :width, 0.1)
    agent_speed=get(agent_params, :s, 1.)
    ag_mvt_noise=get(agent_params, :w, 0.05)
    obs_noise=get(agent_params, :v, 0.05)
    add_agent_to_world(;
                       kworld=kworld, name=agent_params[:name], start_pos=agent_params[:start],
                       ag_flist=agent_params[:flist], ag_sensor_list=agent_params[:elist],
                       dimensions=dims, digits=digits, mdp_horizon_discount=mdp_horizon_discount,
                       agent_width=agent_width, agent_speed=agent_speed, ag_mvt_noise=ag_mvt_noise,
                       obs_noise=obs_noise)
end

"""Populate world with a list of agents at once.
"""
populate_world(kworld::KWorld, kagents::Vector{KAgentMDP}) = map(kag->add_agent_to_world(kworld, kag), kagents)

"""Get total number of agents.
"""
get_num_agents(kworld::KWorld) = length(kworld.inhabitants)