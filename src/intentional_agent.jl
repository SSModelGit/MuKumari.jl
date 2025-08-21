export blindstart_KAgentState, pseudo_agent_placement, KAgentMDP, init_standard_KAgentMDP

@with_kw_noshow struct KAgentMDP <: POMDPs.MDP{KAgentState, Symbol}
    name::String
    start::Matrix # Grid location of starting pose of agent
    dimensions::Tuple # Dimensions of 2D grid-world
    boxworld::GI.Polygon # 2D world constructed from dimensions
    objl::AgentObjectiveLandscape # Agent objective landscape
    obcs::Vector # 2D world obstacles (holes in the traversable region) - separated from landscape for convenience
    obj::Function # objective function: must return two values, [Immediate reward for reaching state (KAgentState), Boolean True if Objective Accomplished]
    world::GI.Polygon # Effectively traversable 2D world (built from boxworld and obcs)
    width::Float64
    s::Float64 # movement speed
    w::Float64 # movement noise (in the angle, i.e. action, not in movement speed)
    menv::MuEnv # Observable environment process of the world
    v::Float64 # variance of environment observation noise process
    γ::Float64 # discount factor
    digits::Integer # rounding factor
end

function init_standard_KAgentMDP(;
    name::String, start::Matrix,
    dimensions::Tuple, objl::AgentObjectiveLandscape, menv::MuEnv,
    digits::Integer=3, agent_width::Float64=0.1, agent_speed::Float64=1., ag_mvt_noise::Float64=0.05, obs_noise::Float64=0.05,
    mdp_horizon_discount::Float64=0.95)
    let d=dimensions, boxworld=GI.Polygon([[(d[1], d[1]), (d[1], d[2]), (d[2], d[2]), (d[2], d[1]), (d[1], d[1])]]), obcs=obcs_from_landscape(objl)
        KAgentMDP(name=name, start=start,
                  dimensions=d, boxworld = boxworld, objl=objl, obcs=obcs,
                  obj=obj_from_landscape(objl; digits=digits),
                  world=GI.Polygon([GI.getexterior(boxworld), map(o->GI.getexterior(o), obcs)...]),
                  width=agent_width,
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

POMDPs.initialobs(mdp::KAgentMDP, s) = Deterministic([state(s)..., z(s)..., t(s)])

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

function POMDPs.gen(mdp::KAgentMDP, s::KAgentState, a::Symbol, rng)
    # add noise to the action taken (both in direction and speed)
    real_a = reshape(round.(rand(rng, MvNormal(action_heading_assoc_kagent[a], mdp.w)), digits=mdp.digits), (1,:)) # real action factoring in noise
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

    # compute reward for reaching the next state (first output of the MDP's defined objective function)
    r = mdp.obj(sp)[1]

    # return required items for the POMDPs.gen function (next state, observation, reward)
    return (sp = sp, r = r)
end