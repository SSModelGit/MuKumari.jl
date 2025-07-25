export KAgentPOMDP, init_standard_KAgentPOMDP, KAgentBeliefUpdater

@with_kw_noshow struct KAgentPOMDP <: POMDPs.POMDP{KAgentState, Symbol, Vector{Float64}}
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

function init_standard_KAgentPOMDP(;
    name::String, start::Matrix,
    dimensions::Tuple, objl::AgentObjectiveLandscape, menv::MuEnv,
    digits::Integer=3, agent_width::Float64=0.1, agent_speed::Float64=1., ag_mvt_noise::Float64=0.05, obs_noise::Float64=0.05,
    mdp_horizon_discount::Float64=0.95)
    let d=dimensions, boxworld=GI.Polygon([[(d[1], d[1]), (d[1], d[2]), (d[2], d[2]), (d[2], d[1]), (d[1], d[1])]]), obcs=obcs_from_landscape(objl)
        KAgentPOMDP(name=name, start=start,
                    dimensions=d, boxworld = boxworld, objl=objl, obcs=obcs,
                    obj=obj_from_landscape(objl; digits=digits),
                    world=GI.Polygon([GI.getexterior(boxworld), map(o->GI.getexterior(o), obcs)...]),
                    width=agent_width,
                    s=agent_speed, w=ag_mvt_noise, menv=menv, v=obs_noise, γ=mdp_horizon_discount,
                    digits=digits)
    end
end

"""
$(SIGNATURES)

Redefine the blind start function for the KAgentPOMDP.
"""
blindstart_KAgentState(pomdp::KAgentPOMDP, x::Matrix) = KAgentState(x, [predict_env(pomdp.menv, x)], Matrix[])

function Base.show(io::IO, pomdp::KAgentPOMDP)
    println(io, "KAgent POMDP")
    println(io, "\tLength of grid-space along the x-dimension: $(pomdp.dimensions)")
    println(io, "\tObjective function: $(pomdp.obj)")
    println(io, "\tEnvironment characteristics observed: $(pomdp.menv.μ_order)")
end

POMDPs.isterminal(pomdp::KAgentPOMDP, s::KAgentState) = pomdp.obj(s)[2]

POMDPs.initialstate(pomdp::KAgentPOMDP) = Deterministic(blindstart_KAgentState(pomdp, pomdp.start))

POMDPs.initialobs(pomdp::KAgentPOMDP, s) = Deterministic([state(s)..., z(s)..., t(s)])

POMDPs.discount(pomdp::KAgentPOMDP) = pomdp.γ

"""
$(SIGNATURES)

Action space of the KAgentPOMDP.

Represents the eight cardinal and diagonal directions, along with "staying in the center".

Currently assumes constant motion speed.
"""
POMDPs.actions(pomdp::KAgentPOMDP) = [:n, :ne, :e, :se, :s, :sw, :w, :nw, :c]

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

shape_state_as_obs(pomdp::KAgentPOMDP, s::KAgentState) = [state(s)..., z(s)...,t(s)...]

function POMDPs.gen(pomdp::KAgentPOMDP, s::KAgentState, a::Symbol, rng)
    # add noise to the action taken (both in direction and speed)
    real_a = reshape(round.(rand(rng, MvNormal(action_heading_assoc_kagent[a], pomdp.w)), digits=pomdp.digits), (1,:)) # real action factoring in noise
    # propagate next location
    xp = @. s.x + real_a * pomdp.s
    # adjust for any collision
    xp = collision_check(s.x, xp, pomdp.world, pomdp.width; debug=false, digits=pomdp.digits)
    # make an observation vector for next location
    zp = push!(copy(s.z), predict_env(pomdp.menv, xp))
    # update the next timestep's history
    hist_p = push!(copy(s.hist), s.x)
    # create state for next time step
    sp = KAgentState(xp, zp, hist_p)

    # POMDP observation refers to state observation. Noise will occur in the position of the vehicle
    # for simplicity doubling noise in observation of position with noise of movement
    o_x = xp .+ reshape(round.(rand(rng, MvNormal([0.0, 0.0], pomdp.w)), digits=pomdp.digits), (1,:))
    # NEW ADDITION: trying out a vectorization of the agent state that captures all immediate information in one vector for RL-ing
    o = [o_x..., z(sp)..., t(sp)]

    # compute reward for reaching the next state (first output of the MDP's defined objective function)
    r = pomdp.obj(sp)[1]

    # return required items for the POMDPs.gen function (next state, observation, reward)
    return (sp = sp, o = o, r = r)
end

"""
$(SIGNATURES)

Belief-state updater for the KAgentPOMDP.

It essentially reconstructs a new deterministic state out of the observation vector.

#TODO: Make this non-deterministic (i.e., observation noise should enter here.)
"""
@with_kw struct KAgentBeliefUpdater <: POMDPs.Updater
    state_dims = 2
    env_dims = 3
end

"""
$(SIGNATURES)

Defines the initial belief for the KAgent POMDP.

For the most part, this should be used like
```
POMDPs.initialize_belief(u, initialstate(pomdp))
```

This will result in the same initial belief distribution as the actual initial state distribution.
"""
POMDPs.initialize_belief(u::KAgentBeliefUpdater, d::Any) = d

function POMDPs.update(bu::KAgentBeliefUpdater, old_b, action, obs)
    old_s = rand(old_b)
    xn = reshape(obs[1:bu.state_dims], (1, bu.state_dims))
    zn = obs[bu.state_dims+1:bu.state_dims+bu.env_dims]
    hist = push!(copy(old_s.hist), old_s.x)
    z = push!(copy(old_s.z), zn)
    Deterministic(KAgentState(xn, z, hist))
end