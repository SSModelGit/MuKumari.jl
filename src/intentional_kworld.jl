export KWorld, create_kworld, add_agent_to_world, get_num_agents

@with_kw_noshow struct KWorld
    solver::Union{MCTSSolver, DPWSolver} # Untyped to allow for a broad array of possible types
    dimensions::Tuple # Dimensions of 2D-world
    inhabitants::Dict{String, T} where T <: Union{KAgentMDP, KAgentPOMDP} = Dict{String, KAgentPOMDP}() # Dictionary of agents operating in this world
    menv::MuEnv # Global environment of the world
    glob_landscape::GlobalObjectiveLandscape # Global objective landscape of the world

    # KWorld(solver, dimensions::Tuple, inhabitants::Dict, menv::MuEnv, glob_landscape::GlobalObjectiveLandscape) = new(solver,
    #                                                                                                                   dimensions,
    #                                                                                                                   inhabitants,
    #                                                                                                                   menv, glob_landscape)
end

inhabitant_names(kworld::KWorld; offset="    ", base="\t") = mapreduce(x->"$(base)$(offset)Agent #$(x[1]): \"$(x[2][1])\"\n",
                                                                       *, enumerate(kworld.inhabitants); init="")

function Base.show(io::IO, kworld::KWorld)
    println(io, "Brief summary of this KWorld:")
    println(io, "\tType of the default solver: $(typeof(kworld.solver))")
    println(io, "\tDimensions of the world: $(kworld.dimensions)")
    # new lines are auto-added by the name list functions
    print(io, "\tCurrent inhabitants:\n$(inhabitant_names(kworld))")
    print(io, "\tList of characteristics:\n$(muenv_characteristics_namelist(kworld.menv; base="\t", offset="    "))")
    print(io, "\tList of objective features present in total:\n$(feature_name_list_from_vec(kworld.glob_landscape.feature_list))")
end

"""
$(SIGNATURES)

Keyword constructor for a new world.

Defaults to no inhabitants (i.e., an empty dictionary.) Use `add_agent_to_world` to populate one-by-one.
"""
function create_kworld(; 
                              solver::Union{MCTSSolver, DPWSolver}, dims::Tuple,
                              menv::MuEnv, gobj::GlobalObjectiveLandscape,
                              inhabitants::Dict=Dict{String, KAgentPOMDP}())
    @info "Initializing inhabitants as POMDPs"
    KWorld(solver, dims, inhabitants, menv, gobj)
end

"""
$(SIGNATURES)

Base function to add a single agent to the world.

Separated in case there exists an already-defined agent MDP that needs to be added.
"""
add_agent_to_world(kworld::KWorld, kagent::Union{KAgentMDP, KAgentPOMDP}) = kworld.inhabitants[kagent.name] = kagent

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
    @info "Currently initializing a KAgentPOMDP! Capacity to instead add a KAgentMDP still unaddressed."
    agent_mdp = init_standard_KAgentPOMDP(name=name, start=start_pos,
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
populate_world(kworld::KWorld, kagents::Vector{T}) where T <: Union{KAgentMDP, KAgentPOMDP} = map(kag->add_agent_to_world(kworld, kag), kagents)

"""Get total number of agents.
"""
get_num_agents(kworld::KWorld) = length(kworld.inhabitants)