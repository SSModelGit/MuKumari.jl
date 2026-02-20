using MuKumari

using LinearAlgebra: norm, normalize

using POMDPTools, MCTS, POMDPLinter

# addressing weird load order bugs
using Plots
using CUDA, cuDNN

# Commenting out CairoMakie due to current issue compiling with Plots and GR_jll
# using CairoMakie
using ProgressMeter

# Make sure to load Plots before Crux, because of some weird load order bug
using Crux

using JLD2: @save, @load

using Flux # Going to add this in now to start forming the networks
using BSON
using Dates

## Environment Feature types:
# Surface:    :surf
# Aerial:     :aer
# Subsurface: :sub
## Agent Feature types:
# Agent 1: :ag1
# Agent 2: :ag2

# since copy(::MCTSSolver) isn't defined for some reason, just use this instead of figuring out a copy extension
base_solver() = MCTSSolver(n_iterations=1000, depth=20, exploration_constant=10.0)

# Sloppy way to force a point within the dimensions (truncate coordinates along both axes to be within dim limits)
minmax_to_dims(x, dims::Tuple; tol=1e-5) = min(max(x, dims[1]+tol), dims[2]-tol)
constrain_to_dims(p::Tuple, dims::Tuple; tol=1e-5) = [minmax_to_dims(p[1], dims; tol=tol), minmax_to_dims(p[2], dims; tol=tol)]

"""
    obcs_gen(flist::Vector{Symbol}, num_obcs::Integer, dims::Tuple;
                  min_dist::Float64=0.5, size_var::Float64=0.1, obc_risk::Float64=10., obc_impact::Float64=10., digits=2)

Randomly generates obstacles distributed throughout space.

No checking if it overlaps with goal points, on purpose.
    Note that the average size of an obstacle is equivalent to the minimum spacing allowed between obstacle centers.
    (i.e. any pair of obstacles can overlap partially at most.)
"""
function obcs_gen(flist::Vector{Symbol}, num_obcs::Integer, dims::Tuple;
                  min_dist::Float64=0.5, size_var::Float64=0.1, obc_risk::Float64=10., obc_impact::Float64=10., digits=2)
    # Initialize count of filled features
    flist_count = Dict([(f, 0) for f in flist])

    # Initialize list of unfilled features
    unfilled_ftypes = filter(f->f.second < num_obcs, flist_count)


    # Initialize occupancy matrix of obstacle centers
    width = Integer(floor((dims[2] - dims[1])/min_dist) + 1) # technically half-width of the obstacle
    obc_centers = zeros(Bool, (width, width)) # matrix is empty so all centers are false, i.e. unoccupied

    # Initialize vector of obstacles
    obcs = []

    # loop until no more features to fill
    while !isempty(unfilled_ftypes)
        # pick center at random
        possible_center = rand(1:width, (1,2))
        if !obc_centers[possible_center...]
            obc_centers[possible_center...] = true # mark center as occupied
            physical_center = (possible_center .- 1.) .* min_dist .+ dims[1] # identify actual location of center in 2D-space
            physical_center = reshape(physical_center, (2,))

            # identify four corners of obstacle using polar coordinates
            angles = deg2rad.([rand(1:90), rand(91:180), rand(181:270), rand(271:360)])
            radius = [(min_dist + rand() * 2 * size_var - size_var) for i in 1:4] # uniform variance \pm size_var around min_dist
            corner_vecs = [[radius[i] * cos(angles[i]), radius[i] * sin(angles[i])] for i in 1:4] # convert to cartesian vectors
            corners = [Tuple(physical_center .+ corner_vecs[i]) for i in 1:4] # determine Tuple cartesian coordinates for corners
            corners = map(v->round.(v, digits=2), corners) # clean up slightly so we avoid nasty issues with floating point precision
            corners = map(v->constrain_to_dims(v, dims), corners) # imperfect way of ensuring obstacles stay in-bounds
            push!(corners, corners[1]) # need to close off the geometry by repeating the first point

            relevant_feature_mask = rand(Bool, (length(unfilled_ftypes),)) # determine which features this obstacle is relevant to
            # construct the obstacle representation for each applicable feature type
            for f in collect(keys(unfilled_ftypes))[relevant_feature_mask]
                flist_count[f] += 1 # update count for no. of obstacles for given feature
                # TODO: add some variance w.r.t. the risk and impact of an obstacle
                push!(obcs, (f, Dict(:poly => copy(corners), :risk => obc_risk, :impact => obc_impact)))
            end

            # update unfilled types
            unfilled_ftypes = filter(f->f.second < num_obcs, flist_count)
        end
    end

    return obcs
end

"""
    goal_gen(flist::Vector{Symbol}, num_goals::Integer, dims::Tuple;
                  size::Float64=0.75, min_dist::Float64=0.5, strength::Float64=10., influence::Float64=5.)

Randomly generates goals distributed throughout space.

No checking if it overlaps with obstacles, on purpose.
    Note that the average size of an goal is separate from the minimum spacing allowed between goal centers (and is by default bigger!)
"""
function goal_gen(flist::Vector{Symbol}, num_goals::Integer, dims::Tuple;
                  size::Float64=0.75, min_dist::Float64=0.5, strength::Float64=10., influence::Float64=5.)
    # count number of applicable features
    num_features = length(flist)
    # Initialize count of filled features
    flist_count = Dict([(f, 0) for f in flist])

    # Initialize list of unfilled features
    unfilled_ftypes = filter(f->f.second < num_goals, flist_count)

    # initialize goal list
    goal_list = []

    # initialize goal occupancy grid
    width = Integer(floor((dims[2] - dims[1])/min_dist) + 1) # technically half-width
    goal_centers = zeros(Bool, (width, width)) # matrix is empty so all centers are false, i.e. unoccupied

    # construct a number of goals. Goals can count towards multiple features (leading to a total count > num_goals)
    while !isempty(unfilled_ftypes)
        # identify features this goal will apply to
        gtypes = flist[rand(Bool, num_features)]

        # pick at random using equidistant spacings across dimensions (using min_dist as the spacing constant)
        possible_goal = rand(1:width, (1,2))
        if !goal_centers[possible_goal...]
            goal_centers[possible_goal...] = true
            
            # convert into physical spacing
            physical_goal = (possible_goal .- 1.) .* min_dist .+ dims[1]

            # loop through all the goal types
            for gtype in gtypes
                flist_count[gtype] += 1
                push!(goal_list, (gtype, Dict(:target=>reshape(physical_goal, (1,2)), :strength=>strength, :influence=>influence, :size=>size)))
            end

            # update unfilled types
            unfilled_ftypes = filter(f->f.second < num_goals, flist_count)
        end
    end
    return goal_list
end

"""
    init_world(dims::Tuple=(0., 10.), flist::Vector{Symbol}=[:aer, :surf, :sub]; num_obcs::Integer=5, num_goals::Integer=3)

Constructs a world to instantiate agents within. Currently uses fixed presets for the world environment model and horizon objectives.

Uses the functions `obcs_gen` and `goal_gen` to randomly populate obstacles and goals within the world's dimensions.
"""
function init_world(dims::Tuple=(0., 10.), flist::Vector{Symbol}=[:aer, :surf, :sub]; num_obcs::Integer=5, num_goals::Integer=3)
    obcs = obcs_gen(flist, num_obcs, dims)
    goals = goal_gen(flist, num_goals, dims)
    urgency = [(:a1, 1.5), (:a2, 1.0), (:a3, 0.5)]

    # Define global objective landscape
    globj_scape = GlobalObjectiveLandscape(; goals=goals, obstacles=obcs, horizons=urgency)

    # define global environment
    menv = let μfs = [(:sin, x->sin(x[1]) + cos(x[2])),
                    (:exp, x->100*exp(-norm(x-[8 8.])^2 / 1.)),
                    (:lin, x->x[1]^2 + x[2])],
            μs = [:sin, :exp, :lin];
        MuEnv(3, μs, Dict(μfs));
    end

    # Define world to hold all agents
    solver = base_solver()
    # solver = DPWSolver(n_iterations=1000, depth=20, exploration_constant=1.0)
    kworld = create_kworld(; solver=solver, dims=dims, gobj=globj_scape, menv=menv)

    return kworld
end

"""
    init_agent(kworld::KWorld, name::String="ag1";
                    ag_flist::Vector{Symbol}, ag_envs::Vector{Symbol}, start::Union{Nothing, Matrix}=nothing,
                    w::Float64=0., v::Float64=0.)

Initializes an agent within a provided world using the specified parameters.

Not all agent parameters can be specified - ex. Agent MDP horizon γ.
    - Read documentation on `MuKumari.add_agent_to_world` for more details.

When a start position is not provided (i.e. the keyword argument is set to `nothing`, as is the default), a random start position is chosen instead.
In case the random position chosen is within a goal point (rendering the problem trivial), the initialization function will try again.
    For simplicity, I implemented checking this by using `POMDPs.isterminal`, which also unfortunately means the entire agent needs to be constructed first.
    In turn, this means that I recursively call the function if the first attempt fails (`start`=`nothing`), and return the first success.
        Likewise, if the provided start value fails, it also triggers the recursive calls.
"""
function init_agent(kworld::KWorld, name::String="ag1";
                    ag_flist::Vector{Symbol}, ag_envs::Vector{Symbol}, start::Union{Nothing, Matrix}=nothing,
                    w::Float64=0., v::Float64=0.)
    if isnothing(start)
        let dims=kworld.dimensions
            # sub-sample from 1 -> 19 (where 0 and 20 represent dimension bounds), then transform back to dimension-appropriate values
            start = (rand(1:(Integer(floor((dims[2] - dims[1])/0.5) - 1)), (1,2))) .* 0.5 .+ dims[1]
        end
    end
    ag_params = Dict(:name  => name,
                    :start => start,
                    :flist => copy(ag_flist),
                    :elist => copy(ag_envs),
                    :w => w, :v => v) # no noise for our simple buddy
    add_agent_to_world(kworld, ag_params; add_safely=false)
    ag_mdp = kworld.inhabitants[name]
    ag_bup = KAgentBeliefUpdater(state_dims=length(ag_params[:start]), env_dims=length(ag_envs))

    # prevent us from crafting an agent that is immediately done (avoid trivial MDPs)
    if isterminal(ag_mdp, rand(initialstate(ag_mdp)))
        # do not keep the start value if provided! It evidently does not work with this problem construction
        delete!(kworld.inhabitants, kworld.inhabitants[name])
        return init_agent(kworld, name; ag_flist=ag_flist, ag_envs=ag_envs, start=nothing, w=w, v=v)
    else
        return ag_mdp, ag_bup
    end
end

function combine_experience_buffers(exp1::ExperienceBuffer, exp2::ExperienceBuffer)
    # define addn. parameters for buffer
    total_elements = exp1.elements + exp2.elements
    # doing a bunch of if statements bc idk what to do if they evaluate to not true
    if (iszero(exp1.total_count) && iszero(exp2.total_count)); total_count = 0; end
    if (isnothing(exp1.priority_params) && isnothing(exp2.priority_params)); priority_params = nothing; end
    if (isempty(exp1.indices) && isempty(exp2.indices)); indices = Array{Int64}[]; end
    if (isone(exp1.next_ind) && isone(exp2.next_ind)); next_ind = 1; end

    # combine data elements
    data = Dict([(k, hcat(exp1.data[k], exp2.data[k])) for k in keys(exp1.data)])
    return ExperienceBuffer(data, total_elements, next_ind, indices, priority_params, total_count)
end

"""
    gen_experience(kworld::KWorld, name::String, ag_flist::Vector{Symbol}, num_instances::Integer=10;
                        max_steps=30, sim_thresh=15, debug_progress=false, updater_offset=1)

Generate expert simulations within a given world, under specified parameters. Agents will be named as `name_#`, where # is the simulation instance number.

Constructs `num_instances` instantations of an agent defined by the feature list parameter `ag_flist` relative to the provided `kworld`.
    Simulates each instance to within `sim_thresh` steps of `max_steps`, using `MuKumari.expert_simulator` as the simulation method.
    Set `debug_progress` to false to not get debug output, and instead get a progress bar. 
        Likewise, leave the `updater_offset` as +2 to the last offset of any progress bars you are using in your main loops.
"""
function gen_experience(kworld::KWorld, name::String, ag_flist::Vector{Symbol}, num_instances::Integer=10;
                        max_steps=30, sim_thresh=15, debug_progress=false, updater_offset=1)
    experiences = []
    total_experience = Any[nothing, nothing]
    generate_showvalues(sn) = () -> [("Instance #", sn)]
    updater = Progress(num_instances; desc="Simulating instances of agent: $(name)...", offset=updater_offset)
    for k in 1:num_instances
        ag_mdp, ag_bup = init_agent(kworld, name*"_"*string(k); ag_flist=ag_flist, ag_envs=[:sin, :exp])
        solver = BeliefMCTSSolver(base_solver(), ag_bup)

        planner = solve(solver, ag_mdp)
        obs_dims = state_space(ag_mdp).dims[1]

        data = expert_simulator(ag_mdp, planner, ag_bup;
                                max_steps=max_steps, sim_limit=sim_thresh, obs_dims=obs_dims,
                                debug_progress=debug_progress, updater_offset=updater_offset+2)
        next!(updater; showvalues=generate_showvalues(k))

        anonymized_location_data = deepcopy(data)
        anonymized_location_data[:s][1:2, :] = zeros(size(data[:s][1:2,:]))
        push!(experiences, (ExperienceBuffer(data, max_steps, 1, Array{Int64}[], nothing, 0),
                            ExperienceBuffer(anonymized_location_data, max_steps, 1, Array{Int64}[], nothing, 0)))
        for i in 1:2
            if isnothing(total_experience[i])
                total_experience[i] = experiences[end][i]
            else
                total_experience[i] = combine_experience_buffers(experiences[end][i], total_experience[i])
            end
        end
    end

    return Dict(:ind_exps=>experiences, :total_exp=>total_experience)
end

"""
    multi_agent_experience_generator(; max_steps=30, sim_thresh=15, num_instances=10, plot_traces=false)

Main function for this file. Made to help generate data for future multi-agent objective inference and planning using the IQ-Learn technique.

* `max_steps`: the maximum number of timestep obversations that should be gathered for any agent instance.
* `sim_thresh`: the max number of steps any agent simulation should allowed to run, even if the agent hasn't reached the terminal state yet.
    * This is also to determine if we have sufficiently approached the max number of observations that should be gathered.
* `num_instances`: Number of agent instantiations to simulate per agent MDP.
    * Note that instantiations only differ by the starting position.
* `plot_traces`: currently unused

Returns: `[kworld, data]`
    * `kworld`: KWorld object, containing information on all MDP problems used (including obstacles, goals, etc.)
    * `data`: Dictionary of all agent experiences. Is a vector of ExperienceBuffers (defined in Crux.jl).
        * Also contains `kworld` under the key "kworld"
        * Also contains a cumulative ExperienceBuffer under "total"
"""
function multi_agent_experience_generator(; max_steps=30, sim_thresh=15, num_instances=10, plot_traces=false, max_agent_count=7)
    # define KWorld object
    dims = (0., 10.)
    kworld = init_world((0., 10.), [:sub, :surf, :aer]; num_obcs=5, num_goals=3)

    # define agents by features relative to the world they will operate within
    ag_flists = Dict("ag1" => [:sub, :a1],
                     "ag2" => [:surf, :a1],
                     "ag3" => [:aer, :a1],
                     "ag4" => [:sub, :surf, :a1],
                     "ag5" => [:surf, :aer, :a1],
                     "ag6" => [:sub, :aer, :a1],
                     "ag7" => [:sub, :surf, :aer, :a1])

    # keep the number within realm of possibility
    max_agent_count = min(length(ag_flists), max_agent_count)

    # initialize data container
    data = Dict{String, Any}("kworld" => kworld)
    total_experience = nothing # cumulative buffer (init as nothing)

    # progress bar stuff
    generate_showvalues(sn) = () -> [("Agent #", sn)]
    updater = Progress(length(ag_flists); desc="Generating expert data...", offset=1)
    for (i, p) in enumerate(ag_flists)
        # generate data for agent
        data[p[1]] = gen_experience(kworld, p[1], p[2], num_instances; max_steps=max_steps, sim_thresh=sim_thresh, updater_offset=3)

        # update cumulative buffer
        if isnothing(total_experience)
            total_experience = data[p[1]][:total_exp]
        else
            total_experience = [combine_experience_buffers(data[p[1]][:total_exp][i], total_experience[i]) for i in 1:2]
        end

        # update progress bar
        next!(updater, showvalues=generate_showvalues(i))
        # if we're crossing the upper limit of simulations, finish.
        if i > max_agent_count; break; end
    end

    # store cumulative buffer
    data["total"] = total_experience

    println("\n"^(1+2*4))
    return kworld, data

    # data = Dict([(k, gen_experience(kworld, agent_mdps[k], agent_beliefs[k]; max_steps=max_steps, sim_thresh=sim_thresh)) for k in keys(agent_mdps)])

    # return kworld, agent_mdps, agent_beliefs, data
end


"""
    write_multi_run_metadata(meta_path, data_path; kwargs...)

Write a simple TOML metadata file describing a multi-run dataset. This helper
is used by tests/examples that generate multi-agent experience bundles so that
a companion metadata file exists next to the generated BSON.
"""
function write_multi_run_metadata(data_path::AbstractString; 
                                  n_agents::Int=1, agent_names::Vector{String}=String[], runs_per_agent::Int=1,
                                  run_index_key::String="ind_exps", run_container_key::String="runs",
                                  full_key::String="full_data", anon_key::String="anon_data",
                                  state_field_sizes::Vector{Int}=[2,2,12,10,1],
                                  state_field_names::Vector{String}=["loc","vel","obcs","goals","time"],
                                  keep_state_fields::Vector{Bool}=[true,true,true,false,true],
                                  anonize_first_rows::Int=2)
    """
    Create a multi-run metadata file for the given data path.
    
    The metadata file is saved following the strict naming convention:
    <data_file>.meta.toml (e.g., "experiment.bson" → "experiment.meta.toml")
    
    # Arguments:
    - `data_path`: Absolute or relative path to the BSON data file
    - `n_agents`: Number of agents in the multi-run experiment
    - `agent_names`: Vector of agent identifiers
    - `runs_per_agent`: Number of runs per agent
    - `run_index_key`: Key identifying the run index in the BSON
    - `run_container_key`: Key for the container of all runs
    - `full_key`, `anon_key`: Keys for full and anonymized data buffers
    - `state_field_sizes`, `state_field_names`, `keep_state_fields`: State metadata
    - `anonize_first_rows`: Number of initial state rows to anonymize
    
    # Returns:
    - The absolute path to the generated metadata file
    """
    
    # Compute metadata path from data path using naming convention
    data_abs = abspath(data_path)
    base_path, _ = splitext(data_abs)
    meta_path = base_path * ".meta.toml"
    
    # ensure folder exists
    dir = dirname(meta_path)
    isdir(dir) || mkpath(dir)

    agent_names = isempty(agent_names) ? ["ag$(i)" for i in 1:n_agents] : agent_names

    toml = """
schema_version = 1
data_path = "$(data_abs)"
format = "bson"
data_type = "multi_run"
created_at = "$(Dates.format(Dates.now(), Dates.ISODateTime))"
created_by = "generated"
notes = "Auto-generated multi-run metadata"

# top-level multi-run info
n_agents = $(n_agents)
agent_names = [$(join(map(a->"\""*a*"\"", agent_names), ","))]
runs_per_agent = $(runs_per_agent)
run_index_key = "$(run_index_key)"
agent_key_pattern = "ag%s"

[loader]
run_container_key = "$(run_container_key)"
agent_entry_key = ""
full_key = "$(full_key)"
anon_key = "$(anon_key)"
expected_keys = ["s","sp","a","r"]
unpack_strategy = "runs-array"

[state]
state_field_sizes = [$(join(state_field_sizes, ", "))]
state_field_names = [$(join(map(s->"\""*s*"\"", state_field_names), ", "))]
keep_state_fields = [$(join(map(b->string(b), keep_state_fields), ", "))]
anonize_first_rows = $(anonize_first_rows)
auto_clean = true
"""

    open(meta_path, "w") do io
        write(io, toml)
    end
    return meta_path
end

"""
    get_experience_data(;max_steps=10000, sim_thresh=15, update_progress=false)

For single-trace data generation.
"""
function get_experience_data(;max_steps=10000, sim_thresh=15, update_progress=false)
    obcs = let obcs = [];
        push!(obcs, (:sub, Dict(:poly => [(0., 0.), (0., 0.5), (0.4, 0.3), (0.5, 0.), (0., 0.)], :risk => 10., :impact => 10.)));
        push!(obcs, (:sub, Dict(:poly => [(4., 4.5), (5., 4.5), (7., 5.), (2., 5.), (4., 4.5)], :risk => 10., :impact => 10.)));
    end
    goals = [
        (:aer, Dict(:target=>[9.5 9.5], :strength=>10., :influence=>5., :size=>0.75)),
        (:surf, Dict(:target=>[8.5 8.5], :strength=>10., :influence=>5., :size=>0.75)),
        (:sub, Dict(:target=>[7.5 9.5], :strength=>10., :influence=>5., :size=>0.75))
    ]
    urgency = [(:ag1, 1.5), (:ag2, 0.5)]

    # Define global objective landscape
    globj_scape = GlobalObjectiveLandscape(; goals=goals, obstacles=obcs, horizons=urgency)

    # define global environment
    menv = let μfs = [(:sin, x->sin(x[1]) + cos(x[2])),
                      (:exp, x->100*exp(-norm(x-[8 8.])^2 / 1.)),
                      (:lin, x->x[1]^2 + x[2])],
               μs = [:sin, :exp, :lin];
               MuEnv(3, μs, Dict(μfs));
    end

    # Define world to hold all agents
    solver = MCTSSolver(n_iterations=1000, depth=20, exploration_constant=10.0)
    # solver = DPWSolver(n_iterations=1000, depth=20, exploration_constant=1.0)
    dims = (0., 10.)
    kworld = create_kworld(; solver=solver, dims=dims, gobj=globj_scape, menv=menv)

    ag1_flist = [:sub, :surf, :ag1]
    ag1_envs = [:sin, :exp]
    ag1_params = Dict(:name  => "ag1",
                    :start => [3. 3.],
                    :flist => ag1_flist,
                    :elist => ag1_envs,
                    :w => 0., :v => 0.) # no noise for our simple buddy
    add_agent_to_world(kworld, ag1_params)
    ag1_mdp = kworld.inhabitants["ag1"]
    ag1_bup = KAgentBeliefUpdater(state_dims=length(ag1_params[:start]), env_dims=length(ag1_envs))
    obs_dims = state_space(ag1_mdp).dims[1]
    solver1 = BeliefMCTSSolver(solver, ag1_bup)

    prog = ProgressUnknown(desc="Constructing MCTS policy tree..."; spinner=true)
    planner1 = solve(solver1, ag1_mdp)

    data = expert_simulator(ag1_mdp, planner1, ag1_bup; max_steps=max_steps, sim_limit=sim_thresh, update_progress=update_progress, obs_dims=obs_dims)

    anonymized_location_data = deepcopy(data)
    anonymized_location_data[:s][1:2, :] = zeros(size(data[:s][1:2,:]))

    return kworld,
           ExperienceBuffer(data, max_steps, 1, Array{Int64}[], nothing, 0),
           ExperienceBuffer(anonymized_location_data, max_steps, 1, Array{Int64}[], nothing, 0)
end

"""
    main(; max_steps=30, sim_thresh=15, num_instances=10, plot_traces=false, max_agent_count=7)

Generate multi-agent experience data and save to persistent BSON with metadata.

This function wraps `multi_agent_experience_generator` and handles saving the generated data
to disk alongside metadata that describes the dataset structure and provenance.

# Arguments:
- `max_steps`: maximum timestep observations per agent instance
- `sim_thresh`: max steps to run any agent simulation
- `num_instances`: number of agent instantiations to simulate per agent MDP
- `plot_traces`: currently unused
- `max_agent_count`: limit on the number of agents to generate (max 7)

# Behavior:
Generates multi-agent experience data and saves to a BSON file following the naming convention:
`<max_steps>_<sim_thresh>_<num_instances>_<max_agent_count>_multi_trace_run.bson`

Automatically creates a companion metadata file (`.meta.toml`) describing the dataset.

# Returns:
- `kworld`: KWorld object with all agent MDPs
- `data`: Dictionary of all agent experiences and metadata
"""
function main(; max_steps=30, sim_thresh=15, num_instances=10, plot_traces=false, max_agent_count=7)
    # Generate the experience data
    kworld, data = multi_agent_experience_generator(; 
                                                     max_steps=max_steps, 
                                                     sim_thresh=sim_thresh,
                                                     num_instances=num_instances, 
                                                     plot_traces=plot_traces,
                                                     max_agent_count=max_agent_count)
    
    # Determine actual number of agents generated
    actual_agent_count = min(7, max_agent_count)  # 7 agents defined in multi_agent_experience_generator
    agent_names = ["ag$i" for i in 1:actual_agent_count]
    
    # Create output directory
    output_dir = joinpath(@__DIR__, "expert_data")
    
    # Generate filename according to convention: <max_steps>_<sim_thresh>_<num_instances>_<max_agent_count>_multi_trace_run.bson
    filename = "$(max_steps)_$(sim_thresh)_$(num_instances)_$(max_agent_count)_multi_trace_run.bson"
    data_path = joinpath(output_dir, filename)
    
    # Save BSON data
    BSON.@save data_path data kworld
    println("Saved BSON data to: $data_path")
    
    # Write metadata with same parameters as generation
    write_multi_run_metadata(data_path; 
                            n_agents=actual_agent_count, 
                            agent_names=agent_names,
                            runs_per_agent=num_instances,
                            run_index_key="ind_exps")
    println("Wrote metadata for dataset")
    
    return kworld, data
end

# # use BSON loader otherwise
# kworld, exp_data, exp_data_anon = get_experience_data(;max_steps=30, sim_thresh=20, update_progress=false)

# mdp = get_agent(kworld, "ag1")

# as = actions(mdp)
# S = state_space(mdp)
# γ = Float32(discount(mdp))
# A() = DiscreteNetwork(Chain(Dense(5, 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)

# Π_iql = OnlineIQLearn(π=A(), 𝒟_demo=exp_data_anon, S=S, γ=γ, N=10000, ΔN=1,
# solve(Π_iql, mdp)


########################### Introducing second agent!!!
# sim_res = main(; plot_traces=true);

# obcs, goals, urgency, globj_scape, menv, solver, dims, kworld, ag1_flist, ag1_envs, ag1_params, ag1_mdp, ag1_bup, solver1, planner1, sim_trace1 = sim_res
# sim_trace1[2]

#= simulate(HistoryRecorder(max_steps=10), ag1_mdp, planner1, ag1_bup) =#

# r_sum = 0.0
# step = 0
# for (b, s, a, o, r) in stepthrough(ag1_mdp, planner1, ag1_bup, "b,s,a,o,r"; max_steps=15)
#     global step += 1
#     println("Step $step")
#     println("b = $(rand(b))")
#     @show s
#     @show a
#     @show o
#     println("Distance to (k-?)nearest obstacle?")
#     @show r
#     global r_sum += r
#     @show r_sum
#     println()
# end

# ag2_flist = [:sub, :aer, :ag2]
# ag2_envs = [:sin, :lin]
# ag2_params = Dict(:name  => "ag2",
#                   :start => [7. 7.],
#                   :flist => ag2_flist,
#                   :elist => ag2_envs)
# add_agent_to_world(kworld, ag2_params)
# ag2_mdp = kworld.inhabitants["ag2"]
# planner2 = solve(solver, ag2_mdp)

# sim_trace1 = stepthrough_sim(ag1_mdp, planner1, 15)
# sim_trace2 = stepthrough_sim(ag2_mdp, planner2, 15)
# ag1_mdp = init_standard_KAgentMDP(; name="agent1",
#            start=[3. 3.], dimensions=(0., 10.),
#            objl=obj_landscape, menv=menv)

# ag1_init_state = blindstart_KAgentState(ag1_mdp, ag1_mdp.start)

# planner = solve(solver, ag1_mdp)

# sim_trace = stepthrough_sim(ag1_mdp, planner, 15)
