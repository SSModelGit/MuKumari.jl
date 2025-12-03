using MuKumari

using LinearAlgebra: norm, normalize
using Combinatorics: powerset

using POMDPTools, MCTS, POMDPLinter
using Match: @match

# addressing weird load order bugs
using Plots
using CUDA, cuDNN

# Commenting out CairoMakie due to current issue compiling with Plots and GR_jll
# using CairoMakie

# Make sure to load Plots before Crux, because of some weird load order bug
using Crux

# Utilizing Gen for generative approach to particle filters
using Gen: @gen, Distribution, logpdf, random, categorical
using Gen: ParticleFilterState, initialize_particle_filter, particle_filter_step!, maybe_resample!, get_traces, get_retval

# Save data
using JLD2: @save, @load
using BSON

# Core of deepnet structure
using Flux

# construct a belief state out of the mdp parameters
beliefstate_for_pomdp(mdp::KAgentPOMDP) = KAgentBeliefUpdater(state_dims=length(mdp.start), env_dims=mdp.menv.M)
# since copy(::MCTSSolver) isn't defined for some reason, just use this instead of figuring out a copy extension
van_solver(params) = MCTSSolver(n_iterations=params[1], depth=20, exploration_constant=params[2])
dpw_solver(params) = DPWSolver(n_iterations=params[1], depth=20, exploration_constant=params[2])
function mcts_solver(pomdp::KAgentPOMDP; solver_params=[:van, 1000, 10.0])
    pomdp_bup = beliefstate_for_pomdp(pomdp)
    𝒮_mcts = @match solver_params[1] begin
        :van => BeliefMCTSSolver(van_solver(solver_params[2:end]), pomdp_bup)
        :dpw => BeliefMCTSSolver(dpw_solver(solver_params[2:end]), pomdp_bup)
        _    => BeliefMCTSSolver(van_solver(solver_params[2:end]), pomdp_bup)
    end
    return 𝒮_mcts
end

# define Deep Q solver (REINFORCE, DQN, and SoftQ)
function deep_q_solver(pomdp::KAgentPOMDP; solver_params=[:all, 10000])
    as = actions(pomdp)
    S = state_space(pomdp)
    A() = Flux.gpu(DiscreteNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as; dev=Flux.gpu))
    V() = Flux.gpu(ContinuousNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, 1))))

    solver_type, N = solver_params[[1, 2]] # can also do solver_params[1:2] if I wanted - keeping this as later reminder for self on how to use indexing
    𝒮_net = @match solver_type begin
        :all => [REINFORCE(π=A(), S=S, N=N, ΔN=500, a_opt=(epochs=5,), interaction_storage=[]),
                 DQN(π=A(), S=S, N=N, interaction_storage=[]),
                 SoftQ(π=A(), α=Float32(0.1), S=S, N=N, ΔN=1, c_opt=(;epochs=5), interaction_storage=[])]
        :reinforce => [REINFORCE(π=A(), S=S, N=N, ΔN=500, a_opt=(epochs=5,), interaction_storage=[])]
        :dqn => [DQN(π=A(), S=S, N=N, interaction_storage=[])]
        :softq => [SoftQ(π=A(), α=Float32(0.1), S=S, N=N, ΔN=1, c_opt=(;epochs=5), interaction_storage=[])]
    end

    if !(solver_type==:all)
        return 𝒮_net[1]
    end
    return 𝒮_net
end

function solver_from_type(pomdp::KAgentPOMDP, type::Symbol=:dpw; solver_params)
    @match type begin
        :mcts => mcts_solver(pomdp;   solver_params=solver_params)
        :dql  => deep_q_solver(pomdp; solver_params=solver_params)
        _     => mcts_solver(pomdp;   solver_params=[:dpw, 1000, 10.0])
    end
end

function deep_q_metrics(pomdp::KAgentPOMDP, 𝒮_net; solver_type::Symbol=:all)
    @time π_net = @match solver_type begin
        :all => map(x->solve(x, pomdp), 𝒮_net)
        _    => solve(𝒮_net, pomdp)
    end
    labels = @match solver_type begin
        :all => ["REINFORCE", "DQN", "SoftQ" ]
        :reinforce => ["REINFORCE"]
        :dqn => ["DQN"]
        :softq => ["SoftQ" ]
    end
    p = plot_learning(𝒮_net, title = "One-shot Agent π Training Curves", 
                        labels = labels)
    
    return π_net, p
end

function quick_policy_compute_for_pomdp(pomdp::KAgentPOMDP; solver_type::Symbol=:mcts, solver_params=[:dpw, 1000, 10.0])
    𝒮_pomdp = solver_from_type(pomdp, solver_type; solver_params=solver_params)
    π_pomdp = solve(𝒮_pomdp, pomdp)
    return 𝒮_pomdp, π_pomdp
end

function quick_IQL(kworld::KWorld, anon_data::ExperienceBuffer; plot_metrics::Bool=false)
    N = anon_data.elements
    mdp = get_agent(kworld, "ag1")

    as = actions(mdp)
    S = state_space(mdp)
    γ = Float32(discount(mdp))
    A() = DiscreteNetwork(Chain(Dense(S.dims[1], 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as; dev=Flux.cpu)

    𝒟_iql = OnlineIQLearn(π=A(), 𝒟_demo=anon_data, S=S, γ=γ, N=anon_data.elements, ΔN=1, c_opt=(;epochs=1),reg=false,gp=false, log=(;period=50))

    π_iql = solve(𝒟_iql, mdp)

    if plot_metrics; f = plot_learning([𝒟_iql,], title="Results of IQ-Learning", labels=["iql",]); else; f = nothing; end
    return π_iql, 𝒟_iql, mdp, f
end

function kworld_for_inference(possible_goals::Vector;
                                known_kworld::Union{KWorld, Nothing}=nothing,
                                known_obcs::Union{Vector, Nothing}=nothing,
                                known_env::Union{MuEnv, Nothing}=nothing, dims::Union{Tuple, Nothing}=nothing)
    if !isnothing(known_kworld)
        known_env = known_kworld.menv
        known_obcs = copy(known_kworld.glob_landscape.obstacles)
        dims = kworld.dimensions
    elseif isnothing(known_kworld) && (isnothing(known_obcs) && isnothing(known_env) && isnothing(dims))
        @error "Have to specify EITHER a known world OR known obstacles & environment & dimensions (world will take precedent)"
    end

    infer_glob_landscape = GlobalObjectiveLandscape(; goals=possible_goals, obstacles=known_obcs, horizons=[])
    println("Global Objective is: ", infer_glob_landscape)
    pseudo_solver = MCTSSolver(n_iterations=1000, depth=20, exploration_constant=10.0) # random default solver to fill in required fields of KWorld
    create_kworld(; solver=pseudo_solver, dims=dims, gobj=infer_glob_landscape, menv=known_env)
end

function construct_q_proposals(infer_kworld; inf_agent_start::Matrix=[3. 3.])
    base_objs = copy(infer_kworld.glob_landscape.goals)
    k = length(base_objs)
    base_obj_names = map(x->x[1], base_objs)

    q_base = Dict([(obj, 1/k) for obj in base_obj_names])
    q_objs = map(powerset(base_obj_names, 1)) do obj
        q_obj = 1
        for comp_obj in obj
            q_obj *= q_base[comp_obj]
        end
        (obj, q_obj)
    end |> Dict
    q_norm = sum(values(q_objs))
    for obj in keys(q_objs); q_objs[obj] /= q_norm; end

    name_to_obj_comp_list = map(enumerate(powerset(base_obj_names, 1))) do (i, p_name)
        ("p"*string(i), p_name)
    end |> Dict
    # name_to_obj_comp_list = Dict([(Symbol("p"*i), obj) for (i, obj) in enumerate(keys(q_objs))])
    proposal_names = collect(keys(name_to_obj_comp_list))
    name_to_q_proposal = Dict([(name, q_objs[name_to_obj_comp_list[name]]) for name in proposal_names])

    obc_names = collect(Set(map(x->x[1], infer_kworld.glob_landscape.obstacles)))
    name_to_proposed_mdp = map(enumerate(name_to_obj_comp_list)) do (i, np)
        add_agent_to_world(infer_kworld, Dict(:name  => string(np[1]),
                                              :start => copy(inf_agent_start),
                                              :flist => vcat(np[2], obc_names),
                                              :elist => copy(infer_kworld.menv.μ_order),
                                              :w => 0., :v => 0.))
        (np[1], infer_kworld.inhabitants[string(np[1])])
    end |> Dict
    return proposal_names, q_objs, name_to_obj_comp_list, name_to_q_proposal, name_to_proposed_mdp
end

function precompute_π_proposals(name_to_proposed_mdp; solver_type=:dql, solver_params=[:softq, 10000])
    name_to_𝒮_proposal = Dict()
    name_to_π_proposal = Dict()
    for (name, mdp) in name_to_proposed_mdp
        println("\nNow computing policy for proposal: ", name)
        name_to_𝒮_proposal[name], name_to_π_proposal[name] = quick_policy_compute_for_pomdp(mdp; solver_type=solver_type, solver_params=solver_params)
    end
    return name_to_𝒮_proposal, name_to_π_proposal
end

struct ScoreΠDist <: Distribution{Nothing}
    prop_names::Vector
    q_objs::Dict
    n_compobj_list::Dict
    n_qprop_list::Dict
    n_propmdp_list::Dict
    n_𝒮_proposals::Dict
    n_π_proposals::Dict
end

function precompute_π_dist(infer_kworld; solver_type=:dql, solver_params=[:softq, 10000])
    prop_names, q_objs, n_compobj_list, n_qprop_list, n_propmdp_list = construct_q_proposals(infer_kworld)
    n_𝒮_proposals, n_π_proposals = precompute_π_proposals(n_propmdp_list; solver_type=solver_type, solver_params=solver_params)

    ScoreΠDist(prop_names, q_objs, n_compobj_list, n_qprop_list, n_propmdp_list, n_𝒮_proposals, n_π_proposals)
end

"""
    prior_sh_entropy_obj(prop_name, component_objectives_dict, q_proposal_dict)

Compute the Shannon Entropy over the prior distribution for the given proposal.

This comes out to be: H(q(z|o))
    NOTE: -H(q(z|o)): sum of the negative log likelihoods of each of the objectives in the proposed objective set, using the proposal distribution
    Compute as: ∑_{z_i∈z}q(z_i|o)*log(q(z_i|o))

Used as a regularizer to the evaluation function, so that less likely proposals receive less emphasis.
"""
function prior_sh_entropy_obj(π_dist::ScoreΠDist, prop_name)
    component_objectives_dict = π_dist.n_compobj_list
    q_proposal_dict = π_dist.q_objs
    mapreduce(n->q_proposal_dict[[n]] * log(q_proposal_dict[[n]]), +, component_objectives_dict[prop_name], init=0)
end

function expected_data_recons_err(π_dist::ScoreΠDist, prop_name, data::ExperienceBuffer; eval_tsteps=100, init_eval_tstep=1)
    mdp = π_dist.n_propmdp_list[prop_name]
    q_zo = π_dist.n_qprop_list[prop_name]
    π_prop = π_dist.n_π_proposals[prop_name]
    all_a_onehot = Flux.onehotbatch(actions(mdp), actions(mdp))

    expectation_sum = 0
    exp_sum_tracker = Matrix{Any}(undef, eval_tsteps, 3)
    for i in 1:eval_tsteps
        if i > data.elements
            break
        end
        recon_val = Crux.value(π_prop, data.data[:s][:,i], all_a_onehot)
        recon_prob = softmax(recon_val .- maximum(recon_val), dims=2) * data.data[:a][:,i]
        # sum log prob (clip value to prevent underflow to -∞)
        log_likelihood_recon_prob = log(max(recon_prob[1], 1e-30))
        expectation_sum += log_likelihood_recon_prob
        exp_sum_tracker[i, :] = [data.data[:s][:,i], log_likelihood_recon_prob, q_zo * expectation_sum]
    end
    return q_zo * expectation_sum, exp_sum_tracker
end

function grid_points(n, dims=(0.,10.))
    dim_span = dims[2] - dims[1]
    dim_shift = dim_span / 20 # 5% shift
    nx = ceil(Int, sqrt(n))
    ny = ceil(Int, n/nx)
    xs = range(dims[1]+dim_shift, dims[2]-dim_shift; length=nx)
    ys = range(dims[1]+dim_shift, dims[2]-dim_shift; length=ny)
    collect(Iterators.take(([x y] for y in ys for x in xs), n))
end

function expected_recons_err_against_iql(π_dist::ScoreΠDist, prop_name, π_iql; eval_num=100)
    mdp = π_dist.n_propmdp_list[prop_name]
    q_zo = π_dist.n_qprop_list[prop_name]
    π_prop = π_dist.n_π_proposals[prop_name]
    all_a_onehot = Flux.onehotbatch(actions(mdp), actions(mdp))

    eval_locations = grid_points(eval_num, mdp.dimensions)
    mdp_states = map(x->blindstart_KAgentState(mdp, x), eval_locations)
    mdp_states_as_obs = map(s->MuKumari.shape_state_as_obs(mdp, s), mdp_states)
    iql_optimal_actions = map(s->action(π_iql, s)[1], mdp_states_as_obs)

    expectation_sum = 0
    exp_sum_tracker = Matrix{Any}(undef, eval_num, 3)
    for i in 1:eval_num
        recon_val = Crux.value(π_prop, mdp_states_as_obs[i], all_a_onehot)
        recon_prob = softmax(recon_val .- maximum(recon_val), dims=2) * Flux.onehot(iql_optimal_actions[i], actions(mdp))
        # sum log prob (clip value to prevent underflow to -∞)
        log_likelihood_recon_prob = log(max(recon_prob[1], 1e-30))
        expectation_sum += log_likelihood_recon_prob
        exp_sum_tracker[i, :] = [eval_locations[i], log_likelihood_recon_prob, q_zo * expectation_sum]
    end
    return q_zo * expectation_sum, exp_sum_tracker
end

"""
    evaluate_proposed_objective(pomdp::KAgentPOMDP, π_proposed, π_infer, data::ExperienceBuffer, q_objectives)

Evaluates the provided policy (π_proposed) under three different evaluation schemes.

Based off Equation (6) from the VAE paper Structural Relational Inference Actor-Critic for Multi-Agent Reinforcement Learning (Zhang et. al.)
Below is a discussion of the three evaluation schemes.

NOTE: q(z|o) refers to likelihood of underlying feature (z) w.r.t. observations (o)
 * Here, z is the proposed objective set, and o is the timeseries of observed actions
 * We can define q(z|o) approximately as the proposal distribution

Scheme 1: E_q[log(p(o|z))]
 * approximately equivalent to the Open-ended SIPS approach of P(g|π,o)/Q(g) (error of reconstruction to true obs weighted by likelihood of reconstruction)
 * Approximate E_q[p(o|z)] as (∑π(a_true) ∀ a ∈ [set of observations]) * q(z|o)
 * This is an adaptation of the Open-ended SIPS approach

Scheme 2: E_q[log(p(o|z))] + H(q(z|o))
 * The combination of Open-ended SIPS with the L_VAE from SRI-AC

Scheme 3: E_q[log(p_iq(o|z))] + H(q(z|o)): Using IQLearn's output as a softer, smoothened, broader point of comparison, instead of directly against data
"""
function evaluate_proposed_objective(π_dist::ScoreΠDist, prop_name, π_iql, data::ExperienceBuffer; eval_steps=100)
    # standard mechanism to evaluate

    eval_1 = expected_data_recons_err(π_dist, prop_name, data; eval_tsteps=eval_steps, init_eval_tstep=1)
    eval_2 = (eval_1[1] + prior_sh_entropy_obj(π_dist, prop_name), eval_1[2])
    eval_2[2][:,2:3] .+= prior_sh_entropy_obj(π_dist, prop_name)
    eval_3 = expected_recons_err_against_iql(π_dist, prop_name, π_iql; eval_num=eval_steps)

    println("Scores under different eval schemes:: Proposal named: ", prop_name)
    println("\tScheme 1: ", eval_1[1], " | Scheme 2: ", eval_2, " | Scheme 3: ", eval_3[1])

    # let s = rand(initialstate(pomdp)), o = rand(initialobs(pomdp, s)), a = Flux.onehot(:nw, actions(pomdp))
    #     action(π_infer, o) # fully unnecessary to construct this, just doing this for example's sake
    #     Crux.value(π_proposed, o, a) # use the observation to produce this
    #     Crux.value(π_proposed, MuKumari.shape_state_as_obs(pomdp, s), a) # alternatively, use the internal func`shape_state_as_obs` function on a state directly
    #     j = 1
    #     Crux.value(π_infer, data.data[:s][:,j], data.data[:a][:,j]) # OR evaluate on a timestep drawn from of the ExperienceBuffer (at time = j)
    # end

    return eval_1, eval_2, eval_3
end

function evaluate_all_proposed_objs(π_dist::ScoreΠDist, π_iql, data::ExperienceBuffer; eval_steps=100)
    prop_evals = Dict{Any, Any}()
    for prop_name in π_dist.prop_names
        prop_evals[prop_name] = evaluate_proposed_objective(π_dist, prop_name, π_iql, data; eval_steps=eval_steps)
    end
    return prop_evals
end

function plot_evaluations_over_timesteps(evals::Tuple)
    # specifically extract the third column (cumulative log likelihood)
    eval_timeseries = map(ev->ev[2][:,3], evals)
    time_axis = 1:size(eval_timeseries[1])[1]

    plt = plot(xlabel="# of Data Points Evaluated", ylabel="Neg. Log Likelihood", lw=2)

    for (i, ts) in enumerate(eval_timeseries)
        plot!(plt, time_axis, ts, label="Scheme $i")
    end

    plt
end

function plot_evaluations_over_timesteps(evals::Dict)
    # specifically extract the third column (cumulative log likelihood)
    # time_axis = 1:size(eval_timeseries[1])[1]
    time_axis = 1:size(evals[first(keys(evals))][1][2][:,3])[1]
    scheme_names = Dict(1=>"Open-ended SIPS", 2=>"Entropy-biased SIPS", 3=>"IQLearn-guided SIPS")

    plt = plot(xlabel="# of Data Points Evaluated", ylabel="Neg. Log Likelihood",
               title="Effectiveness of proposed objective evaluation\nunder various SIPS schemes", lw=2)

    proposals = collect(keys(evals))
    base_colors = palette(:tab10, length(proposals)*2)
    linestyles = (:solid, :dash, :dot)

    for (i, key) in pairs(proposals)
        data = map(ev->ev[2][:,3], evals[key])
        color = base_colors[i]

        for (j, ts) in enumerate(data)
            if j==2
                continue
            end
            plot!(plt, time_axis, ts, label="$key ($(scheme_names[j][1:end-5]))", color=color,
                  linestyle=linestyles[j], linewidth=2)
        end
    end

    plt
end

println("Directory is: ", @__DIR__)

script_dir = @__DIR__

(kworld, data, anon_data) = BSON.load(script_dir*"/single_start_exp.bson")[:data]
anon_data.elements = 996 # manual edit of this specific data file to account for empty end values

π_iql, 𝒟_iql, mdp, f = quick_IQL(kworld, anon_data; plot_metrics=false)

# kworld_infer = kworld_for_inference(kworld.glob_landscape.goals[1:end-1]; known_kworld=kworld)
kworld_infer = kworld_for_inference(kworld.glob_landscape.goals[1:end];
                                    known_obcs=kworld.glob_landscape.obstacles, known_env=mdp.menv, dims=mdp.dimensions)

π_dist = precompute_π_dist(kworld_infer; solver_type=:dql, solver_params=[:softq, 10000])

# forward_estim_solver = :softq
# 𝒮_dqn_metric_net = deep_q_solver(mdp; solver_params=[forward_estim_solver, 10000])
# π_dqn_metric_net, p_dqn_metrics = deep_q_metrics(mdp, 𝒮_dqn_metric_net; solver_type=forward_estim_solver)

# The means of evaluating the Q function is to use the `Crux.value` function