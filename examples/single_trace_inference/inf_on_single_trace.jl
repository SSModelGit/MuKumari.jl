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

using JLD2: @save, @load
using BSON

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
    @time π_net = map(x->solve(x, pomdp), 𝒮_net)
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

function quick_policy_compute_for_objl(pomdp::KAgentPOMDP; solver_type::Symbol=:mcts, solver_params=[:dpw, 1000, 10.0])
    base_solver = solver_from_type(pomdp, solver_type; solver_params=solver_params)

    𝒮_base = solve(base_solver, pomdp)
    return 𝒮_base
end

function evaluate_proposed_objective(pomdp::KAgentPOMDP, π_proposed, π_infer, data::ExperienceBuffer, q_objectives)
    # use equation (6) from the VAE paper Structural Relational Inference Actor-Critic for Multi-Agent Reinforcement Learning (Zhang et. al.)

    # three eval types
    # NOTE: q(z|o) refers to likelihood of underlying feature (z) w.r.t. observations (o)
    ## Here, z is the proposed objective set, and o is the timeseries of observed actions
    ## We can define q(z|o) approximately as the proposal distribution
    # NOTE: -H(q(z|o)): sum of the negative log likelihoods of each of the objectives in the proposed objective set, using the proposal distribution
    ## Compute as: ∑_{z_i∈z}q(z_i|o)*log(q(z_i|o))

    # Type 1: E_q[log(p(o|z))]
    ## approximately equivalent to the Open-ended SIPS approach of P(g|π,o)/Q(g) (error of reconstruction to true obs weighted by likelihood of reconstruction)
    ### Approximate E_q[p(o|z)] as (∑π(a_true) ∀ a ∈ [set of observations]) * q(z|o)
    ### This is an adaptation of the Open-ended SIPS approach

    # Type 2: -H(q(z|o)) - E_q[log(p(o|z))]
    ## The combination of Open-ended SIPS with the L_VAE from SRI-AC

    # Type 3: -H(q(z|o)) - E_q[log(p_iq(o|z))]: Using IQLearn's output as a softer, smoothened, broader point of comparison, instead of directly against data
    ## 

    # standard mechanism to evaluate
    let s = rand(initialstate(pomdp)), o = rand(initialobs(pomdp, s)), a = Flux.onehot(:nw, actions(pomdp))
        action(π_infer, o) # fully unnecessary to construct this, just doing this for example's sake
        Crux.value(π_proposed, o, a) # use the observation to produce this
        Crux.value(π_proposed, MuKumari.shape_state_as_obs(pomdp, s), a) # alternatively, use the internal func`shape_state_as_obs` function on a state directly
        j = 1
        Crux.value(π_infer, data.data[:s][:,j], data.data[:a][:,j]) # OR evaluate on a timestep drawn from of the ExperienceBuffer (at time = j)
    end
end

function construct_q_proposals(base_objs)
    k = length(base_objs)
    q_base = Dict([(obj, 1/k) for obj in base_objs])
    q_objs = map(powerset(base_objs, 1)) do obj
        q_obj = 1
        for comp_obj in obj
            q_obj *= q_base[comp_obj]
        end
        (obj, q_obj)
    end |> Dict
    q_norm = sum(values(q_objs))
    for obj in q_objs; q_objs[obj] /= q_norm; end
    q_objs
end

function quick_IQL(kworld::KWorld, anon_data::ExperienceBuffer; plot_metrics::Bool=false)
    N = anon_data.elements
    mdp = get_agent(kworld, "ag1")

    as = actions(mdp)
    S = state_space(mdp)
    γ = Float32(discount(mdp))
    A() = DiscreteNetwork(Chain(Dense(S.dims[1], 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as; dev=Flux.cpu)

    𝒟_iql = OnlineIQLearn(π=A(), 𝒟_demo=anon_data, S=S, γ=γ, N=anon_data.elements, ΔN=1, c_opt=(;epochs=1),reg=false,gp=false, log=(;period=50))

    solve(𝒟_iql, mdp)

    if plot_metrics; f = plot_learning([𝒟_iql,], title="Results of IQ-Learning", labels=["iql",]); else; f = nothing; end
    return 𝒟_iql, mdp, f
end

println("Directory is: ", @__DIR__)

script_dir = @__DIR__

(kworld, data, anon_data) = BSON.load(script_dir*"/single_start_exp.bson")[:data]
anon_data.elements = 996 # manual edit of this specific data file to account for empty end values

𝒟_iql, mdp, f = quick_IQL(kworld, anon_data; plot_metrics=false)

𝒮_dqn_metric_net = deep_q_solver(mdp; solver_params=[:all, 10000])
π_dqn_metric_net, p_dqn_metrics = deep_q_metrics(mdp, 𝒮_dqn_metric_net; solver_type=:all)

# The means of evaluating the Q function is to use the `Crux.value` function