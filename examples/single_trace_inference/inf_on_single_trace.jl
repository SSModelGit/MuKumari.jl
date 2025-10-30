using MuKumari

using LinearAlgebra: norm, normalize

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

function evaluate_proposed_objectives(pomdp::KAgentPOMDP, 𝒮_proposed, 𝒮_infer)
    # use equation (6) from the VAE paper Structural Relational Inference Actor-Critic for Multi-Agent Reinforcement Learning (Zhang et. al.)
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