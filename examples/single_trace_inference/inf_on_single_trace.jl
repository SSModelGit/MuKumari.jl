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
van_solver() = MCTSSolver(n_iterations=1000, depth=20, exploration_constant=10.0)
dpw_solver() = DPWSolver(n_iterations=100, depth=20, exploration_constant=2.0)
function solver_from_type(mcts_type::Symbol=:dpw)
    @match mcts_type begin
        :van => van_solver()
        :dpw => dpw_solver()
        _    => van_solver()
    end
end
println("Directory is: ", @__DIR__)

script_dir = @__DIR__

(kworld, data, anon_data) = BSON.load(script_dir*"/single_start_exp.bson")[:data]

anon_data.elements = 996

N = anon_data.elements
mdp = get_agent(kworld, "ag1")

as = actions(mdp)
S = state_space(mdp)
γ = Float32(discount(mdp))
A() = DiscreteNetwork(Chain(Dense(S.dims[1], 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as; dev=Flux.cpu)

𝒟_iql = OnlineIQLearn(π=A(), 𝒟_demo=anon_data, S=S, γ=γ, N=anon_data.elements, ΔN=1, c_opt=(;epochs=1),reg=false,gp=false, log=(;period=50))

solve(𝒟_iql, mdp)

f = plot_learning([𝒟_iql,], title="Results of IQ-Learning", labels=["iql",])

# The means of evaluating the Q function is to use the `Crux.value` function

function quick_policy_compute_for_objl(; mcts_type=:dpw)
    base_solver = solver_from_type(mcts_type)
end