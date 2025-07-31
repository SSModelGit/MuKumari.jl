using MuKumari

using LinearAlgebra: norm, normalize

using POMDPTools, MCTS, POMDPLinter

using CairoMakie
using JLD2: save, load

## Environment Feature types:
# Surface:    :surf
# Aerial:     :aer
# Subsurface: :sub
## Agent Feature types:
# Agent 1: :ag1
# Agent 2: :ag2

obcs = let obcs = [];
    push!(obcs, (:sub, Dict(:poly => [(0., 0.), (0., 0.5), (0.4, 0.3), (0.5, 0.), (0., 0.)], :risk => 3., :impact => 10.)));
    push!(obcs, (:sub, Dict(:poly => [(4., 4.5), (5., 4.5), (7., 5.), (2., 5.), (4., 4.5)], :risk => 3., :impact => 10.)));
end
goals = [
    (:aer, Dict(:target=>[9.5 9.5], :strength=>100., :influence=>10., :size=>0.5)),
    (:surf, Dict(:target=>[8.5 8.5], :strength=>100., :influence=>10., :size=>0.5)),
    (:sub, Dict(:target=>[7.5 9.5], :strength=>100., :influence=>10., :size=>0.5))
]
urgency = [(:ag1, 1.5), (:ag2, 0.5)]

# Define global objective landscape
globj_scape = GlobalObjectiveLandscape(; goals=goals, obstacles=obcs, horizons=urgency)

# define global environment
menv = let μfs = [(:sin, x->sin(x[1]) + cos(x[2])), (:exp, x->100*exp(-norm(x-[8 8.])^2 / 1.)), (:lin, x->x[1]^2 + x[2])], μs = [:sin, :exp, :lin];
    MuEnv(3, μs, Dict(μfs));
end

# Define world to hold all agents
solver = MCTSSolver(n_iterations=100, depth=20, exploration_constant=1.0)
dims = (0., 10.)
kworld = create_kworld(; solver=solver, dims=dims, gobj=globj_scape, menv=menv)

# First changes
# # Declaring GP Agents to place in the world
# struct GPAgent
#     name::String
#     pos::Vector{Float64}       # Current 2D position
#     α::Matrix{Float64}         # E×E matrix (α_i)
#     β::Vector{Float64}         # E-vector (β_i)
#     m::Int                     # Number of data points observed
#     r_comm::Float64            # Communication radius (determines neighbors)
# end

# kworld.gp_agents = Dict(
#     "ag1" => init_gpagent("ag1", [3.0, 3.0], E), # position based on the original demo code
#     "ag2" => init_gpagent("ag2", [7.0, 7.0], E)
# )

# using existing framework
# and multi agent demo

ag1_flist = [:sub, :surf, :ag1]    # Features this agent cares about
ag1_envs  = [:sin, :exp]           # Environment processes it can observe


ag1_params = Dict(
    :name  => "ag1",
    :start => [3. 3.],
    :flist => ag1_flist,
    :elist => ag1_envs
)

add_agent_to_world(kworld, ag1_params)

ag2_flist = [:sub, :aer, :ag2]
ag2_envs  = [:lin, :exp]

ag2_params = Dict(
    :name  => "ag2",
    :start => [7. 7.],
    :flist => ag2_flist,
    :elist => ag2_envs
)

add_agent_to_world(kworld, ag2_params)

ag1_mdp = kworld.inhabitants["ag1"]
ag2_mdp = kworld.inhabitants["ag2"]

planner1 = solve(solver, ag1_mdp)
planner2 = solve(solver, ag2_mdp)

# Setup to calculate Basis Φ(x) (feature vectors) at any location
# ulation after rendering the highest peak. We set σ2
# s = 1 and
# Σ = diag([0.02,0.02]) for the Gaussian kernel (7), and we set
# E = 80for E-dimensional estimator (14) and (20).
# Fig. 4 represents the progress over time from k = 0 to k=
# 1500. As shown in Fig. 4(a), four agents searched the map


# const E = 50 # can change to a similar number
# const σ_rbf = 1.0
# const rbf_centers = rand(2, E) .* 10.0  # uniformly in [0,10]×[0,10]

# function Φ(x::Vector{Float64})
#     return exp.(-sum((rbf_centers .- x).^2, dims=1) ./ (2 * σ_rbf^2)) |> vec
# end

# # Lines 6-9
# function update_gp_model!(agent::GPAgent, menv::MuEnv, Φ::Function, σ_ν::Float64)
#     x = agent.pos
#     φ = Φ(x)
#     noise = rand(Normal(0, σ_ν))
#     y = sample_env(menv, x) + noise

#     m = agent.m
#     agent.α .= (1 - 1/m) * agent.α .+ (1/m) * (φ * φ')
#     agent.β .= (1 - 1/m) * agent.β .+ (1/m) * (φ * y)
#     agent.m += 1
# end

# # Lines 11-15
# function communication_step!(agent::GPAgent, neighbors::Vector{GPAgent}, γ::Float64)
#     Δα = zeros(size(agent.α))
#     Δβ = zeros(size(agent.β))

#     for n in neighbors
#         Δα .+= γ * (n.α .- agent.α)
#         Δβ .+= γ * (n.β .- agent.β)
#     end

#     agent.α .+= Δα
#     agent.β .+= Δβ
# end

# # Line 16: Algorithm 2
# function get_neighbors(agent::GPAgent, agents::Dict{String,GPAgent})
#     neighbors = GPAgent[]
#     # not the most efficient
#     for (name, other) in agents
#         if name != agent.name && norm(agent.pos - other.pos) ≤ agent.r_comm
#             push!(neighbors, other)
#         end
#     end
#     return neighbors
# end

# # Pulling it all together (Algorithm 1)
# function run_simulation!(kworld; steps=50, σ_ν=0.1, γ=0.1, step_size=0.5)
#     # All the timesteps
#     for t in 1:steps
#         for agent in values(kworld.gp_agents)
#             # Noise + observation
#             update_gp_model!(agent, kworld.menv, Φ, σ_ν)

#             # Communication
#             neighbors = get_neighbors(agent, kworld.gp_agents)
#             communication_step!(agent, neighbors, γ)

#             # TODO: Replace with algorithm 2
#             grad = estimate_gradient(agent.pos, agent)
#             new_pos = agent.pos + step_size * normalize(grad)

#             # Convert to world bounds [0, 10]×[0, 10]

#         end

#         # visualize_world(kworld, t)
#         # already existing functionality??
#     end
# end


# # ag1_flist = [:sub, :surf, :ag1]
# # ag1_envs = [:sin, :exp]
# # ag1_params = Dict(:name  => "ag1",
# #                   :start => [3. 3.],
# #                   :flist => ag1_flist,
# #                   :elist => ag1_envs)
# # add_agent_to_world(kworld, ag1_params)
# # ag1_mdp = kworld.inhabitants["ag1"]
# # planner1 = solve(solver, ag1_mdp)

# # ag2_flist = [:sub, :aer, :ag2]
# # ag2_envs = [:sin, :lin]
# # ag2_params = Dict(:name  => "ag2",
# #                   :start => [7. 7.],
# #                   :flist => ag2_flist,
# #                   :elist => ag2_envs)
# # add_agent_to_world(kworld, ag2_params)
# # ag2_mdp = kworld.inhabitants["ag2"]
# # planner2 = solve(solver, ag2_mdp)

# # sim_trace1 = stepthrough_sim(ag1_mdp, planner1, 15)
# # sim_trace2 = stepthrough_sim(ag2_mdp, planner2, 15)
# # ag1_mdp = init_standard_KAgentMDP(; name="agent1",
# #            start=[3. 3.], dimensions=(0., 10.),
# #            objl=obj_landscape, menv=menv)

# # ag1_init_state = blindstart_KAgentState(ag1_mdp, ag1_mdp.start)

# # planner = solve(solver, ag1_mdp)

# # sim_trace = stepthrough_sim(ag1_mdp, planner, 15)

# function plot_agents(gp_agents)
#     fig, ax = scatter([], []; axis=(title="Agent Positions", limits=(0,10,0,10)))
#     for (name, a) in gp_agents
#         scatter!(ax, a.pos[1], a.pos[2], label=name)
#     end
#     fig
# end

# plot_agents(kworld.gp_agents)
