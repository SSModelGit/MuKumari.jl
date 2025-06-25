using MuKumari

using LinearAlgebra: norm, normalize

using POMDPTools, MCTS, POMDPLinter
using Crux

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

ag1_flist = [:sub, :surf, :ag1]
ag1_envs = [:sin, :exp]
ag1_params = Dict(:name  => "ag1",
                  :start => [3. 3.],
                  :flist => ag1_flist,
                  :elist => ag1_envs)
add_agent_to_world(kworld, ag1_params)
ag1_mdp = kworld.inhabitants["ag1"]
planner1 = solve(solver, ag1_mdp)

ag2_flist = [:sub, :aer, :ag2]
ag2_envs = [:sin, :lin]
ag2_params = Dict(:name  => "ag2",
                  :start => [7. 7.],
                  :flist => ag2_flist,
                  :elist => ag2_envs)
add_agent_to_world(kworld, ag2_params)
ag2_mdp = kworld.inhabitants["ag2"]
planner2 = solve(solver, ag2_mdp)

sim_trace1 = stepthrough_sim(ag1_mdp, planner1, 15)
sim_trace2 = stepthrough_sim(ag2_mdp, planner2, 15)
# ag1_mdp = init_standard_KAgentMDP(; name="agent1",
#            start=[3. 3.], dimensions=(0., 10.),
#            objl=obj_landscape, menv=menv)

# ag1_init_state = blindstart_KAgentState(ag1_mdp, ag1_mdp.start)

# planner = solve(solver, ag1_mdp)

# sim_trace = stepthrough_sim(ag1_mdp, planner, 15)