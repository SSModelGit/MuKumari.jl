using MuKumari

using LinearAlgebra: norm, normalize

using POMDPTools, MCTS, POMDPLinter

using CairoMakie
using JLD2: save, load

obcs = let obcs = [];
    push!(obcs, Dict(:poly => [(0., 0.), (0., 0.5), (0.4, 0.3), (0.5, 0.), (0., 0.)], :risk => 3., :impact => 10.));
    push!(obcs, Dict(:poly => [(4., 4.5), (5., 4.5), (7., 5.), (2., 5.), (4., 4.5)], :risk => 3., :impact => 10.));
end

goal = Dict(:target=>[9.5 9.5], :strength=>100., :influence=>10., :size=>0.5)

urgency = 1.5

obj_landscape = ObjectiveLandscape(; objectives=[(:goal, goal), (:obc, obcs), (:horz, urgency)])

# obj_landscape = ObjectiveLandscape(; objectives=[(:goal, goal), (:horz, urgency)])

menv = let μfs = [(:sin, x->sin(x[1]) + cos(x[2])), (:exp, x->100*exp(-norm(x-[8 8.])^2 / 1.)), (:lin, x->x[1]^2 + x[2])], μs = [:sin, :exp, :lin];
    MuEnv(3, μs, Dict(μfs));
end

ag1_mdp = init_standard_KAgentMDP(; name="agent1",
           start=[3. 3.], dimensions=(0., 10.),
           objl=obj_landscape, menv=menv)

ag1_init_state = blindstart_KAgentState(ag1_mdp, ag1_mdp.start)

solver = MCTSSolver(n_iterations=100, depth=20, exploration_constant=1.0)
planner = solve(solver, ag1_mdp)

sim_trace = stepthrough_sim(ag1_mdp, planner, 15)