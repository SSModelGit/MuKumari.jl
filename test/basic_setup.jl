using MuKumari

using LinearAlgebra: norm, normalize
using Meshes: Point, Vec, Quadrangle, viz

using POMDPTools, MCTS, POMDPLinter

import CairoMakie as mke
using JLD2: save, load

obcs = let obcs = Quadrangle[];
    push!(obcs, Quadrangle((0., 0.), (0., 0.5), (0.4, 0.3), (0.5, 0.)));
    push!(obcs, Quadrangle((4., 4.5), (5., 4.5), (7., 5.), (2, 5.)));
end

menv = let μfs = [(:sin, x->sin(x[1]) + cos(x[2])), (:exp, x->100*exp(-norm(x-[8 8.])^2 / 1.)), (:lin, x->x[1]^2 + x[2])], μs = [:sin, :exp, :lin];
    MuEnv(3, μs, Dict(μfs));
end

objs = [s->line_to_target_obj(s, [9.5 9.5]), s->time_till_completion_obj(s), s->safety_obj(s, obcs)]

ag1_mdp = init_standard_KAgentMDP(; name="agent1",
           start=[7. 7.],
           dimensions=(0., 10.),
           obj = s->combined_obj(s, objs),
           menv = menv)

ag1_init_state = blindstart_KAgentState(ag1_mdp, ag1_mdp.start)

solver = MCTSSolver(n_iterations=100, depth=20, exploration_constant=5.0)
planner = solve(solver, ag1_mdp)
action(planner, ag1_init_state)