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

## Environment Feature types:
# Surface:    :surf
# Aerial:     :aer
# Subsurface: :sub
## Agent Feature types:
# Agent 1: :ag1
# Agent 2: :ag2

function main(; plot_traces=false)
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
    solver1 = BeliefMCTSSolver(solver, ag1_bup)
    planner1 = solve(solver1, ag1_mdp)
    sim_trace1 = stepthrough_sim(ag1_mdp, planner1, ag1_bup, 15; plot_sim_trace=plot_traces);

    return obcs, goals, urgency, globj_scape, menv, solver, dims, kworld, ag1_flist, ag1_envs, ag1_params, ag1_mdp, ag1_bup, solver1, planner1, sim_trace1
end

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
    solver1 = BeliefMCTSSolver(solver, ag1_bup)

    prog = ProgressUnknown(desc="Constructing MCTS policy tree..."; spinner=true)
    planner1 = solve(solver1, ag1_mdp)

    data = expert_simulator(ag1_mdp, planner1, ag1_bup; max_steps=max_steps, sim_limit=sim_thresh, update_progress=update_progress)

    ExperienceBuffer(data, max_steps, 1, Array{Int64}[], nothing, 0)
end

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
