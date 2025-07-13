export stepthrough_sim

"""
$(SIGNATURES)

Complete a stepthrough of the system.

Returns a list of states and actions over time.
"""
function stepthrough_sim(mdp::KAgentMDP, planner::MCTSPlanner, max_steps::Integer=10; plot_sim_trace=false)
    sim_trace = Any[]

    for (s,a,r) in stepthrough(mdp, planner, "s,a,r", max_steps=max_steps)
        println("in state:\n$s")
        println("took action: $a")
        println("received reward: $r")
        println("--------------------\n")
        push!(sim_trace, [s, a])
    end

    push!(sim_trace, [@gen(:sp)(mdp, sim_trace[end][1], sim_trace[end][2]), :c])

    return sim_trace
end

"""
$(SIGNATURES)

Complete a stepthrough of the system for a KAgentPOMDP.

* Additionally requires a Belief Updater!

Returns a list of states and actions over time.
"""
function stepthrough_sim(mdp::KAgentPOMDP, planner::MCTSPlanner, bup::KAgentBeliefUpdater, max_steps::Integer=10; plot_sim_trace=false)
    sim_trace = Any[]
    step = 0
    for (b,s,a,o,r) in stepthrough(mdp, planner, bup, "b,s,a,o,r", max_steps=max_steps)
        step += 1
        println("Step $step")
        println("in state:\n$s")
        println("took action: $a")
        println("received reward: $r")
        println("--------------------\n")
        push!(sim_trace, [s, a])
    end

    push!(sim_trace, [@gen(:sp)(mdp, sim_trace[end][1], sim_trace[end][2]), :c])

    return sim_trace
end