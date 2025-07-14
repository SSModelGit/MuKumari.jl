export stepthrough_sim, expert_simulator

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

    if plot_sim_trace
        viz_system_sim(mdp, mdp.objl, sim_trace)
    end

    return sim_trace
end

"""
$(SIGNATURES)

Complete a stepthrough of the system for a KAgentPOMDP.

* Additionally requires a Belief Updater!

Returns a list of states and actions over time.
"""
function stepthrough_sim(pomdp::KAgentPOMDP, planner::AbstractMCTSPlanner, bup::KAgentBeliefUpdater, max_steps::Integer=10; plot_sim_trace=false)
    sim_trace = Any[]
    step = 0
    for (b,s,a,o,r) in stepthrough(pomdp, planner, bup, "b,s,a,o,r", max_steps=max_steps)
        step += 1
        println("Step $step")
        println("Has belief: $b")
        println("in state:\n$s")
        println("took action: $a")
        println("Received observation: $o")
        println("received reward: $r")
        println("--------------------\n")
        push!(sim_trace, [s, a])
    end

    push!(sim_trace, [@gen(:sp)(pomdp, sim_trace[end][1], sim_trace[end][2]), :c])

    if plot_sim_trace
        viz_system_sim(pomdp, pomdp.objl, sim_trace)
    end

    return sim_trace
end

"""
$(SIGNATURES)

Data collector for "expert" operation
* Creates a dictionary that matches the needs of an Experience as provided by Crux.jl

Will produce a dictionary with the following fields:

* :a => Boolean Matrix of actions taken each time step; columns correspond to timesteps, rows correspond to one-hot encoding of action space
* :s => Matrix of agent state over time; columns correspond to timesteps; each column slice is a vector of the agent state
* :sp => Matrix of the new agent state after taking action for the given time step; matrix arrangement same as :s
* :expert_val => Matrix of 1.0's, representing that each time step corresponds to an "expert taking action"
* :r => Matrix of reward received each timestep; 1 row per column; 
* :t => Matrix of time step counter; 1 row per column
* :done => Boolean Matrix of terminal status; 1 row per column; true if reached terminal state; false otherwise
"""
function expert_simulator(; max_steps=10000)
    @error "Not yet implemented!"
end