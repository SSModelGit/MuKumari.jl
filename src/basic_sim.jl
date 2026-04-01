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
        if plot_sim_trace
            println("Step $step")
            println("Has belief: $b")
            println("in state:\n$s")
            println("took action: $a")
            println("Received observation: $o")
            println("received reward: $r")
            println("--------------------\n")
        end
        push!(sim_trace, [s, a])
    end

    push!(sim_trace, [@gen(:sp)(pomdp, sim_trace[end][1], sim_trace[end][2]), :c])

    if plot_sim_trace
        f = viz_system_sim(pomdp, pomdp.objl, sim_trace)
        return (sim_trace, f)
    end

    return sim_trace
end

"""
    stepthrough_sim(pomdp::KAgentPOMDP, policy, max_steps::Integer=10; start_state=nothing)

Manually simulate the POMDP using a learned policy from Crux.

# Arguments
- `pomdp::KAgentPOMDP`: The POMDP environment
- `policy`: Learned policy from Crux
- `max_steps::Integer`: Number of steps to simulate (default: 10)
- `start_state`: Optional starting state. If nothing, samples from initialstate(pomdp)
"""
function stepthrough_sim(pomdp::KAgentPOMDP, policy, max_steps::Integer=10; start_state=nothing)
    sim_trace = Any[]
    s = isnothing(start_state) ? rand(initialstate(pomdp)) : start_state

    for step in 1:max_steps
        obs_vec = shape_state_as_obs(pomdp, s)
        a = action(policy, obs_vec)[1]

        sp = @gen(:sp)(pomdp, s, a)

        push!(sim_trace, [s, a])
        s = sp
    end

    push!(sim_trace, [@gen(:sp)(pomdp, sim_trace[end][1], sim_trace[end][2]), :c])

    return sim_trace
end

onehot_action_encoder(pomdp::KAgentPOMDP) = a->onehot(a, actions(pomdp))

function step_info_string(n,i,b,s,a,aoh,o,r)
    l1 = "Iteration $n.$i\n    State:\n    $s\n    Belief:\n    $b\n"
    l2 = "    Action taken: $a | Encoded as: $(collect(transpose(aoh))).T\n"
    l3 = "    Observation of [next] state: $o | Reward received: $r\n"
    l4 = "-----------------------------------------------\n"
    return l1*l2*l3*l4
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
function expert_simulator(pomdp::KAgentPOMDP, planner::AbstractMCTSPlanner, bup::KAgentBeliefUpdater;
                          max_steps=10000, sim_limit=15, obs_dims::Union{Nothing, Integer}=nothing,
                          debug_progress=false, updater_offset=1, nonterminal_system::Bool=false)
    step_counter = 1 # this is used to index arrays; use one-indexing
    sim_counter = 0 # used to track number of sims taken; use zero-indexing

    # Prep action saving
    one1 = onehot_action_encoder(pomdp)
    a_dims = length(actions(pomdp))
    a_list = Matrix{Bool}(undef, a_dims, max_steps)

    # Prep state saving
    if isnothing(obs_dims); obs_dims = bup.state_dims + bup.env_dims + 1; end
    # s_list = Matrix{Float64}(undef, obs_dims, max_steps)
    # sp_list = Matrix{Float64}(undef, obs_dims, max_steps)
    s_list = zeros(Float64, obs_dims, max_steps)
    sp_list = zeros(Float64, obs_dims, max_steps)

    # additional list prep
    expert_val_list = ones(Float32,1,max_steps)
    r_list = Matrix{Float64}(undef, 1, max_steps)
    t_list = Matrix{Int64}(undef, 1, max_steps)
    done_list = Matrix{Bool}(undef, 1, max_steps)

    single_trace = []
    broke = false
    if !debug_progress;
        p1 = Progress(max_steps; desc="Collecting behavior data...", offset=updater_offset);
        generate_showvalues(sn) = () -> [("Step number", sn)]
    end
    while step_counter ≤ max_steps
        sim_counter += 1
        single_trace = empty!(single_trace)
        step = 0
        if !debug_progress; p2 = Progress(sim_limit; desc="Simulation #$(sim_counter)...", offset=updater_offset+2); end
        for (b,s,sp,a,o,r) in stepthrough(pomdp, planner, bup, "b,s,sp,a,o,r", max_steps=sim_limit)
            step += 1
            push!(single_trace, [s,sp,a,one1(a),r,step,POMDPs.isterminal(pomdp, sp)])
            if debug_progress; print(step_info_string(sim_counter, step, b, s, a, single_trace[end][4], o, r));
            else;               next!(p2; showvalues=generate_showvalues(step)); end
        end
        if single_trace[end][end]
            # only treat as an expert if it reaches the goal and terminates
            if step_counter + step - 1 ≤ max_steps
                for (j, i) in enumerate(step_counter:step_counter+step-1)
                    a_list[:,i]   .= single_trace[j][4]
                    s_list[:,i]   .= shape_state_as_obs(pomdp, single_trace[j][1])
                    sp_list[:,i]  .= shape_state_as_obs(pomdp, single_trace[j][2])
                    r_list[1,i]    = single_trace[j][5]
                    t_list[1,i]    = single_trace[j][6]
                    done_list[1,i] = single_trace[j][7]
                end
                step_counter += step
                update!(p1, step_counter; showvalues=generate_showvalues(step_counter))
            elseif max_steps - step_counter + 1 ≤ sim_limit
                # avoid struggling to find a suitable simulation when we're close enough to the end
                break
            end
        elseif nonterminal_system
            for (j, i) in enumerate(step_counter:min(step_counter+step-1, max_steps))
                a_list[:,i]   .= single_trace[j][4]
                s_list[:,i]   .= shape_state_as_obs(pomdp, single_trace[j][1])
                sp_list[:,i]  .= shape_state_as_obs(pomdp, single_trace[j][2])
                r_list[1,i]    = single_trace[j][5]
                t_list[1,i]    = single_trace[j][6]
                done_list[1,i] = single_trace[j][7]
            end
            step_counter += step
            update!(p1, step_counter; showvalues=generate_showvalues(step_counter))
        end
    end
    step_counter = min(step_counter, max_steps)

    return Dict(:a => a_list[:,1:step_counter-1],
                :s => s_list[:,1:step_counter-1],
                :sp => sp_list[:,1:step_counter-1],
                :r => r_list[:,1:step_counter-1],
                :t => t_list[:,1:step_counter-1],
                :expert_val => expert_val_list[:,1:step_counter-1],
                :done => done_list[:,1:step_counter-1])
end