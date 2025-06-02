using CairoMakie: Figure, Axis, DataAspect, Reverse, Colorbar, poly!, heatmap!, arrows!, Circle, Point2f, Vec2f

# Still not sure if this is the best way to handle plot sizing
const global inch::Float64 = 96
const global cm::Float64 = inch / 2.54
const global pt::Float64 = 4 / 3

export viz_system_single_timestep, viz_system_sim, stepthrough_sim

function viz_obcs(ax, obcs::Vector)
    for obc in obcs
        poly!(ax, obc[:poly], color=:red, alpha=0.8)
    end
end

function viz_obj_landscape(ax, objs::ObjectiveLandscape)
    for obj in objs.objectives
        @match obj[1] begin
            :goal => poly!(ax, Circle(Point2f(Tuple(obj[2][:target])), obj[2][:size]), strokecolor=:black, strokewidth=0.0pt, color=:green)
            :obc  => viz_obcs(ax, obj[2])
            _     => println("Ignoring horizon objective.")
        end
    end
end

function viz_agent_worldview(mdp::KAgentMDP, objs::ObjectiveLandscape)
    f = Figure(; figure_padding=(2.0pt, 2.0pt, 2.0pt, 2.0pt))
    ax = Axis(f[1,1], spinewidth=2.0pt, aspect=DataAspect())

    # Visualize world
    poly!(ax, collect(GI.getpoint(mdp.boxworld)), strokecolor=:black, strokewidth=1.0pt, color=:white, alpha=0.80)
    viz_obj_landscape(ax, objs)
    return f, ax
end

function viz_agent_reward_landscape(f, ax, mdp::KAgentMDP, s::KAgentState)
    hm = heatmap!(ax, mdp.dimensions[1]:0.1:mdp.dimensions[2], mdp.dimensions[1]:0.01:mdp.dimensions[2],
                  (x,y)->mdp.obj(pseudo_agent_placement(s, [x y]))[1],
                  alpha=0.5)
    Colorbar(f[1,2], hm)
end

function viz_agent_status(f, ax, mdp::KAgentMDP, s::KAgentState, a::Symbol; show_obj=false)
    poly!(ax, Circle(Point2f(Tuple(s.x)), 0.1pt), strokecolor=:black, strokewidth=0.0pt, color=:blue)
    arrows!(ax, [Point2f(Tuple(s.x))], [Vec2f(Tuple(action_heading_assoc_kagent[a]))], lengthscale = 0.3)
    if show_obj; viz_agent_reward_landscape(f, ax, mdp, s); end
end

function viz_system_single_timestep(mdp::KAgentMDP, s::KAgentState, a::Symbol, objs::ObjectiveLandscape)
    f, ax = viz_agent_worldview(mdp, objs)
    viz_agent_reward_landscape(f, ax, mdp, s)
    viz_agent_status(f, ax, mdp, s, a; show_obj=false)
    f
end

function viz_system_sim(mdp::KAgentMDP, objs::ObjectiveLandscape, sim_trace::Vector)
    f, ax = viz_agent_worldview(mdp, objs)
    viz_agent_reward_landscape(f, ax, mdp, sim_trace[end][1])

    for trace_step in sim_trace
        viz_agent_status(f, ax, mdp, trace_step[1], trace_step[2]; show_obj=false)
    end
    f
end

"""Complete a stepthrough of the system.

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