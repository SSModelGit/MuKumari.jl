using LinearAlgebra: norm

export line_to_target_obj, time_till_completion_obj, safety_obj, combined_obj

"""Rewards based on closeness to goal.
"""
function line_to_target_obj(s::KAgentState, target::Matrix; r_scaling=100., threshold=0.1)
    d_to_target = norm(s.x - target)
    return [r_scaling * d_to_target, d_to_target < threshold]
end

"""Penalizes time spent based on length of stored history.
"""
time_till_completion_obj(s::KAgentState; r_scaling=10.) = [- r_scaling * length(s.hist), missing]

"""Goal-avoidance behavior.

Currently very simple, just checks if in obstacle or not.
"""
function safety_obj(s::KAgentState, obstacles::Vector; r_scaling=10.)
    let x = Point(Tuple(s.x))
        for obstacle in obstacles
            if x ∈ obstacle
                return [-r_scaling, false]
            end
        end
        return [0, missing]
    end
end

combined_reward(r::Vector) = mapreduce(c->c[1], +, r)
combined_termination_check(b::BitVector) = any(b) & all(b)

function combined_obj(s::KAgentState, fs::Vector)
    let fv = map(f->f(s), fs), rv = map(c->c[1], fv), clean_bv = Bool.(skipmissing(map(c->c[2], fv)))
        Any[combined_reward(rv), combined_termination_check(clean_bv)]
    end
end