using LinearAlgebra: norm

export line_to_target_obj, time_till_completion_obj, safety_obj, combined_obj
export obj_from_landscape, obcs_from_landscape

"""Rewards based on closeness to goal.
"""
function line_to_target_obj(s::KAgentState, goal::Dict)
    dist = norm(s.x - goal[:target])
    return Any[goal[:strength] * exp(-dist^2 / goal[:influence]^2), dist < goal[:size]]
end

time_till_completion_obj(s::KAgentState, urgency::Float64) = Any[- urgency * length(s.hist), missing]

"""Goal-avoidance behavior. Takes a vector of obstacle descriptors.

Currently computes the "risk" of the agent in approaching an obstacle.

Each component of the vector is a dictionary holding two objects:
* :poly   => The vector of tuple-coordinates that represent the polygon. Must be closed!
* :risk   => The "risk"-scaling factor; scales the shortest distance between a point and obstacle.
* :impact => The amount by which an agent is penalized for entering an obstacle.

ex: Dict(:poly => [(0.,0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)], :risk => 10.)
"""
function safety_obj(s::KAgentState, obstacles::Vector)
    let x = GI.Point(Tuple(s.x)), total_risk = 0, collided = missing
        for obstacle in obstacles
            dist = GO.distance(x, GI.Polygon([obstacle[:poly]]))
            if dist <= 0.
                collided = true
            end
            total_risk += obstacle[:impact] * exp(-dist^2 / obstacle[:risk])
            # total_risk += dist * obstacle[:risk]
        end
        return [-total_risk, collided]
    end
end

combined_reward(r::Vector) = mapreduce(c->c[1], +, r)
combined_termination_check(b::BitVector) = any(b) & all(b)

function combined_obj(s::KAgentState, fs::Vector; digits=2)
    let fv = map(f->f(s), fs), rv = map(c->c[1], fv), clean_bv = Bool.(skipmissing(map(c->c[2], fv)))
        Any[round(combined_reward(rv); digits=digits), combined_termination_check(clean_bv)]
    end
end

function obj_from_landscape(objs::AgentObjectiveLandscape; digits=2)
    fs = map(objs.objectives) do obj
        @match obj[1] begin
            :goal => s->line_to_target_obj(s, obj[2])
            :obc  => s->safety_obj(s, obj[2])
            :horz => s->time_till_completion_obj(s, obj[2])
        end
    end
    return s->combined_obj(s, fs; digits=digits)
end

function obcs_from_landscape(objs::AgentObjectiveLandscape)
    obcs = map(objs.objectives) do obj
        @match obj[1] begin
            :obc => obj[2]
            _    => nothing
        end
    end
    map(filter(!isnothing, obcs)[1]) do obc
        GI.Polygon([obc[:poly]])
    end
end