using LinearAlgebra: norm
using Match

export ObjectiveLandscape, line_to_target_obj, time_till_completion_obj, safety_obj, combined_obj, obj_from_landscape

"""A struct containing a vector of objectives. With special format.

Each element of the vector is a tuple:
* The first element of the tuple is a symbol indicating objective type
* The second element of the tuple contains the details of the objective itself

An example is:
```
(:goal, Dict(:target=>[9.5 9.5], :strength=>100., :influence=>10., :size=>0.5))
```

The objective types can be:
* Goal-attraction objective: :goal
  * The contents should be a dictionary
* Goal-avoidance objective: :obc
  * The contents should be a vector of all obstacles. Look at obstacle use for more information.
* Time horizon objective: :horz
  * The content is a single float to indicate horizon urgency (relative to strongest reward).
"""
struct ObjectiveLandscape
    objectives::Vector

    ObjectiveLandscape(objectives::Vector) = new(objectives)
end

"""Keyword-based constructor.
"""
ObjectiveLandscape(; objectives) = ObjectiveLandscape(objectives)

"""Rewards based on closeness to goal.
"""
function line_to_target_obj(s::KAgentState, goal::Dict)
    d_to_target = norm(s.x - goal[:target])
    return Any[goal[:strength] * exp(-d_to_target^2 / goal[:influence]), d_to_target < goal[:size]]
end

time_till_completion_obj(s::KAgentState, urgency::Float64) = Any[- urgency * length(s.hist), missing]

"""Goal-avoidance behavior. Takes a vector of obstacle descriptors.

Currently computes the "risk" of the agent in approaching an obstacle.

Each component of the vector is a dictionary holding two objects:
* :poly => The vector of tuple-coordinates that represent the polygon. Must be closed!
* :risk => The "risk"-scaling factor; scales the shortest distance between a point and obstacle.

ex: Dict(:poly => [(0.,0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)], :risk => 10.)
"""
function safety_obj(s::KAgentState, obstacles::Vector)
    let x = GI.Point(Tuple(s.x)), total_risk = 0, collided = missing
        for obstacle in obstacles
            dist = GO.distance(x, GI.Polygon([obstacle[:poly]]))
            if dist <= 0.
                collided = true
            end
            total_risk += dist * obstacle[:risk]
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

function obj_from_landscape(objs::ObjectiveLandscape; digits=2)
    fs = map(objs.objectives) do obj
        @match obj[1] begin
            :goal => s->line_to_target_obj(s, obj[2])
            :obc  => s->safety_obj(s, obj[2])
            :horz => s->time_till_completion_obj(s, obj[2])
        end
    end
    return s->combined_obj(s, fs; digits=digits)
end