export AbstractObjectiveLandscape, AgentObjectiveLandscape, GlobalObjectiveLandscape, tangle_agent_landscape, detailed_global_obj_view
export line_to_target_obj, time_till_completion_obj, safety_obj, combined_obj
export obj_from_landscape, obcs_from_landscape

############################################
### Objective Landscapes
#
# Below we define objective landscapes.
#
# An objective landscape is a container of
# all objective functions used by the agent
# MDPs. It represents the "environment" by
# how it drives the agent's decisions.
############################################

abstract type AbstractObjectiveLandscape end

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

The final component of the struct is `f_types`. This does not need user input.
* This is relevant only when constructing agent landscapes from a global context.
* It will be automatically filled in based on what feature access the user defines for the agent.
"""
@with_kw_noshow struct AgentObjectiveLandscape <: AbstractObjectiveLandscape
    objectives::Vector
    f_types::Vector = []

    AgentObjectiveLandscape(objectives::Vector, f_types) = new(objectives, f_types)
end

function describe_single_obj(obj::Tuple; base="", offset="", eol="")
    s = "$(base)$(offset)|| "*uppercase(string(obj[1]))
    if typeof(obj[2]) <: Dict
        s = mapreduce(v->" | $(v[2][1]): $(v[2][2])", *, enumerate(obj[2]); init=s)
    elseif typeof(obj[2]) <: Float64
        s = s*" | urgency: $(obj[2])"
    end
    return s*" ||$(eol)"
end

function describe_objectives(objs::Vector; base="", offset="    ", eol="\n")
    mapreduce(*, objs; init="") do obj
        @match obj[1] begin
            :goal => describe_single_obj(obj; base=base, offset=offset, eol=eol)
            :obc  => mapreduce(o->describe_single_obj((:obcs, o); base=base, offset=offset, eol=eol), *, obj[2]; init="")
            :horz => describe_single_obj(obj; base=base, offset=offset, eol=eol)
        end
    end
end

function Base.show(io::IO, aobj::AgentObjectiveLandscape)
    println(io, "An agent-specific objective landscape tracking features: $(aobj.f_types)")
    print(io, "Agent objective descriptions:\n$(describe_objectives(aobj.objectives))")
end

"""Global container for all the landscape information regarding obstacles, goals, etc.

Consists of three fields: goals, obstacles, and horizons.
(The feature field is automatically constructed via the keyword constructor.)
* Each field takes an array of tuples. Each tuple consists of:
  * The feature's landscape level, ex. a surface obstacle would be tagged :surface
  * The feature information. The specific syntax is similar to the `AgentObjectiveLandscape`.

Example:
* We have two goals, one "aerial"- and one "surface"-level
* We have two obstacles that are both "subsurface"
* We have no horizons
```
GlobalObjectiveLandscape(
    [(:surface, Dict(:target=>[9.5 9.5], :strength=>100., :influence=>10., :size=>0.5)),
     (:aerial,  Dict(:target=>[7.5 7.5], :strength=>50., :influence=>10., :size=>1.5))],
    [(:sub, Dict(:poly => [(0., 0.), (0., 0.5), (0.4, 0.3), (0.5, 0.), (0., 0.)], :risk => 3., :impact => 10.)),
     (:sub, Dict(:poly => [(4., 4.5), (5., 4.5), (7., 5.), (2., 5.), (4., 4.5)],  :risk => 3., :impact => 10.))],
    []
)
```
"""
struct GlobalObjectiveLandscape <: AbstractObjectiveLandscape
    goals::Vector
    obstacles::Vector
    horizons::Vector
    feature_list::Vector

    GlobalObjectiveLandscape(goals::Vector, obstacles::Vector, horizons::Vector, feature_list::Vector) = new(goals, obstacles, horizons, feature_list)
end

"""Keyword-based constructor for the global landscape.

Automatically constructs the feature list.
"""
function GlobalObjectiveLandscape(; goals::Vector, obstacles::Vector, horizons::Vector)
    feature_set = Set()
    map(f->push!(feature_set, f[1]), Iterators.flatten([goals, obstacles, horizons]))

    GlobalObjectiveLandscape(goals, obstacles, horizons, collect(feature_set))
end

feature_name_list_from_vec(flist::Vector; base=" ", offset="", eol="") = mapreduce(x->"$(base)$(offset)\"$(x)\"$(eol)", *, flist; init="")

function Base.show(io::IO, gobj::GlobalObjectiveLandscape)
    println(io, "Summary of the global objective landscape")
    println(io, "    Goal types: ($(length(gobj.goals)) total)\n    $(feature_name_list_from_vec(map(x->x[1], gobj.goals)))")
    println(io, "    Obstacle types: ($(length(gobj.obstacles)) total)\n    $(feature_name_list_from_vec(map(x->x[1], gobj.obstacles)))")
    println(io, "    Horizon types: ($(length(gobj.horizons)) total)\n    $(feature_name_list_from_vec(map(x->x[1], gobj.horizons)))")
    println(io, "Features tracked in total: ($(length(gobj.feature_list)) total)\n    $(feature_name_list_from_vec(gobj.feature_list))")
end

function detailed_global_obj_view(gobj::GlobalObjectiveLandscape)
    s = string(gobj)*"\n" # first acquire standard summary
    s = s*"Objective details:\n----------------------------\n"
    s = s*"Goals:\n"*mapreduce(x->describe_single_obj(x; base="", offset="    ", eol="\n"), *, gobj.goals;init="")
    s = s*"\n----------------------------\nObstacles:\n"*mapreduce(x->describe_single_obj(x; base="", offset="    ", eol="\n"), *, gobj.obstacles;init="")
    s = s*"\n----------------------------\nHorizons:\n"*mapreduce(x->describe_single_obj(x; base="", offset="    ", eol="\n"), *, gobj.horizons;init="")
    s = s*"\n----------------------------\n"
    print(s)
end

"""Helper function for tangling features from global landscape.

Pulls out the features relevant to a given agent based on the list of accessible features.
"""
tangle_by_feature_access(features::Vector, access_list::Vector) = map(f -> f[2], filter(f -> f[1] ∈ access_list, features))

"""Helper function for satisfying the specific syntax requirements of the `AgentObjectiveLandscape`.
"""
function tupleify_features(f_type::Symbol, fs::Vector)
    @match f_type begin
        :goal => map(f->(:goal, f), fs)
        :obc  => (:obc, fs)
        :horz => map(f->(:horz, f), fs)
    end
end

"""Main function to derive an agent's objective landscape from the global objective landscape.

Requires, in addition to the global landscape, a vector of symbols corresponding to accessible feature types.
"""
function tangle_agent_landscape(gobj::GlobalObjectiveLandscape, f_access::Vector)
    @assert f_access ⊆ gobj.feature_list
    (goals, obcs, horzs) = map(f->tangle_by_feature_access(f, f_access), [gobj.goals, gobj.obstacles, gobj.horizons])
    objectives = [tupleify_features(:goal, goals)..., tupleify_features(:obc, obcs), tupleify_features(:horz, horzs)...]
    AgentObjectiveLandscape(; objectives=objectives, f_types=f_access)
end

############################################
### Objective Functions
#
# Below we define some basic objectives.
#
# Each objective represents a type of intent
# that drives the agent's behavior. For
# example, the "line to target objective"
# indicates the agent wants to approach a
# particular location. The below provided
# functions are parameterizable and
# sufficiently different, so as to capture
# the overall space of possible intents as
# broadly as possible.
############################################

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
                dist = 0.
                # No longer stopping on collision with obstacle
                # collided = true
            end
            total_risk += obstacle[:impact] * exp(-dist^2 / obstacle[:risk])
            # total_risk += dist * obstacle[:risk]
        end
        return [-total_risk, collided]
    end
end

combined_reward(r::Vector) = mapreduce(c->c[1], +, r)
# combined_termination_check(b::BitVector) = any(b) & all(b)
combined_termination_check(b::BitVector) = any(b)

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