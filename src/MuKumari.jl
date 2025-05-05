module MuKumari

using Reexport

export MuEnv, predict_μ, predict_env, update_μf

"""Type for the Mu Environment.

At construction time, the user must provide the `M` environment characteristic functions.

Currently, the implementation does not track time. This may be changed later.
Thus, the functions are internally assumed time-invariant.
To change the functions over time, the user may use the function `update_μf`.
* This is useful when the user manually controls the time-update step in simulation.
* Between each time-step update, use the `update_μf` function to optionally modify the environment.

To predict the environment state, use the associated methods `predict_μ` or `predict_env`.
"""
struct MuEnv
    M::Integer # number of characteristics tracked by the pseudo-environment
    μ_order::Vector{Symbol} # Order of characteristics for vector creation
    μf::Dict{Symbol, Function} # Association between characteristic symbol and characteristic function

    function MuEnv(M::Integer, μ_order::Vector, μf::Dict)
        @assert length(μ_order) == M
        @assert length(μf) == M
        new(M, μ_order, μf)
    end
end

predict_μ(muenv::MuEnv, μ::Symbol, X::Matrix) = muenv.μf[μ](X)

predict_env(muenv::MuEnv, X::Matrix) = reshape([muenv.μf[μ](X) for μ in muenv.μ_order], muenv.M)

update_μf(muenv::MuEnv, μ::Symbol, f::Function) = muenv.μf[μ] = f

include("intentional_agent.jl")
include("basic_objectives.jl")

end
