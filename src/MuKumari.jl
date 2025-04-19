module MuKumari

export MuEnv

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

predict_env(muenv::MuEnv, X::Matrix) = reshape([muenv.μf[μ](X) for μ in muenv.μ_order], muenv.M, 1)

update_μf(muenv::MuEnv, μ::Symbol, f::Function) = muenv.μf[μ] = f

end
