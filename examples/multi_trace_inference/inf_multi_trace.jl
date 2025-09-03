using MuKumari

using LinearAlgebra: norm, normalize

using POMDPTools, MCTS, POMDPLinter

# addressing weird load order bugs
using Plots
using CUDA, cuDNN

# Commenting out CairoMakie due to current issue compiling with Plots and GR_jll
# using CairoMakie

# Make sure to load Plots before Crux, because of some weird load order bug
using Crux

using JLD2: @save, @load
using BSON

using Flux

println("Directory is: ", @__DIR__)

script_dir = @__DIR__

data = BSON.load(script_dir*"/100_15_100_7_multi_trace_run.bson")[:data]

kworld = data["kworld"]
ma_data = data["total"]

function trace_conversion_type1_to_type2(trace::Matrix; n_z::Integer=2, n_obcs::Integer=2, n_goals::Integer=1)
    n_rows = size(trace)[1]-2-n_z-n_obcs*2-n_goals*2
    new_trace = Matrix{Float64}(undef, n_rows, 0) # initialize empty

    new_col = zeros(n_rows)
    for c in eachcol(trace)
        x = c[1:2]
        obcs_idx = 2 + n_z + 1

    end
end

# anon_data.elements = 996

# N = anon_data.elements
# mdp = get_agent(kworld, "ag1")

# as = actions(mdp)
# S = state_space(mdp)
# γ = Float32(discount(mdp))
# A() = DiscreteNetwork(Chain(Dense(S.dims[1], 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)

# 𝒟_iql = OnlineIQLearn(π=A(), 𝒟_demo=anon_data, S=S, γ=γ, N=anon_data.elements, ΔN=1, c_opt=(;epochs=1),reg=false,gp=false, log=(;period=50))

# solve(𝒟_iql, mdp)

# plot_learning([𝒟_iql,], title="Results of IQ-Learning", labels=["iql",])