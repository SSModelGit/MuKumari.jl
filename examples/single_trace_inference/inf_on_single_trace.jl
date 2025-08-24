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

(kworld, data, anon_data) = BSON.load(script_dir*"/single_start_exp.bson")[:data]

anon_data.elements = 996

N = anon_data.elements
mdp = get_agent(kworld, "ag1")

as = actions(mdp)
S = state_space(mdp)
γ = Float32(discount(mdp))
A() = DiscreteNetwork(Chain(Dense(S.dims[1], 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)

𝒟_iql = OnlineIQLearn(π=A(), 𝒟_demo=anon_data, S=S, γ=γ, N=anon_data.elements, ΔN=1, c_opt=(;epochs=1),reg=false,gp=false, log=(;period=50))

solve(𝒟_iql, mdp)

plot_learning([𝒟_iql,], title="Results of IQ-Learning", labels=["iql",])