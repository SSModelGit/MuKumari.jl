using MuKumari

using LinearAlgebra: norm, normalize
using Statistics: std
using Combinatorics: powerset

using POMDPTools, MCTS, POMDPLinter
using Match: @match
using Parameters: @with_kw
using Printf
import GeoInterface as GI

# addressing weird load order bugs
using Plots
using StatsPlots
using Measures
using CUDA, cuDNN

# Commenting out CairoMakie due to current issue compiling with Plots and GR_jll
# using CairoMakie

# Make sure to load Plots before Crux, because of some weird load order bug
using Crux

# Utilizing Gen for generative approach to particle filters
using Gen: @gen, @trace, Distribution, UnknownChange, NoChange, categorical, choicemap, get_choice, get_choices, get_retval
using GenParticleFilters: pf_initialize, pf_rejuvenate!, pf_resample!, pf_update!, effective_sample_size, select, mh
using GenParticleFilters: get_traces, get_log_weights
import Gen

# Save data
using JLD2: @save, @load
using BSON

# Core of deepnet structure
using Flux

# construct a belief state out of the mdp parameters
beliefstate_for_pomdp(mdp::KAgentPOMDP) = KAgentBeliefUpdater(state_dims=length(mdp.start), env_dims=mdp.menv.M)
# since copy(::MCTSSolver) isn't defined for some reason, just use this instead of figuring out a copy extension
van_solver(params) = MCTSSolver(n_iterations=params[1], depth=20, exploration_constant=params[2])
dpw_solver(params) = DPWSolver(n_iterations=params[1], depth=20, exploration_constant=params[2])
function mcts_solver(pomdp::KAgentPOMDP; solver_params=[:van, 1000, 10.0])
    pomdp_bup = beliefstate_for_pomdp(pomdp)
    𝒮_mcts = @match solver_params[1] begin
        :van => BeliefMCTSSolver(van_solver(solver_params[2:end]), pomdp_bup)
        :dpw => BeliefMCTSSolver(dpw_solver(solver_params[2:end]), pomdp_bup)
        _    => BeliefMCTSSolver(van_solver(solver_params[2:end]), pomdp_bup)
    end
    return 𝒮_mcts
end

# define Deep Q solver (REINFORCE, DQN, and SoftQ)
function deep_q_solver(pomdp::KAgentPOMDP; solver_params=[:all, 10000, 2, 512])
    as = actions(pomdp)
    S = state_space(pomdp)
    A() = Flux.gpu(DiscreteNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as; dev=Flux.gpu))
    V() = Flux.gpu(ContinuousNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, 1))))

    solver_type, N, epochs, batch_size = solver_params[[1, 2, 3, 4]] # can also do solver_params[1:2] if I wanted - keeping this as later reminder for self on how to use indexing
    𝒮_net = @match solver_type begin
        :all => [REINFORCE(π=A(), S=S, N=N, ΔN=500, a_opt=(epochs=epochs,), interaction_storage=[]),
                 DQN(π=A(), S=S, N=N, interaction_storage=[]),
                 SoftQ(π=A(), α=Float32(0.1), S=S, N=N, ΔN=1, c_opt=(;epochs=epochs, batch_size=batch_size), interaction_storage=[])]
        :reinforce => [REINFORCE(π=A(), S=S, N=N, ΔN=500, a_opt=(epochs=epochs,), interaction_storage=[])]
        :dqn => [DQN(π=A(), S=S, N=N, interaction_storage=[])]
        :softq => [SoftQ(π=A(), α=Float32(0.1), S=S, N=N, ΔN=1, c_opt=(;epochs=epochs, batch_size=batch_size), interaction_storage=[])]
    end

    if !(solver_type==:all)
        return 𝒮_net[1]
    end
    return 𝒮_net
end

function solver_from_type(pomdp::KAgentPOMDP, type::Symbol=:dpw; solver_params)
    @match type begin
        :mcts => mcts_solver(pomdp;   solver_params=solver_params)
        :dql  => deep_q_solver(pomdp; solver_params=solver_params)
        _     => mcts_solver(pomdp;   solver_params=[:dpw, 1000, 10.0])
    end
end

function deep_q_metrics(pomdp::KAgentPOMDP, 𝒮_net; solver_type::Symbol=:all)
    @time π_net = @match solver_type begin
        :all => map(x->solve(x, pomdp), 𝒮_net)
        _    => solve(𝒮_net, pomdp)
    end
    labels = @match solver_type begin
        :all => ["REINFORCE", "DQN", "SoftQ" ]
        :reinforce => ["REINFORCE"]
        :dqn => ["DQN"]
        :softq => ["SoftQ" ]
    end
    p = plot_learning(𝒮_net, title = "One-shot Agent π Training Curves", 
                        labels = labels)
    
    return π_net, p
end

function quick_policy_compute_for_pomdp(pomdp::KAgentPOMDP; solver_type::Symbol=:mcts, solver_params=[:dpw, 1000, 10.0])
    𝒮_pomdp = solver_from_type(pomdp, solver_type; solver_params=solver_params)
    π_pomdp = solve(𝒮_pomdp, pomdp)
    return 𝒮_pomdp, π_pomdp
end

function quick_IQL(kworld::KWorld, anon_data::ExperienceBuffer; plot_metrics::Bool=false)
    N = anon_data.elements
    mdp = get_agent(kworld, "ag1")

    as = actions(mdp)
    S = state_space(mdp)
    γ = Float32(discount(mdp))
    A() = DiscreteNetwork(Chain(Dense(S.dims[1], 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as; dev=Flux.cpu)

    𝒟_iql = OnlineIQLearn(π=A(), 𝒟_demo=anon_data, S=S, γ=γ, N=anon_data.elements, ΔN=1, c_opt=(;epochs=1),reg=false,gp=false, log=(;period=50))

    π_iql = solve(𝒟_iql, mdp)

    if plot_metrics; f = plot_learning([𝒟_iql,], title="Results of IQ-Learning", labels=["iql",]); else; f = nothing; end
    return π_iql, 𝒟_iql, mdp, f
end

function quick_IQL(mdp::KAgentPOMDP, anon_data::ExperienceBuffer)
    N = anon_data.elements
    as = actions(mdp)
    S = state_space(mdp)
    γ = Float32(discount(mdp))

    A() = DiscreteNetwork(Chain(Dense(S.dims[1], 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as; dev=Flux.cpu)

    𝒟_iql = OnlineIQLearn(π=A(), 𝒟_demo=anon_data, S=S, γ=γ, N=anon_data.elements, ΔN=1, c_opt=(;epochs=1),reg=false,gp=false, log=(;period=50))

    π_iql = solve(𝒟_iql, mdp)

    return π_iql, 𝒟_iql, mdp
end

#################################
# Fourier-mode parameter sampling
#################################

@with_kw struct FourierDiscreteCfg
    Kmax::Int = 10
    λK::Float64 = 0.35            # P(K=k) ∝ exp(-λK*(k-1))

    # frequency grid
    Δf::Float64 = 0.1
    Fmax_i::Int = 10              # bins in -Fmax_i:Fmax_i

    # amplitude grid
    ΔA::Float64 = 0.1
    Amax_i::Int = 1              # bins in 0:Amax_i

    # phase grid
    P::Int = 32                   # bins in 0:P-1

    # optional: bias towards lower |freq|
    freq_mag_decay::Float64 = 0.0
end

@inline f_from_i(i::Int, cfg::FourierDiscreteCfg) = i * cfg.Δf
@inline A_from_i(i::Int, cfg::FourierDiscreteCfg) = i * cfg.ΔA
@inline ϕ_from_i(i::Int, cfg::FourierDiscreteCfg) = 2π * (i / cfg.P)

"""
    K_probs(cfg::FourierDiscreteCfg)

Constructs categorical vector mapping k-count to exponential decay distribution.
"""
function K_probs(cfg::FourierDiscreteCfg)
    ws = exp.(-cfg.λK .* (0:(cfg.Kmax-1)))
    ws ./= sum(ws)
    return ws
end

"""
    freq_bin_support_and_probs(cfg::FourierDiscreteCfg)

TODO: Constructs categorical vector mapping freq to exp decay??
"""
function freq_bin_support_and_probs(cfg::FourierDiscreteCfg)
    supp = collect(-cfg.Fmax_i:cfg.Fmax_i)
    if cfg.freq_mag_decay <= 0
        ws = fill(1.0, length(supp))
    else
        ws = exp.(-cfg.freq_mag_decay .* abs.(supp))
    end
    ws ./= sum(ws)
    return supp, ws
end

"""
    amp_bin_support_and_probs(cfg::FourierDiscreteCfg)

TODO: Constructs categorical vector mapping amplitude to uniform??
"""
function amp_bin_support_and_probs(cfg::FourierDiscreteCfg)
    supp = collect(0:cfg.Amax_i)
    ws = fill(1.0, length(supp))
    ws ./= sum(ws)
    return supp, ws
end

"""
    phase_bin_support_and_probs(cfg::FourierDiscreteCfg)

TODO: Constructs categorical vector mapping ϕ to uniform??
"""
function phase_bin_support_and_probs(cfg::FourierDiscreteCfg)
    supp = collect(0:(cfg.P-1))
    ws = fill(1.0, length(supp))
    ws ./= sum(ws)
    return supp, ws
end

"""
    gen_K(cfg::FourierDiscreteCfg)

Generative function to sample number of Fourier features.
"""
@gen function gen_K(cfg::FourierDiscreteCfg)
    K ~ categorical(K_probs(cfg))   # returns 1..Kmax
    return K
end

"""
    gen_mode_indices(cfg::FourierDiscreteCfg)

Generative function to sample f, A, and ϕ for a Fourier feature.
"""
@gen function gen_mode_indices(cfg::FourierDiscreteCfg)
    f_supp, f_w = freq_bin_support_and_probs(cfg)
    a_supp, a_w = amp_bin_support_and_probs(cfg)
    p_supp, p_w = phase_bin_support_and_probs(cfg)

    fx_idx ~ categorical(f_w)
    fy_idx ~ categorical(f_w)
    A_idx  ~ categorical(a_w)
    ϕ_idx  ~ categorical(p_w)

    return (fx_i = f_supp[fx_idx],
            fy_i = f_supp[fy_idx],
            A_i  = a_supp[A_idx],
            ϕ_i  = p_supp[ϕ_idx])
end


"""
    gen_fourier_bank(cfg::FourierDiscreteCfg)

Composes the Fourier feature sampling process:
* First, samples number of features to be used
* Second, samples the parameters for each feature (f, A, ϕ).

Returns a cached set of keys mapping to each feature and associated params.
"""
@gen function gen_fourier_bank_fixed(cfg::FourierDiscreteCfg)
    # K in 1..Kmax
    K = @trace(gen_K(cfg), :K)

    # supports & probs (precompute once)
    f_supp, f_w = freq_bin_support_and_probs(cfg)
    a_supp, a_w = amp_bin_support_and_probs(cfg)
    p_supp, p_w = phase_bin_support_and_probs(cfg)

    # fixed bank of discrete indices (length Kmax)
    fx_i = Vector{Int}(undef, cfg.Kmax)
    fy_i = Vector{Int}(undef, cfg.Kmax)
    A_i  = Vector{Int}(undef, cfg.Kmax)
    ϕ_i  = Vector{Int}(undef, cfg.Kmax)

    for m in 1:cfg.Kmax
        fx_idx = @trace(categorical(f_w), (:mode, m) => :fx_idx)
        fy_idx = @trace(categorical(f_w), (:mode, m) => :fy_idx)
        A_idx  = @trace(categorical(a_w), (:mode, m) => :A_idx)
        ϕ_idx  = @trace(categorical(p_w), (:mode, m) => :ϕ_idx)

        fx_i[m] = f_supp[fx_idx]
        fy_i[m] = f_supp[fy_idx]
        A_i[m]  = a_supp[A_idx]
        ϕ_i[m]  = p_supp[ϕ_idx]
    end

    # continuous params for the full bank
    fx = f_from_i.(fx_i, Ref(cfg))
    fy = f_from_i.(fy_i, Ref(cfg))
    A  = A_from_i.(A_i,  Ref(cfg))
    ϕ  = ϕ_from_i.(ϕ_i,  Ref(cfg))

    # stable cache key uses only the active prefix (1:K)
    key = (K, fx_i[1:K], fy_i[1:K], A_i[1:K], ϕ_i[1:K])

    return (key=key, K=K, fx=fx, fy=fy, A=A, ϕ=ϕ, fx_i=fx_i, fy_i=fy_i, A_i=A_i, ϕ_i=ϕ_i)
end

"""
    make_fourier_scalar_field(bank; normalize=true)

Returns:
- field(x::Real, y::Real)::Float64

Definition:
  field(x,y) = Σ_{m=1..K} A[m] * cos(fx[m]*x + fy[m]*y + ϕ[m])

If `scaleQ=true`, divides by max(1,K) so magnitude doesn't explode with K.
"""
function make_fourier_scalar_field(bank; scaleQ::Bool=true)
    K  = bank.K
    fx = bank.fx
    fy = bank.fy
    A  = bank.A
    ϕ  = bank.ϕ
    invK = scaleQ ? (1.0 / max(1, K)) : 1.0

    field = function (x::Real, y::Real)
        acc = 0.0
        @inbounds for m in 1:K
            acc += A[m] * cos(fx[m]*x + fy[m]*y + ϕ[m])
        end
        return invK * acc
    end

    return field
end

"""
    make_pomdp_objective_from_field(field; done_mode=:never, done_threshold=Inf)

Converts a scalar field into MuKumari's objective signature:
  obj(s)::Any = Any[reward::Real, done::Bool]

Note that it defaults to just saying `false`, as there is no clear `done` in the open-ended case.
"""
function make_pomdp_objective_from_field(field::Function)
    return (s) -> Any[field(s.x[1,1], s.x[1,2]), false]
end

######################################################
# Define Custom Distribution over Potential Objectives
######################################################

@with_kw struct ScoreΠDist
    ## dynamic/open-ended objective ids (Fourier keys)
    prop_names::Vector = []
    # q_objs::Dict
    # n_compobj_list::Dict
    ## prior weights per proposal (key => weight)
    n_qprop_list::Dict{Any,Float64} = Dict{Any,Float64}()
    ## mdp cache (key => mdp)
    n_propmdp_list::Dict{Any,Any} = Dict{Any,Any}()
    ## solver/policy caches (key => solver, policy)
    n_𝒮_proposals::Dict{Any,Any} = Dict{Any,Any}()
    n_π_proposals::Dict{Any,Any} = Dict{Any,Any}()
    # solver_type::Symbol = :dql
    # solver_params::Vector = [:softq, 10000]
    ## carries action mappings used by inference_model
    mdp_params::Vector = [] # [π_alist, π_a_1hot, π_a_1hotall] (by default)

    ### Open-Ended System Specific
    ## Fourier sampling config
    fourier_cfg::FourierDiscreteCfg = FourierDiscreteCfg()
end

# define getter functions
get_proposal_names(π_dist::ScoreΠDist) = π_dist.prop_names
# get_proposal_component_priors(π_dist::ScoreΠDist) = π_dist.q_objs
# get_proposal_component_objectives(π_dist::ScoreΠDist, proposal) = π_dist.n_compobj_list[proposal]
get_proposal_prior(π_dist::ScoreΠDist, proposal) = π_dist.n_qprop_list[proposal]
get_idxable_proposal_prior_list(π_dist::ScoreΠDist) = [get_proposal_prior(π_dist, p) for p in get_proposal_names(π_dist)]

π_alist(π_dist::ScoreΠDist) = π_dist.mdp_params[1]
π_a_1hot(π_dist::ScoreΠDist) = π_dist.mdp_params[2]
π_a_1hotall(π_dist::ScoreΠDist) = π_dist.mdp_params[3]

# lazy create mdp if missing
ensure_mdp!(π_dist::ScoreΠDist, key) = get!(π_dist.n_propmdp_list, key) do
    @error "ya fucked up, where's the mdp at"
end

# lazy solver
get_𝒮_proposal(π_dist::ScoreΠDist, key) = get!(π_dist.n_𝒮_proposals, key) do
    mdp = ensure_mdp!(π_dist, key)
    # specify Deep Q-learning approach; choose Soft-Q learning, for 2000 iterations (empirically selected)
    solver_from_type(mdp, :dql; solver_params=[:softq, 200, 2, 512])
end

# lazy policy
get_π_proposal(π_dist::ScoreΠDist, key) = get!(π_dist.n_π_proposals, key) do
    𝒮 = get_𝒮_proposal(π_dist, key)
    mdp = ensure_mdp!(π_dist, key)
    solve(𝒮, mdp)
end

store_π_iql(π_dist::ScoreΠDist, π_iql) = push!(π_dist.n_π_proposals, :iql=>π_iql)
get_π_iql(π_dist::ScoreΠDist) = get(π_dist.n_π_proposals, :iql, nothing)

"""
Register a newly seen key into the proposal set, if absent.
Optionally can set a default prior mass here;
TODO-lowprio: simplest is uniform mass then renormalize.
"""
function register_key_if_new!(π_dist::ScoreΠDist, key; prior_mass::Float64=1.0)
    if !(key in π_dist.prop_names)
        push!(π_dist.prop_names, key)
        π_dist.n_qprop_list[key] = prior_mass
    end
    return key
end

# Decide target training budget based on posterior mass
@inline function training_budget(prob::Float64)
    prob ≥ 0.30 && return 1500
    prob ≥ 0.10 && return 800
    prob ≥ 0.03 && return 300
    return 0
end

"""
    surrogate_dataset_from_iql_grid(π_dist, π_iql, mdp;
                                   eval_num=200,
                                   dims=nothing)

Evaluate π_iql on a grid of points across the environment, returning:
- state_data::Matrix{Float64} of size (2, N)  [x;y] per column
- observations::Vector{Int} of length N       action index aidx per point
- eval_locations::Vector{Any} (optional convenience) the original [x y] points

Note: `mdp` should be the `mdp` used for training π_iql! Not another one.
"""
function surrogate_dataset_from_iql_grid(π_dist::ScoreΠDist,
                                        π_iql,
                                        mdp::KAgentPOMDP;
                                        eval_num::Int=200,
                                        dims=nothing)

    # ----- grid sampling (same idea as grid_points in geninf_on_single_trace.jl) -----
    # grid_points(n, dims=(0.,10.)) returns a Vector of 1×2 matrices [x y] :contentReference[oaicite:3]{index=3}
    if isnothing(dims)
        dims = mdp.dimensions
    end
    dim_span  = dims[2] - dims[1]
    dim_shift = dim_span / 20
    nx = ceil(Int, sqrt(eval_num))
    ny = ceil(Int, eval_num / nx)
    xs = range(dims[1] + dim_shift, dims[2] - dim_shift; length=nx)
    ys = range(dims[1] + dim_shift, dims[2] - dim_shift; length=ny)
    eval_locations = collect(Iterators.take(([x y] for y in ys for x in xs), eval_num))

    N = length(eval_locations)

    # ----- build state_data matrix (2 × N) -----
    state_data = Matrix{Float64}(undef, Crux.dim(state_space(mdp))[1], N)
    # @inbounds for i in 1:N
    #     # eval_locations[i] is 1×2; store as x,y rows
    #     state_data[1, i] = Float64(eval_locations[i][1])
    #     state_data[2, i] = Float64(eval_locations[i][2])
    # end

    # ----- map action symbol -> action index (aidx) -----
    alist = π_alist(π_dist)
    a_to_idx = Dict{Any, Int}(a => j for (j, a) in enumerate(alist))

    observations = Vector{Int}(undef, N)

    # ----- evaluate π_iql on each grid state (pattern from expected_recons_err_against_iql) -----
    # expected_recons_err_against_iql does:
    #   s_obs = MuKumari.shape_state_as_obs(mdp, blindstart_KAgentState(mdp, x))
    #   a*    = action(π_iql, s_obs)[1] :contentReference[oaicite:4]{index=4}
    @inbounds for i in 1:N
        s = blindstart_KAgentState(mdp, eval_locations[i])
        obs = MuKumari.shape_state_as_obs(mdp, s)
        state_data[:, i] = copy(obs)
        asymb = action(π_iql, obs)[1]

        idx = get(a_to_idx, asymb, 0)
        idx == 0 && error("π_iql returned action $(asymb) not found in π_alist(π_dist). Check action sets match.")
        observations[i] = idx
    end

    return state_data, observations, eval_locations
end

#############################################################
# Define KAgentPOMDP around open-ended scalar field objective
#############################################################

"""
    build_kagent_pomdp(agent_params::Dict, obj::Function; name="fourier_obj")

Required keys in agent_params:
- :start::Matrix
- :dimensions::Tuple   # (d1, d2), same semantics as MuKumari
- :menv::MuEnv

Optional keys (with defaults aligned to init_standard_KAgentPOMDP):
- :digits::Int
- :agent_width::Float64
- :agent_speed::Float64
- :ag_mvt_noise::Float64
- :obs_noise::Float64
- :mdp_discount::Float64
- :obcs::Vector   # optional obstacles geometry (default empty)
- :goals::Vector  # optional goals geometry (default empty)
"""
function build_kagent_pomdp(agent_params::Dict, obj::Function; name::String="fourier_obj")
    @assert haskey(agent_params, :start)
    @assert haskey(agent_params, :dimensions)
    @assert haskey(agent_params, :menv)
    @assert haskey(agent_params, :obcs)

    start      = agent_params[:start]
    d          = agent_params[:dimensions]
    menv       = agent_params[:menv]
    obcs       = agent_params[:obcs]

    digits     = get(agent_params, :digits, 3)
    width      = get(agent_params, :agent_width, 0.1)
    speed      = get(agent_params, :agent_speed, 1.0)
    ag_noise   = get(agent_params, :ag_mvt_noise, 0.05)
    obs_noise  = get(agent_params, :obs_noise, 0.05)
    γ          = get(agent_params, :mdp_discount, 0.95)

    goals      = get(agent_params, :goals, Any[])

    # Minimal agent landscape placeholder (not used to define obj)
    objl = AgentObjectiveLandscape(objectives=Any[], f_types=Any[])

    # Mirror init_standard_KAgentPOMDP world construction
    boxworld = GI.Polygon([[(d[1], d[1]), (d[1], d[2]), (d[2], d[2]), (d[2], d[1]), (d[1], d[1])]])
    # Note: if obcs are empty, world is just the exterior ring.
    world = isempty(obcs) ? boxworld : GI.Polygon([GI.getexterior(boxworld), map(o -> GI.getexterior(o), obcs)...])

    return KAgentPOMDP(name=name, start=start,
                      dimensions=d, boxworld=boxworld,
                      objl=objl, obcs=obcs, goals=goals,
                      obj=obj,
                      world=world,
                      width=width, s=speed, w=ag_noise,
                      menv=menv, v=obs_noise, γ=γ,
                      digits=digits)
end

##############################################
# Define functions for custom Gen distribution
##############################################

struct ActionDirac <: Gen.Distribution{AbstractVector}
end

Gen.random(::ActionDirac, x::AbstractVector) = x
Gen.logpdf(::ActionDirac, v::AbstractVector, x::AbstractVector) = (argmax(v) == argmax(x)) ? 0.0 : -Inf
Gen.logpdf_grad(::ActionDirac, v, x) = (nothing,)
Gen.has_output_grad(::ActionDirac) = false
Gen.is_discrete(::ActionDirac) = true

const actiondirac = ActionDirac()
(::ActionDirac)(x::AbstractVector) = Gen.random(ActionDirac(), x)

ensure_mdp!(π_dist::ScoreΠDist, key, ff, agent_params::Dict) = get!(π_dist.n_propmdp_list, key) do
    field = make_fourier_scalar_field(ff; scaleQ=true)
    obj   = make_pomdp_objective_from_field(field)

    build_kagent_pomdp(agent_params, obj; name="fourier_" * string(hash(key)))
end

"""
    proposal_boltzmann(π_dist::ScoreΠDist, prop_name, loc::KAgentState)


Computes Boltzmann distribution for π_{prop_name}(s).

* `π_dist`: ScoreΠDist object containing all proposal policies and their associated MDPs.
* `prop_name`: Identifier (e.g. symbol or string) for the proposal whose policy should be evaluated.
* `loc`: Current state/location at which the policy is evaluated.
  * Assumed to be in KAgentState form; `MuKumari.shape_state_as_obs` is used internally to convert it.
  * TODO: Bad behavior to use a non-exported function! Either export or choose different approach.

The policy is evaluated using `Crux.value` on the one-hot action set, producing unnormalized action scores.
These are normalized with a softmax (after subtracting the maximum for numerical stability) to obtain a Boltzmann distribution.

Returns: `boltzmann`
* `boltzmann`::Matrix{Float64} giving the Boltzmann action distribution.
  * Rows correspond to states (only one).
  * Columns correspond to actions.
  * Values are cast to `Float64` (from `Float32`, e.g. when computed on GPU) for compatibility with Gen’s tracing and scoring machinery.
"""
function proposal_boltzmann(π_dist::ScoreΠDist, prop_name, loc; temperature::Float64=1.0)
    mdp = ensure_mdp!(π_dist, prop_name)
    π_prop = get_π_proposal(π_dist, prop_name)
    all_a_onehot = π_a_1hotall(π_dist)

    # assume state location already in obs vec form, otherwise need to use MuKumari.shape_state_as_obs(loc)
    q = Crux.value(π_prop, MuKumari.shape_state_as_obs(mdp, loc), all_a_onehot)
    # Stability + temperature
    T = max(temperature, 1e-6)
    logits = (q .- maximum(q, dims=2)) ./ T

    boltzmann = softmax(logits, dims=2)

    # cast boltzmann distribution into Float64 form, from as the GPU operates in Float32
    return Float64.(boltzmann)
end

Base.copy(s::KAgentState) = KAgentState(copy(s.x), copy(s.z), copy(s.hist))

"""
Decode Fourier key of the form (K, fx_i, fy_i, A_i, ϕ_i) into continuous params.
Assumes fx_i etc are integer vectors of length K (or length Kmax, if fixed-bank).
"""
function decode_fourier_key(key, cfg::FourierDiscreteCfg)
    K, fx_i, fy_i, A_i, ϕ_i = key
    # use only active prefix if vectors are longer
    fx = f_from_i.(fx_i[1:K], Ref(cfg))
    fy = f_from_i.(fy_i[1:K], Ref(cfg))
    A  = A_from_i.(A_i[1:K],  Ref(cfg))
    ϕ  = ϕ_from_i.(ϕ_i[1:K],  Ref(cfg))
    return (K=K, fx=fx, fy=fy, A=A, ϕ=ϕ, fx_i=fx_i[1:K], fy_i=fy_i[1:K], A_i=A_i[1:K], ϕ_i=ϕ_i[1:K])
end

"""
    top_objectives(pf_state, π_dist; topk=10)

Aggregates posterior mass by objective key (= trace return value).
Returns top-k with:
- key
- prob mass
- count
- decoded Fourier params
"""
function top_objectives(pf_state, π_dist::ScoreΠDist; topk::Int = 10)
    traces = get_traces(pf_state)
    logw   = get_log_weights(pf_state)

    # robust normalization (handles large negative logw); if all -Inf => fallback to counts
    finite = isfinite.(logw)
    if !any(finite)
        # no numeric weights available; return empirical counts only
        counts = Dict{Any,Int}()
        for tr in traces
            key = get_retval(tr)
            counts[key] = get(counts, key, 0) + 1
        end
        keys_sorted = sort(collect(keys(counts)), by=k->counts[k], rev=true)
        Kout = min(topk, length(keys_sorted))
        return [(key=keys_sorted[j],
                 prob=NaN,  # explicitly “unknown”
                 count=counts[keys_sorted[j]],
                 params=decode_fourier_key(keys_sorted[j], π_dist.fourier_cfg))
                for j in 1:Kout]
    end

    lw = logw[finite]
    tr = traces[finite]

    m = maximum(lw)
    w = exp.(lw .- m)
    Z = sum(w)
    p = w ./ Z

    mass   = Dict{Any,Float64}()
    counts = Dict{Any,Int}()

    @inbounds for i in eachindex(tr)
        key = get_retval(tr[i])
        mass[key]   = get(mass, key, 0.0) + p[i]
        counts[key] = get(counts, key, 0) + 1
    end

    keys_sorted = sort(collect(keys(mass)), by=k->mass[k], rev=true)
    Kout = min(topk, length(keys_sorted))

    return [(key=keys_sorted[j],
             prob=mass[keys_sorted[j]],
             count=counts[keys_sorted[j]],
             params=decode_fourier_key(keys_sorted[j], π_dist.fourier_cfg))
            for j in 1:Kout]
end

"""
    training_budget(prob; schedule=(200,600,1500), τ=(0.03,0.10,0.30))

Map posterior mass -> target training N.

- If prob ≥ τ3 => schedule[3]
- else if prob ≥ τ2 => schedule[2]
- else if prob ≥ τ1 => schedule[1]
- else => 0  (do not train yet)
"""
function training_budget(prob::Real; schedule::NTuple{3,Int}=(200, 600, 1500),
                         τ::NTuple{3,Float64}=(0.03, 0.10, 0.30))
    p = float(prob)
    if !isfinite(p) || p ≤ 0
        return 0
    elseif p ≥ τ[3]
        return schedule[3]
    elseif p ≥ τ[2]
        return schedule[2]
    elseif p ≥ τ[1]
        return schedule[1]
    else
        return 0
    end
end

"""
    hamming_fourier_key(k1, k2) -> Int

Hamming distance on Fourier *discrete* key representation.

Key format assumed:
    (K::Int, fx_i::Vector{Int}, fy_i::Vector{Int}, A_i::Vector{Int}, ϕ_i::Vector{Int})

Only compare the active prefixes (1:K), and add abs(K1-K2).
"""
function hamming_fourier_key(k1, k2)
    K1, fx1, fy1, A1, ϕ1 = k1
    K2, fx2, fy2, A2, ϕ2 = k2
    d = abs(K1 - K2)

    K = min(K1, K2)
    @inbounds for m in 1:K
        d += (fx1[m] != fx2[m])
        d += (fy1[m] != fy2[m])
        d += (A1[m]  != A2[m])
        d += (ϕ1[m]  != ϕ2[m])
    end

    # treat unmatched tail entries as mismatches
    if K1 != K2
        Kbig = max(K1, K2)
        d += 4 * (Kbig - K)  # each extra mode has 4 discrete indices
    end
    return d
end

"""
    nearest_trained_key(π_dist, key; min_trained=1)

Returns the closest key among those already in π_dist.n_π_proposals and
whose training steps record indicates ≥ min_trained.
Returns `nothing` if none exist.
"""
function nearest_trained_key(π_dist::ScoreΠDist, key; min_trained::Int=1)
    best = nothing
    best_d = typemax(Int)

    # fall back if training bookkeeping not present yet
    steps = get!(π_dist.n_𝒮_proposals, :_trained_steps) do
        Dict{Any,Int}()
    end

    for k in keys(π_dist.n_π_proposals)
        # skip non-keys (e.g. :iql)
        k isa Tuple || continue
        get(steps, k, 0) ≥ min_trained || continue

        d = hamming_fourier_key(key, k)
        if d < best_d
            best = k
            best_d = d
        end
    end
    return best
end

"""
    maybe_refine_policies!(π_dist, pf_state, agent_params;
                           topk=5, schedule=(200,600,1500), τ=(0.03,0.10,0.30))

Look at current PF posterior, choose a target budget per key via training_budget,
and call ensure_policy_trained_to! to escalate only those keys.
"""
function maybe_refine_policies!(π_dist::ScoreΠDist, pf_state, agent_params::Dict;
                               topk::Int=5,
                               schedule::NTuple{3,Int}=(200,600,1500),
                               τ::NTuple{3,Float64}=(0.03,0.10,0.30))
    tops = top_objectives(pf_state, π_dist; topk=topk)
    for item in tops
        # item.prob may be NaN during all -Inf weights; skip in that case
        prob = item.prob
        target = training_budget(prob; schedule=schedule, τ=τ)
        target == 0 && continue
        ensure_policy_trained_to!(π_dist, item.key, agent_params;
                                 target_steps=target, warm_start=true)
    end
    return nothing
end

############################
# Warm start utilities
############################

# Try to get a Flux/Crux model object we can copy params into/out of
_policy_model(π) = hasproperty(π, :model) ? getproperty(π, :model) : π

function _warm_start_params!(π_dest, π_src)
    md = _policy_model(π_dest)
    ms = _policy_model(π_src)

    pd = Flux.params(md)
    ps = Flux.params(ms)

    if length(pd) != length(ps)
        @warn "Warm start skipped (param count mismatch)" nd=length(pd) ns=length(ps)
        return π_dest
    end

    for (d, s) in zip(pd, ps)
        if size(d) != size(s)
            @warn "Warm start skipped (param shape mismatch)" sized=size(d) sizes=size(s)
            return π_dest
        end
    end

    for (d, s) in zip(pd, ps)
        d .= s
    end
    return π_dest
end

############################
# Multi-fidelity training core
############################

"""
    ensure_policy_trained_to!(π_dist, key, agent_params;
                              target_steps, warm_start=true,
                              epochs=2, batch_size=512)

Ensures:
- an MDP exists for key (must already be in n_propmdp_list; created via inference_model)
- a SoftQ solver/policy exists
- training has been run up to `target_steps` (in solver N units)

Uses:
- nearest_trained_key(...) and hamming_fourier_key(...) for warm start
- stores trained steps in π_dist.n_𝒮_proposals[:_trained_steps]::Dict{Any,Int}

Returns: policy object (π_dist.n_π_proposals[key])
"""
function ensure_policy_trained_to!(π_dist::ScoreΠDist, key, agent_params::Dict;
                                  target_steps::Int,
                                  warm_start::Bool=true,
                                  epochs::Int=2,
                                  batch_size::Int=512)

    # bookkeeping dict (stored inside n_𝒮_proposals to avoid struct edits)
    trained = get!(π_dist.n_𝒮_proposals, :_trained_steps) do
        Dict{Any,Int}()
    end
    already = get(trained, key, 0)
    if already ≥ target_steps && haskey(π_dist.n_π_proposals, key)
        return π_dist.n_π_proposals[key]
    end

    # MDP must exist (inference_model should have created it)
    if !haskey(π_dist.n_propmdp_list, key)
        # If key hasn't been instantiated yet, we cannot train it here.
        # (The PF will create it once it samples it.)
        return get(π_dist.n_π_proposals, key, nothing)
    end
    mdp = ensure_mdp!(π_dist, key)

    # Build a solver for the *target* budget.
    solver = solver_from_type(mdp, :dql; solver_params=[:softq, target_steps, epochs, batch_size])

    # Warm start policy network parameters from nearest trained neighbor (if requested)
    if warm_start
        nn = nearest_trained_key(π_dist, key; min_trained=1)
        if nn !== nothing && haskey(π_dist.n_π_proposals, nn)
            try
                if hasproperty(solver, :π)
                    _warm_start_params!(getproperty(solver, :π), π_dist.n_π_proposals[nn])
                end
            catch err
                @warn "Warm start failed; training from scratch" err=err
            end
        end
    end

    # Train policy (solve)
    π = solve(solver, mdp)

    # Cache updated solver/policy and trained steps
    π_dist.n_𝒮_proposals[key] = solver
    π_dist.n_π_proposals[key] = π
    trained[key] = target_steps

    return π
end

@gen function inference_model(N::Int, π_dist::ScoreΠDist, agent_params::Dict, state_data::Matrix)
    # sample discretized Fourier parameters (traceable)
    fourier = @trace(gen_fourier_bank_fixed(π_dist.fourier_cfg), :fourier)
    key = fourier.key
    # register for downstream reporting / priors
    register_key_if_new!(π_dist, key)

    # lazy build mdp/policy (side-effecting cache)
    mdp = ensure_mdp!(π_dist, key, fourier, agent_params)
    _   = get_π_proposal(π_dist, key) # only use this to do lazy-loading as needed

    temp = get(agent_params, :policy_temperature, 1.0)
    for n in 1:N
        s = blindstart_KAgentState(mdp, reshape(state_data[:,n][1:2], (1,2)))
        boltzmann = max.(vec(proposal_boltzmann(π_dist, key, s; temperature=temp)), 0.0)
        boltzmann ./= sum(boltzmann)
        _ = {n => :aidx} ~ categorical(boltzmann)
    end

    return key
end

function particle_filter(observations::Vector{Int}, π_dist::ScoreΠDist, agent_params::Dict, state_data::Matrix,
                         n_particles::Int = 50; ess_thresh::Float64 = 0.5,
                         rejuv_modes::Int = 8, rejuv_recent_actions::Int = 3,
                         resample_alg::Symbol = :residual,
                         refine_every::Int = 5,
                         refine_topk::Int = 5)

    N = length(observations)
    obs_choices = [choicemap((n => :aidx, observations[n])) for n in 1:N]

    state = pf_initialize(inference_model, (1, π_dist, agent_params, state_data), obs_choices[1], n_particles)

    for n in 2:N
        if effective_sample_size(state) < ess_thresh * n_particles
            pf_resample!(state, resample_alg)

            # rejuvenation selection
            sels = Any[:fourier => :K]
            M = min(rejuv_modes, π_dist.fourier_cfg.Kmax)

            for m in 1:M
                push!(sels, (:fourier, :mode, m) => :fx_idx)
                push!(sels, (:fourier, :mode, m) => :fy_idx)
                push!(sels, (:fourier, :mode, m) => :A_idx)
                push!(sels, (:fourier, :mode, m) => :ϕ_idx)
            end

            a_lo = max(1, n - rejuv_recent_actions)
            for τ in a_lo:(n-1)
                push!(sels, (τ => :aidx))
            end

            pf_rejuvenate!(state, mh, (select(sels...),))

            # after a resample/rejuv event is a great time to refine top policies
            maybe_refine_policies!(π_dist, state, agent_params; topk=refine_topk)
        end

        # Update with new observation
        pf_update!(state,
                   (n, π_dist, agent_params, state_data),
                   (UnknownChange(), NoChange(), NoChange(), NoChange()),
                   obs_choices[n])

        # periodic refinement (lightweight)
        if (n % refine_every) == 0
            maybe_refine_policies!(π_dist, state, agent_params; topk=refine_topk)
        end
    end

    return state
end

# @gen function inference_model(T::Int, π_dist::ScoreΠDist, agent_params::Dict)
#     # sample discretized Fourier parameters (traceable)
#     fourier = @trace(gen_fourier_bank_fixed(π_dist.fourier_cfg), :fourier)
#     key = fourier.key
#     # register for downstream reporting / priors
#     register_key_if_new!(π_dist, key)

#     # lazy build mdp/policy (side-effecting cache)
#     mdp = ensure_mdp!(π_dist, key, fourier, agent_params)
#     π   = get_π_proposal(π_dist, key) # only use this to do lazy-loading as needed

#     s = copy(agent_params[:start_state])
#     temp = get(agent_params, :policy_temperature, 1.0)
#     a_all = []
#     for t in 1:N
#         boltzmann = max.(vec(proposal_boltzmann(π_dist, key, s; temperature=temp)), 0.0)
#         boltzmann ./= sum(boltzmann)
#         aidx = {t => :aidx} ~ categorical(boltzmann)

#         asymb = π_alist(π_dist)[aidx]
#         a = π_a_1hot(π_dist)(asymb)
#         {t => :a} ~ actiondirac(a) # record action choice for debugging traces
#         push!(a_all, a)
#         # state transition
#         s = POMDPs.@gen(:sp)(mdp, s, asymb)
#     end

#     return key
# end

################
### Plotting ###
################

############################
### Objective Visualization
############################

"""
    _dims_to_bounds(dimensions) -> (lo, hi)

Standardizes boxworld construction from 2-tuple `dimensions` with corners (d1,d1) and (d2,d2).
"""
@inline function _dims_to_bounds(dimensions)
    lo, hi = dimensions[1], dimensions[2]
    lo <= hi || error("dimensions must satisfy dimensions[1] <= dimensions[2]; got $(dimensions)")
    return lo, hi
end

"""
    _grid_from_mdp(mdp; gridsize=100) -> (xs, ys)

Grid over (x,y) spanning mdp.dimensions.
"""
function _grid_from_mdp(mdp::KAgentPOMDP; gridsize::Int=100)
    lo, hi = _dims_to_bounds(mdp.dimensions)
    xs = range(lo, hi; length=gridsize)
    ys = range(lo, hi; length=gridsize)
    return xs, ys
end

"""
    _state_at_xy(mdp, x, y) -> KAgentState

Constructs a state located at (x,y) using MuKumari's blindstart helper.
"""
@inline function _state_at_xy(mdp::KAgentPOMDP, x::Real, y::Real)
    return blindstart_KAgentState(mdp, reshape([Float64(x), Float64(y)], (1,2)))
end

"""
    objective_grid_from_field(field, xs, ys) -> Matrix

Returns Z where Z[j,i] = field(xs[i], ys[j]) (i = x index, j = y index),
which matches Plots.heatmap(x, y, Z) conventions.
"""
function objective_grid_from_field(field::Function, xs, ys)
    Z = Matrix{Float64}(undef, length(ys), length(xs))
    @inbounds for (j, y) in enumerate(ys)
        for (i, x) in enumerate(xs)
            Z[j,i] = field(x, y)
        end
    end
    return Z
end

"""
    objective_grid_from_mdp(mdp, xs, ys) -> Matrix

Uses mdp.obj(s)[1] as the scalar objective/reward at (x,y).
"""
function objective_grid_from_mdp(mdp::KAgentPOMDP, xs, ys)
    Z = Matrix{Float64}(undef, length(ys), length(xs))
    @inbounds for (j, y) in enumerate(ys)
        for (i, x) in enumerate(xs)
            s = _state_at_xy(mdp, x, y)
            r = mdp.obj(s)[1]
            Z[j,i] = Float64(r)
        end
    end
    return Z
end

"""
    xy_path_from_state_matrix(S; xy_rows=(1,2)) -> (xs, ys)

Extracts the trajectory from `data::ExperienceBufer`

- S is (n_features × T)
- returns vectors length T
"""
function xy_path_from_state_matrix(S::AbstractMatrix; xy_rows::Tuple{Int,Int}=(1,2))
    rx, ry = xy_rows
    T = size(S, 2)
    xs = Vector{Float64}(undef, T)
    ys = Vector{Float64}(undef, T)
    @inbounds for t in 1:T
        xs[t] = Float64(S[rx, t])
        ys[t] = Float64(S[ry, t])
    end
    return xs, ys
end

"""
    greedy_action_symbol_from_boltzmann(π_dist, key, s) -> (a_sym, a_idx)

Uses `proposal_boltzmann(...)` machinery to compute a Boltzmann
distribution over actions and then selects argmax (greedy).

This gives a deterministic rollout for visual comparison.
"""
function greedy_action_symbol_from_boltzmann(π_dist::ScoreΠDist, key, s::KAgentState)
    b = vec(proposal_boltzmann(π_dist, key, s))
    # guard against tiny negatives/nans
    b = max.(b, 0.0)
    if !(isfinite(sum(b))) || sum(b) <= 0
        # fall back to uniform if something went wrong numerically
        b .= 1.0
    end
    aidx = argmax(b)
    asymb = π_alist(π_dist)[aidx]
    return asymb, aidx
end

"""
    rollout_greedy_policy(π_dist, key; start_state, T) -> (xs, ys, states)

Rolls out the policy induced by the proposal's Q-function on its own MDP.
Uses greedy selection from the Boltzmann distribution (argmax over actions).
"""
function rollout_greedy_policy(π_dist::ScoreΠDist, key;
                               start_state::KAgentState,
                               T::Int)
    mdp = ensure_mdp!(π_dist, key)
    s = copy(start_state)
    xs = Vector{Float64}(undef, T)
    ys = Vector{Float64}(undef, T)
    states = Vector{KAgentState}(undef, T)

    @inbounds for t in 1:T
        xs[t] = Float64(s.x[1,1])
        ys[t] = Float64(s.x[1,2])
        states[t] = copy(s)

        asymb, _ = greedy_action_symbol_from_boltzmann(π_dist, key, s)

        # Transition using the POMDP generative step like in inference_model
        s = POMDPs.@gen(:sp)(mdp, s, asymb)
    end

    return xs, ys, states
end

"""
    top_key(pf_state, π_dist) -> (key, prob)

Convenience accessor for top posterior objective key.
"""
function top_key(pf_state, π_dist::ScoreΠDist)
    tops = top_objectives(pf_state, π_dist; topk=1)
    isempty(tops) && error("top_objectives returned empty; cannot plot.")
    return tops[1].key, tops[1].prob
end

"""
    plot_top_objective_with_trajectories(pf_state, π_dist, agent_params;
                                         observed_state_matrix,
                                         gridsize=140,
                                         xy_rows=(1,2),
                                         show_predicted=true,
                                         title_prefix="Top objective")

Heatmap of the inferred top objective function, overlaying:
- observed agent trajectory (from data)
- predicted rollout under the inferred objective's MDP+policy (greedy)

Returns a Plots.jl plot object.
"""
function plot_top_objective_with_trajectories(pf_state, π_dist::ScoreΠDist, agent_params::Dict;
                                             observed_state_matrix::AbstractMatrix,
                                             gridsize::Int=140,
                                             xy_rows::Tuple{Int,Int}=(1,2),
                                             show_predicted::Bool=true,
                                             title_prefix::String="Top objective")

    key, prob = top_key(pf_state, π_dist)

    # Build inferred scalar field from decoded params
    ff = decode_fourier_key(key, π_dist.fourier_cfg)
    field = make_fourier_scalar_field(ff; scaleQ=true)

    # Need an mdp for plotting bounds (use cached/ensured proposal mdp)
    mdp_hat = ensure_mdp!(π_dist, key)
    xs_grid, ys_grid = _grid_from_mdp(mdp_hat; gridsize=gridsize)

    Z = objective_grid_from_field(field, xs_grid, ys_grid)

    # Observed trajectory
    obs_x, obs_y = xy_path_from_state_matrix(observed_state_matrix; xy_rows=xy_rows)
    T = length(obs_x)

    p = heatmap(xs_grid, ys_grid, Z;
               aspect_ratio=1,
               title="$(title_prefix) (posterior ≈ $(prob))",
               xlabel="x", ylabel="y",
               colorbar_title="objective")

    plot!(p, obs_x, obs_y; label="observed", linewidth=3)

    if show_predicted
        start_state = agent_params[:start_state]
        pred_x, pred_y, _ = rollout_greedy_policy(π_dist, key; start_state=start_state, T=T)
        plot!(p, pred_x, pred_y; label="predicted (greedy)", linewidth=3, linestyle=:dash)
    end

    # Mark start/end for quick visual sanity
    scatter!(p, [obs_x[1]], [obs_y[1]]; label="obs start", markersize=6)
    scatter!(p, [obs_x[end]], [obs_y[end]]; label="obs end", markersize=6)

    return p
end

"""
    plot_objective_side_by_side(pf_state, π_dist;
                                observed_mdp,
                                gridsize=140,
                                title_left="Inferred top objective",
                                title_right="Observed MDP objective")

Side-by-side heatmaps:
- inferred top objective field (from Fourier features)
- observed MDP objective (mdp.obj(s)[1])

Returns a Plots.jl plot object with layout (1,2).
"""
function plot_objective_side_by_side(pf_state, π_dist::ScoreΠDist;
                                    observed_mdp::KAgentPOMDP,
                                    gridsize::Int=140,
                                    title_left::String="Inferred top objective",
                                    title_right::String="Observed MDP objective")

    key, prob = top_key(pf_state, π_dist)

    ff = decode_fourier_key(key, π_dist.fourier_cfg)
    field = make_fourier_scalar_field(ff; scaleQ=true)

    # Use observed mdp bounds for both to make comparison apples-to-apples
    xs_grid, ys_grid = _grid_from_mdp(observed_mdp; gridsize=gridsize)

    Z_inf = objective_grid_from_field(field, xs_grid, ys_grid)
    Z_obs = objective_grid_from_mdp(observed_mdp, xs_grid, ys_grid)

    p1 = heatmap(xs_grid, ys_grid, Z_inf;
                 aspect_ratio=1,
                 title="$(title_left)\n(posterior ≈ $(prob))",
                 xlabel="x", ylabel="y",
                 colorbar_title="objective")

    p2 = heatmap(xs_grid, ys_grid, Z_obs;
                 aspect_ratio=1,
                 title=title_right,
                 xlabel="x", ylabel="y",
                 colorbar_title="objective")

    return plot(p1, p2; layout=(1,2))
end

###############
### Testing ###
###############

using BSON
using LinearAlgebra

############################
# Single “run data” struct #
############################

struct RunPack
    run_id::Int                 # top-level run index in the BSON
    agent::String               # "ag1".."ag7"
    inst::Int                   # instance index (k)
    mdp::Any                    # KAgentPOMDP
    full::Any                   # ExperienceBuffer (full)
    anon::Any                   # ExperienceBuffer (anon; used for IQL)
    ann::NamedTuple             # (num_goals, num_obstacles, max_goal_separation)
end

##########################
# Minimal annotations API
##########################

# robust target extraction: goals look like (:aer, Dict(:target=>[x,y], ...))
function _goal_targets(goals)
    ts = Vector{Vector{Float64}}()
    for g in goals
        d = g[2]
        if d isa Dict && haskey(d, :target)
            push!(ts, vec(Float64.(d[:target])))
        end
    end
    return ts
end

function _max_pairwise_dist(X::Vector{Vector{Float64}})
    n = length(X)
    n ≤ 1 && return 0.0
    best = 0.0
    @inbounds for i in 1:n, j in (i+1):n
        best = max(best, norm(X[i] .- X[j]))
    end
    return best
end

function kworld_annotations(kworld)
    gl = getproperty(kworld, :glob_landscape)
    goals = getproperty(gl, :goals)
    obcs  = getproperty(gl, :obstacles)
    return (
        num_goals = length(goals),
        num_obstacles = length(obcs),
        max_goal_separation = _max_pairwise_dist(_goal_targets(goals))
    )
end

#################################
# BSON -> Vector{RunPack} loader
#################################

# Supports:
#  - stored as raw[:data] = (kworld, dataDict)
#  - stored as raw[:runs] = [ ... ]  (each a run dict or (kworld,data))
function _normalize_run_payload(x)
    if x isa Tuple && length(x) == 2
        kw, d = x
        d isa Dict || error("Expected (kworld, Dict) in run payload.")
        dd = deepcopy(d)
        dd["kworld"] = kw
        return dd
    end
    x isa Dict || error("Expected Dict run payload.")
    return x
end

function load_runpacks(bson_path::AbstractString)
    raw = BSON.load(bson_path)

    runs =
        haskey(raw, :runs)  ? raw[:runs]  :
        haskey(raw, "runs") ? raw["runs"] :
        haskey(raw, :data)  ? [raw[:data]] :
        haskey(raw, "data") ? [raw["data"]] :
        [raw]

    packs = RunPack[]
    for (rid, r0) in enumerate(runs)
        run = _normalize_run_payload(r0)

        kworld = haskey(run, "kworld") ? run["kworld"] :
                 haskey(run, :kworld)  ? run[:kworld]  :
                 error("No kworld in run $rid")

        ann = kworld_annotations(kworld)

        # agent keys are Strings like "ag1".."ag7"
        agent_keys = sort([k for k in keys(run) if k isa String && startswith(k, "ag")])

        for agent in agent_keys
            expdict = run[agent]  # Dict(:ind_exps=>..., :total_exp=>...)
            insts = expdict[:ind_exps]

            for k in 1:length(insts)
                full_buf, anon_buf = insts[k]
                full_buf = data_cleaner(full_buf, [2,2,12,10,1],Bool[1,1,1,0,1])
                anon_buf = data_cleaner(anon_buf, [2,2,12,10,1],Bool[1,1,1,0,1])
                name = agent * "_" * string(k)
                mdp  = kworld.inhabitants[name]  # matches generator naming
                push!(packs, RunPack(rid, agent, k, mdp, full_buf, anon_buf, ann))
            end
        end
    end
    return packs
end

#######################################
# Evaluation core (two “modes” of PF)
#######################################

# results are returned as NamedTuples for easy downstream processing
# (keeps code compact; no separate Result type required)
function eval_pack(pack::RunPack;
                   n_particles::Int=50,
                   ess_thresh::Float64=0.7,
                   refine_every::Int=5,
                   refine_topk::Int=5,
                   iql_gridN::Int=100,
                   minN::Int=20)

    mdp = pack.mdp

    # 1) train IQL on anon buffer
    # In geninf_on_rff.jl, quick_IQL(kworld, anon_data) trains using mdp=get_agent(kworld,"ag1").
    # Here we use a minimal per-mdp pattern (works if OnlineIQLearn etc already imported):
    π_iql, 𝒟_iql, _, _ = quick_IQL(mdp, pack.anon)

    # 2) build π_dist action mappings from this mdp’s action set
    as = actions(mdp)
    action_list = [as, a->Flux.onehot(a, as), Flux.onehotbatch(as, as)]
    π_dist = ScoreΠDist(; mdp_params=action_list)

    # 3) agent_params from mdp
    agent_params = agent_params_from_mdp(mdp)
    T = size(pack.full.data[:s], 2) # num of cols in state # TODO: should be ...data.elements

    # 4) Decide on T
    data_slices = (T ≤ minN) ? collect(1:T) : [rand(1:T) for _ in 1:minN]

    ########################
    # Mode A: real dataset
    ########################
    # PF uses (state_data[:,1:T], aidx[1:T]) from the full buffer
    state_data = pack.full.data[:s][:, data_slices]
    obs_aidx   = onehot_cols_to_aidx(pack.full.data[:a][:, data_slices])

    pf_real = particle_filter(obs_aidx, π_dist, agent_params, state_data, n_particles;
                              ess_thresh=ess_thresh, refine_every=refine_every, refine_topk=refine_topk)

    ###############################
    # Mode B: IQL grid surrogate PF
    ###############################
    iql_state_data, iql_obs_aidx, _ = surrogate_dataset_from_iql_grid(π_dist, π_iql, mdp; eval_num=iql_gridN)

    pf_iql = particle_filter(iql_obs_aidx, π_dist, agent_params, iql_state_data, n_particles;
                             ess_thresh=ess_thresh, refine_every=refine_every, refine_topk=refine_topk)

    return (
        pack = pack,
        mdp = mdp,
        agent_params = agent_params,
        π_dist = π_dist,
        π_iql = π_iql,
        pf_real = pf_real,
        pf_iql  = pf_iql,
        real = (state_data=state_data, obs_aidx=obs_aidx),
        iql  = (state_data=iql_state_data, obs_aidx=iql_obs_aidx)
    )
end

#############################
# Metrics (compact + useful)
#############################

# --- degeneracy ---
function pf_degeneracy(pf_state, π_dist; n_particles::Int)
    logw = get_log_weights(pf_state)
    finite = isfinite.(logw)
    all_ninf = !any(finite)

    tops = top_objectives(pf_state, π_dist; topk=5)
    nunique = length(tops)
    collapsed = (nunique == 1) && (!isempty(tops)) && (tops[1].count == n_particles)

    return (all_logw_ninf=all_ninf, nunique=nunique, collapsed=collapsed, ess=effective_sample_size(pf_state))
end

# --- objective reconstruction: z-scored RMSE + correlation on a grid ---
function _zscore(Z)
    μ = mean(Z)
    σ = std(vec(Z))
    σ = (σ ≤ 1e-12) ? 1.0 : σ
    return (Z .- μ) ./ σ
end

function objective_recon_metrics(pf_state, π_dist, mdp; gridsize::Int=120)
    tops = top_objectives(pf_state, π_dist; topk=1)
    isempty(tops) && return (rmse_z=NaN, corr=NaN)

    key = tops[1].key
    ff = decode_fourier_key(key, π_dist.fourier_cfg)
    field = make_fourier_scalar_field(ff; scaleQ=true)

    lo, hi = mdp.dimensions
    xs = range(lo, hi; length=gridsize)
    ys = range(lo, hi; length=gridsize)

    Zhat  = Matrix{Float64}(undef, length(ys), length(xs))
    Ztrue = Matrix{Float64}(undef, length(ys), length(xs))

    @inbounds for (j,y) in enumerate(ys), (i,x) in enumerate(xs)
        Zhat[j,i] = field(x,y)
        s = blindstart_KAgentState(mdp, [x y])
        Ztrue[j,i] = Float64(mdp.obj(s)[1])
    end

    A = vec(_zscore(Zhat))
    B = vec(_zscore(Ztrue))
    rmse = sqrt(mean((A .- B).^2))
    corr = dot(A,B) / (norm(A)*norm(B) + 1e-12)
    return (rmse_z=rmse, corr=corr)
end

# --- “policy matches true actions at true states” for top key (greedy argmax) ---
function policy_match_acc(pf_state, π_dist, agent_params, state_data, obs_aidx)
    tops = top_objectives(pf_state, π_dist; topk=1)
    isempty(tops) && return (acc=NaN, N=0)
    key = tops[1].key
    mdp_hat = ensure_mdp!(π_dist, key)

    temperature = get(agent_params, :policy_temperature, 1.0)

    T = length(obs_aidx)
    pred = Vector{Int}(undef, T)
    @inbounds for t in 1:T
        s = blindstart_KAgentState(mdp_hat, reshape(state_data[:,t][1:2], (1,2)))
        b = vec(proposal_boltzmann(π_dist, key, s; temperature=temperature))
        pred[t] = argmax(b)
    end
    return (acc=mean(pred .== obs_aidx), N=T)
end

##############################################
# Aggregate per-pack results into compact rows
##############################################

function summarize_eval(evals; n_particles::Int, gridsize::Int=120)
    rows_real = NamedTuple[]
    rows_iql  = NamedTuple[]

    for E in evals
        pack = E.pack
        ann  = pack.ann

        # Real-mode metrics
        degR = pf_degeneracy(E.pf_real, E.π_dist; n_particles=n_particles)
        objR = objective_recon_metrics(E.pf_real, E.π_dist, E.mdp; gridsize=gridsize)
        polR = policy_match_acc(E.pf_real, E.π_dist, E.agent_params, E.real.state_data, E.real.obs_aidx)

        push!(rows_real, (
            run_id=pack.run_id, agent=pack.agent, inst=pack.inst,
            num_goals=ann.num_goals, num_obstacles=ann.num_obstacles, max_goal_sep=ann.max_goal_separation,
            obj_rmse_z=objR.rmse_z, obj_corr=objR.corr,
            policy_acc=polR.acc, policy_N=polR.N,
            deg_all_ninf=degR.all_logw_ninf, deg_nunique=degR.nunique, deg_collapsed=degR.collapsed, ess=degR.ess
        ))

        # IQL-surrogate-mode metrics (policy_acc computed against surrogate actions at surrogate states)
        degI = pf_degeneracy(E.pf_iql, E.π_dist; n_particles=n_particles)
        objI = objective_recon_metrics(E.pf_iql, E.π_dist, E.mdp; gridsize=gridsize)
        polI = policy_match_acc(E.pf_iql, E.π_dist, E.agent_params, E.iql.state_data, E.iql.obs_aidx)

        push!(rows_iql, (
            run_id=pack.run_id, agent=pack.agent, inst=pack.inst,
            num_goals=ann.num_goals, num_obstacles=ann.num_obstacles, max_goal_sep=ann.max_goal_separation,
            obj_rmse_z=objI.rmse_z, obj_corr=objI.corr,
            policy_acc=polI.acc, policy_N=polI.N,
            deg_all_ninf=degI.all_logw_ninf, deg_nunique=degI.nunique, deg_collapsed=degI.collapsed, ess=degI.ess
        ))
    end

    return (real=rows_real, iql=rows_iql)
end

#####################
# One-shot entrypoint
#####################

"Feature vector used to diversify ordering."
pack_feat(p) = Float64[p.ann.num_goals, p.ann.num_obstacles, p.ann.max_goal_separation]

"Greedy farthest-next ordering to maximize diversity between consecutive packs."
function diversify_packs(packs::Vector{RunPack})
    n = length(packs)
    n <= 2 && return packs

    F = [pack_feat(p) for p in packs]

    # Normalize each feature dimension for sane distances
    M = reduce(hcat, F)  # 3×n
    μ = mean(M, dims=2)
    σ = std(M, dims=2)
    σ .= max.(σ, 1e-9)
    Mz = (M .- μ) ./ σ

    # Start at an extreme point (max norm) to reduce dependence on initial ordering
    norms = vec(sum(abs2, Mz; dims=1))
    start = argmax(norms)

    order = Int[start]
    remaining = Set(1:n)
    delete!(remaining, start)

    while !isempty(remaining)
        last = order[end]
        best_i = first(remaining)
        best_d = -Inf
        @inbounds for i in remaining
            d = sum(abs2, Mz[:, i] .- Mz[:, last])  # squared L2
            if d > best_d
                best_d = d
                best_i = i
            end
        end
        push!(order, best_i)
        delete!(remaining, best_i)
    end

    return packs[order]
end

function eval_all(packs::Vector{RunPack}; max_tests::Int=1000,
                  kwargs...)
    packs2 = diversify_packs(packs)
    N = min(length(packs2), max_tests)
    out = Vector{Any}(undef, N)
    for i in 1:N
        out[i] = eval_pack(packs2[i]; kwargs...)
    end
    return out
end

function multi_run_test(bson_path::AbstractString;
                        max_tests::Int=1000,
                        n_particles::Int=50,
                        ess_thresh::Float64=0.7,
                        refine_every::Int=5,
                        refine_topk::Int=5,
                        minN::Int=20,
                        iql_gridN::Int=80,
                        gridsize::Int=120)

    packs = load_runpacks(bson_path)
    evals = eval_all(packs;
                     max_tests=max_tests,
                     n_particles=n_particles,
                     ess_thresh=ess_thresh,
                     refine_every=refine_every,
                     refine_topk=refine_topk,
                     minN=minN,
                     iql_gridN=iql_gridN)

    return summarize_eval(evals; n_particles=n_particles, gridsize=gridsize)
end

############################################
# New testing suite: ablation over objectives
############################################

using Random
using Statistics
using LinearAlgebra: norm
using Plots

############################
# 0) Small util helpers
############################

"""
    safe_get_obstacle_count(mdp_or_pack)

Prefer annotations from RunPack when available; else fall back to mdp.obcs length.
"""
function safe_get_obstacle_count(x)
    if hasproperty(x, :ann)
        return getproperty(x.ann, :num_obstacles)
    end
    if hasproperty(x, :obcs)
        return length(getproperty(x, :obcs))
    end
    return missing
end

"""
    randcat(rng, p)

Sample an index in 1:length(p) with probabilities p (assumed nonnegative, not necessarily normalized).
"""
function randcat(rng::AbstractRNG, p::AbstractVector{<:Real})
    s = 0.0
    @inbounds for i in eachindex(p)
        s += float(p[i])
    end
    u = rand(rng) * s
    c = 0.0
    @inbounds for i in eachindex(p)
        c += float(p[i])
        if u <= c
            return Int(i)
        end
    end
    return Int(lastindex(p))  # numerical fallback
end

############################
# ExperienceBuffer creation (exact signature)
############################

"""
    mk_experience_buffer(data::Dict)

Construct exactly: ExperienceBuffer(data, max_steps, 1, Array{Int64}[], nothing, 0)
where max_steps is the number of columns in data[:s].
"""
function mk_experience_buffer(data::Dict{Symbol, Matrix})
    max_steps = size(data[:s], 2)
    return ExperienceBuffer(data, max_steps, 1, Array{Int64}[], nothing, 0)
end

function alloc_buffer_dict(obs_dims::Int, a_dims::Int, max_steps::Int)
    a_list = Matrix{Bool}(undef, a_dims, max_steps)

    s_list  = zeros(Float64, obs_dims, max_steps)
    sp_list = zeros(Float64, obs_dims, max_steps)

    expert_val_list = ones(Float32, 1, max_steps)
    r_list   = Matrix{Float64}(undef, 1, max_steps)
    t_list   = Matrix{Int64}(undef, 1, max_steps)
    done_list = Matrix{Bool}(undef, 1, max_steps)

    # initialize the ones that must be deterministic
    t_list[1, :] .= collect(Int64, 1:max_steps)
    done_list[1, :] .= false

    return Dict(
        :a => a_list,
        :s => s_list,
        :sp => sp_list,
        :r => r_list,
        :t => t_list,
        :expert_val => expert_val_list,
        :done => done_list,
    )
end

"""
    wrap_like(template_buf, data)

Create a new ExperienceBuffer by cloning `template_buf` and replacing `.data`
(and step counters) so Crux/IQL code accepts it.
"""
function wrap_like(template_buf, data::Dict{Symbol,Any})
    buf = deepcopy(template_buf)
    buf.data = data
    if hasproperty(buf, :elements)
        buf.elements = size(data[:s], 2)
    end
    if hasproperty(buf, :max_steps)
        buf.max_steps = size(data[:s], 2)
    end
    return buf
end

"""
    anonymize_buffer_location!(buf)

Zeroes out the first two rows (location dims) of :s and :sp.
Works in-place on ExperienceBuffer (buf.data is a Dict).
"""
function anonymize_buffer_location!(buf)
    @assert hasproperty(buf, :data) "Expected an ExperienceBuffer-like object with `.data`"
    D = buf.data
    @assert haskey(D, :s) && haskey(D, :sp) "Buffer data missing :s or :sp"

    @assert size(D[:s], 1) ≥ 2 && size(D[:sp], 1) ≥ 2 "State obs dim < 2; cannot anonymize first two rows"

    D[:s][1:2, :] .= 0.0
    D[:sp][1:2, :] .= 0.0
    return buf
end

############################
# Rollout to produce ExperienceBuffer for 20 steps
############################

"""
    qpolicy_action(π, mdp, s; temperature=1.0, rng=...)

Same as before: choose action by Boltzmann over Q-values.
Returns (a::Symbol, aidx::Int, probs::Vector{Float64})
"""
function qpolicy_action(π, mdp::KAgentPOMDP, s::KAgentState;
                        temperature::Real=1.0,
                        rng=Random.default_rng())

    as = actions(mdp)
    all_a_onehot = Flux.onehotbatch(as, as)
    obs = MuKumari.shape_state_as_obs(mdp, s)

    q = vec(Crux.value(π, obs, all_a_onehot))
    qmax = maximum(q)
    logits = (q .- qmax) ./ temperature
    p = exp.(logits)
    p ./= sum(p)

    aidx = randcat(rng, p)
    return as[aidx], aidx, Float64.(p)
end

"""
    rollout_experience_buffer(mdp, π; T=20, temperature=1.0, rng=...)
"""
function rollout_experience_buffer(mdp::KAgentPOMDP, π;
                                   T::Int=20,
                                   temperature::Real=1.0,
                                   rng=Random.default_rng())

    as = actions(mdp)
    na = length(as)

    # Determine obs dimension robustly
    s0 = rand(initialstate(mdp))
    obs0 = MuKumari.shape_state_as_obs(mdp, s0)
    obs_dim = length(obs0)

    data = alloc_buffer_dict(obs_dim, na, T)

    s = s0
    for t in 1:T
        a, aidx, _ = qpolicy_action(π, mdp, s; temperature=temperature, rng=rng)

        # transition
        nt = POMDPs.gen(mdp, s, a, rng)
        sp = nt.sp
        r  = nt.r

        # store onehot action column (Bool matrix)
        data[:a][:, t] .= false
        data[:a][aidx, t] .= true

        # store obs-shaped state, next-state
        data[:s][:, t]  .= Float64.(shape_state_as_obs(mdp, s))
        data[:sp][:, t] .= Float64.(shape_state_as_obs(mdp, sp))

        # store reward (1×T)
        data[:r][1, t] = Float64(r)

        s = sp
    end

    return mk_experience_buffer(data)
end

"""
    build_shared_menv(; M=3)
"""
function build_shared_menv(; M::Int=3)
    μfs = [
        (:sin, x->sin(x[1]) + cos(x[2])),
        (:exp, x->100*exp(-norm(x.-[8 8.])^2 / 1.0)),
        (:lin, x->x[1]^2 + x[2])
    ]
    μs = Symbol[μfs[i][1] for i in 1:M]
    return MuEnv(M, μs, Dict(μfs))
end

############################
# 1) Select 25 skeleton MDPs
############################

"""
    select_skeleton_mdps(bson_path; nbins=5, per_bin=5, rng=Random.default_rng())

Loads RunPacks, counts them, bins by obstacle count (least→most), and selects `per_bin` packs per bin.
Returns:
- packs_all
- chosen_packs (length nbins*per_bin)
- bin_info (NamedTuple with boundaries and counts)
"""
function select_skeleton_mdps(bson_path::AbstractString;
                              nbins::Int=5,
                              per_bin::Int=5,
                              rng=Random.default_rng())

    packs_all = load_runpacks(bson_path)
    N_total = length(packs_all)

    # Sort by obstacle count ascending
    obs = [p.ann.num_obstacles for p in packs_all]
    order = sortperm(obs)
    packs_sorted = packs_all[order]
    obs_sorted = obs[order]

    # Split into nbins contiguous bins (equal size as possible)
    bins = Vector{Vector{RunPack}}(undef, nbins)
    idxs = collect(1:N_total)
    # chunk boundaries
    for b in 1:nbins
        lo = floor(Int, (b-1)*N_total/nbins) + 1
        hi = floor(Int, b*N_total/nbins)
        bins[b] = packs_sorted[lo:hi]
    end

    chosen = RunPack[]
    boundaries = NamedTuple[]

    for (b, binpacks) in enumerate(bins)
        binN = length(binpacks)
        if binN == 0
            push!(boundaries, (bin=b, min_obstacles=missing, max_obstacles=missing, count=0))
            continue
        end
        mino = minimum(p.ann.num_obstacles for p in binpacks)
        maxo = maximum(p.ann.num_obstacles for p in binpacks)

        push!(boundaries, (bin=b, min_obstacles=mino, max_obstacles=maxo, count=binN))

        k = min(per_bin, binN)
        picks = randperm(rng, binN)[1:k]
        append!(chosen, binpacks[picks])
    end

    return packs_all, chosen, (total=N_total, nbins=nbins, per_bin=per_bin, boundaries=boundaries)
end

############################
# 2) Generate 30 ablation objectives
############################

"""
Internal: sample discrete Fourier indices with an override for K and with controllable supports.
Returns (key, ff_namedtuple, sweep_tag, sweep_level, cfg_used)
"""
function sample_fourier_key(cfg::FourierDiscreteCfg;
                            K_override::Union{Nothing,Int}=nothing,
                            rng=Random.default_rng())

    # Supports/probs
    Kp = K_probs(cfg)
    freq_supp, freq_w = freq_bin_support_and_probs(cfg)
    amp_supp, amp_w   = amp_bin_support_and_probs(cfg)

    K = isnothing(K_override) ? rand(rng, Categorical(Kp)) : K_override
    K = clamp(K, 1, cfg.Kmax)

    fx_idx = Vector{Int}(undef, K)
    fy_idx = Vector{Int}(undef, K)
    A_idx  = Vector{Int}(undef, K)
    ϕ_idx  = Vector{Int}(undef, K)

    for m in 1:K
        fx_idx[m] = freq_supp[randcat(rng, freq_w)]
        fy_idx[m] = freq_supp[randcat(rng, freq_w)]
        A_idx[m]  = amp_supp[randcat(rng, amp_w)]
        ϕ_idx[m]  = rand(rng, 0:cfg.P-1)
    end

    key = (K, fx_idx, fy_idx, A_idx, ϕ_idx)
    return key
end

"""
    build_ablation_objectives(; rng=..., base_cfg=FourierDiscreteCfg(), levels=10)

Creates 30 objectives total:
- sweep=:K (10 objs): K in [1..10], with narrow freq/amp ranges
- sweep=:freq_range (10 objs): Fmax_i increases, K fixed at 2
- sweep=:amp_range (10 objs): Amax_i increases, K fixed at 2

Returns Vector of NamedTuples with fields:
(id, sweep, level, cfg, key, field, obj)
"""
function build_ablation_objectives(; rng=Random.default_rng(),
                                   base_cfg::FourierDiscreteCfg=FourierDiscreteCfg(),
                                   levels::Int=10)

    out = NamedTuple[]

    # Sweep A: number of features K (keep freq/amp “similar”: small ranges)
    # Choose narrow supports by using small Fmax_i and small Amax_i.
    cfgK = FourierDiscreteCfg(; Kmax=10,
                             λK=base_cfg.λK,
                             Δf=base_cfg.Δf, Fmax_i=3, freq_mag_decay=0.0,
                             ΔA=base_cfg.ΔA, Amax_i=1,
                             P=base_cfg.P)

    for i in 1:levels
        K = i  # 1..10
        key = sample_fourier_key(cfgK; K_override=K, rng=rng)
        # decode indices -> actual values (fx, fy, A, ϕ)
        ff = decode_fourier_key(key, cfgK)
        field = make_fourier_scalar_field(ff; scaleQ=true)
        obj   = make_pomdp_objective_from_field(field)
        push!(out, (id=length(out)+1, sweep=:K, level=K, cfg=cfgK, key=key, field=field, obj=obj))
    end

    # Sweep B: frequency range (keep K=2, amplitude range fixed)
    # “Similar freq values” -> small Fmax_i; “very different” -> large Fmax_i.
    cfgF_base = FourierDiscreteCfg(; Kmax=10,
                                  λK=base_cfg.λK,
                                  Δf=base_cfg.Δf,
                                  Fmax_i=3, freq_mag_decay=0.0,
                                  ΔA=base_cfg.ΔA, Amax_i=base_cfg.Amax_i,  # keep amplitude range fixed
                                  P=base_cfg.P)

    F_levels = round.(Int, range(2, 30; length=levels))  # monotone increase
    for Fmax in F_levels
        cfgF = FourierDiscreteCfg(; Kmax=cfgF_base.Kmax, λK=cfgF_base.λK,
                                 Δf=cfgF_base.Δf, Fmax_i=Fmax, freq_mag_decay=cfgF_base.freq_mag_decay,
                                 ΔA=cfgF_base.ΔA, Amax_i=cfgF_base.Amax_i,
                                 P=cfgF_base.P)
        key = sample_fourier_key(cfgF; K_override=2, rng=rng)
        ff = decode_fourier_key(key, cfgF)
        field = make_fourier_scalar_field(ff; scaleQ=true)
        obj   = make_pomdp_objective_from_field(field)
        push!(out, (id=length(out)+1, sweep=:freq_range, level=Fmax, cfg=cfgF, key=key, field=field, obj=obj))
    end

    # Sweep C: amplitude range (keep K=2, frequency range fixed)
    cfgA_base = FourierDiscreteCfg(; Kmax=10,
                                  λK=base_cfg.λK,
                                  Δf=base_cfg.Δf, Fmax_i=base_cfg.Fmax_i, freq_mag_decay=base_cfg.freq_mag_decay,
                                  ΔA=base_cfg.ΔA, Amax_i=3,
                                  P=base_cfg.P)

    A_levels = round.(Int, range(2, 50; length=levels))
    for Amax in A_levels
        cfgA = FourierDiscreteCfg(; Kmax=cfgA_base.Kmax, λK=cfgA_base.λK,
                                 Δf=cfgA_base.Δf, Fmax_i=cfgA_base.Fmax_i, freq_mag_decay=cfgA_base.freq_mag_decay,
                                 ΔA=cfgA_base.ΔA, Amax_i=Amax,
                                 P=cfgA_base.P)
        key = sample_fourier_key(cfgA; K_override=2, rng=rng)
        ff = decode_fourier_key(key, cfgA)
        field = make_fourier_scalar_field(ff; scaleQ=true)
        obj   = make_pomdp_objective_from_field(field)
        push!(out, (id=length(out)+1, sweep=:amp_range, level=Amax, cfg=cfgA, key=key, field=field, obj=obj))
    end

    return out
end

############################
# 3) Synthesize 30 MDPs from skeletons + shared MuEnv + empty goals
############################

"""
    synthesize_ablation_mdps(skeleton_packs, objectives; shared_menv=build_shared_menv(), rng=...)

For each objective:
- sample one skeleton pack at random from the 25
- extract agent_params_from_mdp(skeleton.mdp)
- override :menv and :goals
- build_kagent_pomdp(agent_params, obj)

Returns Vector of NamedTuples:
(id, sweep, level, mdp, agent_params, skeleton_ref, objrec)
"""
function synthesize_ablation_mdps(skeleton_packs::Vector{RunPack},
                                  objectives::Vector{<:NamedTuple};
                                  shared_menv=build_shared_menv(),
                                  rng=Random.default_rng())

    out = NamedTuple[]
    for objrec in objectives
        sk = rand(rng, skeleton_packs)
        agent_params = agent_params_from_mdp(sk.mdp)

        agent_params[:menv]  = shared_menv
        agent_params[:goals] = Any[]

        mdp_new = build_kagent_pomdp(agent_params, objrec.obj; name="abl_$(objrec.id)")

        push!(out, (id=objrec.id,
                    sweep=objrec.sweep,
                    level=objrec.level,
                    mdp=mdp_new,
                    agent_params=agent_params,
                    skeleton_ref=(run_id=sk.run_id, agent=sk.agent, inst=sk.inst, num_obstacles=sk.ann.num_obstacles),
                    objrec=objrec))
    end
    return out
end

############################
# 4) Train SoftQ
############################

"""
    softq_policy(mdp; N=2000, epochs=2, batch_size=256)

Trains SoftQ via deep_q_solver and returns (solver, policy).
"""
function softq_policy(mdp::KAgentPOMDP; N::Int=2000, epochs::Int=2, batch_size::Int=256)
    𝒮 = deep_q_solver(mdp; solver_params=[:softq, N, epochs, batch_size])
    π = solve(𝒮, mdp)
    return 𝒮, π
end

###########################
# 4.5) Cache-ing!
###########################

@with_kw struct MuEnvSpec
    variant::Symbol = :default_shared
    M::Int = 3
    μ_order::Vector{Symbol} = [:sin, :exp, :lin]
end

function build_shared_menv(spec::MuEnvSpec)
    μfs = [
        (:sin, x->sin(x[1]) + cos(x[2])),
        (:exp, x->100*exp(-norm(x.-[8 8.])^2 / 1.0)),
        (:lin, x->x[1]^2 + x[2])
    ]
    μdict = Dict(μfs)
    return MuEnv(spec.M, spec.μ_order, μdict)
end

"""
BSON payload structure:
cache = Dict(
  :meta => ...,
  :muenv_spec => MuEnvSpec(...),
  :records => Vector{Dict} with per-objective:
      id, sweep, level,
      cfg (FourierDiscreteCfg serialized ok),
      key (Tuple K, fx_i, fy_i, A_i, ϕ_i),
      agent_params_core (Dict without :menv / :start_state),
      skeleton_ref,
      full_data (Dict{Symbol,Matrix}),
      anon_data (Dict{Symbol,Matrix})
)
"""

function reconstruct_mdp_from_cache(rec::Dict, muenv_spec::MuEnvSpec)
    cfg = rec[:cfg]
    key = rec[:key]

    bank = decode_fourier_key(key, cfg)                 # returns bank with fx, fy, A, ϕ
    field = make_fourier_scalar_field(bank; scaleQ=true)
    obj = make_pomdp_objective_from_field(field)

    menv = build_shared_menv(muenv_spec)

    agent_params = deepcopy(rec[:agent_params_core])
    agent_params[:menv]  = menv
    agent_params[:goals] = Any[]
    # Start state should be consistent and avoid BSON-loaded mdp.menv:
    x0 = agent_params[:start]
    agent_params[:start_state] = KAgentState(x0, [predict_env(menv, x0)], Matrix[])

    mdp = build_kagent_pomdp(agent_params, obj; name="abl_$(rec[:id])")
    return mdp, agent_params
end

function generate_and_cache_ablation_data(bson_path::String;
                                          cache_path::String,
                                          rng::AbstractRNG,
                                          shared_muenv_spec::MuEnvSpec=MuEnvSpec(),
                                          nbins::Int=5, per_bin::Int=5,
                                          levels::Int=10,
                                          T::Int=20)

    packs_all, skeletons, bin_info = select_skeleton_mdps(bson_path; nbins=nbins, per_bin=per_bin, rng=rng)

    objectives = build_ablation_objectives(; rng=rng, levels=levels)

    # IMPORTANT: do NOT call agent_params_from_mdp in a way that touches BSON-loaded mdp.menv
    mdprecs = synthesize_ablation_mdps(skeletons, objectives;
                                       shared_menv=build_shared_menv(shared_muenv_spec),
                                       rng=rng)

    records = Vector{Dict}(undef, length(mdprecs))

    for (i, rec) in enumerate(mdprecs)
        mdp = rec.mdp

        # Train SoftQ for generation
        _, π_softq = softq_policy(mdp; N=2000, epochs=2, batch_size=256)

        temperature = get(rec.agent_params, :policy_temperature, 2.0)
        full_buf = rollout_experience_buffer(mdp, π_softq; T=T, temperature=temperature, rng=rng)

        # Create anon_buf by copying data with Dict{Symbol,Matrix} typing
        full_data = full_buf.data
        anon_data = Dict{Symbol, Matrix}(k => copy(v) for (k,v) in full_data)
        # zero out first two rows of :s and :sp
        anon_data[:s][1:2, :] .= 0.0
        anon_data[:sp][1:2, :] .= 0.0

        # Store a BSON-safe “core” agent_params (NO :menv and NO :start_state)
        ap = deepcopy(rec.agent_params)
        pop!(ap, :menv, nothing)
        pop!(ap, :start_state, nothing)

        records[i] = Dict(
            :id => rec.id,
            :sweep => rec.sweep,
            :level => rec.level,
            :cfg => rec.objrec.cfg,
            :key => rec.objrec.key,
            :agent_params_core => ap,
            :skeleton_ref => rec.skeleton_ref,
            :full_data => Dict{Symbol, Matrix}(k => copy(v) for (k,v) in full_data),
            :anon_data => anon_data,
        )
    end

    cache = Dict(
        :meta => Dict(
            :source_bson => bson_path,
            :nbins => nbins, :per_bin => per_bin,
            :n_skeletons => length(skeletons),
            :n_objectives => length(objectives),
            :T => T,
            :bin_info => bin_info,
        ),
        :muenv_spec => shared_muenv_spec,
        :records => records,
    )

    BSON.@save cache_path cache
    return cache
end

function load_ablation_cache(cache_path::String)
    d = BSON.load(cache_path)
    @assert haskey(d, :cache) "Expected BSON to contain key :cache"
    return d[:cache]
end

############################
# 5) Run Mode A vs Mode B + metrics, per ablation MDP
############################

"""
    eval_ablation_mdp(rec; n_particles=..., iql_gridN=..., minN=20, ...)

rec is one element from synthesize_ablation_mdps output.
Returns NamedTuple with all metrics for modeA and modeB plus identifiers.
"""
function eval_ablation_mdp(rec; n_particles::Int=50, ess_thresh::Float64=0.7, refine_every::Int=5, refine_topk::Int=5,
                           iql_gridN::Int=100, minN::Int=20, gridsize::Int=120, rng=Random.default_rng())

    mdp = rec.mdp

    # 1) Train SoftQ for data generation (Mode A “real” dataset)
    _, π_softq = softq_policy(mdp; N=2000, epochs=2, batch_size=256)

    # 2) Generate experience (full + anon identical here)
    temperature = get(rec.agent_params, :policy_temperature, 2.0)
    full_buf = rollout_experience_buffer(mdp, π_softq; T=minN, temperature=temperature, rng=rng)

    anon_data = Dict{Symbol,Matrix}(k => copy(v) for (k,v) in full_buf.data)
    anon_buf  = ExperienceBuffer(anon_data, size(anon_data[:s], 2), 1, Array{Int64}[], nothing, 0)
    anonymize_buffer_location!(anon_buf)

    # 3) Train IQL (Mode B surrogate driver)
    π_iql, 𝒟_iql, _ = quick_IQL(mdp, anon_buf)  # uses helper

    # 4) Build π_dist with action mappings
    as = actions(mdp)
    action_list = [as, a->Flux.onehot(a, as), Flux.onehotbatch(as, as)]
    π_dist = ScoreΠDist(; mdp_params=action_list)

    # 5) Mode A PF
    T = size(full_buf.data[:s], 2)
    data_slices = (T ≤ minN) ? collect(1:T) : [rand(rng, 1:T) for _ in 1:minN]
    state_dataA = full_buf.data[:s][:, data_slices]
    obs_aidxA   = onehot_cols_to_aidx(full_buf.data[:a][:, data_slices])

    pfA = particle_filter(obs_aidxA, π_dist, rec.agent_params, state_dataA, n_particles;
                          ess_thresh=ess_thresh, refine_every=refine_every, refine_topk=refine_topk)

    # 6) Mode B PF (IQL grid surrogate)
    iql_state_data, iql_obs_aidx, _ = surrogate_dataset_from_iql_grid(π_dist, π_iql, mdp; eval_num=iql_gridN)

    pfB = particle_filter(iql_obs_aidx, π_dist, rec.agent_params, iql_state_data, n_particles;
                          ess_thresh=ess_thresh, refine_every=refine_every, refine_topk=refine_topk)

    # 7) Metrics for both modes
    degA = pf_degeneracy(pfA, π_dist; n_particles=n_particles)
    objA = objective_recon_metrics(pfA, π_dist, mdp; gridsize=gridsize)
    polA = policy_match_acc(pfA, π_dist, rec.agent_params, state_dataA, obs_aidxA)

    degB = pf_degeneracy(pfB, π_dist; n_particles=n_particles)
    objB = objective_recon_metrics(pfB, π_dist, mdp; gridsize=gridsize)
    polB = policy_match_acc(pfB, π_dist, rec.agent_params, iql_state_data, iql_obs_aidx)

    return (
        id=rec.id, sweep=rec.sweep, level=rec.level,
        skeleton_ref=rec.skeleton_ref,
        # Mode A:
        A=(deg=degA, obj=objA, pol=polA),
        # Mode B:
        B=(deg=degB, obj=objB, pol=polB),
    )
end


"""
Run PF + metrics only, using cached buffers.
This reruns quick_IQL (Mode B) from anon_data, but avoids regenerating the trajectories.
"""
function eval_ablation_from_cache(cache::Dict;
                                  n_particles::Int=50,
                                  ess_thresh::Float64=0.7,
                                  refine_every::Int=5,
                                  refine_topk::Int=5,
                                  iql_gridN::Int=120,
                                  gridsize::Int=120,
                                  ess_min_frac::Float64=0.25,   # NEW
                                  rng::AbstractRNG=Random.default_rng())

    muenv_spec = cache[:muenv_spec]
    records = cache[:records]

    evals = Vector{Any}(undef, length(records))
    ess_min = ess_min_frac * n_particles

    for (i, rec) in enumerate(records)
        mdp, agent_params = reconstruct_mdp_from_cache(rec, muenv_spec)

        full_data = Dict{Symbol, Matrix}(rec[:full_data])
        anon_data = Dict{Symbol, Matrix}(rec[:anon_data])

        full_buf = ExperienceBuffer(full_data, size(full_data[:s],2), 1, Array{Int64}[], nothing, 0)
        anon_buf = ExperienceBuffer(anon_data, size(anon_data[:s],2), 1, Array{Int64}[], nothing, 0)

        π_iql, 𝒮_iql, _ = quick_IQL(mdp, anon_buf)

        as = actions(mdp)
        action_list = [as, a->Flux.onehot(a, as), Flux.onehotbatch(as, as)]
        π_dist = ScoreΠDist(; mdp_params=action_list)

        # Mode A PF inputs
        state_dataA = full_buf.data[:s]
        obs_aidxA   = onehot_cols_to_aidx(full_buf.data[:a])
        lobs = Int64(length(obs_aidxA) * 0.1)

        pfA = particle_filter(obs_aidxA[1:lobs], π_dist, agent_params, state_dataA[:, 1:lobs], n_particles;
                              ess_thresh=ess_thresh, refine_every=refine_every, refine_topk=refine_topk)

        # Mode B PF inputs: TODO!!!
        # iql_state_data, iql_obs_aidx, _ = surrogate_dataset_from_iql_grid(π_dist, π_iql, mdp; eval_num=iql_gridN)

        pfB = particle_filter(obs_aidxA, π_dist, agent_params, state_dataA, n_particles*3;
                              ess_thresh=ess_thresh, refine_every=refine_every, refine_topk=refine_topk)

        # Degeneracy first
        degA = pf_degeneracy(pfA, π_dist; n_particles=n_particles)
        degB = pf_degeneracy(pfB, π_dist; n_particles=n_particles)

        badA = degA.collapsed || (degA.ess < ess_min)
        badB = degB.collapsed || (degB.ess < ess_min)

        # Only compute other metrics if not degenerate; else NaN them
        objA = badA ? (rmse_z=NaN, corr=NaN) : objective_recon_metrics(pfA, π_dist, mdp; gridsize=gridsize)
        polA = badA ? (acc=NaN,)            : policy_match_acc(pfA, π_dist, agent_params, state_dataA, obs_aidxA)

        objB = badB ? (rmse_z=NaN, corr=NaN) : objective_recon_metrics(pfB, π_dist, mdp; gridsize=gridsize)
        polB = badB ? (acc=NaN,)             : policy_match_acc(pfB, π_dist, agent_params, state_dataA, obs_aidxA)

        keyA, probA = badA ? (nothing, NaN) : top_key(pfA, π_dist)
        keyB, probB = badB ? (nothing, NaN) : top_key(pfB, π_dist)


        evals[i] = (
            id=rec[:id], sweep=rec[:sweep], level=rec[:level],
            skeleton_ref=rec[:skeleton_ref],
            A=(deg=degA, bad=badA, obj=objA, pol=polA, top_key=keyA, top_prob=probA),
            B=(deg=degB, bad=badB, obj=objB, pol=polB, top_key=keyB, top_prob=probB),
        )
    end

    return evals
end

############################
# 6) Run full ablation + aggregate + plots
############################

"""
    run_ablation_suite(bson_path; ...)

End-to-end:
1) select 25 skeletons from bins
2) build 30 objectives
3) synthesize 30 MDPs
4) eval each (Mode A vs Mode B metrics)
Returns:
- meta info
- eval records (vector)
- grouped summaries
"""
function run_ablation_suite(bson_path::AbstractString;
                            nbins::Int=5,
                            per_bin::Int=5,
                            rng=Random.default_rng(),
                            shared_menv=build_shared_menv(),
                            n_particles::Int=50,
                            ess_thresh::Float64=0.7,
                            refine_every::Int=5,
                            refine_topk::Int=5,
                            iql_gridN::Int=120,
                            minN::Int=20,
                            gridsize::Int=120)

    packs_all, skeletons, bin_info = select_skeleton_mdps(bson_path; nbins=nbins, per_bin=per_bin, rng=rng)

    objectives = build_ablation_objectives(; rng=rng, levels=10)
    mdprecs = synthesize_ablation_mdps(skeletons, objectives; shared_menv=shared_menv, rng=rng)

    evals = Vector{Any}(undef, length(mdprecs))
    for (i, rec) in enumerate(mdprecs)
        evals[i] = eval_ablation_mdp(rec;
                                     n_particles=n_particles,
                                     ess_thresh=ess_thresh,
                                     refine_every=refine_every,
                                     refine_topk=refine_topk,
                                     iql_gridN=iql_gridN,
                                     minN=minN,
                                     gridsize=gridsize,
                                     rng=rng)
    end

    return (meta=(bin_info=bin_info,
                  n_skeletons=length(skeletons),
                  n_objectives=length(objectives),
                  n_mdps=length(mdprecs)),
            evals=evals)
end

"""
    summarize_ablation(evals)

Produces per-sweep, per-level arrays for each metric comparing Mode A vs Mode B.
Returns a Dict keyed by sweep => summary NamedTuple.
"""
# function summarize_ablation(evals)
#     sweeps = unique(e.sweep for e in evals)
#     out = Dict{Symbol,Any}()

#     for sw in sweeps
#         Es = filter(e->e.sweep==sw, evals)
#         levels = sort(unique(e.level for e in Es))

#         # helper to mean over replicates at same level (here usually 1 per level)
#         function agg(f)
#             [mean([f(e) for e in Es if e.level==lv]) for lv in levels]
#         end

#         # Degeneracy: use ESS + collapse flags
#         essA = agg(e->e.A.deg.ess)
#         essB = agg(e->e.B.deg.ess)
#         colA = agg(e->e.A.deg.collapsed ? 1.0 : 0.0)
#         colB = agg(e->e.B.deg.collapsed ? 1.0 : 0.0)

#         # Objective recon:
#         rmseA = agg(e->e.A.obj.rmse_z)
#         rmseB = agg(e->e.B.obj.rmse_z)
#         corA  = agg(e->e.A.obj.corr)
#         corB  = agg(e->e.B.obj.corr)

#         # Policy match:
#         accA  = agg(e->e.A.pol.acc)
#         accB  = agg(e->e.B.pol.acc)

#         out[sw] = (levels=levels,
#                    essA=essA, essB=essB,
#                    collapsedA=colA, collapsedB=colB,
#                    rmseA=rmseA, rmseB=rmseB,
#                    corrA=corA, corrB=corB,
#                    accA=accA, accB=accB)
#     end

#     return out
# end

nanmean(v) = isempty(v) ? NaN : mean(v)

function summarize_ablation(evals)
    sweeps = unique(e.sweep for e in evals)
    out = Dict{Symbol,Any}()

    for sw in sweeps
        Es = filter(e->e.sweep==sw, evals)
        levels = sort(unique(e.level for e in Es))

        # NaN-safe aggregation over replicates at each level
        function agg(f)
            [begin
                vals = [f(e) for e in Es if e.level==lv]
                vals = filter(x -> !(ismissing(x) || (x isa Real && isnan(x))), vals)
                nanmean(vals)
             end for lv in levels]
        end

        # Degeneracy
        essA = agg(e->e.A.deg.ess)
        essB = agg(e->e.B.deg.ess)
        colA = agg(e->e.A.deg.collapsed ? 1.0 : 0.0)
        colB = agg(e->e.B.deg.collapsed ? 1.0 : 0.0)

        badA = agg(e->e.A.bad ? 1.0 : 0.0)
        badB = agg(e->e.B.bad ? 1.0 : 0.0)

        # Objective recon
        rmseA = agg(e->e.A.obj.rmse_z)
        rmseB = agg(e->e.B.obj.rmse_z)
        corA  = agg(e->e.A.obj.corr)
        corB  = agg(e->e.B.obj.corr)

        # Policy match
        accA  = agg(e->e.A.pol.acc)
        accB  = agg(e->e.B.pol.acc)

        out[sw] = (levels=levels,
                   essA=essA, essB=essB,
                   collapsedA=colA, collapsedB=colB,
                   badA=badA, badB=badB,
                   rmseA=rmseA, rmseB=rmseB,
                   corrA=corA, corrB=corB,
                   accA=accA, accB=accB)
    end

    return out
end

"""
    plot_ablation_summaries(sumdict)

Creates bar plots per sweep comparing Mode A vs Mode B for:
- ESS (degeneracy)
- RMSE_z and Corr (objective recon)
- policy accuracy (policy match)

Returns Dict sweep => Dict(metric_name => plot)
"""
# function plot_ablation_summaries(sumdict::Dict{Symbol,Any})
#     plots = Dict{Symbol,Any}()
#     labels = [MODE_LABELS[:A] MODE_LABELS[:B]]

#     for (sw, S) in sumdict
#         lv = S.levels

#         # 1) degeneracy: ESS
#         p_ess = bar(string.(lv), [S.essA S.essB],
#                     label=["Mode A" "Mode B"],
#                     title="Sweep $(sw): ESS",
#                     xlabel="sweep level", ylabel="ESS")

#         # 2) objective recon: RMSE_z
#         p_rmse = bar(string.(lv), [S.rmseA S.rmseB],
#                      label=["Mode A" "Mode B"],
#                      title="Sweep $(sw): objective RMSE_z",
#                      xlabel="sweep level", ylabel="RMSE_z")

#         # 3) objective recon: Corr
#         p_corr = bar(string.(lv), [S.corrA S.corrB],
#                      label=["Mode A" "Mode B"],
#                      title="Sweep $(sw): objective corr",
#                      xlabel="sweep level", ylabel="corr")

#         # 4) policy match: accuracy
#         p_acc = bar(string.(lv), [S.accA S.accB],
#                     label=["Mode A" "Mode B"],
#                     title="Sweep $(sw): policy match acc",
#                     xlabel="sweep level", ylabel="accuracy")

#         plots[sw] = Dict(:ess=>p_ess, :rmse=>p_rmse, :corr=>p_corr, :acc=>p_acc)
#     end

#     return plots
# end

const METHOD_LABELS = ["Open-Ended SIPS", "IQ-SIPS"]

degmask_from_summary(metricA::Vector, metricB::Vector, colA::Vector, colB::Vector) = (
    ((colA .>= 0.5) .| isnan.(Float64.(metricA))),   # degenerate A
    ((colB .>= 0.5) .| isnan.(Float64.(metricB)))    # degenerate B
)
replace_nan_with_zero(v::Vector) = [isnan(Float64(x)) ? 0.0 : Float64(x) for x in v]

# Pick a small nonzero height that scales with the plot.
function default_deg_height(yA_plot::Vector{<:Real}, yB_plot::Vector{<:Real}; ylims=nothing)
    # Prefer ylims if provided (best for ACC/ESS)
    if ylims !== nothing
        ymin, ymax = ylims
        yr = max(ymax - ymin, eps(Float64))
        return 0.03 * yr
    end

    # Otherwise infer from data scale (RMSE often)
    ys = vcat(yA_plot, yB_plot)
    ymax = maximum(ys)
    if !isfinite(ymax) || ymax ≤ 0
        return 0.05
    end
    return max(0.03 * ymax, 1e-6)
end

# Draw diagonal hatch lines over a rectangular bar region.
# This works on any Plots backend.
function hatch_rect!(p, x_left::Real, x_right::Real, y0::Real, y1::Real;
                     spacing_frac::Real=0.18, linecolor=:black, linewidth::Real=1.5, direction::Symbol=:/)
    w = x_right - x_left
    h = y1 - y0
    if w ≤ 0 || h ≤ 0
        return p
    end

    spacing = spacing_frac * w
    # We draw a family of parallel lines that intersect the rectangle.
    # direction = :/ means rising left->right, :\ means falling left->right.
    if direction == :/
        # Lines: y = (h/w)*(x - c) + y0; sweep c
        cmin = x_left - h * (w/h)  # safe over-sweep
        cmax = x_right
        cs = collect(cmin:spacing:cmax)
        for c in cs
            # segment endpoints clipped to rectangle
            # compute intersection with bottom/top edges
            x0 = c
            y_at_xleft  = y0 + (h/w) * (x_left - c)
            y_at_xright = y0 + (h/w) * (x_right - c)

            # candidate points on left/right edges
            pts = Tuple{Float64,Float64}[]
            if y0 ≤ y_at_xleft ≤ y1
                push!(pts, (x_left, y_at_xleft))
            end
            if y0 ≤ y_at_xright ≤ y1
                push!(pts, (x_right, y_at_xright))
            end
            # intersections with bottom/top edges
            x_at_y0 = c
            x_at_y1 = c + (w/h)*h  # c + w
            # Actually for this parameterization, easier: solve for x given y:
            # y = y0 + (h/w)(x - c) => x = c + (w/h)(y - y0)
            x_bot = c + (w/h)*(0.0)
            x_top = c + (w/h)*(h)
            if x_left ≤ x_bot ≤ x_right
                push!(pts, (x_bot, y0))
            end
            if x_left ≤ x_top ≤ x_right
                push!(pts, (x_top, y1))
            end

            if length(pts) ≥ 2
                # pick two farthest points (simple: first two after unique)
                (xA,yA),(xB,yB) = pts[1], pts[2]
                plot!(p, [xA,xB], [yA,yB]; color=linecolor, linewidth=linewidth, label=nothing)
            end
        end
    else
        # direction == :\ : mirror by swapping left/right in the slope sign
        # Use same approach but slope negative.
        spacing = spacing_frac * w
        cs = collect((x_left):spacing:(x_right + h*(w/h)))
        for c in cs
            # line: y = y0 + (h/w)*(c - x)
            y_at_xleft  = y0 + (h/w) * (c - x_left)
            y_at_xright = y0 + (h/w) * (c - x_right)

            pts = Tuple{Float64,Float64}[]
            if y0 ≤ y_at_xleft ≤ y1
                push!(pts, (x_left, y_at_xleft))
            end
            if y0 ≤ y_at_xright ≤ y1
                push!(pts, (x_right, y_at_xright))
            end

            # Solve for x on bottom/top: y = y0 + (h/w)*(c - x) => x = c - (w/h)(y - y0)
            x_bot = c - (w/h)*(0.0)
            x_top = c - (w/h)*(h)
            if x_left ≤ x_bot ≤ x_right
                push!(pts, (x_bot, y0))
            end
            if x_left ≤ x_top ≤ x_right
                push!(pts, (x_top, y1))
            end

            if length(pts) ≥ 2
                (xA,yA),(xB,yB) = pts[1], pts[2]
                plot!(p, [xA,xB], [yA,yB]; color=linecolor, linewidth=linewidth, label=nothing)
            end
        end
    end

    return p
end

# Convert sweep levels (stored as bin max indices) to interpretable labels in physical units.
# Uses the cfg stored in cache per-record (best, because it reflects the actual sweep).
function sweep_tick_labels_from_cache(cache::Dict, sw::Symbol, levels::Vector{Int})
    recs = cache[:records]

    # Helper: find cfg for a (sweep, level)
    function cfg_for(sw, lv)
        for r in recs
            if r[:sweep] == sw && r[:level] == lv
                return r[:cfg]
            end
        end
        error("No cache record found for sweep=$(sw), level=$(lv)")
    end

    labels = String[]
    for lv in levels
        cfg = cfg_for(sw, lv)
        if sw == :K
            push!(labels, string(lv))  # K itself is meaningful
        elseif sw == :freq_range
            # lv is Fmax_i; Δf is physical step
            halfspan = lv * cfg.Δf
            width = 2 * halfspan
            push!(labels, @sprintf("%.2f", width))  # show width, not index
            # alternatively: push!(labels, "±$(round(halfspan,digits=2))")
        elseif sw == :amp_range
            # lv is Amax_i; ΔA is physical step
            amax = lv * cfg.ΔA
            push!(labels, @sprintf("%.2f", amax))
        else
            push!(labels, string(lv))
        end
    end
    return labels
end

function pretty_xlabel(sw::Symbol)
    sw == :K && return "K (number of Fourier features)"
    sw == :freq_range && return "Frequency range width, 2Fₘₐₓ (units)"
    sw == :amp_range && return "Amplitude maximum, Aₘₐₓ (units)"
    return "Sweep level"
end

function pretty_title(sw::Symbol, metric::Symbol)
    sweep_name = sw == :K ? "K Sweep" :
                 sw == :freq_range ? "Frequency Range Sweep" :
                 sw == :amp_range ? "Amplitude Range Sweep" : string(sw)
    metric_name = metric == :ess ? "Effective Sample Size (ESS)" :
                  metric == :rmse ? "Objective Reconstruction Error (RMSE)" :
                  metric == :acc ? "Policy Match Accuracy" : string(metric)
    return "$(sweep_name): $(metric_name)"
end

function pretty_ylabel(metric::Symbol)
    metric == :ess && return "ESS (particles)"
    metric == :rmse && return "RMSE (objective value)"
    metric == :acc && return "Accuracy (fraction)"
    return string(metric)
end

# Core grouped-bar helper (this is the key fix).
# Use numeric x positions + dodge + explicit xticks.
# function grouped_bars(level_labels::Vector{String}, yA::Vector, yB::Vector;
#                       title::String, xlabel::String, ylabel::String,
#                       ylims=nothing)

#     n = length(level_labels)
#     @assert length(yA) == n && length(yB) == n

#     x = 1:n
#     Y = hcat(yA, yB)  # N×2 -> two series at each x (grouped)

#     p = bar(x, Y;
#         bar_position=:dodge,
#         legend=:topright,
#         label=METHOD_LABELS,
#         xticks=(x, level_labels),
#         xrotation=25,
#         title=title,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         size=(950, 560),
#         dpi=220,
#         framestyle=:box,
#         gridalpha=0.15,
#         left_margin=12mm, right_margin=6mm,
#         top_margin=10mm, bottom_margin=12mm
#     )

#     if ylims !== nothing
#         ylims!(p, ylims)
#     end

#     return p
# end

function grouped_bars(level_labels::Vector{String}, yA::Vector, yB::Vector;
                      title::String, xlabel::String, ylabel::String,
                      ylims=nothing)

    n = length(level_labels)
    @assert length(yA) == n && length(yB) == n

    # IMPORTANT: numeric x; labels supplied via xticks
    x = 1:n

    # Y must be n×2 where each column is a method (A, B)
    Y = hcat(yA, yB)

    p = groupedbar(
        x, Y;
        bar_position = :dodge,      # side-by-side
        label = METHOD_LABELS,
        xticks = (x, level_labels),
        xrotation = 25,
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        size = (950, 560),
        dpi = 220,
        framestyle = :box,
        gridalpha = 0.15,
        legend = :topright,
        left_margin = 12mm, right_margin = 6mm,
        top_margin = 10mm, bottom_margin = 12mm
    )

    if ylims !== nothing
        ylims!(p, ylims)
    end

    return p
end

function grouped_bars_with_degenerate_overlay(
    level_labels::Vector{String},
    yA::Vector, yB::Vector,
    degA::AbstractVector{Bool}, degB::AbstractVector{Bool};
    title::String, xlabel::String, ylabel::String,
    ylims=nothing,
    deg_height::Union{Nothing,Float64}=nothing
)
    n = length(level_labels)
    @assert length(yA)==n && length(yB)==n
    @assert length(degA)==n && length(degB)==n

    x = 1:n
    yA_plot = replace_nan_with_zero(yA)
    yB_plot = replace_nan_with_zero(yB)
    Y = hcat(yA_plot, yB_plot)

    p = groupedbar(
        x, Y;
        bar_position=:dodge,
        label=METHOD_LABELS,
        xticks=(x, level_labels),
        xrotation=25,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        size=(950, 560),
        dpi=220,
        framestyle=:box,
        gridalpha=0.15,
        legend=:topright,
        left_margin=12mm, right_margin=6mm,
        top_margin=10mm, bottom_margin=12mm
    )

    if ylims !== nothing
        ylims!(p, ylims)
    end

    # Choose a small visible marker height
    h = deg_height === nothing ? default_deg_height(yA_plot, yB_plot; ylims=ylims) : deg_height

    # Approximate dodge geometry for 2-series groupedbar:
    dx = 0.18
    bw = 0.32

    # Draw small bars + hatch lines
    for i in 1:n
        if degA[i]
            xc = x[i] - dx
            # draw outline bar
            bar!(p, [xc], [h]; bar_width=bw, fillalpha=0.0, linecolor=:black, linewidth=2, label=nothing)
            # hatch over the rectangle
            hatch_rect!(p, xc - bw/2, xc + bw/2, 0.0, h; direction=:/, linewidth=1.2)
        end
        if degB[i]
            xc = x[i] + dx
            bar!(p, [xc], [h]; bar_width=bw, fillalpha=0.0, linecolor=:black, linewidth=2, label=nothing)
            hatch_rect!(p, xc - bw/2, xc + bw/2, 0.0, h; direction=:\, linewidth=1.2)
        end
    end

    return p
end


# """
#     make_ablation_barplots(out)

# Given `out = ablation_main(...)`, returns Dict[sweep][metric] => plot,
# with 9 plots total (3 sweeps × 3 metrics).
# """
# function make_ablation_barplots(out)
#     sumdict = out.summaries
#     cache = out.cache

#     plots = Dict{Symbol,Dict{Symbol,Any}}()

#     for (sw, S) in sumdict
#         levels = S.levels
#         tick_labels = sweep_tick_labels_from_cache(cache, sw, levels)

#         p_ess = grouped_bars(tick_labels, S.essA, S.essB;
#             title=pretty_title(sw, :ess),
#             xlabel=pretty_xlabel(sw),
#             ylabel=pretty_ylabel(:ess),
#             ylims=(0, out.meta[:n_particles])
#         )

#         p_rmse = grouped_bars(tick_labels, S.rmseA, S.rmseB;
#             title=pretty_title(sw, :rmse),
#             xlabel=pretty_xlabel(sw),
#             ylabel=pretty_ylabel(:rmse)
#         )

#         p_acc = grouped_bars(tick_labels, S.accA, S.accB;
#             title=pretty_title(sw, :acc),
#             xlabel=pretty_xlabel(sw),
#             ylabel=pretty_ylabel(:acc),
#             ylims=(0, 1)
#         )

#         plots[sw] = Dict(:ess=>p_ess, :rmse=>p_rmse, :acc=>p_acc)
#     end

#     return plots
# end

function make_ablation_barplots(out)
    sumdict = out.summaries
    cache   = out.cache
    plots   = Dict{Symbol,Dict{Symbol,Any}}()

    for (sw, S) in sumdict
        levels = S.levels
        tick_labels = sweep_tick_labels_from_cache(cache, sw, levels)

        # Degeneracy masks per metric: use collapsed flags + NaNs in that metric
        degA_ess,  degB_ess  = degmask_from_summary(S.essA,  S.essB,  S.collapsedA, S.collapsedB)
        degA_rmse, degB_rmse = degmask_from_summary(S.rmseA, S.rmseB, S.collapsedA, S.collapsedB)
        degA_acc,  degB_acc  = degmask_from_summary(S.accA,  S.accB,  S.collapsedA, S.collapsedB)

        p_ess = grouped_bars_with_degenerate_overlay(
            tick_labels, S.essA, S.essB, degA_ess, degB_ess;
            title=pretty_title(sw, :ess),
            xlabel=pretty_xlabel(sw),
            ylabel=pretty_ylabel(:ess),
            ylims=(0, out.meta[:n_particles])
        )

        p_rmse = grouped_bars_with_degenerate_overlay(
            tick_labels, S.rmseA, S.rmseB, degA_rmse, degB_rmse;
            title=pretty_title(sw, :rmse),
            xlabel=pretty_xlabel(sw),
            ylabel=pretty_ylabel(:rmse)
        )

        p_acc = grouped_bars_with_degenerate_overlay(
            tick_labels, S.accA, S.accB, degA_acc, degB_acc;
            title=pretty_title(sw, :acc),
            xlabel=pretty_xlabel(sw),
            ylabel=pretty_ylabel(:acc),
            ylims=(0, 1)
        )

        plots[sw] = Dict(:ess=>p_ess, :rmse=>p_rmse, :acc=>p_acc)
    end

    return plots
end

##########################
# Helpers: record lookup #
##########################

"""
    cache_record_for_eval(cache, e)

Find the cache record corresponding to eval entry e.
Supports either direct index by id (1..N) or lookup by rec[:id].
"""
function cache_record_for_eval(cache::Dict, e)
    records = cache[:records]
    # fast path if ids are 1..N in order
    if 1 ≤ e.id ≤ length(records) && haskey(records[e.id], :id) && records[e.id][:id] == e.id
        return records[e.id]
    end
    # fallback lookup
    for r in records
        if r[:id] == e.id
            return r
        end
    end
    error("No cache record found for eval id=$(e.id)")
end

"""
    best_eval_by_accuracy(evals; requireA=false, requireB=true)

Select eval with highest IQ-SIPS accuracy subject to degeneracy constraints.
Skips NaNs.
"""
function best_eval_by_accuracy(evals; requireA::Bool=false, requireB::Bool=true)
    best = nothing
    best_acc = -Inf

    for e in evals
        if requireB && get(e.B, :bad, false)
            continue
        end
        if requireA && get(e.A, :bad, false)
            continue
        end

        acc = get(e.B.pol, :acc, NaN)
        if isnan(acc)
            continue
        end

        if acc > best_acc
            best_acc = acc
            best = e
        end
    end

    best === nothing && error("No eval matched constraints requireA=$requireA requireB=$requireB")
    return best
end

#############################
# Objective grid utilities  #
#############################

"""
    objective_grid_from_key(key, cfg, xs, ys)

Build objective scalar field from Fourier key+cfg and evaluate on grid.
"""
function objective_grid_from_key(key, cfg, xs, ys)
    bank = decode_fourier_key(key, cfg)
    field = make_fourier_scalar_field(bank; scaleQ=true)
    return objective_grid_from_field(field, xs, ys)
end

#########################################
# Plot 1: True objective + tracks       #
#########################################

"""
Plot 1:
- Heatmap of true objective
- Overlay: observed trajectory from cache full_data[:s]
- Overlay: rollout under IQ-SIPS top inferred objective (greedy) from same start, same horizon

Returns a Plots.jl plot.
"""
function plot_true_objective_vs_iqsips_rollout(cache::Dict, e;
                                               gridsize::Int=180,
                                               xy_rows::Tuple{Int,Int}=(1,2))
    rec = cache_record_for_eval(cache, e)
    muenv_spec = cache[:muenv_spec]

    mdp, agent_params = reconstruct_mdp_from_cache(rec, muenv_spec)

    # Grid + true objective
    xs, ys = _grid_from_mdp(mdp; gridsize=gridsize)
    Z_true = objective_grid_from_mdp(mdp, xs, ys)

    # Observed trajectory from cached data
    Sobs = rec[:full_data][:s]
    obs_x, obs_y = xy_path_from_state_matrix(Sobs; xy_rows=xy_rows)
    T = length(obs_x)

    # IQ-SIPS inferred rollout (requires top_key)
    keyB = e.B.top_key
    probB = get(e.B, :top_prob, NaN)

    # Build π_dist for rollout helper
    as = actions(mdp)
    action_list = [as, a->Flux.onehot(a, as), Flux.onehotbatch(as, as)]
    π_dist = ScoreΠDist(; mdp_params=action_list)

    # -------------------- FIX: ensure MDP exists for this key --------------------
    cfgB = rec[:cfg]  # FourierDiscreteCfg used during ablation
    ffB  = decode_fourier_key(keyB, cfgB)
    ensure_mdp!(π_dist, keyB, ffB, agent_params)   # populates n_propmdp_list[keyB]
    # ---------------------------------------------------------------------------

    pred_x, pred_y, _ = rollout_greedy_policy(π_dist, keyB; start_state=agent_params[:start_state], T=10)

    p = heatmap(xs, ys, Z_true;
        aspect_ratio=1,
        dpi=220,
        title="True Objective vs IQ-SIPS Inferred Behavior (posterior ≈ $(isnan(probB) ? "?" : string(round(probB, digits=3))))",
        xlabel="x (world units)",
        ylabel="y (world units)",
        colorbar_title="Objective value")

    plot!(p, obs_x, obs_y; label="Observed trajectory", linewidth=3)
    plot!(p, pred_x, pred_y; label="IQ-SIPS rollout (greedy, top key)", linewidth=3, linestyle=:dash)

    scatter!(p, [obs_x[1]], [obs_y[1]]; label="Start", markersize=6)
    scatter!(p, [obs_x[end]], [obs_y[end]]; label="End", markersize=6)

    return p
end

#########################################
# Plot 2: Objective heatmap triptych    #
#########################################

"""
Plot 2:
- Heatmap true objective
- Heatmap inferred objective (Open-Ended SIPS top key)
- Heatmap inferred objective (IQ-SIPS top key)

Returns a 1x3 Plots.jl layout plot.
"""
function plot_objective_triptych(cache::Dict, e;
                                 gridsize::Int=180)
    rec = cache_record_for_eval(cache, e)
    muenv_spec = cache[:muenv_spec]

    mdp, _ = reconstruct_mdp_from_cache(rec, muenv_spec)

    xs, ys = _grid_from_mdp(mdp; gridsize=gridsize)

    # True objective from reconstructed mdp
    Z_true = objective_grid_from_mdp(mdp, xs, ys)

    # Inferred objectives from stored keys
    cfg = rec[:cfg]   # FourierDiscreteCfg used to decode keys
    keyA = e.A.top_key
    keyB = e.B.top_key

    probA = get(e.A, :top_prob, NaN)
    probB = get(e.B, :top_prob, NaN)

    Z_A = objective_grid_from_key(keyA, cfg, xs, ys)
    Z_B = objective_grid_from_key(keyB, cfg, xs, ys)

    p_true = heatmap(xs, ys, Z_true;
        aspect_ratio=1, dpi=220,
        title="True Objective",
        xlabel="x (world units)", ylabel="y (world units)",
        colorbar_title="Objective")

    p_A = heatmap(xs, ys, Z_A;
        aspect_ratio=1, dpi=220,
        title="Open-Ended SIPS (posterior ≈ $(isnan(probA) ? "?" : string(round(probA, digits=3))))",
        xlabel="x (world units)", ylabel="y (world units)",
        colorbar_title="Objective")

    p_B = heatmap(xs, ys, Z_B;
        aspect_ratio=1, dpi=220,
        title="IQ-SIPS (posterior ≈ $(isnan(probB) ? "?" : string(round(probB, digits=3))))",
        xlabel="x (world units)", ylabel="y (world units)",
        colorbar_title="Objective")

    return plot(p_true, p_A, p_B; layout=(1,3), size=(1500, 480))
end

#########################################
# Driver: make both figures             #
#########################################

"""
    make_final_inference_figures(out; ...)

Produces:
1) Plot 1: best accuracy run with IQ-SIPS non-degenerate
2) Plot 2: best accuracy run with both methods non-degenerate

Returns a NamedTuple with plots and selected evals.
"""
function make_final_inference_figures(out;
                                      gridsize::Int=180,
                                      xy_rows::Tuple{Int,Int}=(1,2))
    cache = out.cache
    evals = out.evals

    # (1) best accuracy with IQ-SIPS non-degenerate
    e1 = best_eval_by_accuracy(evals; requireA=false, requireB=true)
    p1 = plot_true_objective_vs_iqsips_rollout(cache, e1; gridsize=gridsize, xy_rows=xy_rows)

    # (2) best accuracy with both non-degenerate
    e2 = best_eval_by_accuracy(evals; requireA=true, requireB=true)
    p2 = plot_objective_triptych(cache, e2; gridsize=gridsize)

    return (p1=p1, p2=p2, best_iqsips=e1, best_both=e2)
end

############################################
# Convenience single-call entrypoint
############################################

"""
    ablation_main(bson_path; kwargs...)

Runs the suite and returns:
- raw results
- summaries
- plots
"""
# function ablation_main(bson_path::AbstractString; kwargs...)
#     res = run_ablation_suite(bson_path; kwargs...)
#     sums = summarize_ablation(res.evals)
#     pls  = plot_ablation_summaries(sums)
#     return (res=res, summaries=sums, plots=pls)
# end

"""
ablation_main:
- mode=:generate  -> generate buffers, save cache, then evaluate from cache
- mode=:load      -> load cache and evaluate only
"""
function ablation_main(bson_path::String;
                       script_dir::String,
                       mode::Symbol = :generate,
                       cache_filename::String = "ablation_cache.bson",
                       rng::AbstractRNG = Random.default_rng(),
                       shared_muenv_spec::MuEnvSpec = MuEnvSpec(),
                       n_particles::Int=50,
                       minN::Int=20,           # still used by other paths if needed
                       iql_gridN::Int=120,
                       gridsize::Int=120)

    cache_path = joinpath(script_dir, cache_filename)

    cache = if mode == :generate
        generate_and_cache_ablation_data(bson_path;
            cache_path=cache_path,
            rng=rng,
            shared_muenv_spec=shared_muenv_spec,
            T=minN
        )
    elseif mode == :load
        load_ablation_cache(cache_path)
    else
        error("Unknown mode=$mode (use :generate or :load)")
    end

    evals = eval_ablation_from_cache(cache;
        n_particles=n_particles,
        iql_gridN=iql_gridN,
        gridsize=gridsize,
        rng=rng
    )

    sums = summarize_ablation(evals)

    out = (
        cache_path = cache_path,
        cache = cache,
        evals = evals,
        summaries = sums,
        meta = Dict(:n_particles => n_particles, :iql_gridN => iql_gridN, :gridsize => gridsize)
    )
    # Save the entire out wholesale
    BSON.@save joinpath(script_dir, "ablation_out_wholesale.bson") out

    return out
end

#################
### Scripting ###
#################

"""
    data_cleaner(data::ExperienceBuffer, state_field_sizes::Vector{Int64}=[2, 2, 12, 10, 1], keep_state_fields::Vector{Bool}=Bool[1,1,1,0,1])

Helper for cleaning up older data in ways necessary to keep code running.
"""
function data_cleaner(data::ExperienceBuffer, state_field_sizes::Vector{Int64}=[2, 2, 12, 10, 1], keep_state_fields::Vector{Bool}=Bool[1,1,1,0,1])
    # verify elements keeps right size
    actual_size = size(data.data[:s])[2]
    if data.elements ≠ actual_size
        data.elements = actual_size
    end

    # clean state and next-state vectors
    idx = 1
    keep_idxs = []
    for (i, field_size) in enumerate(state_field_sizes)
        if keep_state_fields[i]
            append!(keep_idxs, idx:(idx+field_size-1))
        end
        idx += field_size
    end
    data.data[:s] = data.data[:s][keep_idxs, :]
    data.data[:sp] = data.data[:sp][keep_idxs, :]

    return data
end

"""
    onehot_cols_to_aidx(A::AbstractMatrix) -> Vector{Int}

Convert action matrix A (nactions × T) where each column is one-hot
(or nearly one-hot) into indices aidx[t] ∈ 1:nactions.

Uses argmax per column. Returns vector of integer indices.
"""
function onehot_cols_to_aidx(A::AbstractMatrix; tol::Real=1e-8)
    na, T = size(A)
    aidx = Vector{Int}(undef, T)
    @inbounds for t in 1:T
        col = view(A, :, t)
        # index of maximum entry (should map as idx within actions(mdp))
        aidx[t] = argmax(col)
    end
    return aidx
end

menv = let μfs = [(:sin, x->sin(x[1]) + cos(x[2])),
                  (:exp, x->100*exp(-norm(x-[8 8.])^2 / 1.)),
                  (:lin, x->x[1]^2 + x[2])],
                  μs = [:sin, :exp, :lin];
    MuEnv(3, μs, Dict(μfs));
end

"""
    agent_params_from_mdp(mdp::KAgentPOMDP) -> Dict{Symbol,Any}

Extracts all non-objective agent and environment parameters from an existing
`KAgentPOMDP`, so that new POMDPs can be constructed with identical dynamics,
geometry, noise, discounting, etc., but a different objective function.

The returned dictionary is compatible with `build_kagent_pomdp(agent_params, obj)`.
"""
function agent_params_from_mdp(mdp::KAgentPOMDP)
    return Dict(
        # --- required ---
        :start        => mdp.start,
        :start_state  => KAgentState(mdp.start, [predict_env(menv, mdp.start)], Matrix[]),
        :dimensions   => mdp.dimensions,
        :menv         => menv,

        # --- dynamics / noise ---
        :agent_width  => mdp.width,
        :agent_speed  => mdp.s,
        :ag_mvt_noise => mdp.w,
        :obs_noise    => mdp.v,
        :mdp_discount => mdp.γ,

        # --- geometry ---
        :obcs         => mdp.obcs,
        :goals        => Any[],

        # --- misc ---
        :digits       => mdp.digits,
        :policy_temperature => 2.0
    )
end

println("Directory is: ", @__DIR__)

script_dir = @__DIR__
res_dir = script_dir*"/res"

bson_path = script_dir*"/100_15_100_7_multi_trace_run.bson"

rng = MersenneTwister(0)

out = ablation_main(bson_path;
    script_dir=script_dir,
    mode=:load,   # :generate or :load
    rng=rng,
    n_particles=50,
    minN=150, # specifies number of data points to rollout
    iql_gridN=100,
    gridsize=120
)

# d = BSON.load(joinpath(script_dir, "ablation_out_wholesale.bson"))
# out = d[:out]
plots = make_ablation_barplots(out)

# Example: show plots
display(plots[:K][:acc])
display(plots[:K][:ess])
display(plots[:K][:rmse])
display(plots[:freq_range][:acc])
display(plots[:freq_range][:ess])
display(plots[:freq_range][:rmse])
display(plots[:amp_range][:acc])
display(plots[:amp_range][:ess])
display(plots[:amp_range][:rmse])

for (sw, pd) in plots
    for (metric, p) in pd
        savefig(p, joinpath(res_dir, "$(sw)_$(metric).png"))
    end
end

figs = make_final_inference_figures(out; gridsize=200)
display(figs.p1); display(figs.p2)
savefig(figs.p1, joinpath(res_dir, "final_true_vs_iqsips_rollout.png"))
savefig(figs.p2, joinpath(res_dir, "final_objective_triptych.png"))

# res = multi_run_test(script_dir*"/100_15_100_7_multi_trace_run.bson"; max_tests=200)

# (kworld, data, anon_data) = BSON.load(script_dir*"/single_start_exp.bson")[:data]
# data = data_cleaner(data, [2,2,12,10,1], Bool[1,1,1,0,1])
# anon_data = data_cleaner(anon_data, [2,2,12,10,1], Bool[1,1,1,0,1])

# π_iql, 𝒟_iql, mdp, f = quick_IQL(kworld, anon_data; plot_metrics=false)
# action_list = [actions(mdp), a->Flux.onehot(a, actions(mdp)), Flux.onehotbatch(actions(mdp), actions(mdp))]

# π_dist = ScoreΠDist(; mdp_params = action_list)

# # relevant_data = anon_data.data[:a][:,1:12]
# relevant_data = onehot_cols_to_aidx(anon_data.data[:a][:,1:12])
# start_state = blindstart_KAgentState(mdp, reshape(data.data[:s][:,1][1:2], (1,2)))
# agent_params = agent_params_from_mdp(mdp)
# state_data = data.data[:s][:,1:12]

# iql_state_data, iql_obs_aidx, iql_locs = surrogate_dataset_from_iql_grid(π_dist, π_iql, mdp; eval_num=400)

# filter_state = particle_filter(relevant_data, π_dist, agent_params, state_data, 80; ess_thresh=0.7)

# tops = top_objectives(filter_state, π_dist; topk=10)
# # top objective evaluation
# p_traj = plot_top_objective_with_trajectories(filter_state, π_dist, agent_params;
#                                               observed_state_matrix=state_data, xy_rows=(1,2),
#                                               gridsize=160, show_predicted=true, title_prefix="Top inferred objective")
# # compare top objective against the true objective map
# p_side = plot_objective_side_by_side(filter_state, π_dist; observed_mdp=mdp, gridsize=160)