using MuKumari

using LinearAlgebra: norm, normalize
using Combinatorics: powerset

using POMDPTools, MCTS, POMDPLinter
using Match: @match
using Parameters: @with_kw
import GeoInterface as GI

# addressing weird load order bugs
using Plots
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

#################################
# Fourier-mode parameter sampling
#################################

@with_kw struct FourierDiscreteCfg
    Kmax::Int = 24
    λK::Float64 = 0.35            # P(K=k) ∝ exp(-λK*(k-1))

    # frequency grid
    Δf::Float64 = 0.1
    Fmax_i::Int = 30              # bins in -Fmax_i:Fmax_i

    # amplitude grid
    ΔA::Float64 = 0.1
    Amax_i::Int = 50              # bins in 0:Amax_i

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
Decode your Fourier key of the form (K, fx_i, fy_i, A_i, ϕ_i) into continuous params.
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

Your code treats `dimensions` as a 2-tuple (d1, d2) and constructs the boxworld
with corners (d1,d1) and (d2,d2). This helper just standardizes that.
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
This matches your usage pattern at the bottom of the file.
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

Your cleaned data uses `data.data[:s]` with x,y in the first two rows
(after your data_cleaner trimming). This helper extracts the trajectory.

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

Uses your existing proposal_boltzmann(...) machinery to compute a Boltzmann
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

#########################
### Example call-sites ###
#########################

# After running:
#   filter_state = particle_filter(...)
# You likely have:
#   mdp           :: KAgentPOMDP        (observed)
#   agent_params  :: Dict              (constructed from mdp, includes :start_state)
#   data.data[:s] :: Matrix (features × T)
#
# Example: plot inferred heatmap + observed vs predicted trajectories over first 12 steps
#
# observed_state_matrix = data.data[:s][:, 1:12]
# p_traj = plot_top_objective_with_trajectories(filter_state, π_dist, agent_params;
#                                               observed_state_matrix=observed_state_matrix,
#                                               gridsize=160,
#                                               xy_rows=(1,2),
#                                               show_predicted=true,
#                                               title_prefix="Top inferred objective")
# display(p_traj)
#
# Example: side-by-side objective sanity check
# p_side = plot_objective_side_by_side(filter_state, π_dist; observed_mdp=mdp, gridsize=160)
# display(p_side)


#################
### Scripting ###
#################

"""

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
        :start_state  => rand(initialstate(mdp)),
        :dimensions   => mdp.dimensions,
        :menv         => mdp.menv,

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

(kworld, data, anon_data) = BSON.load(script_dir*"/single_start_exp.bson")[:data]
data = data_cleaner(data, [2,2,12,10,1], Bool[1,1,1,0,1])
anon_data = data_cleaner(anon_data, [2,2,12,10,1], Bool[1,1,1,0,1])
# anon_data.elements = 996 # manual edit of this specific data file to account for empty end values

π_iql, 𝒟_iql, mdp, f = quick_IQL(kworld, anon_data; plot_metrics=false)
action_list = [actions(mdp), a->Flux.onehot(a, actions(mdp)), Flux.onehotbatch(actions(mdp), actions(mdp))]

π_dist = ScoreΠDist(; mdp_params = action_list)

# relevant_data = anon_data.data[:a][:,1:12]
relevant_data = onehot_cols_to_aidx(anon_data.data[:a][:,1:12])
start_state = blindstart_KAgentState(mdp, reshape(data.data[:s][:,1][1:2], (1,2)))
agent_params = agent_params_from_mdp(mdp)
state_data = data.data[:s][:,1:12]

iql_state_data, iql_obs_aidx, iql_locs = surrogate_dataset_from_iql_grid(π_dist, π_iql, mdp; eval_num=400)

filter_state = particle_filter(relevant_data, π_dist, agent_params, state_data, 100; ess_thresh=0.7)