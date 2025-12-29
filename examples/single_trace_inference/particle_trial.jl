using Gen, GenParticleFilters
using Parameters: @with_kw

struct PseudoNormal <: Gen.Distribution{Float64}
end
Gen.random(fd::PseudoNormal, y, w) = normal(y, w)
Gen.logpdf(fd::PseudoNormal, value, y, w) = logpdf(Gen.Normal(), value, y, w)
Gen.logpdf_grad(fd::PseudoNormal, value, y, w) = (nothing,)
Gen.has_output_grad(fd::PseudoNormal) = false
Gen.is_discrete(fd::PseudoNormal) = false

# Gen.random(fd::PseudoNormal, y) = rand(Normal(y, fd.w))
# Gen.logpdf(fd::PseudoNormal, y) = pdf(Normal(y, fd.w))
# Gen.logpdf_grad(fd::PseudoNormal, y) = (nothing,)
# Gen.has_output_grad(fd::PseudoNormal) = false
# Gen.is_discrete(fd::PseudoNormal) = false
const pseudonormal = PseudoNormal()
(::PseudoNormal)(y, w) = random(PseudoNormal(), y, w)

@gen function object_motion(T::Int)
    y, moving = 0, false
    y_obs_all = Float64[]
    for t=1:T
        moving = {t => :moving} ~ bernoulli(moving ? 0.75 : 0.25)
        vel_y = moving ? sin(t) : 0.0
        y = {t => :y} ~ normal(y + vel_y, 0.01)
        y_obs = {t => :y_obs} ~ normal(y, 0.25)
        push!(y_obs_all, y_obs)
    end
    return y_obs_all
end

@gen function pseudo_object_motion(T::Int)
    y, moving = 0, false
    y_obs_all = Float64[]
    for t=1:T
        moving = {t => :moving} ~ bernoulli(moving ? 0.75 : 0.25)
        vel_y = moving ? sin(t) : 0.0
        y = {t => :y} ~ pseudonormal(y + vel_y, 0.01)
        y_obs = {t => :y_obs} ~ pseudonormal(y, 0.25)
        push!(y_obs_all, y_obs)
    end
    return y_obs_all
end

function particle_filter(observations, n_particles, ess_thresh=0.5; which_motion_model=:default)
    # Initialize particle filter with first observation
    n_obs = length(observations)
    obs_choices = [choicemap((t => :y_obs, observations[t])) for t=1:n_obs]
    if which_motion_model==:default
        state = pf_initialize(object_motion, (1,), obs_choices[1], n_particles)
    elseif which_motion_model==:pseudo
        state = pf_initialize(pseudo_object_motion, (1,), obs_choices[1], n_particles)
    end
    # Iterate across timesteps
    for t=2:n_obs
        # Resample and rejuvenate if the effective sample size is too low
        if effective_sample_size(state) < ess_thresh * n_particles
            # Perform residual resampling, pruning low-weight particles
            pf_resample!(state, :residual)
            # Perform a rejuvenation move on past choices
            rejuv_sel = select(t-1=>:moving, t-1=>:y, t=>:moving, t=>:y)
            pf_rejuvenate!(state, mh, (rejuv_sel,))
        end
        # Update filter state with new observation at timestep t
        pf_update!(state, (t,), (UnknownChange(),), obs_choices[t])
    end
    return state
end

# Generate synthetic dataset of object motion
constraints = choicemap([(t => :moving, t > 5) for t in 1:10]...)
trace, _ = generate(object_motion, (10,), constraints)
observations = get_retval(trace)
# Run particle filter with 100 particles
normal_state = particle_filter(observations, 1000; which_motion_model=:pseudo)
normal_res = [mean(normal_state, 5=>:moving) |> x->round(x, digits=2), var(normal_state, 5=>:moving) |> x->round(x, digits=2),
              mean(normal_state, 6=>:moving) |> x->round(x, digits=2), var(normal_state, 6=>:moving) |> x->round(x, digits=2)]
pseudo_state = particle_filter(observations, 1000; which_motion_model=:pseudo)
pseudo_res = [mean(pseudo_state, 5=>:moving) |> x->round(x, digits=2), var(pseudo_state, 5=>:moving) |> x->round(x, digits=2),
              mean(pseudo_state, 6=>:moving) |> x->round(x, digits=2), var(pseudo_state, 6=>:moving) |> x->round(x, digits=2)]
println("Normal results::\nLikelihood of being in motion @ t=5:")
println("\tMean: ", normal_res[1], "\n\tVar: ", normal_res[2])
println("Likelihood of being in motion @ t=6:")
println("\tMean: ", normal_res[3], "\n\tVar: ", normal_res[4])

println("Pseudo results::\nLikelihood of being in motion @ t=5:")
println("\tMean: ", pseudo_res[1], "\n\tVar: ", pseudo_res[2])
println("Likelihood of being in motion @ t=6:")
println("\tMean: ", pseudo_res[3], "\n\tVar: ", pseudo_res[4])