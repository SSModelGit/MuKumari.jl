# MuKumari

This is intended as an easy-to-use playground for quickly setting up multi-agent scenarios in simulated environments with user-specifiable characteristics.

### TODOs:

Steps necessary to get Crux's IQLearn working:

* 1. Fix the state/observation vectorization conversion via the POMDPs `convert_s` function.
  * Proposition 1.1: Change the MDP type into a POMDP and use the `initialobs` to vectorize the state into some observations.
  * Proposition 1.2: Remove the observation output from the `gen`, making `KAgentMDP` a true MDP; self-define `convert_s` to vectorize the state.
* 2. Decide what to do w.r.t. the `ContinuousSpace` converter via Crux's `statespace` function
  * Proposition 2.1: Leave the default values for `μ` and `σ`, in hopes that the statespace will be refined by the network over time
  * Proposition 2.2: Edit the values to somehow better match the world (ex. make `μ=5.` and `σ=5.`, etc.)
* 3. Generate "expert trajectories", wrapping simulator output in an Crux's `ExperienceBuffer` type.