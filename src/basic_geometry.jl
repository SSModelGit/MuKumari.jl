"""collision_check(xs::Matrix, xp::Matrix, pgon, width; debug::Bool=true, digits=3)

[DEPRECATED] Collision checking function. Stops the agent at the first possible collision.

Note that the agent will stop  at an agent's width away from the obstacle.

Intentionally not exported (only an internal-use function).
"""
function collision_check(xs::Matrix, xp::Matrix, pgon, width; debug::Bool=true, digits=3)
    if debug
        dist = GO.distance(GI.Point(Tuple(xp)), pgon)
        println("Point under question: ", xp, "| Stated distance: ", dist, " | Polygon: ", pgon)
    end

    # is it outside the boundaries of the traversible world?
    movement = GI.LineString([GI.Point(Tuple(xs)), GI.Point(Tuple(xp))])
    boundary_intersect = GO.intersection(movement, GI.getexterior(pgon); target=GI.PointTrait())
    if debug; println("Boundary crossings: ", boundary_intersect); end

    # or did it happen to cut through an obstacle?
    through_hole = filter(!isempty, map(GI.gethole(pgon)) do hole
        GO.intersection(movement, hole; target=GI.PointTrait())
    end) |> Iterators.flatten |> collect

    # Put it all together
    total_intersects = vcat(boundary_intersect, through_hole)
    if debug; println("Through holes are: ", through_hole); println("Concated: ", total_intersects); end
    # unique_intersects = unique(x->map(d->round(d, digits=digits), x), total_intersects)
    unique_intersects = unique(x->round.(x; digits=digits), total_intersects)
    if !isempty(unique_intersects)
        # find the closest intersection to the starting position
        intersect_info = map(unique_intersects) do isect
            isect_mat = reshape(collect(isect), (1, :))
            [isect_mat, norm(xs - isect_mat)]
        end
        intersects_by_dists = sort(intersect_info, by=x->x[2])
        if debug; println("Intersection info: ", intersects_by_dists); end

        # take closest intersection point
        nearest_collision, col_dist = intersects_by_dists[1]
        if debug; println("Nearest collision: ", nearest_collision); println("Distance to nearest collision: ", col_dist); end
        vec_reduction_frac = (col_dist - width) / col_dist
        xp = (nearest_collision .- xs) .* vec_reduction_frac .+ xs
    end

    return round.(xp; digits=digits)
end

"""risk_check(xs::Matrix, xp::Matrix, pgon, width; debug::Bool=true, digits=3)

[CURRENT] Collision checking function. ONLY CONSIDERS THE WALLS OF THE ENVIRONMENT.

Note that the agent will stop  at an agent's width away from the WALL.

Intentionally not exported (only an internal-use function).

TODO: Most of the code remains unchanged from the deprecated version, to avoid errors. Should be streamlined later.
"""
function risk_check(xs::Matrix, xp::Matrix, pgon, width; debug::Bool=true, digits=3)
    if debug
        dist = GO.distance(GI.Point(Tuple(xp)), pgon)
        println("Point under question: ", xp, "| Stated distance: ", dist, " | Polygon: ", pgon)
    end

    # is it outside the boundaries of the traversible world?
    movement = GI.LineString([GI.Point(Tuple(xs)), GI.Point(Tuple(xp))])
    if debug; println("Movement: ", movement); end
    boundary_intersect = GO.intersection(movement, GI.getexterior(pgon); target=GI.PointTrait())
    if debug; println("Boundary crossings: ", boundary_intersect); end

    # Identify unique intersects (avoid corner shenaniganery)
    unique_intersects = unique(x->round.(x; digits=digits), boundary_intersect)
    if debug; println("Unique: ", unique_intersects); end
    if !isempty(unique_intersects)
        ## find the closest intersection to the starting position
        # compute distance to each intersect, make list of [[intersect, distance_to_intersect], ...]
        intersect_info = map(unique_intersects) do isect
            isect_mat = reshape(collect(isect), (1, :))
            [isect_mat, norm(xs - isect_mat)]
        end

        # sort by the second value (distance_to_intersect)
        intersects_by_dists = sort(intersect_info, by=x->x[2])
        if debug
            println("intersect info: ", intersect_info)
            println("intersect info by dists: ", intersects_by_dists)
            println("Intersection info: ", intersects_by_dists)
        end

        # take closest intersection point
        nearest_collision, col_dist = intersects_by_dists[1]
        if debug; println("Nearest collision: ", nearest_collision); println("Distance to nearest collision: ", col_dist); end

        # NaN prevention squad
        # if the intersection is at/near the start, avoid 0/0
        if !isfinite(col_dist) || col_dist ≤ eps(Float64)
            # cannot meaningfully move; keep current position
            xp = xs
        else
            vec_reduction_frac = (col_dist - width) / col_dist
            # If closer than width already, don't step past
            if !isfinite(vec_reduction_frac) || vec_reduction_frac ≤ 0.0
                xp = xs
            else
                xp = (nearest_collision .- xs) .* vec_reduction_frac .+ xs
            end
        end
        # never return NaNs (brute force but whatever)
        if any(!isfinite, xp)
            xp = xs
        end
    end

    return round.(xp; digits=digits)
end

"""Returns a vector of the **distances** to the nearest k geometries.

Also provides a quadrant breakdown of how many geometries are present in each quadrant.
"""
function nearest_k_geometries(loc::Matrix, geometries::Vector, k::Integer)
    tuple_loc = Tuple(loc)
    # if loc is invalid, return a fixed-shape neutral observation - brute force but whatever
    if any(!isfinite, tuple_loc)
        field_count = zeros(8)
        centers = fill((0.0, 0.0), Int(k))
        return centers, field_count
    end
    point = GI.Point(tuple_loc)
    field_count = zeros(8)

    # find centers of geometries
    if typeof(geometries[1]) <: GI.Polygon
        centers = [GO.centroid(pgon) for pgon in geometries]
    else
        centers = [p.geom for p in geometries]
    end

    # complete field count
    angles = [atan((c .- tuple_loc)...) for c in centers]
    # 11th hour NaN catcher
    if any(!isfinite, angles)
        centers0 = fill((0.0, 0.0), Int(k))
        return centers0, zeros(8)
    end
    if any(isnan.(angles));
        println("Location: ", loc)
        println("Tuple-ified: ", tuple_loc)
        println("Centers of geometries: ", centers)
    end
    int_angles = (Integer.(round.(rad2deg.(angles))) .+ 360) .% 360 # determine rounded angle to geometry
    # increment field count
    for a in int_angles; field_count[div(a, 45)+1] += 1; end

    # complete distance list
    distance_list = sort(by=x->norm(x .- tuple_loc), centers)

    return distance_list[1:k], field_count
end

# """
#     nearest_k_maxima(pomdp::KAgentPOMDP, s::KAgentState, objective_func::Function, k::Integer; r=10.0, n=201)

# Dense-grid search for local maxima of a scalar objective function around `loc`.
# If an octant bin has zero detected interior maxima, attempt a pseudo-maximum on the octant bisector using
# `clip_bisector_endpoint` and a 1-step monotone check along the ray.

# Inputs
# - s::KAgentState: agent location as a 1×2 matrix, e.g. [x y]
# - objective_func::Function: scalar field objective; accepts KAgentState.
# - k::Integer: number of nearest maxima to return
# - r::Float64: search half-width in each direction
# - n::Integer: grid points per axis (odd gives a center point)

# Returns
# - NamedTuple:
#     (vectors, distances, counts, maxima_xy)

#     vectors   :: Vector{SVector{2,Float64}}  # Δx,Δy to k nearest maxima (globally)
#     distances :: Vector{Float64}             # Euclidean norms of vectors
#     counts    :: Vector{Int} length 8        # number of maxima in each angular quadrant
#     maxima_xy :: Matrix{Float64} (nmax × 2)  # all maxima locations found (x,y)

# Quadrant convention (8 bins):
# - Compute θ = atan(Δy, Δx) in [-π, π)
# - Bin index b = floor((θ + π) / (2π/8)) + 1 ∈ {1..8}
# """
# function nearest_k_maxima(pomdp::KAgentPOMDP,
#                           s::KAgentState,
#                           objective_func::Function,
#                           k::Integer;
#                           r::Float64=10.0,
#                           n::Integer=201)

#     # Agent location (your codebase: s.x is 1×2 and destructuring works)
#     x0, y0 = s.x

#     # Grid
#     xs = range(x0 - r, x0 + r; length=n)
#     ys = range(y0 - r, y0 + r; length=n)

#     # Evaluate objective at arbitrary location by pseudo-placing the agent.
#     # (Pseudo placement helper exists in MuKumari.) :contentReference[oaicite:3]{index=3}
#     eval_obj(x, y) = objective_func(pseudo_agent_placement(s, reshape([x, y], 1, 2)))

#     # Field samples
#     F = Array{Float64}(undef, n, n)
#     @inbounds for i in 1:n
#         xi = xs[i]
#         for j in 1:n
#             F[i, j] = float(eval_obj(xi, ys[j]))
#         end
#     end

#     # Strict interior maxima (8-neighborhood)
#     max_is = Int[]
#     max_js = Int[]
#     @inbounds for i in 2:(n-1)
#         for j in 2:(n-1)
#             v = F[i, j]
#             if v > F[i-1, j-1] && v > F[i-1, j] && v > F[i-1, j+1] &&
#                v > F[i,   j-1]                 && v > F[i,   j+1] &&
#                v > F[i+1, j-1] && v > F[i+1, j] && v > F[i+1, j+1]
#                 push!(max_is, i)
#                 push!(max_js, j)
#             end
#         end
#     end

#     # Convert maxima indices to coordinates
#     nmax = length(max_is)
#     maxima_xy = Matrix{Float64}(undef, nmax, 2)
#     @inbounds for t in 1:nmax
#         maxima_xy[t, 1] = xs[max_is[t]]
#         maxima_xy[t, 2] = ys[max_js[t]]
#     end

#     # Bin into 8 octants + store vector/distances
#     counts = zeros(Int, 8)
#     dxs    = Vector{Float64}(undef, nmax)
#     dys    = Vector{Float64}(undef, nmax)
#     dists  = Vector{Float64}(undef, nmax)

#     bin_width = 2π / 8
#     @inbounds for t in 1:nmax
#         dx = maxima_xy[t, 1] - x0
#         dy = maxima_xy[t, 2] - y0
#         dxs[t] = dx
#         dys[t] = dy
#         d = hypot(dx, dy)
#         dists[t] = d

#         if d == 0.0
#             counts[1] += 1
#         else
#             θ = atan(dy, dx)  # [-π, π)
#             b = Int(floor((θ + π) / bin_width)) + 1
#             b = (b > 8) ? 8 : b
#             counts[b] += 1
#         end
#     end

#     # ------------------------------------------------------------
#     # Pseudo-max addendum (only for bins with zero interior maxima)
#     # ------------------------------------------------------------
#     # One-step "preceding point" step size along the ray:
#     # Use ~one grid cell (conservative).
#     Δ = 0.5 * min(xs[2] - xs[1], ys[2] - ys[1])

#     # For dedup (snap-to-grid indices)
#     seen = Set{Tuple{Int,Int}}((max_is[t], max_js[t]) for t in 1:nmax)

#     # Snap an (x,y) to nearest grid indices
#     function nearest_grid_index(x::Float64, y::Float64)
#         dxg = xs[2] - xs[1]
#         dyg = ys[2] - ys[1]
#         i = Int(round((x - xs[1]) / dxg)) + 1
#         j = Int(round((y - ys[1]) / dyg)) + 1
#         return clamp(i, 1, n), clamp(j, 1, n)
#     end

#     for b in 1:8
#         counts[b] != 0 && continue  # only if no maxima in that octant

#         # Bisector angle for octant b
#         θc = -π + (b - 0.5) * bin_width
#         ux = cos(θc)
#         uy = sin(θc)

#         # Ray endpoint (may lie outside); clip to box boundary using your routine
#         p0 = reshape([x0, y0], 1, 2)
#         p1 = reshape([x0 + r*ux, y0 + r*uy], 1, 2)
#         p_saved = clip_bisector_endpoint(pomdp, p0, p1)  # uses pomdp.dimensions :contentReference[oaicite:4]{index=4}

#         xsaved = float(p_saved[1,1])
#         ysaved = float(p_saved[1,2])

#         # Preceding point along the ray direction
#         xprev = xsaved - Δ*ux
#         yprev = ysaved - Δ*uy

#         v_saved = float(eval_obj(xsaved, ysaved))
#         v_prev  = float(eval_obj(xprev,  yprev))

#         # Pseudo-max criterion: objective increases as we move toward boundary point
#         if v_prev < v_saved
#             ib, jb = nearest_grid_index(xsaved, ysaved)
#             if !((ib, jb) in seen)
#                 push!(max_is, ib)
#                 push!(max_js, jb)
#                 push!(seen, (ib, jb))

#                 # append to maxima_xy and distance arrays
#                 maxima_xy = vcat(maxima_xy, reshape([xs[ib], ys[jb]], 1, 2))
#                 push!(dxs, xs[ib] - x0)
#                 push!(dys, ys[jb] - y0)
#                 push!(dists, hypot(dxs[end], dys[end]))
#                 counts[b] += 1
#             end
#         end
#     end

#     # Refresh nmax after pseudo additions
#     nmax = size(maxima_xy, 1)

#     # Select k nearest maxima (globally)
#     if nmax == 0
#         return (vectors = Vector{NTuple{2,Float64}}(),
#                 distances = Float64[],
#                 counts = counts,
#                 maxima_xy = maxima_xy)
#     end

#     kk  = min(k, nmax)
#     ord = partialsortperm(dists, 1:kk)

#     vectors = Vector{NTuple{2,Float64}}(undef, kk)
#     nearest_dists = Vector{Float64}(undef, kk)
#     @inbounds for ii in 1:kk
#         t = ord[ii]
#         vectors[ii] = (dxs[t], dys[t])
#         nearest_dists[ii] = dists[t]
#     end

#     return (vectors = vectors,
#             distances = nearest_dists,
#             counts = counts,
#             maxima_xy = maxima_xy)
# end

# # One-shot "slide-to-box" for a bisector ray endpoint, written in matrix form.
# # Points are 1×2 matrices: p[1,1]=x, p[1,2]=y
# #
# # Assumptions:
# # - Box boundary is axis-aligned: [d1,d2] × [d1,d2]
# # - Ray is along one of 8 bisectors: slope ∈ {0, ±1, ±∞}
# # Convention:
# # - If both x and y violate bounds, resolve x first.

# function clip_bisector_endpoint(pomdp::KAgentPOMDP, p0::Matrix, p1::Matrix)
#     d1 = float(pomdp.dimensions[1])
#     d2 = float(pomdp.dimensions[2])

#     @inline inbounds(p) = (d1 ≤ p[1,1] ≤ d2 ? 0 : 2) + (d1 ≤ p[1,2] ≤ d2 ? 0 : 1)

#     inbound_check = inbounds(p1)
#     # Fast path
#     if inbound_check == 0
#         return p1
#     end

#     # Direction and slope
#     d = p1 .- p0
#     dx = d[1,1]
#     dy = d[1,2]
#     m  = (dx == 0.0) ? Inf : (dy / dx)

#     # We compute p2 by fixing one coordinate to the violated boundary and sliding the other.
#     p2 = copy(p1)

#     if inbound_check ≥ 2
#         # Fix x first (also when both violated)
#         xB = (p1[1,1] < d1) ? d1 : d2
#         Δx = xB - p1[1,1]
#         p2[1,1] = xB

#         if isfinite(m)
#             p2[1,2] = p1[1,2] + m * Δx
#         else
#             # vertical ray: x shouldn't be the violated dimension when moving from inside,
#             # but keep y unchanged as a safe fallback.
#             p2[1,2] = clamp(p1[1,2], d1, d2)
#         end

#         # If y still violates, fix y and slide x (second and final slide)
#         if p2[1,2] < d1 || p2[1,2] > d2
#             yB = (p2[1,2] < d1) ? d1 : d2
#             Δy = yB - p2[1,2]
#             p2[1,2] = yB

#             if (abs(m) < 1e-12) || !isfinite(m)
#                 # horizontal: x unchanged
#                 p2[1,1] = p2[1,1]
#             else
#                 p2[1,1] = p2[1,1] + (Δy / m)
#             end
#         end

#     else
#         # y violated (and x not violated): fix y first
#         yB = (p1[1,2] < d1) ? d1 : d2
#         Δy = yB - p1[1,2]
#         p2[1,2] = yB

#         if abs(m) < 1e-12 || !isfinite(m)
#             # horizontal / vertical ray: y constant; keep x
#             p2[1,1] = clamp(p1[1,1], d1, d2)
#         else
#             p2[1,1] = p1[1,1] + (Δy / m)
#         end

#         # If x now violates, fix x and slide y (second and final slide)
#         if p2[1,1] < d1 || p2[1,1] > d2
#             xB = (p2[1,1] < d1) ? d1 : d2
#             Δx = xB - p2[1,1]
#             p2[1,1] = xB

#             if isfinite(m)
#                 p2[1,2] = p2[1,2] + m * Δx
#             else
#                 # vertical: y unchanged
#                 p2[1,2] = p2[1,2]
#             end
#         end
#     end

#     # Final safety clamp (avoid weird floating point issues this way)
#     return [clamp(p2[1,1], d1, d2) clamp(p2[1,2], d1, d2)]
# end