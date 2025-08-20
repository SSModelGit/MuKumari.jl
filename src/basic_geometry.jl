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
        vec_reduction_frac = (col_dist - width) / col_dist
        xp = (nearest_collision .- xs) .* vec_reduction_frac .+ xs
    end

    return round.(xp; digits=digits)
end

"""Returns a vector of the **distances** to the nearest k geometries.

Also provides a quadrant breakdown of how many geometries are present in each quadrant.
"""
function nearest_k_geometries(loc::Matrix, geometries::Vector, k::Integer)
    tuple_loc = Tuple(loc)
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