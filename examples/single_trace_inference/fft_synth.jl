# using FFTW
using Plots

"Create an Nx×Ny periodic grid over [0,Lx)×[0,Ly)."
function make_grid(Nx::Int, Ny::Int; Lx::Float64=10.0, Ly::Float64=10.0)
    xs = range(0.0, Lx; length=Nx+1)[1:end-1]
    ys = range(0.0, Ly; length=Ny+1)[1:end-1]
    return xs, ys
end

"""
Create a single 2D Fourier-like component field:
    F(x,y) = A * cos(2π*(kx/Lx*x + ky/Ly*y) + ϕ)

Returns:
  - xs, ys : grid vectors
  - F      : Nx×Ny matrix where F[ix,iy] corresponds to (xs[ix], ys[iy])
"""
function component_field(A::Float64, ϕ::Float64, kx::Float64, ky::Float64,
                         Nx::Int, Ny::Int; Lx::Float64=10.0, Ly::Float64=10.0)
    xs, ys = make_grid(Nx, Ny; Lx=Lx, Ly=Ly)
    F = Matrix{Float64}(undef, Nx, Ny)

    αx = 2π * (kx / Lx)
    αy = 2π * (ky / Ly)

    @inbounds for ix in 1:Nx
        x = xs[ix]
        for iy in 1:Ny
            y = ys[iy]
            F[ix, iy] = A * cos(αx*x + αy*y + ϕ)
        end
    end

    return xs, ys, F
end

"Sum multiple component fields (all same size)."
composite_sum(fields::Vector{Matrix{Float64}}) = reduce(+, fields)

# "Circular convolution via FFT (periodic boundary conditions)."
# function fft_convolve(F::AbstractMatrix, G::AbstractMatrix)
#     H = ifft(fft(F) .* fft(G))
#     return real.(H)
# end

############################
# Plotting helper functions
############################

"Plot a 2D scalar field as a heatmap."
function plot_heatmap(xs, ys, F; title="Scalar Field", kwargs...)
    heatmap(
        xs, ys, F';                # transpose so axes align as x-horizontal, y-vertical
        xlabel="x", ylabel="y",
        title=title,
        colorbar=true,
        aspect_ratio=:equal,
        kwargs...
    )
end

"Plot a 2D scalar field as a 3D surface."
function plot_surface(xs, ys, F; title="Scalar Field (3D)", kwargs...)
    surface(
        xs, ys, F';                # transpose for correct orientation
        xlabel="x", ylabel="y", zlabel="value",
        title=title,
        kwargs...
    )
end

############################
# Example usage
############################

Nx, Ny = 128, 128
Lx, Ly = 10.0, 10.0

xs, ys, c1 = component_field(1.0, 0.0,  1.0,  0.0, Nx, Ny; Lx=Lx, Ly=Ly)
_,  _,  c2 = component_field(0.7, 1.2, -0.0,  1.0, Nx, Ny; Lx=Lx, Ly=Ly)
_,  _,  c3 = component_field(0.4, 2.5,  0.0,  0.0, Nx, Ny; Lx=Lx, Ly=Ly)

F = composite_sum([c1, c2, c3])

p1 = plot_heatmap(xs, ys, F; title="Composite Field (Heatmap)")
p2 = plot_surface(xs, ys, F; title="Composite Field (Surface)")

display(p1)
display(p2)