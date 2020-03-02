module ArrayAllez

include("cache.jl")

include("inplace.jl")

@init @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
	include("inplace-flux.jl")
	include("prod+cumprod.jl")
    #= # not sure these work
    using .Tracker: TrackedReal, TrackedArray, track, @grad
    *ˡ(x::TrackedReal, y) = track(*ˡ, x, y)
    *ˡ(x::TrackedArray, y) = track(*ˡ, x, y)
    *ʳ(x, y::TrackedArray) = track(*ʳ, x, y)
    *ʳ(x, y::TrackedReal) = track(*ʳ, x, y)
    @grad *ˡ(A, B) = (data(A) * data(B)),  Δ -> (data(Δ) * data(B)', nothing)
    @grad *ʳ(A, B) = (data(A) * data(B)),  Δ -> (nothing, data(A)' * data(Δ))
    =#
end
include("inplace-zygote.jl")

include("dropdims.jl")

include("odot.jl")

end
