module ArrayAllez

include("cache.jl")

include("inplace.jl")

include("odot.jl")

@init @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
	include("inplace-flux.jl")
	include("prod+cumprod.jl")
end
include("inplace-zygote.jl")

include("dropdims.jl")

end
