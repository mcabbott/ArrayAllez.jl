module ArrayAllez

include("cache.jl")

include("inplace.jl")

@init @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
	include("inplace-flux.jl")
	include("prod+cumprod.jl")
end
@init @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    IVERBOSE && @info "ArrayAllez loaded in-place code for Zygote... or tried!"
end


end