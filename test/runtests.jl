using ArrayAllez
using Test

using Flux
using Flux.Tracker: TrackedArray, gradcheck, back!, data, grad

gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...) ## from Flux tests
gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)

@testset "exp + log" begin

	@test gradtest(exp0, (2,3))
	@test gradtest(sum∘exp_, (2,3))
	@test gradtest(sum∘exp!∘copy, (2,3))

	p = param(randn(2,3));
	back!(sum(exp.(p)))
	pg = p.grad
	p.grad[:] .= 0;
	back!(sum(exp!(p)))
	@test p.grad == pg

	p.grad[:] .= 0;
	back!(sum(exp!!(p)))
	@test p.grad == pg

	@test gradtest(log0, rand(2,3))
	@test gradtest(log_, rand(2,3))
	@test_broken gradtest(log!∘copy, rand(2,3))

	p = param(rand(2,3));
	back!(sum(log.(p)))
	pg = p.grad
	p.grad[:] .= 0;
	back!(sum(log!(p)))
	@test p.grad == pg

	# p.grad[:] .= 0;
	# back!(sum(log!!(p)))
	# @test p.grad == pg

	@test gradcheck(A -> scale0(A,4) |> sum, rand(2,3))
	@test gradcheck(A -> scale_(A,4) |> sum, rand(2,3))

end
@testset "scale + inv" begin

	m = randn(3,7)
	v = randn(3)
	r = randn(7)'

	@test gradtest(z -> scale_(z,9), m)
	@test gradtest(z -> scale_(z,v), m)
	@test gradtest(z -> scale_(z,r), m)

	# @test gradtest(z -> iscale_(z,9), m)
	# @test gradtest(z -> iscale_(z,v), m)
	# @test gradtest(z -> iscale_(z,r), m)

	# @test gradcheck(z -> sum(inv_(z)), m)
	# @test gradcheck(z -> sum(inv_(z,9)), m)


	# @test gradcheck(z -> sum(scale_(m,z)), v) # crash?
	# @test gradcheck(z -> sum(scale_(m,z)), r)
	#
	# @test gradcheck(z -> sum(iscale_(m,z)), v)
	# @test gradcheck(z -> sum(iscale_(m,z)), r)

end
@testset "prod + cumprod" begin # https://github.com/FluxML/Flux.jl/pull/524

	@test gradtest(x -> prod(x, dims=(2, 3)), (3,4,5))
	@test gradtest(x -> prod(x, dims=1), (3,4,5))
	@test gradtest(x -> prod(x, dims=1), (3,))
	@test gradtest(x -> prod(x), (3,4,5))
	@test gradtest(x -> prod(x), (3,))
	
	rzero(dims...) = (x = rand(dims...); x[2]=0; x)
	@test gradtest(x -> prod(x, dims=(2, 3)), rzero(3,4,5))
	@test gradtest(x -> prod(x, dims=1), rzero(3,4,5))
	@test gradtest(x -> prod(x, dims=1), rzero(3,))
	@test gradtest(x -> prod(x), rzero(3,4,5))
	@test gradtest(x -> prod(x), rzero(3,))
	
	@test gradtest(x -> cumsum(x, dims=2), (3,4,5))
	@test gradtest(x -> cumsum(x, dims=1), (3,))
	@test gradtest(x -> cumsum(x), (3,))
	
	@test gradtest(x -> cumprod(x, dims=2), (3,4,5))
	@test gradtest(x -> cumprod(x, dims=1), (3,))
	@test gradtest(x -> cumprod(x), (3,))
	@test gradtest(x -> cumprod(x, dims=2), rzero(3,4,5))
	@test gradtest(x -> cumprod(x, dims=1), rzero(3,))
	@test gradtest(x -> cumprod(x), rzero(3,))

end
