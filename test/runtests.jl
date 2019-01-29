using ArrayAllez
using Test

@testset "simple" begin
    @testset "small" begin

        m = rand(3,7)
        v = randn(3)
        r = randn(7)'

        @test exp!(copy(m)) == exp_(m) == exp0(m) == exp_(:test, m)
        @test exp!(copy(v)) == exp_(v) == exp0(v) == exp_(:test, v)
        @test exp!(copy(r)) == exp_(r) == exp0(r) == exp_(:test, r)

        @test exp!(copy(m)) == exp_(m) == exp0(m) == exp_(:test, m)
        @test exp!(copy(v)) == exp_(v) == exp0(v) == exp_(:test, v)
        @test exp!(copy(r)) == exp_(r) == exp0(r) == exp_(:test, r)

        @test inv!(copy(m)) == inv_(m) == inv0(m) == inv_(:test, m)
        @test inv!(copy(v)) == inv_(v) == inv0(v) == inv_(:test, v)
        @test inv!(copy(r)) == inv_(r) == inv0(r) == inv_(:test, r)


        @test scale!(copy(m),π) == scale_(m,π) == scale0(m,π)
        @test scale!(copy(v),π) == scale_(v,π) == scale0(v,π)
        @test scale!(copy(r),π) == scale_(r,π) == scale0(r,π)

        @test scale!(copy(m),v) == scale_(m,v) == scale0(m,v)
        @test scale!(copy(v),v) == scale_(v,v) == scale0(v,v)

        @test scale!(copy(m),r) == scale_(m,r) == scale0(m,r)
        @test scale!(copy(r),r) == scale_(r,r) == scale0(r,r)


        @test iscale!(copy(m),π) ≈ iscale_(m,π) ≈ iscale0(m,π)
        @test iscale!(copy(v),π) ≈ iscale_(v,π) ≈ iscale0(v,π)
        @test iscale!(copy(r),π) ≈ iscale_(r,π) ≈ iscale0(r,π)

        @test iscale!(copy(m),v) ≈ iscale_(m,v) ≈ iscale0(m,v)
        @test iscale!(copy(v),v) ≈ iscale_(v,v) ≈ iscale0(v,v)

        @test iscale!(copy(m),r) ≈ iscale_(m,r) ≈ iscale0(m,r)
        @test iscale!(copy(r),r) ≈ iscale_(r,r) ≈ iscale0(r,r)

    end
    @testset "large" begin

        m = rand(300,700);
        v = randn(300);
        r = randn(700)';

        @test exp!(copy(m)) ≈ exp_(m) ≈ exp0(m) ≈ exp_(:test, m)
        @test exp!(copy(v)) ≈ exp_(v) ≈ exp0(v) ≈ exp_(:test, v)
        @test exp!(copy(r)) ≈ exp_(r) ≈ exp0(r) ≈ exp_(:test, r)

        @test exp!(copy(m)) ≈ exp_(m) ≈ exp0(m) ≈ exp_(:test, m)
        @test exp!(copy(v)) ≈ exp_(v) ≈ exp0(v) ≈ exp_(:test, v)
        @test exp!(copy(r)) ≈ exp_(r) ≈ exp0(r) ≈ exp_(:test, r)

        @test inv!(copy(m)) ≈ inv_(m) ≈ inv0(m) ≈ inv_(:test, m)
        @test inv!(copy(v)) ≈ inv_(v) ≈ inv0(v) ≈ inv_(:test, v)
        @test inv!(copy(r)) ≈ inv_(r) ≈ inv0(r) ≈ inv_(:test, r)

    end
end

@testset "odot" begin

    c = rand(3)
    cc = rand(3,3)

    @test cc ⊙ c == cc * c
    @test cc ⊙ cc == cc * cc
    @test c' ⊙ cc == c' * cc

    ccc = rand(3,3,3)
    Ic = reshape(ccc,9,3)
    cI = reshape(ccc,3,9)

    @test vec(ccc ⊙ ccc) == vec(Ic * cI)
    @test vec(ccc ↓ ccc) == vec(Ic * cI)

end

# using AppleAccelerate

using Flux
using Flux.Tracker: TrackedArray, gradcheck, back!, data, grad

gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...) ## from Flux tests
gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)

@testset "gradients" begin

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
        @test gradtest(log!∘copy, rand(2,3))

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

        m = rand(3,7)
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
end