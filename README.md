# ArrayAllez.jl

[![Build Status](https://travis-ci.org/mcabbott/ArrayAllez.jl.svg?branch=master)](https://travis-ci.org/mcabbott/ArrayAllez.jl)

```
] add ArrayAllez
```

### `⊙ = \odot`

Matrix multiplication, on the last index of one tensor & the first index of the next:

```julia
three = rand(2,2,5);
mat = rand(5,2);

p1 = three ⊙ mat

p2 = reshape(reshape(three,:,5) * mat ,2,2,2) # same

using Einsum
@einsum p3[i,j,k] := three[i,j,s] * mat[s,k]  # same
```

There are also variants `⊙ˡ, ⊙ʳ` with different gradient definitions,
specifying that only what's on the left (or right) needs to be tracked. 
(Likewise `*ˡ, *ʳ` for ordinary `*`.)

### `bmm == ⨱ (\timesbar)`

Batched matrix multiplication, which understands all trailing dimensions:

```julia
four = rand(2,3,8,9);
three = rand(3,8,9);

size(four ⨱ three) == (2, 8, 9)
(four ⨱ three)[:,1,1] ≈ four[:,:,1,1] * three[:,1,1]

using Einsum
@einsum out[i,x,y] := four[i,j,x,y] * three[j,x,y];
out ≈ four ⨱ three
```

Corresponding `⨱ˡ, ⨱ʳ` are not yet defined.

### `dimnames`

Both `⊙` and `⨱` will propagate names from [NamedDims.jl](https://github.com/invenia/NamedDims.jl).

### `log! ∘ exp!`

This began as a way to more conveniently choose between [Yeppp!](https://github.com/JuliaMath/Yeppp.jl) 
and [AppleAccelerate](https://github.com/JuliaMath/AppleAccelerate.jl)
and [IntelVectorMath](https://github.com/JuliaMath/IntelVectorMath.jl),
without requiring that any by installed. 
The fallback version is just a loop, with `@threads` for large enough arrays.

```julia
x = rand(1,100);

y = exp0(x)  # precisely = exp.(x)
x ≈ log!(y)  # in-place, just a loop

using AppleAccelerate  # or using IntelVectorMath, or using Yeppp

y = exp!(x)  # with ! mutates
x = log_(y)  # with _ copies
```

Besides `log!` and `exp!`, there is also `scale!` which understands rows/columns. 
And `iscale!` which divides, and `inv!` which is an element-wise inverse.
All have non-mutating versions ending `_` instead of `!`, and simple broadcast-ed versions with `0`.

```julia
m = ones(3,7)
v = rand(3)
r = rand(7)'

scale0(m, 99)  # simply m .* 99
scale_(m, v)   # like m .* v but using rmul!
iscale!(m, r)  # like m ./ r but mutating.
m
```

### `∇`

These commands all make some attempt to define gradients for use with 
[Tracker](https://github.com/FluxML/Tracker.jl) ans 
[Zygote](https://github.com/FluxML/Zygote.jl), but caveat emptor. 
There is also an `exp!!` which mutates both its forward input and its backward gradient, 
which may be a terrible idea.

```julia
using Tracker
x = param(randn(5));
y = exp_(x)

Tracker.back!(sum_(exp!(x)))
x.data == y # true
x.grad
```

This package also defines gradients for `prod` (overwriting an incorrect one) and `cumprod`, 
as in [this PR](https://github.com/FluxML/Flux.jl/pull/524). 

### `Array_`

An experiment with [LRUCache](https://github.com/JuliaCollections/LRUCache.jl) for working space:

```julia
x = rand(2000)' # turns off below this size

copy_(:copy, x)
similar_(:sim, x)
Array_{Float64}(:new, 5,1000) # @btime 200 ns, 32 bytes

inv_(:inv, x) # most of the _ functions can opt-in
```

### `@dropdims`

This macro wraps reductions like `sum(A; dims=...)` in `dropdims()`.
It understands things like this:

```julia
@dropdims sum(10 .* randn(2,10); dims=2) do x
    trunc(Int, x)
end
```

### See Also

* [Vectorize.jl](https://github.com/rprechelt/Vectorize.jl) is a more comprehensive wrapper. 

* [Strided.jl](https://github.com/Jutho/Strided.jl) adds `@threads` to broadcasting. 

* [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl) adds AVX black magic.
