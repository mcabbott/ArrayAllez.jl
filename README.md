# ArrayAllez.jl

[![Travis CI](https://travis-ci.com/mcabbott/ArrayAllez.jl.svg?branch=master)](https://travis-ci.com/mcabbott/ArrayAllez.jl)
[![Github CI](https://github.com/mcabbott/ArrayAllez.jl/workflows/CI/badge.svg)](https://github.com/mcabbott/ArrayAllez.jl/actions?query=workflow%3ACI+branch%3Amaster)

```
] add ArrayAllez
```

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

### Removed

This package used to provide two functions generalising matrix multiplication. They are now better handled by other packages:

* `TensorCore.boxdot` contracts neighbours: `rand(2,3,5) ⊡ rand(5,7,11) |> size == (2,3,7,11)`
* `NNlib.batched_mul` keeps a batch dimension: `rand(2,3,10) ⊠ rand(3,5,10) |> size == (2,5,10)`

### See Also

* [Vectorize.jl](https://github.com/rprechelt/Vectorize.jl) is a more comprehensive wrapper. 

* [Strided.jl](https://github.com/Jutho/Strided.jl) adds `@threads` to broadcasting. 

* [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl) adds AVX black magic.
