# ArrayAllez.jl

[![Build Status](https://travis-ci.org/mcabbott/ArrayAllez.jl.svg?branch=master)](https://travis-ci.org/mcabbott/ArrayAllez.jl)

```
] add https://github.com/mcabbott/ArrayAllez.jl

add  Yeppp  Flux  https://github.com/platawiec/AppleAccelerate.jl#julia07
```

### `log! ∘ exp!`

This began as a way to more conveniently choose between [Yeppp!](https://github.com/JuliaMath/Yeppp.jl) 
and [AppleAccelerate](https://github.com/JuliaMath/AppleAccelerate.jl). Or neither, just `@threads`: 

```julia
x = rand(5);

y = exp.(x)  # = exp0(x) 

using Yeppp  # or using AppleAccelerate

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

These commands all make some attempt to define [Flux](https://github.com/FluxML/Flux.jl) gradients, 
but caveat emptor. There is also an `exp!!` which mutates both its forward input and its backward gradient, 
which may be a terrible idea.

```julia
using Flux
x = param(rand(5));
y = exp_(x)

Flux.back!(sum(exp!(x)))
x.grad
```

This package also defines gradients for `prod` (overwriting an incorrect one) and `cumprod`, 
as in [this PR](https://github.com/FluxML/Flux.jl/pull/524). 

### `Array_`

An experiment with [LRUCache](https://github.com/JuliaCollections/LRUCache.jl):

```julia
x = rand(2000)' # below this size, falls back

copy_(:copy, x)
similar_(:sim, x)
Array_{Float64}(:new, 5,1000) # @btime 200 ns, 32 bytes

inv_(:inv, x) # most of the _ functions can opt-in
```

### See Also

* [Vectorize.jl](https://github.com/rprechelt/Vectorize.jl) is a more comprehensive wrapper, including Intel MKL. 

* [Strided.jl](https://github.com/Jutho/Strided.jl) adds @threads to broadcasting. 


