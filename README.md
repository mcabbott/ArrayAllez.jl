# ArrayAllez.jl

[![Build Status](https://travis-ci.org/mcabbott/ArrayAllez.jl.svg?branch=master)](https://travis-ci.org/mcabbott/ArrayAllez.jl)

```
] add https://github.com/mcabbott/ArrayAllez.jl

add  Yeppp  Flux  https://github.com/platawiec/AppleAccelerate.jl#julia07
```

## `log! ∘ exp!`

Convenient vectorised operations... from [Yeppp!](https://github.com/JuliaMath/Yeppp.jl) or [AppleAccelerate](https://github.com/JuliaMath/AppleAccelerate.jl), or just using `@threads`: 

```julia
x = rand(5);

y = exp.(x)

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

## `∇`

These commands all define `Flux` gradients.

```julia
using Flux
x = param(rand(5))

y = exp!(x)

Flux.back!(sum(y))
x.grad
```
There is also an `exp!!` which mutates both its forward input and its backward gradient, which may be a terrible idea.

## `Array_`

An experiment with [LRUCache](https://github.com/JuliaCollections/LRUCache.jl):

```julia
x = rand(2000)' # below this size, falls back

copy_(:copy, x)
similar_(:sim, x)
Array_{Float64}(:new, 2,1000)

inv_(:inv, x) # all the _ functions can opt-in
```
