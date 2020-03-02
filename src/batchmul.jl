#========== batchmul ==========#

export bmm, ⨱, batchmul!

"""
    Y = bmm(A,B)  # Y[i,k,b...] := A[i,j,b...] * B[j,k,b...]
    Y = A ⨱ B     # \\timesbar

Batched matrix multiplication.

Given two arrays of the same `ndims`, the first two dimensions of each refer to the matrices,
while all trailing dimensions are batch indices, and must agree. If the second array has
one less dimension, then this it is regarded as a batch of vectors instead.

This just inserts the necessary reshapes before calling `NNlib.batched_mul`.
And it allows you to indicate transposed matrices by `Transpose`.

```
julia> A, B = rand(2,3,8,9), rand(3,4,8,9);

julia> size(A ⨱ B)
(2, 4, 8, 9)

julia> (A ⨱ B)[:,:,8,9] == A[:,:,8,9] * B[:,:,8,9]
true

julia> X, Y = rand(2,3,8,9), rand(2,8,9);

julia> size(Transpose(X) ⨱ Y)
(3, 8, 9)

julia> Transpose(ones(2,3,8,9))
3×2×8×9 Transpose{Float64,Array{Float64,4}} ... just an input flag for bmm / batchmul!
```

See also `odot == ⊙` for contraction of neighbouring indices instead.
"""
function bmm end

const ⨱ = bmm

# step 1 is to remove Adjoint or Transpose

_BMM_TYPES = [(:AbstractArray, identity, identity),
    (:Transpose, transpose, parent),
    (:Adjoint, adjoint, parent)
]
for (AT,f,v) in _BMM_TYPES, (BT,g,w) in _BMM_TYPES
    @eval bmm(A::$AT, B::$BT) = _bmm($f, $v(A), $g, $w(B))
end

# step 2 is to reshape

for N in 4:10

    @eval function _bmm(f::Function, A::AbstractArray{TA,$N}, g::Function, B::AbstractArray{TB,$N}) where {TA,TB}
        Cdims = ntuple($N-2) do d
            size(A, d+2) == size(B, d+2) || throw(DimensionMismatch(string(
                "batch dimensions must match, got ", size(A), " ⨱ ", size(B))))
            size(A, d+2)
        end
        A3 = reshape(A, size(A,1), size(A,2), :)
        B3 = reshape(B, size(B,1), size(B,2), :)
        C3 = _bmm(f, A3, g, B3)
        reshape(C3, size(C3,1), size(C3,2), Cdims...)
    end

end
for N in 2:10

    @eval function _bmm(f::Function, A::AbstractArray{TA,$N}, g::Function, B::AbstractArray{TB,$(N-1)}) where {TA,TB}
        Cdims = ntuple($N-2) do d
            size(A, d+2) == size(B, d+1) || throw(DimensionMismatch(
                "mismatch in input sizes, in dimension $(d+2)"))
            size(A, d+2)
        end
        A3 = reshape(A, size(A,1), size(A,2), :)
        B3 = reshape(B, size(B,1), 1, :)
        C3 = _bmm(f, A3, g, B3)
        reshape(C3, size(C3,1), Cdims...)
    end

end

# step 3 is to call whatever actually does the work

using NNlib

_bmm(f::Function, A::AbstractArray{TA,2}, g::Function, B::AbstractArray{TB,2}) where {TA,TB} =
    f(A) * g(B)

_bmm(f::Function, A::AbstractArray{TA,3}, g::Function, B::AbstractArray{TB,3}) where {TA,TB} =
    NNlib.batched_mul(_wrap(f,A), _wrap(g,B))

_wrap(::typeof(identity), A) = A
_wrap(::typeof(transpose), A) = NNlib.batched_transpose(A)
_wrap(::typeof(adjoint), A) = NNlib.batched_adjoint(A)

for N in 3:10, AT in [:Adjoint, :Transpose]
    @eval begin
        Base.size(A::$AT{T,Array{T,$N}} where {T}) =
            (size(parent(A),2), size(parent(A),1), size(parent(A))[3:end]...)

        Base.show(io::IO, m::MIME"text/plain", A::$AT{T,Array{T,$N}} where {T}) =
            println(io, summary(A), "\nJust as a marker for ⨱, i.e. ArrayAllez.bmm")
    end
end

#=
"""
    batchmul!(Y,A,B)      # Y[..,k] .= A[..,k] * B[..,k]
    batchmul!(Y,A,B,α,β)  # Y[..,k] .= α .* A[..,k] * B[..,k] .+ β .* Y[..,k]

Batched matrix multiplication. Calls `batched_gemm!` when possible,
otherwise falls back to slicing. Indices beyond `A`'s 3rd are additional batch dimensions.
Tries to stick closely to `mul!` as possible.

To transpose `A` or `B` on the un-batched indices, call `batchmul!(Y,Transpose(A),B)` etc.
These illegal wrappers are unwrapped to pass `_batchmul!` some flags,
and ultimately `batched_gemm!` its letters `'N'` or `'T'`,
or `'C'` from `Adjoint(A)` for complex numbers.
"""
function batchmul! end

=#

#========== names ==========#

_bmm(f::Function, A::NamedDimsArray, g::Function, B::AbstractArray) = _bmm_named(f, A, g, B)
_bmm(f::Function, A::AbstractArray, g::Function, B::NamedDimsArray) = _bmm_named(f, A, g, B)
_bmm(f::Function, A::NamedDimsArray, g::Function, B::NamedDimsArray) = _bmm_named(f, A, g, B)

function _bmm_named(f, A, g, B)
    LA = _names(f, A)
    LB = _names(g, B)
    if LA[2] != :_ && LB[1] != :_
        LA[2] == LB[1] || throw(ArgumentError(string(
            "contracted names must match, got ", LA, " ⨱ ", LB)))
    end
    for d in 3:ndims(A)
        LA[d] == :_ && continue
        lB = LB[d+ndims(B)-ndims(A)]
        lB == :_ && continue
        LA[d] == lB || throw(ArgumentError(string(
            "batch names must match, got ", LA, " ⨱ ", LB)))
    end
    LC = ntuple(ndims(B)) do d
        d==1 && return LA[1]
        ndims(A) == ndims(B) + 1 && return LA[d+1]
        d==2 && return LB[2]
        LA[d]
    end
    C = _bmm(f, NamedDims.unname(A), g, NamedDims.unname(B))
    NamedDimsArray(C, LC)
end

_names(::typeof(identity), A) = dimnames(A)
function _names(::Union{typeof(transpose), typeof(adjoint)}, A)
    LA = dimnames(A)
    ntuple(ndims(A)) do d
        d==1 && return LA[2]
        d==2 && return LA[1]
        LA[d]
    end
end

#========== left, right ==========#

