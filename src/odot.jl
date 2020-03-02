
#========== odot ==========#

export odot, ⊙, dotmul!

"""
    odot(A,B) = A ⊙ B    # \\odot

Generalised matrix multiplication: Contracts the last index of `A` with the first index of `B`,
for any `ndims(A)` & `ndims(B)`. If both are vectors, then it returns a scalar `== sum(A .* B)`.
Left-associative `A⊙B⊙C = (A⊙B)⊙C` like `*`.
```
julia> A = rand(3,4,5); B = rand(5,6,7);

julia> size(A ⊙ B)
(3, 4, 6, 7)

julia> typeof(rand(5) ⊙ rand(5))
Float64
```
See also `dotmul!(Y,A,B)`, which is to `⊙` as `mul!` is to `*`.
And `bmm == ⨱` for batched matrix multiplication.
"""
odot(A::AbstractArray, B::AbstractArray) =
    reshape(_squash_left(A) * _squash_right(B), _odot_size(A,B))

const ⊙ = odot

_squash_left(A::AbstractArray) = reshape(A, :,size(A,ndims(A)))
_squash_left(A::AbstractMatrix) = A

_squash_right(B::AbstractArray) = reshape(B, size(B,1),:)
_squash_right(B::AbstractVecOrMat) = B

_odot_size(A::AbstractArray{T,N}, B::AbstractArray{S,M}) where {T,N,S,M} =
    ntuple(i -> i<N ? size(A, i) : size(B, i-N+2), Val(N+M-2))

# These can skip final reshape:
odot(A::AbstractMatrix, B::AbstractVecOrMat) = A*B

# These produce scalar output:
odot(A::AbstractVector{<:Real}, B::AbstractVector) = dot(A,B)
odot(A::AbstractVector{<:Number}, B::AbstractVector) = transpose(A)*B
odot(A::AbstractVector, B::AbstractVector) = first(permutedims(A)*B)

# Multiplication by a scalar:
odot(A::AbstractArray, b::Number) = A*b
odot(a::Number, B::AbstractArray) = a*B
odot(a::Number, b::Number) = a*b

"""
    dotmul!(Y, A, B)

In-place version of `odot`, i.e. `Y .= A ⊙ B`.
"""
function dotmul!(Y::AbstractArray, A::AbstractArray, B::AbstractArray)
    sz = prod(size(A)[1:end-1]), prod(size(B)[2:end])
    mul!(reshape(Y, sz), _squash_left(A), _squash_right(B))
    Y
end

function dotmul!(Y::AbstractArray, A::AbstractArray, B::AbstractArray, α::Number, β::Number=zero(eltype(Y)))
    sz = prod(size(A)[1:end-1]), prod(size(B)[2:end])
    mul!(reshape(Y, sz), _squash_left(A), _squash_right(B), α, β)
    Y
end

#========== names ==========#

using NamedDims

odot(A::NamedDimsArray, B::NamedDimsArray) = named_odot(A, B)
odot(A::NamedDimsArray, B::AbstractArray) = named_odot(A, B)
odot(A::AbstractArray, B::NamedDimsArray) = named_odot(A, B)

function named_odot(A, B)
    LA, LB = NamedDims.dimnames(A), NamedDims.dimnames(B)
    last(LA) == first(LB) || throw(ArgumentError(string(
        "contracted names must match! got ", LA, " ⊙ ", LB )))

    C = odot(NamedDims.unname(A), NamedDims.unname(B))
    C isa AbstractArray || return C # scalar case

    LC = ntuple(ndims(A) + ndims(B) - 2) do d
        d < ndims(A) ? LA[d] : LB[d-ndims(A)+2]
    end
    NamedDimsArray(C, LC)
end

#========== left + right ==========#

export *ˡ, *ʳ, ⊙ˡ, ⊙ʳ

lrdoc = """
    P *ˡ X  # \\^l
    X *ʳ P  # \\^r

These are just ordinary multiplication, but differ in that their Zygote gradients
assume only `P` needs to be tracked, whether on the left or the right,
and `X` is a constant.

    P ⊙ˡ X
    X ⊙ʳ P

Similar, but with `odot` instead of `*`.
"""

@doc lrdoc
*ˡ(x,y) = *(x,y)
@doc lrdoc
*ʳ(x,y) = *(x,y)

using ZygoteRules: @adjoint

@adjoint (A *ˡ B) = (A * B), Δ -> (Δ * B', nothing)
@adjoint (A *ʳ B) = (A * B), Δ -> (nothing, A' * Δ)

dot_l(x::AbstractArray, y::AbstractArray) = dot(x, y)
dot_r(x::AbstractArray, y::AbstractArray) = dot(x, y)
@adjoint dot_l(x::AbstractArray, y::AbstractArray) = dot(x, y), Δ->(Δ .* y, nothing)
@adjoint dot_r(x::AbstractArray, y::AbstractArray) = dot(x, y), Δ->(nothing, Δ .* x)

@doc lrdoc
⊙ˡ(A::AbstractArray, B::AbstractArray) =
    reshape(_squash_left(A) *ˡ _squash_right(B), _odot_size(A,B))

@doc lrdoc
⊙ʳ(A::AbstractArray, B::AbstractArray) =
    reshape(_squash_left(A) *ʳ _squash_right(B), _odot_size(A,B))

# These can skip final reshape:
⊙ˡ(A::AbstractMatrix, B::AbstractVecOrMat) = A *ˡ B
⊙ʳ(A::AbstractMatrix, B::AbstractVecOrMat) = A *ʳ B

# These produce scalar output:
⊙ˡ(A::AbstractVector{<:Real}, B::AbstractVector) = dot_l(A, B)
⊙ʳ(A::AbstractVector{<:Real}, B::AbstractVector) = dot_r(A, B)
⊙ˡ(A::AbstractVector{<:Number}, B::AbstractVector) = transpose(A) *ˡ B
⊙ʳ(A::AbstractVector{<:Number}, B::AbstractVector) = transpose(A) *ʳ B
⊙ˡ(A::AbstractVector, B::AbstractVector) = first(permutedims(A) *ˡ B)
⊙ʳ(A::AbstractVector, B::AbstractVector) = first(permutedims(A) *ʳ B)

# Multiplication by a scalar:
⊙ˡ(A::AbstractArray, b::Number) = A *ˡ b
⊙ʳ(A::AbstractArray, b::Number) = A *ʳ b
⊙ˡ(a::Number, B::AbstractArray) = a *ˡ B
⊙ʳ(a::Number, B::AbstractArray) = a *ʳ B
⊙ˡ(a::Number, b::Number) = a *ˡ b
⊙ʳ(a::Number, b::Number) = a *ʳ b

