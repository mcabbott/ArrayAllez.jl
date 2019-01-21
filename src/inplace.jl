
#========== exp! and log! ==========#

export exp0, exp_, exp!, exp!!
export log0, log_, log!, log!!

"""
    exp!(A)
    exp_(A) = exp!(similar(A), A)
    exp0(A) ≈ exp.(A)
Element-wise in-place exponential, and friends.
Multi-threaded when `length(A) >= 100`.
Will be handled by `Yeppp` or `AppleAccelerate` if you load one of them. 
"""
function exp! end

exp0(A) = similar(A) .= exp.(A) # maps Adjoint -> Adjoint etc

@doc @doc(exp!)
exp_(A) = exp!(similar(A), A)
exp!(A) = exp!(A, A)
exp!!(A) = exp!(A) # differs in gradient

function exp!(B, A)
    @assert size(A)==size(B)
    if length(A) < 100
        B .= exp.(A)
    else
        Threads.@threads for I in eachindex(A)
            @inbounds B[I] = exp1(A[I])
        end
    end
    B
end

"""
    log!(A)
    log_(A) ≈ log!(similar(A), A)
    log0(A) = log.(A)
Element-wise in-place natural logarithm, and friends.
Multi-threaded when `length(A) >= 100`.
Will be handled by `Yeppp` or `AppleAccelerate` if you load one of them.
"""
function log! end

log0(A) = similar(A) .= log.(A)

@doc @doc(log!)
log_(A) = log!(similar(A), A)
log!(A) = log!(A, A)
log!!(A) = log!(A) # differs in gradient

function log!(B, A)
    @assert size(A)==size(B)
    if length(A) < 100
        B .= log.(A)
    else
        Threads.@threads for I in eachindex(A)
            @inbounds B[I] = log1(A[I])
        end
    end
    B
end

# These are a little faster than Julia's built-in functions?
exp1(x::Float64) = ccall(:exp, Cdouble, (Cdouble,), x)
exp1(x) = exp(x)
log1(x::Float64) = ccall(:log, Cdouble, (Cdouble,), x)
log1(x) = log(x)

# Versions which use cache
exp_(name::Symbol, A) = exp!(similar_(name, A), A)
log_(name::Symbol, A) = log!(similar_(name, A), A)

#========== inv! and scale! ==========#

export inv0, inv_, inv!, inv!!
export scale0, scale_, scale!, scale!!
export iscale0, iscale_, iscale!, iscale!!

using LinearAlgebra: Adjoint, Transpose

const ARVector = Union{Adjoint{<:Any, <:AbstractVector}, Transpose{<:Any, <:AbstractVector}}
const RVector = Union{Adjoint{<:Any, <:Vector}, Transpose{<:Any, <:Vector}}

"""
    inv!(A) ≈ 1 ./ A
    inv!(A, b::Number) ≈ b ./ A
And `inv_(A)` which copies, and `inv0(A)` simple broadcasting.
Multi-threaded when `length(A) >= 1000`.
Will be handled by `AppleAccelerate` if you load it.
"""
function inv! end

inv0(A::AbstractArray, b::Number=1) = similar(A) .= b ./ A # maps Adjoint -> Adjoint etc

@doc @doc(inv!)
inv_(b::Number) = 1/b # for iscale_
inv_(A::AbstractArray, b::Number=1) = inv!(similar(A), A, b)

inv!(b::Number) = 1/b
function inv!(C::AbstractArray, A::AbstractArray, b::Number=1)
    @assert size(A)==size(C)
    if length(A) < 1000
        C .= b ./ A
    else
        Threads.@threads for I in eachindex(A)
            @inbounds C[I] = b / A[I]
        end
    end
    A
end

inv_(name::Symbol, A::AbstractArray, b::Number=1) = inv!(similar_(name, A), A, b)

"""
    scale!(A, b::Number) ≈ A .* b
    scale!(A, v::Vector) ≈ A .* v       # A::Matrix
    scale!(A, r::Adjoint) ≈ A .* r      # r::RowVector / Transpose etc.
    scale!(A, B) ≈ A .* B
For each of these, there is also also `scale_(A, ...)` non-mutating but perhaps accellerated,
and `scale0(A, ...)` simple broadcasting.
"""
function scale! end

using LinearAlgebra

scale0(A::AbstractArray, b) = similar(A) .= A .* b
scale_(A::Array, b::Number) = rmul!(copy(A), b)
scale!(A::Array, b::Number) = rmul!(A, b)
scale!!(A::Array, b) = scale!(A,b) ## differs in gradient

@doc @doc(scale!)
scale_(A::Matrix, v::Vector) = lmul!(Diagonal(v), copy(A))
scale!(A::Matrix, v::Vector) = lmul!(Diagonal(v), A)

scale_(A::Matrix, r::RVector) = rmul!(copy(A), Diagonal(transpose(r)))
scale!(A::Matrix, r::RVector) = rmul!(A, Diagonal(transpose(r)))

scale_(A::AbstractArray{T,N}, B::AbstractArray{T,N}) where {T,N} = similar(A) .= A .* B
scale!(A::AbstractArray{T,N}, B::AbstractArray{T,N}) where {T,N} = A .= A .* B

# scale0(A::AbstractArray, b, cdef...) = Broadcast.broadcast(*,A, b, cdef...)
# scale_(A::AbstractArray, b, cdef...) = scale_(scale_(A, b), cdef...)
# scale!(A::AbstractArray, b, cdef...) = scale!(scale!(A, b), cdef...)

scale_(name::Symbol, A::Array, b::Number) = rmul!(copy_(name, A), b)
scale_(name::Symbol, A::Matrix, v::Vector) = lmul!(Diagonal(v), copy_(name, A))
scale_(name::Symbol, A::Matrix, r::RVector) = rmul!(copy_(name, A), Diagonal(transpose(r)))
scale_(name::Symbol, A::AbstractArray{T,N}, B::AbstractArray{T,N}) where {T,N} = similar_(name, A) .= A .* B
scale_(name::Symbol, A::AbstractArray, b, cdef...) = scale_(name, scale_(name, A, b), cdef...)


"""
    iscale!(A, b::Number) ≈ A ./ b
    iscale!(A, v::Vector) ≈ A ./ v      # A::Matrix
    iscale!(A, r::Adjoint) ≈ A ./ r     # r::RowVector / Transpose etc.
    iscale!(A, B) ≈ A ./ B
For each of these, there is also `iscale_(A, ...)` non-mutating but perhaps accellerated,
and `iscale0(A, ...)` simple broadcasting.
Finally there is `iscale!!(A, x)` which mutate both arguments, wihch may be a terrible idea.
"""
function iscale! end

iscale0(A::AbstractArray, b) = similar(A) .= A ./ b

@doc @doc(iscale!)
iscale_(A::AbstractArray, b) = scale_(A, inv_(b))
iscale!(A::AbstractArray, b) = scale!(A, inv_(b))
iscale!!(A::AbstractArray, b) = scale!(A, inv!(b))

iscale_(name::Symbol, A::AbstractArray, b) = scale_(name, A, inv_(name, b))

#========== sum_ ==========#

export sum_

sum_(A::AbstractArray) = sum(A) # differs only in backward pass

#========== Accelerators ==========#

const CFloat = Union{Float64, Float32}
const CFloatArray{N} = Array{<:CFloat, N}
const CFloatMatrix = Matrix{<:CFloat}

IVERBOSE = false

VEC = ""
function load_note(str)
    global VEC
    if VEC == ""
        @info "ArrayAllez loaded code for $str"
        VEC = str
    else
        @warn "ArrayAllez loaded code for $str, perhaps overwriting $VEC"
        VEC *= " then $str"
    end
end

using Requires

@init @require Yeppp = "6310b701-1812-5374-a82f-9f6f2d54a40a" begin
    using .Yeppp

    # exp_(A::CFloatArray) = Yeppp.exp(A)
    exp!(B::CFloatArray, A::CFloatArray) = Yeppp.exp!(B, A)

    # log_(A::CFloatArray) = Yeppp.log(A)
    log!(B::CFloatArray, A::CFloatArray) = Yeppp.log!(B, A)

    scale_(A::Array{T,N}, B::Array{T,N}) where {T<:CFloat,N} = Yeppp.multiply(A,B)
    scale!(A::Array{T,N}, B::Array{T,N}) where {T<:CFloat,N} = Yeppp.multiply!(A,A,B)

    IVERBOSE && load_note("Yeppp")
end

@init @require AppleAccelerate = "13e28ba4-7ad8-5781-acae-3021b1ed3924" begin
    using .AppleAccelerate

    exp_(A::CFloatArray) = AppleAccelerate.exp(A)
    exp!(B::CFloatArray, A::CFloatArray) = AppleAccelerate.exp!(B, A)

    log_(A::CFloatArray) = AppleAccelerate.log(A)
    log!(B::CFloatArray, A::CFloatArray) = AppleAccelerate.log!(B, A)

    inv_(A::CFloatArray) = AppleAccelerate.rec(A)
    inv!(A::CFloatArray) = AppleAccelerate.rec!(A, A)

    scale_(A::Vector{T}, B::Vector{T}) where {T<:CFloat} = AppleAccelerate.vmul(A,B)
    scale!(A::Vector{T}, B::Vector{T}) where {T<:CFloat} = AppleAccelerate.vmul!(A,A,B)

    scale_(A::Array{T,N}, B::Array{T,N}) where {T<:CFloat,N} =
        begin AppleAccelerate.vmul(vec(A),vec(B)); A end  # vmul is literally only vectors
    scale!(A::Array{T,N}, B::Array{T,N}) where {T<:CFloat,N} =
        begin AppleAccelerate.vmul!(vec(A),vec(A),vec(B)); A end

    iscale_(A::Vector{T}, B::Vector{T}) where {T<:CFloat} = AppleAccelerate.vdiv(A,B)
    iscale!(A::Vector{T}, B::Vector{T}) where {T<:CFloat} = AppleAccelerate.vdiv!(A,A,B)

    iscale_(A::Array{T,N}, B::Array{T,N}) where {T<:CFloat,N} =
        begin AppleAccelerate.vdiv(vec(A),vec(B)); A end
    iscale!(A::Array{T,N}, B::Array{T,N}) where {T<:CFloat,N} =
        begin AppleAccelerate.vdiv!(vec(A),vec(A),vec(B)); A end

    IVERBOSE && load_note("AppleAccelerate")
end

#========== The End ==========#
