
#=
On Julia 1.0, it was worth using threaded loop for exp & log above about 100.
But on 1.2 & 1.3, it looks like it only pays above about 5000.
And once you have @avx, the crossing must be above 2^20
=#

using ArrayAllez, BenchmarkTools
Threads.nthreads()
ArrayAllez.TH_EXP

using LoopVectorization
using AppleAccelerate

function exp_t(A)
    B = similar(A)
    Threads.@threads for I in eachindex(A)
        @inbounds B[I] = exp(A[I])
    end
    B
end
function exp_x(A)
    B = similar(A)
    @avx for I in eachindex(A)
        B[I] = exp(A[I])
    end
    B
end


times_exp = []
@time for p in 20 #6:2:14
    r = rand(2^p)
    t0 = 1e6 * @belapsed exp0($r)
    t1 = 1e6 * @belapsed exp_t($r)
    t2 = 1e6 * @belapsed exp_x($r)
    t3 = 1e6 * @belapsed exp_($r)
    push!(times_exp, (length = 2^p, bcast = t0, thread = t1, avx = t2, lib = t3))
end
times_exp # in micro-seconds

function log_t(A)
    B = similar(A)
    Threads.@threads for I in eachindex(A)
        @inbounds B[I] = log(A[I])
    end
    B
end
function log_x(A)
    B = similar(A)
    @avx for I in eachindex(A)
        B[I] = log(A[I])
    end
    B
end

times_log = []
@time for p in 4:2:8 # 18:2:20 # 6:2:14
    r = rand(2^p)
    t0 = 1e6 * @belapsed log0($r)
    t1 = 1e6 * @belapsed log_t($r)
    t2 = 1e6 * @belapsed log_x($r)
    t3 = 1e6 * @belapsed log_($r)
    push!(times_log, (length = 2^p, bcast = t0, thread = t1, avx = t2, lib = t3))
end
times_log

