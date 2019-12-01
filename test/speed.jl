
#=
On Julia 1.0, it was worth using threaded loop for exp & log above about 100.
But on 1.2 & 1.3, it looks like it only pays above about 5000.
=#

using ArrayAllez, BenchmarkTools
Threads.nthreads()
ArrayAllez.TH_EXP

function exp_t(A)
    B = similar(A)
    Threads.@threads for I in eachindex(A)
        @inbounds B[I] = exp(A[I])
    end
    B
end

times_exp = []
@time for p in 6:2:14
    r = rand(2^p)
    t0 = 1e6 * @belapsed exp0($r)
    t1 = 1e6 * @belapsed exp_t($r)
    t2 = 1e6 * @belapsed exp_($r)
    push!(times_exp, (length = 2^p, bcast = t0, thread = t1, lib = t2))
end
times_exp

function log_t(A)
    B = similar(A)
    Threads.@threads for I in eachindex(A)
        @inbounds B[I] = log(A[I])
    end
    B
end

times_log = []
@time for p in 6:2:14
    r = rand(2^p)
    t0 = 1e6 * @belapsed log0($r)
    t1 = 1e6 * @belapsed log_t($r)
    t2 = 1e6 * @belapsed log_($r)
    push!(times_log, (length = 2^p, bcast = t0, thread = t1, lib = t2))
end
times_log

