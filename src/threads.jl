
"""
This is the replacement for Julia's `@threads` macro proposed in
https://github.com/JuliaLang/julia/pull/35003
"""
macro threads(args...)
    na = length(args)
    if na != 1
        throw(ArgumentError("wrong number of arguments in @threads"))
    end
    ex = args[1]
    if !isa(ex, Expr)
        throw(ArgumentError("need an expression argument to @threads"))
    end
    if ex.head === :for
        if ex.args[1] isa Expr && ex.args[1].head === :(=)
            return _threadsfor(ex.args[1], ex.args[2])
        else
            throw(ArgumentError("nested outer loops are not currently supported by @threads"))
        end
    else
        throw(ArgumentError("unrecognized argument to @threads"))
    end
end

function _threadsfor(iter_stmt, lbody)
    loopvar   = iter_stmt.args[1]
    iter      = iter_stmt.args[2]
    rng = gensym(:rng)
    out = quote
        Base.@sync for $rng in $(Iterators.partition)($iter, $(length)($iter) ÷ $(nthreads)())
            Base.Threads.@spawn begin
                Base.@sync for $loopvar in $rng
                    $lbody
                end
            end
        end
    end
    esc(out)
end


