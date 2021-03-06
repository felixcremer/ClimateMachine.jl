module VariableTemplates

export varsize, Vars, Grad, @vars, varsindex

using StaticArrays

function varsindex(::Type{S}, insym::Symbol) where {S <: NamedTuple}
    offset = 0
    for varsym in fieldnames(S)
        T = fieldtype(S, varsym)
        if T <: Real
            offset += 1
            varrange = offset:offset
        elseif T <: SHermitianCompact
            LT = StaticArrays.lowertriangletype(T)
            N = length(LT)
            varrange = offset .+ (1:N)
            offset += N
        elseif T <: StaticArray
            N = length(T)
            varrange = offset .+ (1:N)
            offset += N
        else
            varrange = offset .+ (1:varsize(T))
            offset += varsize(T)
        end
        if insym == varsym
            return varrange
        end
    end
    error("symbol '$insym' not found")
end

"""
    varsize(S)

The number of elements specified by the template type `S`.
"""
varsize(::Type{T}) where {T <: Real} = 1
varsize(::Type{Tuple{}}) = 0
varsize(::Type{NamedTuple{(), Tuple{}}}) = 0
varsize(::Type{SVector{N, T}}) where {N, T <: Real} = N

include("var_names.jl")

# TODO: should be possible to get rid of @generated
@generated function varsize(::Type{S}) where {S}
    types = fieldtypes(S)
    isempty(types) ? 0 : sum(varsize, types)
end

function process_vars!(syms, typs, expr)
    if expr isa LineNumberNode
        return
    elseif expr isa Expr && expr.head == :block
        for arg in expr.args
            process_vars!(syms, typs, arg)
        end
        return
    elseif expr.head == :(::)
        push!(syms, expr.args[1])
        push!(typs, expr.args[2])
        return
    else
        error("Invalid expression")
    end
end

macro vars(args...)
    syms = Any[]
    typs = Any[]
    for arg in args
        process_vars!(syms, typs, arg)
    end
    :(NamedTuple{$(tuple(syms...)), Tuple{$(esc.(typs)...)}})
end

struct GetVarError <: Exception
    sym::Symbol
end
struct SetVarError <: Exception
    sym::Symbol
end


"""
    Vars{S,A,offset}(array::A)

Defines property overloading for `array` using the type `S` as a template. `offset` is used to shift the starting element of the array.
"""
struct Vars{S, A, offset}
    array::A
end
Vars{S}(array) where {S} = Vars{S, typeof(array), 0}(array)

Base.parent(v::Vars) = getfield(v, :array)
Base.eltype(v::Vars) = eltype(parent(v))
Base.propertynames(::Vars{S}) where {S} = fieldnames(S)
Base.similar(v::Vars{S, A, offset}) where {S, A, offset} =
    Vars{S, A, offset}(similar(parent(v)))

@generated function Base.getproperty(
    v::Vars{S, A, offset},
    sym::Symbol,
) where {S, A, offset}
    expr = quote
        Base.@_inline_meta
        array = parent(v)
    end
    for k in fieldnames(S)
        T = fieldtype(S, k)
        if T <: Real
            retexpr = :($T(array[$(offset + 1)]))
            offset += 1
        elseif T <: SHermitianCompact
            LT = StaticArrays.lowertriangletype(T)
            N = length(LT)
            retexpr = :($T($LT($([:(array[$(offset + i)]) for i in 1:N]...))))
            offset += N
        elseif T <: StaticArray
            N = length(T)
            retexpr = :($T($([:(array[$(offset + i)]) for i in 1:N]...)))
            offset += N
        else
            retexpr = :(Vars{$T, A, $offset}(array))
            offset += varsize(T)
        end
        push!(expr.args, :(
            if sym == $(QuoteNode(k))
                return @inbounds $retexpr
            end
        ))
    end
    push!(expr.args, :(throw(GetVarError(sym))))
    expr
end

@generated function Base.setproperty!(
    v::Vars{S, A, offset},
    sym::Symbol,
    val,
) where {S, A, offset}
    expr = quote
        Base.@_inline_meta
        array = parent(v)
    end
    for k in fieldnames(S)
        T = fieldtype(S, k)
        if T <: Real
            retexpr = :(array[$(offset + 1)] = val)
            offset += 1
        elseif T <: SHermitianCompact
            LT = StaticArrays.lowertriangletype(T)
            N = length(LT)
            retexpr = :(
                array[($(offset + 1)):($(offset + N))] .= $T(val).lowertriangle
            )
            offset += N
        elseif T <: StaticArray
            N = length(T)
            retexpr = :(array[($(offset + 1)):($(offset + N))] .= val[:])
            offset += N
        else
            offset += varsize(T)
            continue
        end
        push!(expr.args, :(
            if sym == $(QuoteNode(k))
                return @inbounds $retexpr
            end
        ))
    end
    push!(expr.args, :(throw(SetVarError(sym))))
    expr
end

"""
    Grad{S,A,offset}(array::A)

Defines property overloading along slices of the second dimension of `array` using the type `S` as a template. `offset` is used to shift the starting element of the array.
"""
struct Grad{S, A, offset}
    array::A
end
Grad{S}(array) where {S} = Grad{S, typeof(array), 0}(array)

Base.parent(g::Grad) = getfield(g, :array)
Base.eltype(g::Grad) = eltype(parent(g))
Base.propertynames(::Grad{S}) where {S} = fieldnames(S)
Base.similar(g::Grad{S, A, offset}) where {S, A, offset} =
    Grad{S, A, offset}(similar(parent(g)))

@generated function Base.getproperty(
    v::Grad{S, A, offset},
    sym::Symbol,
) where {S, A, offset}
    if A <: SubArray
        M = size(fieldtype(A, 1), 1)
    else
        M = size(A, 1)
    end
    expr = quote
        Base.@_inline_meta
        array = parent(v)
    end
    for k in fieldnames(S)
        T = fieldtype(S, k)
        if T <: Real
            retexpr = :(SVector{$M, $T}(
                $([:(array[$i, $(offset + 1)]) for i in 1:M]...),
            ))
            offset += 1
        elseif T <: StaticArray
            N = length(T)
            retexpr = :(SMatrix{$M, $N, $(eltype(T))}(
                $([:(array[$i, $(offset + j)]) for i in 1:M, j in 1:N]...),
            ))
            offset += N
        else
            retexpr = :(Grad{$T, A, $offset}(array))
            offset += varsize(T)
        end
        push!(expr.args, :(
            if sym == $(QuoteNode(k))
                return @inbounds $retexpr
            end
        ))
    end
    push!(expr.args, :(throw(GetVarError(sym))))
    expr
end

@generated function Base.setproperty!(
    v::Grad{S, A, offset},
    sym::Symbol,
    val,
) where {S, A, offset}
    if A <: SubArray
        M = size(fieldtype(A, 1), 1)
    else
        M = size(A, 1)
    end
    expr = quote
        Base.@_inline_meta
        array = parent(v)
    end
    for k in fieldnames(S)
        T = fieldtype(S, k)
        if T <: Real
            retexpr = :(array[:, $(offset + 1)] .= val)
            offset += 1
        elseif T <: StaticArray
            N = length(T)
            retexpr = :(array[:, ($(offset + 1)):($(offset + N))] .= val)
            offset += N
        else
            offset += varsize(T)
            continue
        end
        push!(expr.args, :(
            if sym == $(QuoteNode(k))
                return @inbounds $retexpr
            end
        ))
    end
    push!(expr.args, :(throw(SetVarError(sym))))
    expr
end

end # module
