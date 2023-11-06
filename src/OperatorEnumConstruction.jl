module OperatorEnumConstructionModule

import ..OperatorEnumModule: AbstractOperatorEnum, OperatorEnum, GenericOperatorEnum
import ..EquationModule: string_tree, Node
import ..EvaluateEquationModule: eval_tree_array
import ..EvaluateEquationDerivativeModule: eval_grad_tree_array, _zygote_gradient
import ..EvaluationHelpersModule: _grad_evaluator

"""Used to set a default value for `operators` for ease of use."""
@enum AvailableOperatorTypes::UInt8 begin
    IsNothing
    IsOperatorEnum
    IsGenericOperatorEnum
end

# These constants are purely for convenience. Internal code
# should make use of `Node`, `string_tree`, `eval_tree_array`,
# and `eval_grad_tree_array` directly.

const LATEST_OPERATORS = Ref{Union{Nothing,AbstractOperatorEnum}}(nothing)
const LATEST_OPERATORS_TYPE = Ref{AvailableOperatorTypes}(IsNothing)
const LATEST_UNARY_OPERATOR_MAPPING = Dict{Function,fieldtype(Node{Float64}, :op)}()
const LATEST_BINARY_OPERATOR_MAPPING = Dict{Function,fieldtype(Node{Float64}, :op)}()
const LATEST_MULTINARY_OPERATOR_MAPPING = Dict{Function,fieldtype(Node{Float64}, :op)}()
const ALREADY_DEFINED_UNARY_OPERATORS = (;
    operator_enum=Dict{Function,Bool}(), generic_operator_enum=Dict{Function,Bool}()
)
const ALREADY_DEFINED_BINARY_OPERATORS = (;
    operator_enum=Dict{Function,Bool}(), generic_operator_enum=Dict{Function,Bool}()
)
const ALREADY_DEFINED_MULTINARY_OPERATORS = (;
    operator_enum=Dict{Function,Bool}(), generic_operator_enum=Dict{Function,Bool}()
)
const LATEST_VARIABLE_NAMES = Ref{Vector{String}}(String[])

function Base.show(io::IO, tree::Node)
    latest_operators_type = LATEST_OPERATORS_TYPE.x
    if latest_operators_type == IsNothing
        return print(io, string_tree(tree; variable_names=LATEST_VARIABLE_NAMES.x))
    elseif latest_operators_type == IsOperatorEnum
        latest_operators = LATEST_OPERATORS.x::OperatorEnum
        return print(
            io, string_tree(tree, latest_operators; variable_names=LATEST_VARIABLE_NAMES.x)
        )
    else
        latest_operators = LATEST_OPERATORS.x::GenericOperatorEnum
        return print(
            io, string_tree(tree, latest_operators; variable_names=LATEST_VARIABLE_NAMES.x)
        )
    end
end
function (tree::Node)(X; kws...)
    Base.depwarn(
        "The `tree(X; kws...)` syntax is deprecated. Use `tree(X, operators; kws...)` instead.",
        :Node,
    )
    latest_operators_type = LATEST_OPERATORS_TYPE.x
    if latest_operators_type == IsNothing
        error("Please use the `tree(X, operators; kws...)` syntax instead.")
    elseif latest_operators_type == IsOperatorEnum
        latest_operators = LATEST_OPERATORS.x::OperatorEnum
        return tree(X, latest_operators; kws...)
    else
        latest_operators = LATEST_OPERATORS.x::GenericOperatorEnum
        return tree(X, latest_operators; kws...)
    end
end

function _grad_evaluator(tree::Node, X; kws...)
    Base.depwarn(
        "The `tree'(X; kws...)` syntax is deprecated. Use `tree'(X, operators; kws...)` instead.",
        :Node,
    )
    latest_operators_type = LATEST_OPERATORS_TYPE.x
    # return _grad_evaluator(tree, X, $operators; kws...)
    if latest_operators_type == IsNothing
        error("Please use the `tree'(X, operators; kws...)` syntax instead.")
    elseif latest_operators_type == IsOperatorEnum
        latest_operators = LATEST_OPERATORS.x::OperatorEnum
        return _grad_evaluator(tree, X, latest_operators; kws...)
    else
        error("Gradients are not implemented for `GenericOperatorEnum`.")
    end
end

function set_default_variable_names!(variable_names::Vector{String})
    return LATEST_VARIABLE_NAMES.x = variable_names
end

function create_evaluation_helpers!(operators::OperatorEnum)
    LATEST_OPERATORS.x = operators
    return LATEST_OPERATORS_TYPE.x = IsOperatorEnum
end
function create_evaluation_helpers!(operators::GenericOperatorEnum)
    LATEST_OPERATORS.x = operators
    return LATEST_OPERATORS_TYPE.x = IsGenericOperatorEnum
end

function lookup_op(@nospecialize(f), ::Val{degree}) where {degree}
    if degree == 1
        mapping = LATEST_UNARY_OPERATOR_MAPPING
    elseif degree == 2
        mapping = LATEST_BINARY_OPERATOR_MAPPING
    elseif degree > 2
        mapping = LATEST_MULTINARY_OPERATOR_MAPPING
    else
        # Should never happen
    end
    if !haskey(mapping, f)
        error(
            "Convenience constructor for `Node` using operator `$(f)` is out-of-date. " *
            "Please create an `OperatorEnum` (or `GenericOperatorEnum`) with " *
            "`define_helper_functions=true` and pass `$(f)`.",
        )
    end
    return mapping[f]
end

function _extend_unary_operator(f::Symbol, type_requirements)
    quote
        quote
            function $($f)(l::Node{T})::Node{T} where {T<:$($type_requirements)}
                return if (l.degree == 0 && l.constant)
                    Node(T; val=$($f)(l.val::T))
                else
                    latest_op_idx = $($lookup_op)($($f), Val(1))
                    Node(latest_op_idx, l)
                end
            end
        end
    end
end

function _extend_binary_operator(f::Symbol, type_requirements, build_converters)
    quote
        quote
            function $($f)(l::Node{T}, r::Node{T}) where {T<:$($type_requirements)}
                if (l.degree == 0 && l.constant && r.degree == 0 && r.constant)
                    Node(T; val=$($f)(l.val::T, r.val::T))
                else
                    latest_op_idx = $($lookup_op)($($f), Val(2))
                    Node(latest_op_idx, l, r)
                end
            end
            function $($f)(l::Node{T}, r::T) where {T<:$($type_requirements)}
                if l.degree == 0 && l.constant
                    Node(T; val=$($f)(l.val::T, r))
                else
                    latest_op_idx = $($lookup_op)($($f), Val(2))
                    Node(latest_op_idx, l, Node(T; val=r))
                end
            end
            function $($f)(l::T, r::Node{T}) where {T<:$($type_requirements)}
                if r.degree == 0 && r.constant
                    Node(T; val=$($f)(l, r.val::T))
                else
                    latest_op_idx = $($lookup_op)($($f), Val(2))
                    Node(latest_op_idx, Node(T; val=l), r)
                end
            end
            if $($build_converters)
                # Converters:
                function $($f)(
                    l::Node{T1}, r::Node{T2}
                ) where {T1<:$($type_requirements),T2<:$($type_requirements)}
                    T = promote_type(T1, T2)
                    l = convert(Node{T}, l)
                    r = convert(Node{T}, r)
                    return $($f)(l, r)
                end
                function $($f)(
                    l::Node{T1}, r::T2
                ) where {T1<:$($type_requirements),T2<:$($type_requirements)}
                    T = promote_type(T1, T2)
                    l = convert(Node{T}, l)
                    r = convert(T, r)
                    return $($f)(l, r)
                end
                function $($f)(
                    l::T1, r::Node{T2}
                ) where {T1<:$($type_requirements),T2<:$($type_requirements)}
                    T = promote_type(T1, T2)
                    l = convert(T, l)
                    r = convert(Node{T}, r)
                    return $($f)(l, r)
                end
            end
        end
    end
end

function _extend_multinary_operator(f::Symbol, type_requirements)####, build_converters)
    quote
        quote
            function $($f)(allargs::Union{Node{<:$($type_requirements)}, $($type_requirements)}...)
                if any(i->!(!(typeof(i)<:Node) || (i.degree == 0 && i.constant)), allargs) #### This evaluates to there if there are ANY non-constant (ie features) in here
                    latest_op_idx = $($lookup_op)($($f), Val(3))
                    Ns = []
                    for arg in allargs
                        if typeof(arg)<:Node
                            push!(Ns, arg)
                        else
                            push!(Ns, Node(;val=arg))
                        end
                    end
                    Node(latest_op_idx, Ns)
                else
                    #### All below are constant so we can evaluate
                    Vs = []
                    for arg in allargs
                        if typeof(arg)<:Node
                            push!(Vs, arg.val)
                        else
                            push!(Vs, val)
                        end
                    end
                    Node(; val=$($f)(Tuple(val for val in Vs)))
                end
            end
            # Lol I'm not doing the converters yet, maybe if I HAVE to to get it 
            #   to build or if I end up wanting to make this public I will...
            # if $($build_converters)
            #     # Converters:
            #     function $($f)(
            #         l::Node{T1}, r::Node{T2}
            #     ) where {T1<:$($type_requirements),T2<:$($type_requirements)}
            #         T = promote_type(T1, T2)
            #         l = convert(Node{T}, l)
            #         r = convert(Node{T}, r)
            #         return $($f)(l, r)
            #     end
            #     function $($f)(
            #         l::Node{T1}, r::T2
            #     ) where {T1<:$($type_requirements),T2<:$($type_requirements)}
            #         T = promote_type(T1, T2)
            #         l = convert(Node{T}, l)
            #         r = convert(T, r)
            #         return $($f)(l, r)
            #     end
            #     function $($f)(
            #         l::T1, r::Node{T2}
            #     ) where {T1<:$($type_requirements),T2<:$($type_requirements)}
            #         T = promote_type(T1, T2)
            #         l = convert(T, l)
            #         r = convert(Node{T}, r)
            #         return $($f)(l, r)
            #     end
            # end
        end
    end
end

function _extend_operators(operators, skip_user_operators, kws, __module__::Module)
    empty_old_operators =
        if length(kws) == 1 && :empty_old_operators in map(x -> x.args[1], kws)
            @assert kws[1].head == :(=)
            kws[1].args[2]
        elseif length(kws) > 0
            error(
                "You passed the keywords $(kws), but only `empty_old_operators` is supported.",
            )
        else
            true
        end
    muliinary_ex = _extend_multinary_operator(:f, :type_requirements)#, :build_converters)
    binary_ex = _extend_binary_operator(:f, :type_requirements, :build_converters)
    unary_ex = _extend_unary_operator(:f, :type_requirements)
    return quote
        local type_requirements
        local build_converters
        local multinary_exists
        local binary_exists
        local unary_exists
        if isa($operators, $OperatorEnum)
            type_requirements = Number
            build_converters = true
            multinary_exists = $(ALREADY_DEFINED_MULTINARY_OPERATORS).operator_enum
            binary_exists = $(ALREADY_DEFINED_BINARY_OPERATORS).operator_enum
            unary_exists = $(ALREADY_DEFINED_UNARY_OPERATORS).operator_enum
        else
            type_requirements = Any
            build_converters = false
            multinary_exists = $(ALREADY_DEFINED_MULTINARY_OPERATORS).generic_operator_enum
            binary_exists = $(ALREADY_DEFINED_BINARY_OPERATORS).generic_operator_enum
            unary_exists = $(ALREADY_DEFINED_UNARY_OPERATORS).generic_operator_enum
        end
        if $(empty_old_operators)
            # Trigger errors if operators are not yet defined:
            empty!($(LATEST_MULTINARY_OPERATOR_MAPPING))
            empty!($(LATEST_BINARY_OPERATOR_MAPPING))
            empty!($(LATEST_UNARY_OPERATOR_MAPPING))
        end
        for (op, func) in enumerate($(operators).multinops)
            local f = Symbol(func[1])
            local skip = false
            if isdefined(Base, f)
                f = :(Base.$(f))
            elseif $(skip_user_operators)
                skip = true
            else
                f = :($($__module__).$(f))
            end
            $(LATEST_MULTINARY_OPERATOR_MAPPING)[func[1]] = op
            skip && continue
            # Avoid redefining methods:
            if (!haskey(unary_exists, func[1])) && (!haskey(binary_exists, func[1]))
                eval($muliinary_ex)
                unary_exists[func[1]] = true
                binary_exists[func[1]] = true
            end
        end
        for (op, func) in enumerate($(operators).binops)
            local f = Symbol(func)
            local skip = false
            if isdefined(Base, f)
                f = :(Base.$(f))
            elseif $(skip_user_operators)
                skip = true
            else
                f = :($($__module__).$(f))
            end
            $(LATEST_BINARY_OPERATOR_MAPPING)[func] = op
            skip && continue
            # Avoid redefining methods:
            if (!haskey(unary_exists, func)) && (!haskey(multinary_exists, func))
                eval($binary_ex)
                unary_exists[func] = true
                multinary_exists[func] = true
            end
        end
        for (op, func) in enumerate($(operators).unaops)
            local f = Symbol(func)
            local skip = false
            if isdefined(Base, f)
                f = :(Base.$(f))
            elseif $(skip_user_operators)
                skip = true
            else
                f = :($($__module__).$(f))
            end
            $(LATEST_UNARY_OPERATOR_MAPPING)[func] = op
            skip && continue
            # Avoid redefining methods:
            if (!haskey(binary_exists, func)) && (!haskey(multinary_exists, func))
                eval($unary_ex)
                binary_exists[func] = true
                multinary_exists[func] = true
            end
        end
    end
end

"""
    @extend_operators operators [kws...]

Extends all operators defined in this operator enum to work on the
`Node` type. While by default this is already done for operators defined
in `Base` when you create an enum and pass `define_helper_functions=true`,
this does not apply to the user-defined operators. Thus, to do so, you must
apply this macro to the operator enum in the same module you have the operators
defined.
"""
macro extend_operators(operators, kws...)
    ex = _extend_operators(operators, false, kws, __module__)
    expected_type = AbstractOperatorEnum
    return esc(
        quote
            if !isa($(operators), $expected_type)
                error("You must pass an operator enum to `@extend_operators`.")
            end
            $ex
        end,
    )
end

"""
    @extend_operators_base operators [kws...]

Similar to `@extend_operators`, but only extends operators already
defined in `Base`.
"""
macro extend_operators_base(operators, kws...)
    ex = _extend_operators(operators, true, kws, __module__)
    expected_type = AbstractOperatorEnum
    return esc(
        quote
            if !isa($(operators), $expected_type)
                error("You must pass an operator enum to `@extend_operators_base`.")
            end
            $ex
        end,
    )
end

"""
    OperatorEnum(; multinary_operators=[], binary_operators=[], unary_operators=[],
                   enable_autodiff::Bool=false, define_helper_functions::Bool=true,
                   empty_old_operators::Bool=true)

Construct an `OperatorEnum` object, defining the possible expressions. This will also
redefine operators for `Node` types, as well as `show`, `print`, and `(::Node)(X)`.
It will automatically compute derivatives with `Zygote.jl`.

# Arguments
- `binary_operators::Vector{Function}`: A vector of functions, each of which is a binary
  operator.
- `unary_operators::Vector{Function}`: A vector of functions, each of which is a unary
  operator.
- `enable_autodiff::Bool=false`: Whether to enable automatic differentiation.
- `define_helper_functions::Bool=true`: Whether to define helper functions for creating
   and evaluating node types. Turn this off when doing precompilation. Note that these
   are *not* needed for the package to work; they are purely for convenience.
- `empty_old_operators::Bool=true`: Whether to clear the old operators.
"""
function OperatorEnum(;
    multinary_operators=[],
    binary_operators=[],
    unary_operators=[],
    enable_autodiff::Bool=false,
    define_helper_functions::Bool=true,
    empty_old_operators::Bool=true,
)
    @assert length(binary_operators) > 0 || length(unary_operators) > 0 || length(multinary_operators) > 0

    
    multinary_operators_new = Vector{Tuple{Function, Int64}}()
    for (op, arity) in multinary_operators
        push!(multinary_operators_new, (op, arity))
    end
    multinary_operators = multinary_operators_new
    binary_operators = Function[op for op in binary_operators]
    unary_operators = Function[op for op in unary_operators]

    diff_multinary_operators = Vector{Tuple{Function, Int64}}()
    diff_binary_operators = Function[]
    diff_unary_operators = Function[]

    if enable_autodiff
        for (op, arity) in multinary_operators
            push!(diff_multinary_operators, (_zygote_gradient(op, Val(3)), arity))
        end
        for op in binary_operators
            push!(diff_binary_operators, _zygote_gradient(op, Val(2)))
        end
        for op in unary_operators
            push!(diff_unary_operators, _zygote_gradient(op, Val(1)))
        end
    end

    operators = OperatorEnum(
        multinary_operators, binary_operators, unary_operators, diff_multinary_operators, diff_binary_operators, diff_unary_operators
        # binary_operators, unary_operators, diff_binary_operators, diff_unary_operators
    )

    if define_helper_functions
        @extend_operators_base operators empty_old_operators = empty_old_operators
        create_evaluation_helpers!(operators)
    end

    return operators
end

"""
    GenericOperatorEnum(; binary_operators=[], unary_operators=[],
                          define_helper_functions::Bool=true, empty_old_operators::Bool=true)

Construct a `GenericOperatorEnum` object, defining possible expressions.
Unlike `OperatorEnum`, this enum one will work arbitrary operators and data types.
This will also redefine operators for `Node` types, as well as `show`, `print`,
and `(::Node)(X)`.

# Arguments
- `binary_operators::Vector{Function}`: A vector of functions, each of which is a binary
  operator.
- `unary_operators::Vector{Function}`: A vector of functions, each of which is a unary
  operator.
- `define_helper_functions::Bool=true`: Whether to define helper functions for creating
   and evaluating node types. Turn this off when doing precompilation. Note that these
   are *not* needed for the package to work; they are purely for convenience.
- `empty_old_operators::Bool=true`: Whether to clear the old operators.
"""
function GenericOperatorEnum(;
    multinary_operators=[],
    binary_operators=[],
    unary_operators=[],
    define_helper_functions::Bool=true,
    empty_old_operators::Bool=true,
)
    @assert length(binary_operators) > 0 || length(unary_operators) > 0 || length(multinary_operators) > 0

    multinary_operators_new = Vector{Tuple{Function, Int64}}()
    for (op, arity) in multinary_operators
        push!(multinary_operators_new, (op, arity))
    end
    multinary_operators = multinary_operators_new
    binary_operators = Function[op for op in binary_operators]
    unary_operators = Function[op for op in unary_operators]

    operators = GenericOperatorEnum(multinary_operators, binary_operators, unary_operators)
    # operators = GenericOperatorEnum(binary_operators, unary_operators)

    if define_helper_functions
        @extend_operators_base operators empty_old_operators = empty_old_operators
        create_evaluation_helpers!(operators)
    end

    return operators
end

end
