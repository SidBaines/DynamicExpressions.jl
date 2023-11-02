module EvaluateEquationDerivativeModule

import LoopVectorization: indices, @turbo
import ..EquationModule: Node
import ..OperatorEnumModule: OperatorEnum
import ..UtilsModule: @return_on_false2, @maybe_turbo, is_bad_array, fill_similar
import ..EquationUtilsModule: count_constants, index_constants, NodeIndex
import ..EvaluateEquationModule: deg0_eval

_zygote_gradient(args...) = error("Please load the Zygote.jl package.")

function assert_autodiff_enabled(operators::OperatorEnum)
    if length(operators.diff_binops) == 0 && length(operators.diff_unaops) == 0 && length(operators.diff_multinops) == 0
        error(
            "Found no differential operators. Did you forget to set `enable_autodiff=true` when creating the `OperatorEnum`?",
        )
    end
end

"""
    eval_diff_tree_array(tree::Node{T}, cX::AbstractMatrix{T}, operators::OperatorEnum, direction::Integer; turbo::Bool=false)

Compute the forward derivative of an expression, using a similar
structure and optimization to eval_tree_array. `direction` is the index of a particular
variable in the expression. e.g., `direction=1` would indicate derivative with
respect to `x1`.

# Arguments

- `tree::Node`: The expression tree to evaluate.
- `cX::AbstractMatrix{T}`: The data matrix, with each column being a data point.
- `operators::OperatorEnum`: The operators used to create the `tree`. Note that `operators.enable_autodiff`
    must be `true`. This is needed to create the derivative operations.
- `direction::Integer`: The index of the variable to take the derivative with respect to.
- `turbo::Bool`: Use `LoopVectorization.@turbo` for faster evaluation.

# Returns

- `(evaluation, derivative, complete)::Tuple{AbstractVector{T}, AbstractVector{T}, Bool}`: the normal evaluation,
    the derivative, and whether the evaluation completed as normal (or encountered a nan or inf).
"""
function eval_diff_tree_array(
    tree::Node{T},
    cX::AbstractMatrix{T},
    operators::OperatorEnum,
    direction::Integer;
    turbo::Bool=false,
)::Tuple{AbstractVector{T},AbstractVector{T},Bool} where {T<:Number}
    assert_autodiff_enabled(operators)
    # TODO: Implement quick check for whether the variable is actually used
    # in this tree. Otherwise, return zero.
    return _eval_diff_tree_array(
        tree, cX, operators, direction, (turbo ? Val(true) : Val(false))
    )
end
function eval_diff_tree_array(
    tree::Node{T1},
    cX::AbstractMatrix{T2},
    operators::OperatorEnum,
    direction::Integer;
    turbo::Bool=false,
) where {T1<:Number,T2<:Number}
    T = promote_type(T1, T2)
    @warn "Warning: eval_diff_tree_array received mixed types: tree=$(T1) and data=$(T2)."
    tree = convert(Node{T}, tree)
    cX = T.(cX)
    return eval_diff_tree_array(tree, cX, operators, direction; turbo=turbo)
end

function _eval_diff_tree_array(
    tree::Node{T},
    cX::AbstractMatrix{T},
    operators::OperatorEnum,
    direction::Integer,
    ::Val{turbo},
)::Tuple{AbstractVector{T},AbstractVector{T},Bool} where {T<:Number,turbo}
    evaluation, derivative, complete = if tree.degree == 0
        diff_deg0_eval(tree, cX, direction)
    elseif tree.degree == 1
        diff_deg1_eval(
            tree,
            cX,
            operators.unaops[tree.op],
            operators.diff_unaops[tree.op],
            operators,
            direction,
            Val(turbo),
        )
    elseif tree.degree == 2
        diff_deg2_eval(
            tree,
            cX,
            operators.binops[tree.op],
            operators.diff_binops[tree.op],
            operators,
            direction,
            Val(turbo),
        )
    else#if degree > 2
        diff_degn_eval(
            tree,
            cX,
            operators.multinops[tree.op],
            operators.diff_multinops[tree.op],
            operators,
            direction,
            Val(turbo),
        )
    end
    @return_on_false2 complete evaluation derivative
    return evaluation, derivative, !(is_bad_array(evaluation) || is_bad_array(derivative))
end

function diff_deg0_eval(
    tree::Node{T}, cX::AbstractMatrix{T}, direction::Integer
)::Tuple{AbstractVector{T},AbstractVector{T},Bool} where {T<:Number}
    const_part = deg0_eval(tree, cX)[1]
    derivative_part = if ((!tree.constant) && tree.feature == direction)
        fill_similar(one(T), cX, axes(cX, 2))
    else
        fill_similar(zero(T), cX, axes(cX, 2))
    end
    return (const_part, derivative_part, true)
end

function diff_deg1_eval(
    tree::Node{T},
    cX::AbstractMatrix{T},
    op::F,
    diff_op::dF,
    operators::OperatorEnum,
    direction::Integer,
    ::Val{turbo},
)::Tuple{AbstractVector{T},AbstractVector{T},Bool} where {T<:Number,F,dF,turbo}
    (cumulator, dcumulator, complete) = _eval_diff_tree_array(
        tree.children[1], cX, operators, direction, Val(turbo)
    )
    @return_on_false2 complete cumulator dcumulator

    # TODO - add type assertions to get better speed:
    @maybe_turbo turbo for j in indices((cumulator, dcumulator))
        x = op(cumulator[j])::T
        dx = diff_op(cumulator[j])::T * dcumulator[j]

        cumulator[j] = x
        dcumulator[j] = dx
    end
    return (cumulator, dcumulator, true)
end

function diff_deg2_eval(
    tree::Node{T},
    cX::AbstractMatrix{T},
    op::F,
    diff_op::dF,
    operators::OperatorEnum,
    direction::Integer,
    ::Val{turbo},
)::Tuple{AbstractVector{T},AbstractVector{T},Bool} where {T<:Number,F,dF,turbo}
    (cumulator, dcumulator, complete) = _eval_diff_tree_array(
        tree.children[1], cX, operators, direction, Val(turbo)
    )
    @return_on_false2 complete cumulator dcumulator
    (array2, dcumulator2, complete2) = _eval_diff_tree_array(
        tree.children[2], cX, operators, direction, Val(turbo)
    )
    @return_on_false2 complete2 array2 dcumulator2

    @maybe_turbo turbo for j in indices((cumulator, dcumulator, array2, dcumulator2))
        x = op(cumulator[j], array2[j])::T

        first, second = diff_op(cumulator[j], array2[j])::Tuple{T,T}
        dx = first * dcumulator[j] + second * dcumulator2[j]

        cumulator[j] = x
        dcumulator[j] = dx
    end
    return (cumulator, dcumulator, true)
end

function diff_degn_eval(
    tree::Node{T},
    cX::AbstractMatrix{T},
    op::F,
    diff_op::dF,
    operators::OperatorEnum,
    direction::Integer,
    ::Val{turbo},
)::Tuple{AbstractVector{T},AbstractVector{T},Bool} where {T<:Number,F,dF,turbo}
    cumulators = []
    dcumulators = []
    for (cn, child) in enumerate(tree.children)
        (cumulator, dcumulator, complete) = _eval_diff_tree_array(
            child, cX, operators, direction, Val(turbo)
        )
        @return_on_false2 complete cumulator dcumulator
        push!(cumulators, cumulator)
        push!(dcumulators, dcumulator)
    end

    # @maybe_turbo turbo for j in indices((cumulator, dcumulator, array2, dcumulator2))
    cumulator_l = similar(cumulators[1])
    dcumulator_l = similar(cumulators[1])
    # @maybe_turbo turbo for j in indices((cumulators[1]))
    # Again, for some reason this fails when we use maybe_turbo
    for j in indices((cumulators[1]))
        inps = Tuple(cumulator[j] for cumulator in cumulators)
        x = op(inps...)::T

        derivs = diff_op(inps...)::Tuple
        dx = 0 
        for dn in indices(dcumulators)
            dx += derivs[dn] * dcumulators[dn]
        end

        cumulator_l[j] = x
        dcumulator_l[j] = dx
    end
    return (cumulator_l, dcumulator_l, true)
end

"""
    eval_grad_tree_array(tree::Node{T}, cX::AbstractMatrix{T}, operators::OperatorEnum; variable::Bool=false, turbo::Bool=false)

Compute the forward-mode derivative of an expression, using a similar
structure and optimization to eval_tree_array. `variable` specifies whether
we should take derivatives with respect to features (i.e., cX), or with respect
to every constant in the expression.

# Arguments

- `tree::Node{T}`: The expression tree to evaluate.
- `cX::AbstractMatrix{T}`: The data matrix, with each column being a data point.
- `operators::OperatorEnum`: The operators used to create the `tree`. Note that `operators.enable_autodiff`
    must be `true`. This is needed to create the derivative operations.
- `variable::Bool`: Whether to take derivatives with respect to features (i.e., `cX` - with `variable=true`),
    or with respect to every constant in the expression (`variable=false`).
- `turbo::Bool`: Use `LoopVectorization.@turbo` for faster evaluation.

# Returns

- `(evaluation, gradient, complete)::Tuple{AbstractVector{T}, AbstractMatrix{T}, Bool}`: the normal evaluation,
    the gradient, and whether the evaluation completed as normal (or encountered a nan or inf).
"""
function eval_grad_tree_array(
    tree::Node{T},
    cX::AbstractMatrix{T},
    operators::OperatorEnum;
    variable::Bool=false,
    turbo::Bool=false,
)::Tuple{AbstractVector{T},AbstractMatrix{T},Bool} where {T<:Number}
    assert_autodiff_enabled(operators)
    n_gradients = variable ? size(cX, 1) : count_constants(tree)
    index_tree = index_constants(tree, UInt16(0))
    return eval_grad_tree_array(
        tree,
        Val(n_gradients),
        index_tree,
        cX,
        operators,
        (variable ? Val(true) : Val(false)),
        (turbo ? Val(true) : Val(false)),
    )
end

function eval_grad_tree_array(
    tree::Node{T},
    ::Val{n_gradients},
    index_tree::NodeIndex,
    cX::AbstractMatrix{T},
    operators::OperatorEnum,
    ::Val{variable},
    ::Val{turbo},
)::Tuple{
    AbstractVector{T},AbstractMatrix{T},Bool
} where {T<:Number,variable,turbo,n_gradients}
    evaluation, gradient, complete = _eval_grad_tree_array(
        tree, Val(n_gradients), index_tree, cX, operators, Val(variable), Val(turbo)
    )
    @return_on_false2 complete evaluation gradient
    return evaluation, gradient, !(is_bad_array(evaluation) || is_bad_array(gradient))
end

function eval_grad_tree_array(
    tree::Node{T1},
    cX::AbstractMatrix{T2},
    operators::OperatorEnum;
    variable::Bool=false,
    turbo::Bool=false,
) where {T1<:Number,T2<:Number}
    T = promote_type(T1, T2)
    return eval_grad_tree_array(
        convert(Node{T}, tree),
        convert(AbstractMatrix{T}, cX),
        operators;
        variable=variable,
        turbo=turbo,
    )
end

function _eval_grad_tree_array(
    tree::Node{T},
    ::Val{n_gradients},
    index_tree::NodeIndex,
    cX::AbstractMatrix{T},
    operators::OperatorEnum,
    ::Val{variable},
    ::Val{turbo},
)::Tuple{
    AbstractVector{T},AbstractMatrix{T},Bool
} where {T<:Number,variable,turbo,n_gradients}
    if tree.degree == 0
        grad_deg0_eval(tree, Val(n_gradients), index_tree, cX, Val(variable))
    elseif tree.degree == 1
        grad_deg1_eval(
            tree,
            Val(n_gradients),
            index_tree,
            cX,
            operators.unaops[tree.op],
            operators.diff_unaops[tree.op],
            operators,
            Val(variable),
            Val(turbo),
        )
    elseif tree.degree == 2
        grad_deg2_eval(
            tree,
            Val(n_gradients),
            index_tree,
            cX,
            operators.binops[tree.op],
            operators.diff_binops[tree.op],
            operators,
            Val(variable),
            Val(turbo),
        )
    else#if tree.degree > 2
        grad_degn_eval(
            tree,
            Val(n_gradients),
            index_tree,
            cX,
            operators.multinops[tree.op],
            operators.diff_multinops[tree.op],
            operators,
            Val(variable),
            Val(turbo),
        )
    end
end

function grad_deg0_eval(
    tree::Node{T},
    ::Val{n_gradients},
    index_tree::NodeIndex,
    cX::AbstractMatrix{T},
    ::Val{variable},
)::Tuple{AbstractVector{T},AbstractMatrix{T},Bool} where {T<:Number,variable,n_gradients}
    const_part = deg0_eval(tree, cX)[1]

    zero_mat = if typeof(cX) <: Array
        zeros(T, n_gradients, size(cX, 2))
    else
        hcat((fill_similar(zero(T), cX, axes(cX, 2)) for _ in 1:n_gradients)...)'
    end

    if variable == tree.constant
        return (const_part, zero_mat, true)
    end

    index = variable ? tree.feature : index_tree.constant_index
    derivative_part = zero_mat
    derivative_part[index, :] .= one(T)
    return (const_part, derivative_part, true)
end

function grad_deg1_eval(
    tree::Node{T},
    ::Val{n_gradients},
    index_tree::NodeIndex,
    cX::AbstractMatrix{T},
    op::F,
    diff_op::dF,
    operators::OperatorEnum,
    ::Val{variable},
    ::Val{turbo},
)::Tuple{
    AbstractVector{T},AbstractMatrix{T},Bool
} where {T<:Number,F,dF,variable,turbo,n_gradients}
    (cumulator, dcumulator, complete) = eval_grad_tree_array(
        tree.children[1], Val(n_gradients), index_tree.children[1], cX, operators, Val(variable), Val(turbo)
    )
    @return_on_false2 complete cumulator dcumulator

    @maybe_turbo turbo for j in indices((cumulator, dcumulator), (1, 2))
        x = op(cumulator[j])::T
        dx = diff_op(cumulator[j])::T

        cumulator[j] = x
        for k in indices(dcumulator, 1)
            dcumulator[k, j] = dx * dcumulator[k, j]
        end
    end
    return (cumulator, dcumulator, true)
end

function grad_deg2_eval(
    tree::Node{T},
    ::Val{n_gradients},
    index_tree::NodeIndex,
    cX::AbstractMatrix{T},
    op::F,
    diff_op::dF,
    operators::OperatorEnum,
    ::Val{variable},
    ::Val{turbo},
)::Tuple{
    AbstractVector{T},AbstractMatrix{T},Bool
} where {T<:Number,F,dF,variable,turbo,n_gradients}
    (cumulator1, dcumulator1, complete) = eval_grad_tree_array(
        tree.children[1], Val(n_gradients), index_tree.children[1], cX, operators, Val(variable), Val(turbo)
    )
    @return_on_false2 complete cumulator1 dcumulator1
    (cumulator2, dcumulator2, complete2) = eval_grad_tree_array(
        tree.children[2], Val(n_gradients), index_tree.children[2], cX, operators, Val(variable), Val(turbo)
    )
    @return_on_false2 complete2 cumulator1 dcumulator1

    @maybe_turbo turbo for j in indices(
        (cumulator1, cumulator2, dcumulator1, dcumulator2), (1, 1, 2, 2)
    )
        c1 = cumulator1[j]
        c2 = cumulator2[j]
        x = op(c1, c2)::T
        dx1, dx2 = diff_op(c1, c2)::Tuple{T,T}
        cumulator1[j] = x
        for k in indices((dcumulator1, dcumulator2), (1, 1))
            dcumulator1[k, j] = dx1 * dcumulator1[k, j] + dx2 * dcumulator2[k, j]
        end
    end

    return (cumulator1, dcumulator1, true)
end

@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)

function grad_degn_eval(
    tree::Node{T},
    ::Val{n_gradients},
    index_tree::NodeIndex,
    cX::AbstractMatrix{T},
    op::F,
    diff_op::dF,
    operators::OperatorEnum,
    ::Val{variable},
    ::Val{turbo},
)::Tuple{
    AbstractVector{T},AbstractMatrix{T},Bool
} where {T<:Number,F,dF,variable,turbo,n_gradients}
    cumulators = []
    dcumulators = []
    for cn in indices(tree.children)
        (cumulator, dcumulator, complete) = eval_grad_tree_array(
            tree.children[cn], Val(n_gradients), index_tree.children[cn], cX, operators, Val(variable), Val(turbo)
        )
        @return_on_false2 complete cumulator dcumulator
        push!(cumulators, cumulator)
        push!(dcumulators, dcumulator)
    end

    cumulator1=similar(cumulators[1])
    dcumulator1=similar(dcumulators[1])
    # @maybe_turbo turbo for j in indices(
    for j in indices(
        tuplejoin(Tuple(cumulator for cumulator in cumulators), Tuple(dcumulator for dcumulator in dcumulators)), tuplejoin(Tuple(1 for i in 1:length(cumulators)), Tuple(2 for i in 1:length(dcumulators)))
    )
        inps=Tuple(cumulator[j] for cumulator in cumulators)
        x = op(inps...)::T
        dxs = diff_op(inps...)::Tuple{T,T}
        cumulator1[j] = x
        for k in indices(Tuple(dcumulator for dcumulator in dcumulators), Tuple(1 for i in 1:length(dcumulators)))
            dcumulator1[k, j] = 0
            for dn in indices(dcumulators)
                dcumulator1[k, j] += dxs[dn] * dcumulators[dn][k, j]
            end
        end
    end

    return (cumulator1, dcumulator1, true)
end

end
