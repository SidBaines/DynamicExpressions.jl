module SimplifyEquationModule

import ..EquationModule: AbstractExpressionNode, constructorof, Node, copy_node, set_node!
import ..EquationUtilsModule: tree_mapreduce, is_node_constant
import ..OperatorEnumModule: AbstractOperatorEnum
import ..UtilsModule: isbad, isgood

_una_op_kernel(f::F, l::T) where {F,T} = f(l)
_bin_op_kernel(f::F, l::T, r::T) where {F,T} = f(l, r)
_any_op_kernel(f::F, cs::NTuple) where {F} = f(cs...)

is_commutative(::typeof(*)) = true
is_commutative(::typeof(+)) = true
is_commutative(_) = false

is_subtraction(::typeof(-)) = true
is_subtraction(_) = false

# This is only defined for `Node` as it is not possible for
# `GraphNode`.
function combine_operators(tree::Node{T}, operators::AbstractOperatorEnum) where {T}
    # NOTE: (const (+*-) const) already accounted for. Call simplify_tree! before.
    # ((const + var) + const) => (const + var)
    # ((const * var) * const) => (const * var)
    # ((const - var) - const) => (const - var)
    # (want to add anything commutative!)
    # TODO - need to combine plus/sub if they are both there.
    if tree.degree == 0
        return tree
    elseif tree.degree == 1
        tree.children[1] = combine_operators(tree.children[1], operators)
    elseif tree.degree == 2
        tree.children[1] = combine_operators(tree.children[1], operators)
        tree.children[2] = combine_operators(tree.children[2], operators)
    else
        for cn in 1:tree.degree
            tree.children[cn] = combine_operators(tree.children[cn], operators)
        end
    end

    top_level_constant =
        tree.degree == 2 && (is_node_constant(tree.children[1]) || is_node_constant(tree.children[2]))
    if tree.degree == 2 && is_commutative(operators.binops[tree.op]) && top_level_constant
        # TODO: Does this break SymbolicRegression.jl due to the different names of operators?
        op = tree.op
        # Put the constant in r. Need to assume var in left for simplification assumption.
        if is_node_constant(tree.children[1])
            tmp = tree.children[2]
            tree.children[2] = tree.children[1]
            tree.children[1] = tmp
        end
        topconstant = tree.children[2].val
        # Simplify down first
        below = tree.children[1]
        if below.degree == 2 && below.op == op
            if is_node_constant(below.children[1])
                tree = below
                tree.children[1].val = _bin_op_kernel(operators.binops[op], tree.children[1].val, topconstant)
            elseif is_node_constant(below.children[2])
                tree = below
                tree.children[2].val = _bin_op_kernel(operators.binops[op], tree.children[2].val, topconstant)
            end
        end
    end

    if tree.degree == 2 && is_subtraction(operators.binops[tree.op]) && top_level_constant

        # Currently just simplifies subtraction. (can't assume both plus and sub are operators)
        # Not commutative, so use different op.
        if is_node_constant(tree.children[1])
            if tree.children[2].degree == 2 && tree.op == tree.children[2].op
                if is_node_constant(tree.children[2].children[1])
                    #(const - (const - var)) => (var - const)
                    l = tree.children[1]
                    r = tree.children[2]
                    simplified_const = (r.children[1].val - l.val) #neg(sub(l.val, r.children[1].val))
                    tree.children[1] = tree.children[2].children[2]
                    tree.children[2] = l
                    tree.children[2].val = simplified_const
                elseif is_node_constant(tree.children[2].children[2])
                    #(const - (var - const)) => (const - var)
                    l = tree.children[1]
                    r = tree.children[2]
                    simplified_const = l.val + r.children[2].val #plus(l.val, r.children[2].val)
                    tree.children[2] = tree.children[2].children[1]
                    tree.children[1].val = simplified_const
                end
            end
        else #tree.children[2] is a constant
            if tree.children[1].degree == 2 && tree.op == tree.children[1].op
                if is_node_constant(tree.children[1].children[1])
                    #((const - var) - const) => (const - var)
                    l = tree.children[1]
                    r = tree.children[2]
                    simplified_const = l.children[1].val - r.val#sub(l.children[1].val, r.val)
                    tree.children[2] = tree.children[1].children[2]
                    tree.children[1] = r
                    tree.children[1].val = simplified_const
                elseif is_node_constant(tree.children[1].children[2])
                    #((var - const) - const) => (var - const)
                    l = tree.children[1]
                    r = tree.children[2]
                    simplified_const = r.val + l.children[2].val #plus(r.val, l.children[2].val)
                    tree.children[1] = tree.children[1].children[1]
                    tree.children[2].val = simplified_const
                end
            end
        end
    end
    return tree
end

function combine_children!(operators, p::N, c::N...) where {T,N<:AbstractExpressionNode{T}}
    all(is_node_constant, c) || return p
    vals = map(n -> n.val, c)
    all(isgood, vals) || return p
    out = if length(c) == 1
        _una_op_kernel(operators.unaops[p.op], vals...)
    elseif length(c) == 2
        _bin_op_kernel(operators.binops[p.op], vals...)
    else
        _any_op_kernel(operators.anyops[p.op], vals...)
    end
    isgood(out) || return p
    new_node = constructorof(N)(T; val=convert(T, out))
    set_node!(p, new_node)
    return p
end

# Simplify tree
function simplify_tree!(tree::AbstractExpressionNode, operators::AbstractOperatorEnum)
    tree = tree_mapreduce(
        identity,
        (p, c...) -> combine_children!(operators, p, c...),
        tree,
        constructorof(typeof(tree));
    )
    return tree
end

end
