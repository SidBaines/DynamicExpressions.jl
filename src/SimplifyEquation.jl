module SimplifyEquationModule

import ..EquationModule: Node, copy_node
import ..OperatorEnumModule: AbstractOperatorEnum
import ..UtilsModule: isbad, isgood

_una_op_kernel(f::F, l::T) where {F,T} = f(l)
_bin_op_kernel(f::F, l::T, r::T) where {F,T} = f(l, r)
_multin_op_kernel(f::F, cdrn::Tuple) where {F} = f(cdrn)

# Simplify tree
function combine_operators(tree::Node{T}, operators::AbstractOperatorEnum) where {T}
    # NOTE: (const (+*-) const) already accounted for. Call simplify_tree before.
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
    end

    top_level_constant = tree.degree == 2 && (tree.children[1].constant || tree.children[2].constant)
    if tree.degree == 2 &&
        (operators.binops[tree.op] == (*) || operators.binops[tree.op] == (+)) &&
        top_level_constant

        # TODO: Does this break SymbolicRegression.jl due to the different names of operators?

        op = tree.op
        # Put the constant in r. Need to assume var in left for simplification assumption.
        if tree.children[1].constant
            tmp = tree.children[2]
            tree.children[2] = tree.children[1]
            tree.children[1] = tmp
        end
        topconstant = tree.children[2].val::T
        # Simplify down first
        below = tree.children[1]
        if below.degree == 2 && below.op == op
            if below.children[1].constant
                tree = below
                tree.children[1].val = _bin_op_kernel(
                    operators.binops[op], tree.children[1].val::T, topconstant
                )
            elseif below.children[2].constant
                tree = below
                tree.children[2].val = _bin_op_kernel(
                    operators.binops[op], tree.children[2].val::T, topconstant
                )
            end
        end
    end

    if tree.degree == 2 && operators.binops[tree.op] == (-) && top_level_constant
        # Currently just simplifies subtraction. (can't assume both plus and sub are operators)
        # Not commutative, so use different op.
        if tree.children[1].constant
            if tree.children[2].degree == 2 && operators.binops[tree.children[2].op] == (-)
                if tree.children[2].children[1].constant
                    #(const - (const - var)) => (var - const)
                    l = tree.children[1]
                    r = tree.children[2]
                    simplified_const = -(l.val::T - r.children[1].val::T) #neg(sub(l.val, r.children[1].val))
                    tree.children[1] = tree.children[2].children[2]
                    tree.children[2] = l
                    tree.children[2].val = simplified_const
                elseif tree.children[2].children[2].constant
                    #(const - (var - const)) => (const - var)
                    l = tree.children[1]
                    r = tree.children[2]
                    simplified_const = l.val::T + r.children[2].val::T #plus(l.val, r.children[2].val)
                    tree.children[2] = tree.children[2].children[1]
                    tree.children[1].val = simplified_const
                end
            end
        else #tree.children[2].constant is true
            if tree.children[1].degree == 2 && operators.binops[tree.children[1].op] == (-)
                if tree.children[1].children[1].constant
                    #((const - var) - const) => (const - var)
                    l = tree.children[1]
                    r = tree.children[2]
                    simplified_const = l.children[1].val::T - r.val::T#sub(l.children[1].val, r.val)
                    tree.children[2] = tree.children[1].children[2]
                    tree.children[1] = r
                    tree.children[1].val = simplified_const
                elseif tree.children[1].children[2].constant
                    #((var - const) - const) => (var - const)
                    l = tree.children[1]
                    r = tree.children[2]
                    simplified_const = r.val::T + l.children[2].val::T #plus(r.val, l.children[2].val)
                    tree.children[1] = tree.children[1].children[1]
                    tree.children[2].val = simplified_const
                end
            end
        end
    end
    return tree
end

# Simplify tree
# TODO: This will get much more powerful with the tree-map functions.
function simplify_tree(tree::Node{T}, operators::AbstractOperatorEnum) where {T}
    if tree.degree == 1
        tree.children[1] = simplify_tree(tree.children[1], operators)
        if tree.children[1].degree == 0 && tree.children[1].constant
            l = tree.children[1].val::T
            if isgood(l)
                out = _una_op_kernel(operators.unaops[tree.op], l)
                if isbad(out)
                    return tree
                end
                return Node(T; val=convert(T, out))
            end
        end
    elseif tree.degree == 2
        tree.children[1] = simplify_tree(tree.children[1], operators)
        tree.children[2] = simplify_tree(tree.children[2], operators)
        constantsBelow = (
            tree.children[1].degree == 0 && tree.children[1].constant && tree.children[2].degree == 0 && tree.children[2].constant
        )
        if constantsBelow
            # NaN checks:
            l = tree.children[1].val::T
            r = tree.children[2].val::T
            if isbad(l) || isbad(r)
                return tree
            end

            # Actually compute:
            out = _bin_op_kernel(operators.binops[tree.op], l, r)
            if isbad(out)
                return tree
            end
            return Node(T; val=convert(T, out))
        end
    elseif tree.degree > 2
        for cn in 1:length(tree.children)
            tree.children[cn] = simplify_tree(tree.children[cn], operators)
        end
        constantsBelow = all(i->(i.degree == 0 && i.constant), tree.children)
        if constantsBelow
            # NaN checks
            vs = Vector{T}
            for cn in 1:legnth(tree.children)
                push!(vs,tree.children[cn].val::T)
            end
            if any(v->isbad(v), vs)
                return tree
            end

            # Actually compute:
            out = _multin_op_kernel(operators.multinops[tree.op], Tuple(v for v in vs))
            if isbad(out)
                return tree
            end
            return Node(T; val=convert(T, out))
        end
    end
    return tree
end

end
