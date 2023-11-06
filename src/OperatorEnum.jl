module OperatorEnumModule

abstract type AbstractOperatorEnum end

"""
    OperatorEnum

Defines an enum over operators, along with their derivatives.
# Fields
- `multinops`: A tuple *2-tuples of* of (n-ary operator, n) (where n can be any integer >= 3). Need the 2-tuple to pass information about the (fixed, per operator) arity of the operator as the second argument
- `binops`: A tuple of binary operators. Scalar input type.
- `unaops`: A tuple of unary operators. Scalar input type.
- `diff_multinops`: A tuple of Zygote-computed derivatives of the n-ary operators (where n can be any integer >= 3).
- `diff_binops`: A tuple of Zygote-computed derivatives of the binary operators.
- `diff_unaops`: A tuple of Zygote-computed derivatives of the unary operators.
"""
struct OperatorEnum <: AbstractOperatorEnum
    multinops::Vector{Tuple{Function, Int64}}
    binops::Vector{Function}
    unaops::Vector{Function}
    diff_multinops::Vector{Function}
    diff_binops::Vector{Function}
    diff_unaops::Vector{Function}
end

"""
    OperatorEnum

Defines an enum over operators, along with their derivatives.
# Fields
- `multinops`: A tuple *2-tuples of* of (n-ary operator, n) (where n can be any integer >= 3). Need the 2-tuple to pass information about the (fixed, per operator) arity of the operator as the second argument
- `binops`: A tuple of binary operators.
- `unaops`: A tuple of unary operators.
- `diff_multinops`: A tuple of Zygote-computed derivatives of the n-ary operators (where n can be any integer >= 3).
- `diff_binops`: A tuple of Zygote-computed derivatives of the binary operators.
- `diff_unaops`: A tuple of Zygote-computed derivatives of the unary operators.
"""
struct GenericOperatorEnum <: AbstractOperatorEnum
    multinops::Vector{Tuple{Function, Int64}}
    binops::Vector{Function}
    unaops::Vector{Function}
end

end
