module OperatorEnumModule

abstract type AbstractOperatorEnum end

"""
    OperatorEnum

Defines an enum over operators, along with their derivatives.
# Fields
- `multiops`: A tuple of n-ary operators (where n can be any integer >= 3).
- `binops`: A tuple of binary operators. Scalar input type.
- `unaops`: A tuple of unary operators. Scalar input type.
- `diff_multiops`: A tuple of Zygote-computed derivatives of the n-ary operators (where n can be any integer >= 3).
- `diff_binops`: A tuple of Zygote-computed derivatives of the binary operators.
- `diff_unaops`: A tuple of Zygote-computed derivatives of the unary operators.
"""
struct OperatorEnum <: AbstractOperatorEnum
    multiops::Vector{Function}
    binops::Vector{Function}
    unaops::Vector{Function}
    diff_multiops::Vector{Function}
    diff_binops::Vector{Function}
    diff_unaops::Vector{Function}
end

"""
    OperatorEnum

Defines an enum over operators, along with their derivatives.
# Fields
- `multiops`: A tuple of n-ary operators (where n can be any integer >= 3).
- `binops`: A tuple of binary operators.
- `unaops`: A tuple of unary operators.
- `diff_multiops`: A tuple of Zygote-computed derivatives of the n-ary operators (where n can be any integer >= 3).
- `diff_binops`: A tuple of Zygote-computed derivatives of the binary operators.
- `diff_unaops`: A tuple of Zygote-computed derivatives of the unary operators.
"""
struct GenericOperatorEnum <: AbstractOperatorEnum
    multiops::Vector{Function}
    binops::Vector{Function}
    unaops::Vector{Function}
end

end
