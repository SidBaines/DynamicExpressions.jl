module OperatorEnumModule

abstract type AbstractOperatorEnum end

"""
    OperatorEnum

Defines an enum over operators, along with their derivatives.
# Fields
- `binops`: A tuple of binary operators. Scalar input type.
- `unaops`: A tuple of unary operators. Scalar input type.
"""
struct OperatorEnum{A,B,U} <: AbstractOperatorEnum
    anyops::A
    binops::B
    unaops::U
end

"""
    GenericOperatorEnum

Defines an enum over operators, along with their derivatives.
# Fields
- `binops`: A tuple of binary operators.
- `unaops`: A tuple of unary operators.
"""
struct GenericOperatorEnum{A,B,U} <: AbstractOperatorEnum
    anyops::A
    binops::B
    unaops::U
end

end
