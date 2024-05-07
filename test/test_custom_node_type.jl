using DynamicExpressions
using Test

mutable struct MyCustomNode{A,B} <: AbstractNode
    degree::Int
    val1::A
    val2::B
    # l::MyCustomNode{A,B}
    # r::MyCustomNode{A,B}
    children::Vector{MyCustomNode{A,B}}

    MyCustomNode(val1, val2) = new{typeof(val1),typeof(val2)}(0, val1, val2)
    MyCustomNode(val1, val2, l) = new{typeof(val1),typeof(val2)}(1, val1, val2, [l])
    MyCustomNode(val1, val2, l, r) = new{typeof(val1),typeof(val2)}(2, val1, val2, [l,r])
    MyCustomNode(val1, val2, children::Vector) = new{typeof(val1),typeof(val2)}(length(children), val1, val2, children)
end

node1 = MyCustomNode(1.0, 2)

@test typeof(node1) == MyCustomNode{Float64,Int}
@test node1.degree == 0
@test count_depth(node1) == 1
@test count_nodes(node1) == 1

node2 = MyCustomNode(1.5, 3, node1)

@test typeof(node2) == MyCustomNode{Float64,Int}
@test node2.degree == 1
@test node2.children[1].degree == 0
@test count_depth(node2) == 2
@test count_nodes(node2) == 2

node2 = MyCustomNode(1.5, 3, node1, node1)

@test count_depth(node2) == 2
@test count_nodes(node2) == 3
@test sum(t -> t.val1, node2) == 1.5 + 1.0 + 1.0
@test sum(t -> t.val2, node2) == 3 + 2 + 2
@test count(t -> t.degree == 0, node2) == 2
