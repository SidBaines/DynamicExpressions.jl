module EquationUtilsModule

import Compat: Returns
import ..EquationModule: AbstractNode, Node, copy_node, tree_mapreduce, any, filter_map

"""
    count_nodes(tree::AbstractNode)::Int

Count the number of nodes in the tree.
"""
count_nodes(tree::AbstractNode) = tree_mapreduce(_ -> 1, +, tree)
# This code is given as an example. Normally we could just use sum(Returns(1), tree).

"""
    count_depth(tree::AbstractNode)::Int

Compute the max depth of the tree.
"""
function count_depth(tree::AbstractNode)
    return tree_mapreduce(Returns(1), (p, child...) -> p + max(child...), tree)
end

"""
    is_node_constant(tree::Node)::Bool

Check if the current node in a tree is constant.
"""
@inline is_node_constant(tree::Node) = tree.degree == 0 && tree.constant

"""
    count_constants(tree::Node)::Int

Count the number of constants in a tree.
"""
count_constants(tree::Node) = count(is_node_constant, tree)

"""
    has_constants(tree::Node)::Bool

Check if a tree has any constants.
"""
has_constants(tree::Node) = any(is_node_constant, tree)

"""
    has_operators(tree::Node)::Bool

Check if a tree has any operators.
"""
has_operators(tree::Node) = tree.degree != 0

"""
    is_constant(tree::Node)::Bool

Check if an expression is a constant numerical value, or
whether it depends on input features.
"""
is_constant(tree::Node) = all(t -> t.degree != 0 || t.constant, tree)

"""
    get_constants(tree::Node{T})::Vector{T} where {T}

Get all the constants inside a tree, in depth-first order.
The function `set_constants!` sets them in the same order,
given the output of this function.
"""
function get_constants(tree::Node{T}) where {T}
    return filter_map(is_node_constant, t -> (t.val::T), tree, T)
end

"""
    set_constants!(tree::Node{T}, constants::AbstractVector{T}) where {T}

Set the constants in a tree, in depth-first order.
The function `get_constants` gets them in the same order,
"""
function set_constants!(tree::Node{T}, constants::AbstractVector{T}) where {T}
    if tree.degree == 0
        if tree.constant
            tree.val = constants[1]
        end
    elseif tree.degree == 1
        set_constants!(tree.children[1], constants)
    elseif tree.degree == 2
        numberLeft = count_constants(tree.children[1])
        set_constants!(tree.children[1], constants)
        set_constants!(tree.children[2], @view constants[(numberLeft + 1):end])
    else
        numberLeft=0
        for child in enumerate(tree.children)
            set_constants!(child, @view constants[(numberLeft + 1):end])
            numberLeft += count_constants(child)
        end
    end
    return nothing
end

## Assign index to nodes of a tree
# This will mirror a Node struct, rather
# than adding a new attribute to Node.
mutable struct NodeIndex
    constant_index::UInt16  # Index of this constant (if a constant exists here)
    children::Vector{NodeIndex}

    NodeIndex() = new()
end

function index_constants(tree::Node)::NodeIndex
    return index_constants(tree, UInt16(0))
end

function index_constants(tree::Node, left_index)::NodeIndex
    index_tree = NodeIndex()
    index_constants!(tree, index_tree, left_index)
    return index_tree
end

# Count how many constants to the left of this node, and put them in a tree
function index_constants!(tree::Node, index_tree::NodeIndex, left_index)
    if tree.degree == 0
        if tree.constant
            index_tree.constant_index = left_index + 1
        end
    elseif tree.degree == 1
        index_tree.constant_index = count_constants(tree.children[1])
        index_tree.children = [NodeIndex()]
        index_constants!(tree.children[1], index_tree.children[1], left_index)
    elseif tree.degree == 2
        index_tree.children = [NodeIndex(), NodeIndex()]
        index_constants!(tree.children[1], index_tree.children[1], left_index)
        index_tree.constant_index = count_constants(tree.children[1])
        left_index_here = left_index + index_tree.constant_index
        index_constants!(tree.children[2], index_tree.children[2], left_index_here)
    else
        index_tree.children = []
        for n in 1:length(tree.children)
            push!(index_tree, NodeIndex())
        end
        for n in 1:length(tree.children)
            if n == 1
                left_index_here = left_index
                index_constants!(tree.children[n], index_tree.children[n], left_index_here)
                index_tree.constant_index = count_constants(tree.children[n])
            else
                left_index_here += count_constants(tree.children[n-1])
                index_constants!(tree.children[n], index_tree.children[n], left_index_here)
            end
        end
    end
    return nothing
end

end
