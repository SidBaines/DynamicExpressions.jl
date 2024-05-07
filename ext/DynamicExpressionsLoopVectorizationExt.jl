module DynamicExpressionsLoopVectorizationExt

using LoopVectorization: @turbo
using DynamicExpressions: AbstractExpressionNode
using DynamicExpressions.UtilsModule: ResultOk, fill_similar
using DynamicExpressions.EvaluateEquationModule: @return_on_check
import DynamicExpressions.EvaluateEquationModule:
    deg1_eval,
    deg2_eval,
    degany_eval,
    deg1_l2_ll0_lr0_eval,
    deg1_l1_ll0_eval,
    deg2_l0_r0_eval,
    deg2_l0_eval,
    deg2_r0_eval,
    degany_allc0_eval
import DynamicExpressions.ExtensionInterfaceModule:
    _is_loopvectorization_loaded, bumper_kern1!, bumper_kern2!

_is_loopvectorization_loaded(::Int) = true

# TODO Maybe add the below? I removed it for now because I don't really know what turbo is doing / applying it 'naively' as below causes tests to fail
# function degany_eval( # TODO: NB I haven't tried to optimise this; feels like there's probably a lot to gain here
#     # cumulators::AbstractVector{AbstractVector{T}}, op::F, ::Val{false} # TODO: Why won't it work with AbstractVector??
#     cumulators::Vector{Vector{T}}, op::F, ::Val{true}
# )::ResultOk where {T<:Number,F}
#     @turbo for j in eachindex(cumulators[1])
#         x = op((cumulator[j] for cumulator in cumulators)...)
#         cumulators[1][j] = x
#     end
#     return ResultOk(cumulators[1], true)
# end

function deg2_eval(
    cumulator_l::AbstractVector{T}, cumulator_r::AbstractVector{T}, op::F, ::Val{true}
)::ResultOk where {T<:Number,F}
    @turbo for j in eachindex(cumulator_l)
        x = op(cumulator_l[j], cumulator_r[j])
        cumulator_l[j] = x
    end
    return ResultOk(cumulator_l, true)
end

function deg1_eval(
    cumulator::AbstractVector{T}, op::F, ::Val{true}
)::ResultOk where {T<:Number,F}
    @turbo for j in eachindex(cumulator)
        x = op(cumulator[j])
        cumulator[j] = x
    end
    return ResultOk(cumulator, true)
end

function deg1_l2_ll0_lr0_eval(
    tree::AbstractExpressionNode{T}, cX::AbstractMatrix{T}, op::F, op_l::F2, ::Val{true}
) where {T<:Number,F,F2}
    if tree.children[1].children[1].constant && tree.children[1].children[2].constant
        val_ll = tree.children[1].children[1].val
        val_lr = tree.children[1].children[2].val
        @return_on_check val_ll cX
        @return_on_check val_lr cX
        x_l = op_l(val_ll, val_lr)::T
        @return_on_check x_l cX
        x = op(x_l)::T
        @return_on_check x cX
        return ResultOk(fill_similar(x, cX, axes(cX, 2)), true)
    elseif tree.children[1].children[1].constant
        val_ll = tree.children[1].children[1].val
        @return_on_check val_ll cX
        feature_lr = tree.children[1].children[2].feature
        cumulator = similar(cX, axes(cX, 2))
        @turbo for j in axes(cX, 2)
            x_l = op_l(val_ll, cX[feature_lr, j])
            x = op(x_l)
            cumulator[j] = x
        end
        return ResultOk(cumulator, true)
    elseif tree.children[1].children[2].constant
        feature_ll = tree.children[1].children[1].feature
        val_lr = tree.children[1].children[2].val
        @return_on_check val_lr cX
        cumulator = similar(cX, axes(cX, 2))
        @turbo for j in axes(cX, 2)
            x_l = op_l(cX[feature_ll, j], val_lr)
            x = op(x_l)
            cumulator[j] = x
        end
        return ResultOk(cumulator, true)
    else
        feature_ll = tree.children[1].children[1].feature
        feature_lr = tree.children[1].children[2].feature
        cumulator = similar(cX, axes(cX, 2))
        @turbo for j in axes(cX, 2)
            x_l = op_l(cX[feature_ll, j], cX[feature_lr, j])
            x = op(x_l)
            cumulator[j] = x
        end
        return ResultOk(cumulator, true)
    end
end

function deg1_l1_ll0_eval(
    tree::AbstractExpressionNode{T}, cX::AbstractMatrix{T}, op::F, op_l::F2, ::Val{true}
) where {T<:Number,F,F2}
    if tree.children[1].children[1].constant
        val_ll = tree.children[1].children[1].val
        @return_on_check val_ll cX
        x_l = op_l(val_ll)::T
        @return_on_check x_l cX
        x = op(x_l)::T
        @return_on_check x cX
        return ResultOk(fill_similar(x, cX, axes(cX, 2)), true)
    else
        feature_ll = tree.children[1].children[1].feature
        cumulator = similar(cX, axes(cX, 2))
        @turbo for j in axes(cX, 2)
            x_l = op_l(cX[feature_ll, j])
            x = op(x_l)
            cumulator[j] = x
        end
        return ResultOk(cumulator, true)
    end
end

function deg2_l0_r0_eval(
    tree::AbstractExpressionNode{T}, cX::AbstractMatrix{T}, op::F, ::Val{true}
) where {T<:Number,F}
    if tree.children[1].constant && tree.children[2].constant
        val_l = tree.children[1].val
        @return_on_check val_l cX
        val_r = tree.children[2].val
        @return_on_check val_r cX
        x = op(val_l, val_r)::T
        @return_on_check x cX
        return ResultOk(fill_similar(x, cX, axes(cX, 2)), true)
    elseif tree.children[1].constant
        cumulator = similar(cX, axes(cX, 2))
        val_l = tree.children[1].val
        @return_on_check val_l cX
        feature_r = tree.children[2].feature
        @turbo for j in axes(cX, 2)
            x = op(val_l, cX[feature_r, j])
            cumulator[j] = x
        end
        return ResultOk(cumulator, true)
    elseif tree.children[2].constant
        cumulator = similar(cX, axes(cX, 2))
        feature_l = tree.children[1].feature
        val_r = tree.children[2].val
        @return_on_check val_r cX
        @turbo for j in axes(cX, 2)
            x = op(cX[feature_l, j], val_r)
            cumulator[j] = x
        end
        return ResultOk(cumulator, true)
    else
        cumulator = similar(cX, axes(cX, 2))
        feature_l = tree.children[1].feature
        feature_r = tree.children[2].feature
        @turbo for j in axes(cX, 2)
            x = op(cX[feature_l, j], cX[feature_r, j])
            cumulator[j] = x
        end
        return ResultOk(cumulator, true)
    end
end

# op(x, y) for x variable/constant, y arbitrary
function deg2_l0_eval(
    tree::AbstractExpressionNode{T},
    cumulator::AbstractVector{T},
    cX::AbstractArray{T},
    op::F,
    ::Val{true},
) where {T<:Number,F}
    if tree.children[1].constant
        val = tree.children[1].val
        @return_on_check val cX
        @turbo for j in eachindex(cumulator)
            x = op(val, cumulator[j])
            cumulator[j] = x
        end
        return ResultOk(cumulator, true)
    else
        feature = tree.children[1].feature
        @turbo for j in eachindex(cumulator)
            x = op(cX[feature, j], cumulator[j])
            cumulator[j] = x
        end
        return ResultOk(cumulator, true)
    end
end

function deg2_r0_eval(
    tree::AbstractExpressionNode{T},
    cumulator::AbstractVector{T},
    cX::AbstractArray{T},
    op::F,
    ::Val{true},
) where {T<:Number,F}
    if tree.children[2].constant
        val = tree.children[2].val
        @return_on_check val cX
        @turbo for j in eachindex(cumulator)
            x = op(cumulator[j], val)
            cumulator[j] = x
        end
        return ResultOk(cumulator, true)
    else
        feature = tree.children[2].feature
        @turbo for j in eachindex(cumulator)
            x = op(cumulator[j], cX[feature, j])
            cumulator[j] = x
        end
        return ResultOk(cumulator, true)
    end
end

# TODO Maybe add the below? I removed it for now because I don't really know what turbo is doing / applying it 'naively' as below causes tests to fail
# function degany_allc0_eval(
#     tree::AbstractExpressionNode{T}, cX::AbstractMatrix{T}, op::F, ::Val{true}
# ) where {T<:Number,F}
#     # TODO maybe there's room for improvement here? I did this quite naively
#     if all(child.constant for child in tree.children)
#         vals = Zeros(T, tree.degree)
#         for (cn, child) in enumerate(tree.children)
#             vals[cn] = child.val
#             @return_on_check vals[cn] cX
#         end
#         x = op(vals...)
#         @return_on_check x cX
#         return ResultOk(fill_similar(x, cX, axes(cX, 2)), true)
#     else
#         cumulator = similar(cX, axes(cX, 2))
#         @turbo for j in axes(cX, 2)
#             x = op((child.constant ? child.val : cX[child.feature, j] for child in tree.children)...)
#             cumulator[j] = x
#         end
#         return ResultOk(cumulator, true)
#     end
# end

## Interface with Bumper.jl
function bumper_kern1!(op::F, cumulator, ::Val{true}) where {F}
    @turbo @. cumulator = op(cumulator)
    return cumulator
end
function bumper_kern2!(op::F, cumulator1, cumulator2, ::Val{true}) where {F}
    @turbo @. cumulator1 = op(cumulator1, cumulator2)
    return cumulator1
end

end
