module DynamicExpressionsZygoteExt

import Zygote: gradient
import DynamicExpressions.EvaluateEquationDerivativeModule: _zygote_gradient

function _zygote_gradient(op::F, ::Val{1}) where {F}
    function (x)
        out = gradient(op, x)[1]
        return out === nothing ? zero(x) : out
    end
end
function _zygote_gradient(op::F, ::Val{2}) where {F}
    function (x, y)
        (∂x, ∂y) = gradient(op, x, y)
        return (∂x === nothing ? zero(x) : ∂x, ∂y === nothing ? zero(y) : ∂y)
    end
end

# This is sketchy, but I'm not sure it's used. ToDo (Sid) Should check this if it's going to be made public
function _zygote_gradient(op::F, ::Val{3}) where {F}
    function args...
        dargs = gradient(op, args)
        return (dargs[n] === nothing ? zero(args[n]) : dargs[n] for n in 1:length(args))
    end
end

end
