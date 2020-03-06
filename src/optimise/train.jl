using Juno
import Zygote: gradient


"""
    update!(opt, p, g)

Perform an update step of the parameters `ps` (or the single parameter `p`) 
according to optimizer `opt`  and the gradients `gs` (the gradient `g`).

As a result, the parameters are mutated and the optimizer's internal state may change. 

  update!(x, x̄)
  
Update the array `x` according to `x .-= x̄`.
"""
function update!(x::AbstractArray, x̄)
  x .-= x̄
  return
end

# skip, if gradient is nothing
update!(x::AbstractArray, x̄::Nothing) = nothing
update!(opt, x::AbstractArray, x̄::Nothing) = nothing
update!(opt, m::M, ∇m::Nothing) where M = nothing

function update!(opt, x::AbstractArray, x̄)
  x .-= apply!(opt, x, x̄)
  return
end

# NOTE: since there won't be real loop in a struct
#       we could always flatten it, which is a bit
#       faster.
@generated function update!(opt, m::M, ∇m) where M
  body = Expr(:block)
  for each in fieldnames(M)
    each = QuoteNode(each)
    push!(body.args, :(update!(opt, getfield(m, $each), getfield(∇m, $each))))
  end
  return body
end

# Callback niceties
call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)

struct StopException <: Exception end

"""
    stop()

Call `Flux.stop()` in a callback to indicate when a callback condition is met.
This would trigger the train loop to stop and exit.

```julia
# Example callback:

cb = function ()
  accuracy() > 0.9 && Flux.stop()
end
```
"""
function stop()
  throw(StopException())
end
