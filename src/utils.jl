using Flux

function get_params_count(model)
    size = 0
    ps = Flux.params(model)
    for layers in ps
        size += length(layer)
    end
    return size
end

function load_weights_from_array!(model,weights)
    if get_params_count(model) > length(weights)
        throw("Your weights vector is not long enough")
    elseif get_params_count(model) < length(weights)
        @warn("Your weights vector have more element than you have parameters to change")
    end
    ps = Flux.params(model)
    layer_idx = 1
    curr_idx = 1
    for layer in ps
        for i in eachindex(layer)
            ps[layer_idx][elt_idx] = weights[curr_idx]
            curr_idx += 1
        end
        layer_idx +=1
    end
end
