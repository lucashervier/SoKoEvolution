using Flux

function get_params_count(model)
    size = 0
    ps = Flux.params(model)
    for layer in ps
        size += length(layer)
    end
    return size
end

function load_weights_from_array!(model,weights)
    nb_params = get_params_count(model)
    nb_weight = length(weights)
    if nb_params > nb_weight
        throw("Your weight vector is not long enough")
    elseif nb_params < nb_weight
        @warn("Your weight vector have more element than you have parameters to change")
    end
    ps = Flux.params(model)
    layer_idx = 1
    curr_idx = 1
    for layer in ps
        for i in eachindex(layer)
            ps[layer_idx][i] = weights[curr_idx]
            curr_idx += 1
        end
        layer_idx +=1
    end
end
