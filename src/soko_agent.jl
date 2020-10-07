import Cambrian: ind_parse
"SokoAgent : Individual class using a floating point genotype in [0, 1]"
struct SokoAgent <: Cambrian.Individual
    genes::Array{Float64}
    fitness::Array{Float64}
    width::Int64
    height::Int64
    nb_object::Int64
    model
end

function SokoAgent(model,width::Int64,height::Int64,nb_object::Array{String})::SokoAgent
    nb_params = get_params_count(model)
    SokoAgent(rand(nb_params), -Inf*ones(1),width,height,nb_object,deepcopy(model))
end

function SokoAgent(model,cfg::NamedTuple)::SokoAgent
    width = cfg.width
    height = cfg.height
    nb_object = cfg.nb_object
    nb_params = get_params_count(model)
    SokoAgent(rand(nb_params), -Inf*ones(cfg.d_fitness),width,height,nb_object,deepcopy(model))
end

function SokoAgent(genes::Array{Float64}, cfg::NamedTuple)::SokoAgent
    width = cfg.width
    height = cfg.height
    nb_object = cfg.nb_object
    nb_params = get_params_count(model)
    if length(genes) == nb_params
        SokoAgent(genes, -Inf*ones(cfg.d_fitness),width,height,nb_object,deepcopy(model))
    else
        throw("The size of the genes you provided doesn't match with the nb of parameters of your model")
    end
end

function SokoAgent(genes::Array{Float64}, model, cfg::NamedTuple)::SokoAgent
    width = cfg.width
    height = cfg.height
    nb_object = cfg.nb_object
    nb_params = get_params_count(model)
    if length(genes) == nb_params
        SokoAgent(genes, -Inf*ones(cfg.d_fitness),width,height,nb_object,deepcopy(model))
    else
        throw("The size of the genes you provided doesn't match with the nb of parameters of your model")
    end
end

function SokoAgent(st::String,model)
    dict = ind_parse(st)
    SokoAgent(Float64.(dict["genes"]), Float64.(dict["fitness"]), dict["width"], dict["height"],dict["nb_object"],deepcopy(model))
end

function get_child(parent::SokoAgent, genes::AbstractArray)
    typeof(parent)(genes,  -Inf*ones(1),parent.width,parent.height,parent.nb_object,deepcopy(parent.model))
end

"""
    mutate(parent::SokoAgent, m_rate::Float64)

To use, define
    mutate(parent::SokoAgent) = mutate(parent, m_rate)
"""
function mutate(parent::SokoAgent, m_rate::Float64)
    inds = rand(length(parent.genes)) .> m_rate
    genes = rand(length(parent.genes))
    genes[inds] = parent.genes[inds]
    get_child(parent, genes)
end

"""
    transcript_sokoagent_genes(sokoagent::SokoAgent, model)

This function allow to translate the genes as the weights of our model.
"""
function transcript_sokoagent_genes!(sokoagent::SokoAgent)
    load_weights_from_array!(sokoagent.model,sokoagent.genes)
end

"""
    choose_action(observation,sokoagent::SokoAgent)
Apply the model to our observation and choose the action idx with the maximum value.
"""
function choose_action(observation,sokoagent::SokoAgent)
    obs = permutedims(observation,[3,2,1])
    obs = Float32.(reshape(obs,(sokoagent.width,sokoagent.height,sokoagent.nb_object,1)))

    output = sokoagent.model(obs)
    return argmax(output)[1]
end

"""
    choose_action(observation,sokoagent::SokoAgent,frame_history_len::Int64)
Apply the model to our observation and choose the action idx with the maximum value.
"""
function choose_action(observations,sokoagent::SokoAgent,frame_history_len::Int64)
    input = Array{Float32}(undef, size(permutedims(observations[1],[3,2,1]))..., frame_history_len)
    for i in eachindex(observations)
        obs = permutedims(observations[i],[3,2,1])
        obs = Float32.(obs)
        input[:,:,:,i] = obs
    end
    output = sokoagent.model(input)
    return argmax(output)[1]
end

function save_ind(sokoagent::SokoAgent,path)
    f = open(path, "w+")
    write(f,"""{"genes":""")
    write(f, string(sokoagent.genes))
    write(f,""","fitness":""")
    write(f, string(sokoagent.fitness))
    write(f,""","width":""")
    write(f, string(sokoagent.width))
    write(f,""","height":""")
    write(f, string(sokoagent.height))
    write(f,""","nb_object":""")
    write(f, string(sokoagent.nb_object))
    write(f,"""}""")
    close(f)
end
