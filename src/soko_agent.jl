function get_child(parent::Individual, genes::AbstractArray)
    typeof(parent)(genes,  -Inf*ones(1),parent.width,parent.height,parent.nb_object,deepcopy(parent.model))
end

"SokoAgent : Individual class using a floating point genotype in [0, 1]"
struct SokoAgent <: Cambrian.Individual
    genes::Array{Float64}
    fitness::Array{Float64}
    width::Int64
    height::Int64
    nb_object::Int64
    model::Any
end

function SokoAgent(model,width::Int64,height::Int64,nb_object::Array{String})::SokoAgent
    nb_params = get_params_count(model)
    SokoAgent(rand(nb_params), -Inf*ones(1),width,height,nb_object,deepcopy(model))
end

function SokoLvlIndividual(model,cfg::NamedTuple)::SokoAgent
    width = cfg.width
    height = cfg.height
    nb_object = cfg.nb_object
    SokoAgent(rand(nb_params), -Inf*ones(cfg.d_fitness),width,height,nb_object,deepcopy(model))
end

function SokoAgent(model,genes::Array{Float64}, cfg::NamedTuple)::SokoAgent
    width = cfg.width
    height = cfg.height
    nb_object = cfg.nb_object
    if length(genes) == get_params_count(model)
        SokoAgent(genes, -Inf*ones(cfg.d_fitness),width,height,nb_object,deepcopy(model))
    else
        throw("The size of the genes you provided doesn't match with the nb of parameters of your model")
    end
end

"""
    mutate(parent::SokoAgent, m_rate::Float64)

To use, define
    mutate(parent::SokoLvlIndividual) = mutate(parent, m_rate)
"""
function mutate(parent::SokoAgent, m_rate::Float64)
    inds = rand(length(parent.genes)) .> m_rate
    genes = rand(length(parent.genes))
    genes[inds] = parent.genes[inds]
    get_child(parent, genes)
end

"""
    transcript_sokoagent_genes(sokoagent::SokoAgent)

This function allow to translate the genes as the weights of our model.
"""
function transcript_sokolvl_genes!(sokoagent::SokoAgent)
    load_weights_from_array!(sokoagent.model,sokoagent.genes)
end

function choose_action(observation,sokoagent::SokoAgent)
    obs = reshape(observation,(sokoagent.width,sokoagent.height,sokoagent.nb_object,1))
    output = sokoagent.model(obs)
    return argmax(output)
end
