import Cambrian: ind_parse

"ContinuousSokoLvl : Individual class using an Array of float as genes, those
genes will represents the weights of a Neural Network"
struct ContinuousSokoLvl <: Cambrian.Individual
    genes::Array{Float64}
    fitness::Array{Float64}
    width::Int64
    height::Int64
    objects_char_list::Array{String}
    agent_idx::Int64
    model
    output_map::Array{String}
end

function ContinuousSokoLvl(model,width::Int64,height::Int64,objects_char_list::Array{String},agent_idx::Int64)::ContinuousSokoLvl
    nb_params = get_params_count(model)
    ContinuousSokoLvl(rand(nb_params), -Inf*ones(1),width,height,objects_char_list,agent_idx,deepcopy(model),[""""""])
end

function ContinuousSokoLvl(model,cfg::NamedTuple)::ContinuousSokoLvl
    width = cfg.width
    height = cfg.height
    objects_char_list = cfg.objects_char_list
    agent_idx = cfg.agent_idx
    nb_params = get_params_count(model)
    ContinuousSokoLvl(rand(nb_params), -Inf*ones(cfg.d_fitness),width,height,objects_char_list,agent_idx,deepcopy(model),[""""""])
end

function ContinuousSokoLvl(genes::Array{Float64}, model, cfg::NamedTuple)::ContinuousSokoLvl
    width = cfg.width
    height = cfg.height
    objects_char_list = cfg.objects_char_list
    agent_idx = cfg.agent_idx
    nb_params = get_params_count(model)
    if length(genes) == nb_params
        ContinuousSokoLvl(genes, -Inf*ones(cfg.d_fitness),width,height,objects_char_list,agent_idx,deepcopy(model),[""""""])
    else
        throw("The size of the genes you provided doesn't match with the nb of parameters of your model")
    end
end

function ContinuousSokoLvl(genes::Array{Float64}, model, cfg::NamedTuple)::ContinuousSokoLvl
    width = cfg.width
    height = cfg.height
    objects_char_list = cfg.objects_char_list
    agent_idx = cfg.agent_idx
    nb_params = get_params_count(model)
    if length(genes) == nb_params
        ContinuousSokoLvl(genes, -Inf*ones(cfg.d_fitness),width,height,objects_char_list,agent_idx,deepcopy(model),[""""""])
    else
        throw("The size of the genes you provided doesn't match with the nb of parameters of your model")
    end
end

function ContinuousSokoLvl(st::String,model)
    dict = ind_parse(st)
    ContinuousSokoLvl(Float64.(dict["genes"]), Float64.(dict["fitness"]), dict["width"], dict["height"],dict["objects_char_list"],dict["agent_idx"],deepcopy(model),dict["output_map"])
end

function get_child(parent::ContinuousSokoLvl, genes::AbstractArray)
    typeof(parent)(genes,  -Inf*ones(1),parent.width,parent.height,parent.objects_char_list,parent.agent_idx,deepcopy(parent.model),[""""""])
end

"""
    mutate(parent::ContinuousSokoLvl, m_rate::Float64)

To use, define
    mutate(parent::ContinuousSokoLvl) = mutate(parent, m_rate)
"""
function mutate(parent::ContinuousSokoLvl, m_rate::Float64)
    inds = rand(length(parent.genes)) .> m_rate
    genes = rand(length(parent.genes))
    genes[inds] = parent.genes[inds]
    get_child(parent, genes)
end

"""
    apply_continuoussokolvl_genes(continuoussokolvl::ContinuousSokoLvl)

This function allow to translate the genes as the weights of our model.
"""
function apply_continuoussokolvl_genes!(continuoussokolvl::ContinuousSokoLvl)
    load_weights_from_array!(continuoussokolvl.model,continuoussokolvl.genes)
end

"""
"""
function write_map!(continuoussokolvl::ContinuousSokoLvl)
    # first get the size of the level
    width = continuoussokolvl.width
    height = continuoussokolvl.height
    # get the list of objects the level may include
    objects_char_list = continuoussokolvl.objects_char_list
    # to know if the agent was selected
    agent_idx = continuoussokolvl.agent_idx
    agent_in_place = false

    lvl_str = """"""

    for y_pos in 1:height
        for x_pos in 1:width
            input = [x_pos,y_pos]
            # our model got 5 output: box,wall,holes,agent and floor
            object_at_x_y = ""
            output = continuoussokolvl.model(input)
            idx_object = argmax(output)[1]
            if !(agent_in_place)&&(idx_object==agent_idx)
                agent_in_place = true
            elseif (agent_in_place)&&(idx_object==agent_idx)
                output[agent_idx] = 0
                idx_object = argmax(output)[1]
            end
            object_at_x_y = objects_char_list[idx_object]
            lvl_str = string(lvl_str,object_at_x_y)
        end
        lvl_str = string(lvl_str,"\n")
    end
    # check if there is at least one agent otherwise put one in the middle
    if !(agent_in_place)
        # x_agent = rand(1:width)
        # y_agent = rand(1:height)
        x_agent = Int(round(width/2))
        y_agent = Int(round(height/2))
        lvl_vec = Vector{Char}(lvl_str)
        lvl_vec[x_agent+(y_agent-1)*5] = 'A'
        lvl_str = String(lvl_vec)
    end
    continuoussokolvl.output_map[1] = lvl_str
    return lvl_str
end

function from_matrix_map_to_str(matrix_map::Array{Int64,2},objects_char_list::Array{String})
    width, height = size(matrix_map)
    lvl_str = """"""
    for y_pos in 1:height
        for x_pos in 1:width
            object_idx = matrix_map[x_pos,y_pos]
            object_at_x_y = objects_char_list[object_idx]
            lvl_str = string(lvl_str,object_in_x_y)
        end
        lvl_str = string(lvl_str,"\n")
    end
    return  lvl_str
end

function write_map2(continuoussokolvl::ContinuousSokoLvl)
    # first get the size of the level
    width = continuoussokolvl.width
    height = continuoussokolvl.height
    # get the list of objects the level may include
    objects_char_list = continuoussokolvl.objects_char_list
    # to know where to place agent
    x_agents = []
    y_agents = []

    matrix_map = zeros(Int,continuoussokolvl.width,continuoussokolvl.height)

    for y_pos in 1:height
        for x_pos in 1:width
            input = [x_pos,y_pos]
            # our model got 5 output: box,wall,holes,agent and floor
            object_at_x_y = 0
            output = continuoussokolvl.model(input)
            idx_object = argmax(output)[1]
            if idx_object == agent_idx
                push!(x_agents,x_pos)
                push!(y_agents,x_pos)
                matrix_map[x_pos,y_pos] = 5
            else
                matrix_map[x_pos,y_pos] = output
            end
        end
    end
    x_agent = round(mean(x_agents))
    y_agent = round(mean(y_agents))

    matrix_map[x_agent,y_agent] = agent_idx
    lvl_str = from_matrix_map_to_str(matrix_map,objects_char_list)
    continuoussokolvl.output_map = lvl_str
    return lvl_str
end

function save_ind(continuoussokolvl::ContinuousSokoLvl,path)
    f = open(path, "w+")
    write(f,"""{"genes":""")
    write(f, string(continuoussokolvl.genes))
    write(f,""","fitness":""")
    write(f, string(continuoussokolvl.fitness))
    write(f,""","width":""")
    write(f, string(continuoussokolvl.width))
    write(f,""","height":""")
    write(f, string(continuoussokolvl.height))
    write(f,""","objects_char_list":""")
    write(f, string(continuoussokolvl.objects_char_list))
    write(f,""","agent_idx":""")
    write(f, string(continuoussokolvl.agent_idx))
    write(f,""","output_map":""")
    write(f, string(continuoussokolvl.output_map))
    write(f,"""}""")
    close(f)
end
