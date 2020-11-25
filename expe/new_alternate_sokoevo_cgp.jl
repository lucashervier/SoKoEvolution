using Griddly
using Cambrian
using EvolutionaryStrategies
using CartesianGeneticProgramming
using Flux
using Statistics
using Random
using Formatting
using CSV
using DataFrames
using ProgressBars
using Logging
import Cambrian: mutate, crossover, populate, save_gen, generation, log_gen, step!
import EvolutionaryStrategies: snes_populate, snes_generation

include("../src/cgp_sokolvl.jl")
include("../src/utils.jl")
include("../src/soko_agent.jl")
#----------------------------Named Parameters----------------------------------#
game_name = "sokoban3"
expe_name = "new_alternate_sokoevo_cgp"
#----------------------------Griddly Resources---------------------------------#
image_path = joinpath(@__DIR__,"..","resources","images")
shader_path = joinpath(@__DIR__,"..","resources","shaders")
gdy_path = joinpath(@__DIR__,"..","resources","games")
gdy_reader = Griddly.GDYReader(image_path,shader_path)
#--------------------------Griddly Initialization------------------------------#
grid = Griddly.load!(gdy_reader,joinpath(gdy_path,"Single-Player/GVGAI/$game_name.yaml"))
game = Griddly.create_game(grid,Griddly.VECTOR)
player1 = Griddly.register_player!(game,"Tux", Griddly.VECTOR)
Griddly.init!(game)
#--------------------------Configuration for Envs------------------------------#
# if direct encoding for envs
cfg_envs = Cambrian.get_config("cfg/sokoevo_cgp_envs.yaml")
# overrides of the mutate function
mutate(i::CGPSokoLvl) = CGPSokoLvl(i.width,i.height,i.objects_char_list,i.agent_idx,[""],goldman_mutate(cfg_envs,i.cgp))
#-------------------------Configuration for Agents-----------------------------#
cfg_agent = Cambrian.get_config("cfg/sokoevo_continuous_agents.yaml")
# specify your Agent model here
agent_model = Chain(
                    Conv((3,3),5=>1,pad=(1,1),relu),
                    Flux.flatten,
                    RNN(64,64),
                    Dense(64,4),
                    softmax
                    )
println("nb_params:$(get_params_count(agent_model))")
# overrides of the mutate function
mutate(i::SokoAgent) = mutate(i, cfg_agent.p_mutation)
# sNES overrides
populate(e::sNES{SokoAgent}) = snes_populate(e)
generation(e::sNES{SokoAgent}) = snes_generation(e)
#---------------------------Cambrian Helpers-----------------------------------#
selection(pop::Array{<:Individual}) = Cambrian.tournament_selection(pop, cfg_envs.tournament_size)
# overrides save_gen for a coevolution
function save_gen(e1::AbstractEvolution,e2::AbstractEvolution;id1="envs",id2="agents")
    path1 = Formatting.format("gens/$expe_name/{1}/{2:06d}",id1, e1.gen)
    mkpath(path1)
    sort!(e1.population)
    for i in eachindex(e1.population)
        path_ind = Formatting.format("{1}/{2:04d}.dna", path1, i)
        save_ind(e1.population[i],path_ind)
    end
    path2 = Formatting.format("gens/$expe_name/{1}/{2:06d}",id2, e2.gen)
    mkpath(path2)
    sort!(e2.population)
    for i in eachindex(e2.population)
        path_ind = Formatting.format("{1}/{2:04d}.dna", path2, i)
        save_ind(e2.population[i],path_ind)
    end
end
# # overrides log_gen if you want to add some infos in logs file, to adapt with
# # your experiment, ignore if the classic logs suit you
function log_gen(e::CGPSokoLvlEvolution)
    best = sort(e.population)[end]
    lvl_str = best.output_map[1]
    nb_boxes,nb_holes,nb_objectives,nb_walls,initial_connectivity_number = get_detailed_info(lvl_str)
    random_reward = evaluate_random(lvl_str)
    for d in 1:e.config.d_fitness
        maxs = map(i->i.cgp.fitness[d], e.population)
        with_logger(e.logger) do
            @info Formatting.format("{1:05d},{2:e},{3:e},{4:e},{5:e},{6:e},{7:e},{8:e},{9:e},{10:e}",
                                    e.gen, maximum(maxs), mean(maxs), std(maxs),nb_boxes,nb_holes,nb_objectives,initial_connectivity_number,random_reward,nb_walls)
        end
    end
    flush(e.logger.stream)
end

function log_gen(e::sNES{SokoAgent})
    for d in 1:e.config.d_fitness
        maxs = map(i->i.fitness[d], e.population)
        with_logger(e.logger) do
            @info Formatting.format("{1:05d},{2:e},{3:e},{4:e}",
                                    e.gen, maximum(maxs), mean(maxs), std(maxs))
        end
    end
    flush(e.logger.stream)
end
#---------------------------Evaluate Helpers-----------------------------------#
# get the performance of a random agent on the given level
function evaluate_random(lvl_str::String)
    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)

    total_reward = 0

    for step in 1:200
        dir = rand(1:4)
        reward, done = Griddly.step_player!(player1,"move", [dir])
        total_reward += reward
        if done==1
            break
        end
    end

    return total_reward
end

# get some specific information on a level, change it according to your needs
function get_detailed_info(lvl_str::String)
    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)
    initial_observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))

    nb_boxes = count_items(1,initial_observation)
    nb_holes = count_items(4,initial_observation)
    nb_walls = count_items(2,initial_observation)
    nb_objectives = min(nb_boxes,nb_holes)

    initial_connectivity = get_connectivity_map(initial_observation)
    initial_connectivity_number = length(keys(initial_connectivity))

    return [nb_boxes,nb_holes,nb_objectives,nb_walls,initial_connectivity_number]
end

# the provided agent will make 200 steps on the given level, you can modify it
# according to your needs
function play_lvl(lvl_str::String,agent::SokoAgent)
    transcript_sokoagent_genes!(agent)
    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)

    total_reward = 0
    observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))

    for step in 1:200
        dir = choose_action(observation,agent)
        reward, done = Griddly.step_player!(player1,"move", [dir])
        observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
        total_reward += reward
        if done==1
            break
        end
    end
    return total_reward
end

function play_lvl2(lvl_str::String,agent::SokoAgent)
    transcript_sokoagent_genes!(agent)
    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)

    total_reward = 0
    nb_step = 0

    first_observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
    observation = first_observation

    for step in 1:200
        dir = choose_action(observation,agent)
        reward, done = Griddly.step_player!(player1,"move", [dir])
        nb_step += 1
        observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
        total_reward += reward
        if done==1
            break
        end
    end

    return [total_reward,nb_step]
end

function get_local_evaluation(envs::CGPSokoLvlEvolution,agents::sNES{SokoAgent})
    envs_size = length(envs.population)
    agents_size = length(agents.population)
    local_eval = zeros(envs_size,agents_size)
    for i in 1:envs_size
        lvl_str = write_map!(envs.population[i])
        for j in 1:agents_size
            _,_,nb_objectives,_ = get_detailed_info(lvl_str)
            if nb_objectives == 0
                local_eval[i,j] = 0
            else
                random_reward = evaluate_random(lvl_str)
                agent_reward, nb_step = play_lvl2(lvl_str,agents.population[j])
                local_eval[i,j] = (agent_reward-random_reward)/nb_objectives + (200-nb_step)/200
            end
        end
    end
    return local_eval
end

# The fitness function of your envs depending on your local evaluation
function fitness_env(idx_env::Int64,local_eval::Array{Float64})
    score = maximum(local_eval[idx_env,:])-minimum(local_eval[idx_env,:])
    return score
end

# The fitness function of your agents depending on your local evaluation
function fitness_agent(idx_agent::Int64,local_eval::Array{Float64})
    return [mean(local_eval[:,idx_agent])]
end

# evaluate function for a coevolution based on a score matrix
function evaluate(e1::AbstractEvolution, e2::AbstractEvolution;save_localfit=true)
    local_eval = get_local_evaluation(e1,e2)
    if save_localfit
        path = "localfit/$expe_name"
        mkpath(path)
        CSV.write(Formatting.format("$path/gen-{1:06d}",e1.gen),  DataFrame(local_eval), header=false)
    end
    objectives_max = e1.config.objectives_max
    connectivity_max = e1.config.connectivity_max
    for i in eachindex(e1.population)
        lvl_str = e1.population[i].output_map[1]
        nb_boxes,nb_holes,nb_objectives,_,initial_connectivity = get_detailed_info(lvl_str)
        if nb_objectives<=0
            e1.population[i].cgp.fitness[:] = [-10]
        elseif initial_connectivity<=0
            e1.population[i].cgp.fitness[:] = [-10]
        else
            score = e1.fitness(i,local_eval)
            objectives = - abs((nb_boxes - nb_holes))/nb_objectives -(nb_objectives>objectives_max)
            connectivity = -(initial_connectivity>connectivity_max)
            diff = (nb_objectives<objectives_max) * 0.25 * nb_objectives/objectives_max
            e1.population[i].cgp.fitness[:] = [score + objectives + connectivity]
        end
    end
    for i in eachindex(e2.population)
        e2.population[i].fitness[:] = e2.fitness(i,local_eval)
    end
end

#-----------------------Cambrian step! and run!--------------------------------#
function step!(e1::AbstractEvolution,e2::AbstractEvolution,switch::String,count_switch::Int)
    e1.gen += 1
    e2.gen += 1
    count_switch += 1
    # we change our environments only every life_env gen
    if e1.gen > 1 && switch=="envs" && count_switch%e1.config.life_env==0
        populate(e2)
        switch = "agents"
        count_switch = 0
    elseif e1.gen > 1 && switch=="envs"
        populate(e1)
    elseif e1.gen > 1 && switch=="agents" && count_switch%e2.config.life_env==0
        populate(e1)
        switch = "envs"
        count_switch = 0
    elseif e1.gen > 1 && switch=="agents"
        populate(e2)
    end

    evaluate(e1,e2)
    generation(e2)

    if ((e1.config.log_gen > 0) && mod(e1.gen, e1.config.log_gen) == 0)
        log_gen(e1)
        log_gen(e2)
    end

    if ((e1.config.save_gen > 0) && mod(e1.gen, e1.config.save_gen) == 0)
        save_gen(e1,e2)
    end

    return switch,count_switch
end

function run!(e1::AbstractEvolution,e2::AbstractEvolution)
    # to save both overall best envs gen and overall best agents gen
    overall_best_agent = -Inf
    overall_best_env = -Inf
    best_envs_envs = deepcopy(e1)
    best_envs_agents = deepcopy(e2)
    best_agents_agents = deepcopy(e2)
    best_agents_envs = deepcopy(e1)

    switch = "envs"
    count_switch = 0

    archives_envs_fitness = -Inf
    archives_envs_envs = deepcopy(e1)
    archives_envs_agents = deepcopy(e2)
    archives_agents_fitness = -Inf
    archives_agents_envs = deepcopy(e1)
    archives_agents_agents = deepcopy(e2)

    for i in tqdm((e1.gen+1):e1.config.n_gen)
        old_switch = switch
        switch, count_switch = step!(e1,e2,switch, count_switch)
        best_agent = sort(e2.population)[end]
        best_env = sort(e1.population)[end]

        if best_agent.fitness[1]>overall_best_agent
            overall_best_agent = best_agent.fitness[1]
            best_agents_agents = deepcopy(e2)
            best_agents_envs = deepcopy(e1)
        end
        if best_env.cgp.fitness[1]>overall_best_env
            overall_best_env = best_env.cgp.fitness[1]
            best_envs_envs = deepcopy(e1)
            best_envs_agents = deepcopy(e2)
        end

        if switch == "envs" == old_switch && best_env.cgp.fitness[1] > archives_envs_fitness
            archives_envs_envs = deepcopy(e1)
            archives_envs_agents = deepcopy(e2)
            archives_envs_fitness = best_env.cgp.fitness[1]
        elseif switch == "agents" == old_switch && best_agent.fitness[1] > archives_agents_fitness
            archives_agents_envs = deepcopy(e1)
            archives_agents_agents = deepcopy(e2)
            archives_agents_fitness = best_agent.fitness[1]
        elseif old_switch != switch
            if old_switch == "envs" && best_env.cgp.fitness[1] > archives_envs_fitness
                archives_envs_envs = deepcopy(e1)
                archives_envs_agents = deepcopy(e2)
                save_gen(archives_envs_envs,archives_envs_agents;id1="archives_envs/envs",id2="archives_envs/agents")
                archives_envs_fitness = -Inf
            elseif old_switch == "envs"
                save_gen(archives_envs_envs,archives_envs_agents;id1="archives_envs/envs",id2="archives_envs/agents")
                archives_envs_fitness = -Inf
            elseif old_switch == "agents" && best_agent.fitness[1] > archives_agents_fitness
                archives_agents_envs = deepcopy(e1)
                archives_agents_agents = deepcopy(e2)
                save_gen(archives_agents_envs,archives_agents_agents;id1="archives_agents/envs",id2="archives_agents/agents")
                archives_agents_fitness = -Inf
            else
                save_gen(archives_agents_envs,archives_agents_agents;id1="archives_agents/envs",id2="archives_agents/agents")
                archives_agents_fitness = -Inf
            end
        end
    end
    save_gen(best_envs_envs,best_envs_agents;id1="overall_best_envs/envs",id2="overall_best_envs/agents")
    save_gen(best_agents_envs,best_agents_agents;id1="overall_best_agents/envs",id2="overall_best_agents/agents")
end
#------------------------------------Main--------------------------------------#
envs = CGPSokoLvlEvolution(cfg_envs,fitness_env;logfile=string("logs/","$expe_name/envs", ".csv"))
agents = sNES{SokoAgent}(agent_model,cfg_agent,fitness_agent;logfile=string("logs/","$expe_name/agents", ".csv"))

run!(envs,agents)
