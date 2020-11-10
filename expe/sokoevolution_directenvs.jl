using Griddly
using Cambrian
using EvolutionaryStrategies
using Flux
using Statistics
using Random
using Formatting
using CSV
using DataFrames
using ProgressBars
using Logging
import Cambrian: mutate, crossover, populate, save_gen, generation, log_gen
import EvolutionaryStrategies: snes_populate, snes_generation

include("../src/sokolvl_individual.jl")
include("../src/utils.jl")
include("../src/soko_agent.jl")
#----------------------------Named Parameters----------------------------------#
game_name = "sokoban3"
expe_name = "new_fitness_direct_envs4"
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
cfg_envs = Cambrian.get_config("cfg/sokoevo_envs.yaml")
# overrides of the mutate function
mutate(i::SokoLvlIndividual) = mutate(i, cfg_envs.p_mutation)
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
function log_gen(e::GAEvo{SokoLvlIndividual})
    best = sort(e.population)[end]
    lvl_str = transcript_sokolvl_genes(best)
    nb_boxes,nb_holes,nb_objectives,initial_connectivity_number,random_reward = get_detailed_info(lvl_str)
    for d in 1:e.config.d_fitness
        maxs = map(i->i.fitness[d], e.population)
        with_logger(e.logger) do
            @info Formatting.format("{1:05d},{2:e},{3:e},{4:e},{5:e},{6:e},{7:e},{8:e},{9:e}",
                                    e.gen, maximum(maxs), mean(maxs), std(maxs),nb_boxes,nb_holes,nb_objectives,initial_connectivity_number,random_reward)
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
    nb_objectives = min(nb_boxes,nb_holes)

    initial_connectivity = get_connectivity_map(initial_observation)
    initial_connectivity_number = length(keys(initial_connectivity))

    random_reward = evaluate_random(lvl_str)

    return [nb_boxes,nb_holes,nb_objectives,initial_connectivity_number,random_reward]
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
    no_boxes_moved = 1 # true
    observation = first_observation

    for step in 1:200
        dir = choose_action(observation,agent)
        reward, done = Griddly.step_player!(player1,"move", [dir])
        nb_step += 1
        observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
        total_reward += reward
        if done==1
            no_boxes_moved = 0 # false
            break
        end
    end

    # to know if any boxes has moved during playtime
    if (no_boxes_moved==1)&&has_the_box_moved(first_observation,observation,1)
        no_boxes_moved == 0 #false
    end

    return [total_reward,nb_step,no_boxes_moved]
end

# fill a matrix with every agent performance on every levels
function get_local_evaluation(envs::GAEvo{SokoLvlIndividual},agents::sNES{SokoAgent})
    envs_size = length(envs.population)
    agents_size = length(agents.population)
    local_eval = zeros(envs_size,agents_size)
    # optional constraints shaping your local fitness
    objectives_max = envs.config.objectives_max
    connectivity_max = envs.config.connectivity_max
    for i in 1:envs_size
        env = envs.population[i]
        lvl_str = transcript_sokolvl_genes(env)
        # get optional information to shape your local fitness
        nb_boxes,nb_holes,nb_objectives,initial_connectivity_number,random_reward = get_detailed_info(lvl_str)
        if nb_objectives==0
            for j in 1:agents_size
                local_eval[i,j] = -10
            end
        elseif initial_connectivity_number == 0
            for j in 1:agents_size
                local_eval[i,j] = -10
            end
        else
            # add fitness factors depending only on envs level shape
            objectives_reward = - abs((nb_boxes - nb_holes))/nb_objectives - (nb_objectives>objectives_max)
            connectivity_reward = - (initial_connectivity_number>connectivity_max)
            for j in 1:agents_size
                # the "main" reward is the agent reward induced by game's rules
                agent_reward, nb_step, no_boxes_moved = play_lvl2(lvl_str,agents.population[j])
                # optimize the number of steps
                step_reward = (200-nb_step)/200
                # penalty for "lazy" agents
                boxes_moving_reward = - 2*no_boxes_moved
                scoring_reward = (agent_reward - random_reward)/nb_objectives
                # in this pos add your local evaluation of an agent "j" on lvl "i"
                local_eval[i,j] = scoring_reward + step_reward + objectives_reward + connectivity_reward + boxes_moving_reward
            end
        end
    end
    return local_eval
end

# The fitness function of your envs depending on your local evaluation

# function fitness_env(idx_env::Int64,local_eval::Array{Float64})
#     return [maximum(local_eval[idx_env,:])-minimum(local_eval[idx_env,:])]
# end

function fitness_env(idx_env::Int64,local_eval::Array{Float64})
    return [min(maximum(local_eval[idx_env,:])-minimum(local_eval[idx_env,:]),mean(local_eval[idx_env,:]))]
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
    for i in eachindex(e1.population)
        e1.population[i].fitness[:] = e1.fitness(i,local_eval)
    end
    for i in eachindex(e2.population)
        e2.population[i].fitness[:] = e2.fitness(i,local_eval)
    end
end

#-----------------------Cambrian step! and run!--------------------------------#
function step!(e1::AbstractEvolution,e2::AbstractEvolution)
    e1.gen += 1
    e2.gen += 1
    # we change our environments only every life_env gen
    if e1.gen > 1 && e1.gen%e1.config.life_env==0
        populate(e1)
        populate(e2)
    elseif e1.gen > 1
        populate(e2)
    end
    # for the population of environment we need to apply constraint
    for i in eachindex(e1.population)
        apply_sokolvl_constraint!(e1.population[i])
        # # you can add other construction constraint
        # apply_box_holes_constraint!(e1.population[i],7,1,3)
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
end

function run!(e1::AbstractEvolution,e2::AbstractEvolution)
    # to save both overall best envs gen and overall best agents gen
    overall_best_agent = -Inf
    overall_best_env = -Inf
    best_envs_envs = deepcopy(e1)
    best_envs_agents = deepcopy(e2)
    best_agents_agents = deepcopy(e2)
    best_agents_envs = deepcopy(e1)

    for i in tqdm((e1.gen+1):e1.config.n_gen)
        step!(e1,e2)
        best_agent = sort(e2.population)[end]
        best_env = sort(e1.population)[end]

        if best_agent.fitness[1]>overall_best_agent
            overall_best_agent = best_agent.fitness[1]
            best_agents_agents = deepcopy(e2)
            best_agents_envs = deepcopy(e1)
        end
        if best_env.fitness[1]>overall_best_env
            overall_best_env = best_env.fitness[1]
            best_envs_envs = deepcopy(e1)
            best_envs_agents = deepcopy(e2)
        end
    end
    save_gen(best_envs_envs,best_envs_agents;id1="overall_best_envs/envs",id2="overall_best_envs/agents")
    save_gen(best_agents_envs,best_agents_agents;id1="overall_best_agents/envs",id2="overall_best_agents/agents")
end
#------------------------------------Main--------------------------------------#
envs = GAEvo{SokoLvlIndividual}(cfg_envs,fitness_env;logfile=string("logs/","$expe_name/envs", ".csv"))
agents = sNES{SokoAgent}(agent_model,cfg_agent,fitness_agent;logfile=string("logs/","$expe_name/agents", ".csv"))

run!(envs,agents)
