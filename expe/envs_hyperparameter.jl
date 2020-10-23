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
import Cambrian: mutate, crossover, populate, save_gen, generation, ga_populate, step!, log_gen
import EvolutionaryStrategies: snes_populate, snes_generation

include("../src/soko_agent.jl")
include("../src/continuous_sokolvl.jl")
include("../src/utils.jl")
#-----------------------------Configuration-----------------------------#
image_path = joinpath(@__DIR__,"..","resources","images")
shader_path = joinpath(@__DIR__,"..","resources","shaders")
gdy_path = joinpath(@__DIR__,"..","resources","games")
gdy_reader = Griddly.GDYReader(image_path,shader_path)

grid = Griddly.load!(gdy_reader,joinpath(gdy_path,"Single-Player/GVGAI/sokoban3.yaml"))
game = Griddly.create_game(grid,Griddly.VECTOR)
player1 = Griddly.register_player!(game,"Tux", Griddly.VECTOR)
Griddly.init!(game)

cfg_envs = Cambrian.get_config("cfg/sokoevo_continuous_envs.yaml")

envs_model = Chain(
                    Dense(2,16),
                    Dense(16,32),
                    Dense(32,16),
                    Dense(16,5),
                    softmax
                    )
println("nb_params:$(get_params_count(envs_model))")

agent_model = Chain(
                    Conv((3,3),5=>1,pad=(1,1),relu),
                    Flux.flatten,
                    RNN(64,64),
                    Dense(64,4),
                    softmax
                    )
# Overrides of the mutate function
mutate(i::ContinuousSokoLvl) = mutate(i, cfg_envs.p_mutation)

selection(pop::Array{<:Individual}) = Cambrian.tournament_selection(pop, cfg_envs.tournament_size)
#----------------------------------sNES Helpers--------------------------------#
populate(e::sNES{ContinuousSokoLvl}) = snes_populate(e)
generation(e::sNES{ContinuousSokoLvl}) = snes_generation(e)
#--------------------------------"Expert" Player-------------------------------#
expert_path = "..//Buboresults//Buboresults//gens//sokoevolution_existinglvl_sokoagent_fitness2//overall//trial_1//3151//0015.dna"
expert_string = read("$expert_path", String)
expert = SokoAgent(expert_string,agent_model)
#--------------------------------Evaluate helper-------------------------------#
# Helpers for the evaluate function
function fitness_env(env::ContinuousSokoLvl)
    apply_continuoussokolvl_genes!(env)
    lvl_str = write_map!(env)

    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)

    total_reward = 0

    first_observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
    nb_objectives = min(count_items(1,first_observation),count_items(4,first_observation))
    if nb_objectives==0
        return [0,0,-10]
    end
    initial_connectivity = get_connectivity_map(first_observation)
    initial_connectivity_number = length(keys(initial_connectivity))
    if initial_connectivity_number==0
        return [0,-10,-1]
    end
    nb_objectives = min(count_items(1,first_observation),count_items(4,first_observation))
    no_boxes_moved = 1 # true
    observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
    for step in 1:100
        dir = choose_action(observation,expert)
        reward, done = Griddly.step_player!(player1,"move", [dir])
        observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
        total_reward += reward
        if done==1
            no_boxes_moved = 0 # false
            break
        end
    end
    final_connectivity = get_connectivity_map(observation)
    final_connectivity_number = length(keys(final_connectivity))
    if (no_boxes_moved==1)&&has_the_box_moved(first_observation,observation,1)
        no_boxes_moved == 0 #false
    end
    fitness1 = total_reward/nb_objectives
    fitness2 = 0.5*(initial_connectivity_number-final_connectivity_number)/max(initial_connectivity_number,final_connectivity_number)
    fitness3 = -1*no_boxes_moved
    return [fitness1,fitness2,fitness3]
end

function evaluate_with_detailed(e::AbstractEvolution)
    best_fitness = -Inf
    best_detailed_fitness = []
    for i in eachindex(e.population)
        detailed_fitness = e.fitness(e.population[i])
        fitness = sum(detailed_fitness)

        if fitness > best_fitness
            best_fitness = fitness
            best_detailed_fitness = detailed_fitness
        end
        e.population[i].fitness[:] = [fitness]
    end
    return best_detailed_fitness
end

function log_gen(e::AbstractEvolution,best_detailed_fitness)
    for d in 1:e.config.d_fitness
        maxs = map(i->i.fitness[d], e.population)
        with_logger(e.logger) do
            @info Formatting.format("{1:04d},{2:e},{3:e},{4:e},{5:e},{6:e},{7:e}",
                                    e.gen, maximum(maxs), mean(maxs), std(maxs),best_detailed_fitness[1],best_detailed_fitness[2],best_detailed_fitness[3])
        end
    end
    flush(e.logger.stream)
end

# overrides save_gen for sNES population
function save_gen(e::sNES{ContinuousSokoLvl})
    path = Formatting.format("gens/envs_hyperparameter/{1:04d}", e.gen)
    mkpath(path)
    sort!(e.population)
    for i in eachindex(e.population)
        path_ind = Formatting.format("{1}/{2:04d}.dna", path, i)
        save_ind(e.population[i],path_ind)
    end
end

function step!(e::sNES{ContinuousSokoLvl})
    e.gen += 1

    if e.gen > 1
        populate(e)
    end

    best_detailed_fitness = evaluate_with_detailed(e)
    generation(e)

    if ((e.config.log_gen > 0) && mod(e.gen, e.config.log_gen) == 0)
        log_gen(e,best_detailed_fitness)
    end
end
# overrides run! to add saving options
function run!(e::sNES{ContinuousSokoLvl})
    best_gen_fitness = -Inf
    e_best = deepcopy(e)
    for i in tqdm((e.gen+1):e.config.n_gen)
        step!(e)
        best_env = sort(e.population)[end]
        if best_env.fitness[1] >= best_gen_fitness
            best_gen_fitness = best_env.fitness[1]
            e_best = deepcopy(e)
            save_gen(e)
        end
    end
end
#------------------------------------Main------------------------------------#
envs = sNES{ContinuousSokoLvl}(envs_model,cfg_envs,fitness_env;logfile=string("logs/","envs_hyperparameter//envs_hyperparameter", ".csv"))

run!(envs)

best_env = sort(envs.population)[end]
println("Final fitness envs: ", best_env.fitness[1])
