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
import Cambrian: mutate, crossover, populate, save_gen, generation
import EvolutionaryStrategies: snes_populate, snes_generation

include("../src/continuous_sokolvl.jl")
include("../src/utils.jl")
include("../src/soko_agent.jl")
#-----------------------------Configuration-----------------------------#
image_path = joinpath(@__DIR__,"..","resources","images")
shader_path = joinpath(@__DIR__,"..","resources","shaders")
gdy_path = joinpath(@__DIR__,"..","resources","games")
gdy_reader = Griddly.GDYReader(image_path,shader_path)

grid = Griddly.load!(gdy_reader,joinpath(gdy_path,"Single-Player/GVGAI/sokoban3.yaml"))
game = Griddly.create_game(grid,Griddly.VECTOR)
player1 = Griddly.register_player!(game,"Tux", Griddly.VECTOR)
Griddly.init!(game)

experience_name = "sokoevolution_newfitness"

cfg_envs = Cambrian.get_config("cfg/sokoevo_continuous_envs.yaml")
cfg_agent = Cambrian.get_config("cfg/sokoevo_continuous_agents.yaml")

agent_model = Chain(
                    Conv((3,3),5=>1,pad=(1,1),relu),
                    Flux.flatten,
                    RNN(64,64),
                    Dense(64,4),
                    softmax
                    )
println("nb_params:$(get_params_count(agent_model))")

envs_model = Chain(
                    Dense(2,16),
                    Dense(16,32),
                    Dense(32,16),
                    Dense(16,5),
                    softmax
                    )
println("nb_params:$(get_params_count(envs_model))")
# Overrides of the mutate function
mutate(i::SokoAgent) = mutate(i, cfg_agent.p_mutation)
mutate(i::ContinuousSokoLvl) = mutate(i, cfg_envs.p_mutation)

selection(pop::Array{<:Individual}) = Cambrian.tournament_selection(pop, cfg_envs.tournament_size)
#-----------------------------sNES Helpers-------------------------------------#
# For agents
populate(e::sNES{SokoAgent}) = snes_populate(e)
generation(e::sNES{SokoAgent}) = snes_generation(e)
# For agents
populate(e::sNES{ContinuousSokoLvl}) = snes_populate(e)
generation(e::sNES{ContinuousSokoLvl}) = snes_generation(e)
#---------------------------Evaluate Helpers-----------------------------------#
function play_env(env::ContinuousSokoLvl,agent::SokoAgent;nb_step = 200)

    apply_continuoussokolvl_genes!(env)
    lvl_str = write_map!(env)

    transcript_sokoagent_genes!(agent)

    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)

    total_reward = 0

    first_observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))

    nb_objectives = min(count_items(1,first_observation),count_items(4,first_observation))
    if nb_objectives==0
        return -10
    end

    initial_connectivity = get_connectivity_map(first_observation)
    initial_connectivity_number = length(keys(initial_connectivity))
    if initial_connectivity_number==0
        return -10
    end

    no_boxes_moved = 1 # true
    nb_steps = 0

    observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
    for step in 1:100
        dir = choose_action(observation,agent)
        nb_steps += 1
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
    fitness4 = 0.5*(200-nb_steps)/200

    fitness = fitness1+fitness2+fitness3+fitness4
    return fitness
end

function get_local_evaluation(envs::sNES{ContinuousSokoLvl},agents::sNES{SokoAgent})
    envs_size = length(envs.population)
    agents_size = length(agents.population)
    local_eval = zeros(envs_size,agents_size)
    for i in 1:envs_size
        for j in 1:agents_size
            local_eval[i,j] = play_env(envs.population[i],agents.population[j])
        end
    end
    return local_eval
end

function fitness_env(idx_env::Int64,local_eval::Array{Float64})
    return [maximum(local_eval[idx_env,:])-minimum(local_eval[idx_env,:])]
end

function fitness_agent(idx_agent::Int64,local_eval::Array{Float64})
    return [mean(local_eval[:,idx_agent])]
end

# overrides evaluate function
function evaluate(e1::AbstractEvolution, e2::AbstractEvolution)
    local_eval = get_local_evaluation(e1,e2)
    path = "localfit/$experience_name"
    mkpath(path)
    file_name = Formatting.format("gen-{1:05d}.csv",e1.gen)
    CSV.write("$path/$file_name",  DataFrame(local_eval), header=false)
    for i in eachindex(e1.population)
        e1.population[i].fitness[:] = e1.fitness(i,local_eval)
    end
    for i in eachindex(e2.population)
        e2.population[i].fitness[:] = e2.fitness(i,local_eval)
    end
end

# overrides save_gen for 2 evolution
function save_gen(e1::AbstractEvolution,e2::AbstractEvolution;id1="envs",id2="agents")
    path1 = Formatting.format("gens/$experience_name/{1}/{2:05d}",id1, e1.gen)
    mkpath(path1)
    sort!(e1.population)
    for i in eachindex(e1.population)
        path_ind = Formatting.format("{1}/{2:04d}.dna", path1, i)
        save_ind(e1.population[i],path_ind)
    end
    path2 = Formatting.format("gens/$experience_name/{1}/{2:05d}",id2, e2.gen)
    mkpath(path2)
    sort!(e2.population)
    for i in eachindex(e2.population)
        path_ind = Formatting.format("{1}/{2:04d}.dna", path2, i)
        save_ind(e2.population[i],path_ind)
    end
end

# add a step! and run function for 2 evolution
function step!(e1::AbstractEvolution,e2::AbstractEvolution)
    e1.gen += 1
    e2.gen += 1

    if e1.gen > 1 && e1.gen%e1.config.life_env==0
        populate(e1)
        populate(e2)
    elseif e1.gen > 1
        populate(e2)
    end

    evaluate(e1,e2)
    generation(e1)
    generation(e2)

    if ((e1.config.log_gen > 0) && mod(e1.gen, e1.config.log_gen) == 0)
        log_gen(e1)
        log_gen(e2)
    end
    if ((e1.config.save_gen > 0) && mod(e1.gen, e1.config.save_gen) == 0)
        save_gen(e1,e2)
    end
end

"Call step!(e1,e2) e1.config.n_gen times consecutively"
function run!(e1::AbstractEvolution,e2::AbstractEvolution)
    for i in tqdm((e1.gen+1):e1.config.n_gen)
        step!(e1,e2)
    end
end

#------------------------------------Main------------------------------------#
envs = sNES{ContinuousSokoLvl}(envs_model,cfg_envs,fitness_env;logfile=string("logs/","$experience_name/envs", ".csv"))
agents = sNES{SokoAgent}(agent_model,cfg_agent,fitness_agent;logfile=string("logs/","$experience_name/agents", ".csv"))

run!(envs,agents)
