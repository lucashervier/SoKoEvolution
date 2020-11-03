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
using JSON
import Cambrian: mutate, crossover, populate, save_gen, generation
import EvolutionaryStrategies: snes_populate, snes_generation
using Logging

include("../src/utils.jl")
include("../src/soko_agent.jl")
#-----------------------------Configuration-----------------------------#
expe_name = "gen_nb_needed"

image_path = joinpath(@__DIR__,"..","resources","images")
shader_path = joinpath(@__DIR__,"..","resources","shaders")
gdy_path = joinpath(@__DIR__,"..","resources","games")
gdy_reader = Griddly.GDYReader(image_path,shader_path)

grid = Griddly.load!(gdy_reader,joinpath(gdy_path,"Single-Player/GVGAI/sokoban3.yaml"))
game = Griddly.create_game(grid,Griddly.VECTOR)
player1 = Griddly.register_player!(game,"Tux", Griddly.VECTOR)
Griddly.init!(game)

levels_path = "cfg/set_known_envs_2.json"
levels_string = read(levels_path,String)
levels_dict = JSON.Parser.parse(levels_string)

cfg_agent = Cambrian.get_config("cfg/sokoevo_continuous_agents.yaml")
agent_model = Chain(
                    Conv((3,3),5=>1,pad=(1,1),relu),
                    Flux.flatten,
                    RNN(64,64),
                    Dense(64,4),
                    softmax
                    )
println("nb_params:$(get_params_count(agent_model))")

# Overrides of the mutate function
mutate(i::SokoAgent) = mutate(i, cfg_agent.p_mutation)

selection(pop::Array{<:Individual}) = Cambrian.tournament_selection(pop, cfg_envs.tournament_size)
#-----------------------------sNES Helpers-------------------------------------#
populate(e::sNES{SokoAgent}) = snes_populate(e)
generation(e::sNES{SokoAgent}) = snes_generation(e)
#-----------------------------other parameters---------------------------------#
life_envs = 50
#-----------------------------Evaluate Function--------------------------------#
function fitness_lvl(agent::SokoAgent,lvl_string::String,nb_objectives::Int)
    transcript_sokoagent_genes!(agent)
    Griddly.load_level_string!(grid,lvl_string)
    Griddly.reset!(game)
    total_reward = 0
    nb_step = 0
    no_boxes_moved = 1 # true
    first_observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
    observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
    for step in 1:200
        dir = choose_action(observation,agent)
        observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
        reward, done = Griddly.step_player!(player1,"move", [dir])
        nb_step += 1
        total_reward += reward
        if done==1
            no_boxes_moved = 0
            break
        end
    end
    if has_the_box_moved(first_observation,observation,1)
        no_boxes_moved = 0 # false
    end
    fitness1 = total_reward/nb_objectives
    fitness2 = (200-nb_step)/200
    fitness3 = -2*no_boxes_moved
    return [fitness1+fitness2+fitness3]
end

function evaluate(e::AbstractEvolution,lvl_string::String,nb_objectives::Int)
    for i in eachindex(e.population)
        e.population[i].fitness[:] = e.fitness(e.population[i],lvl_string,nb_objectives)
    end
end

# add a step! and run function for this evolution strategies
function step!(e::AbstractEvolution,lvl_string::String,nb_objectives::Int)
    e.gen += 1
    if e.gen > 1
        populate(e)
    end

    evaluate(e,lvl_string,nb_objectives)
    generation(e)
    if ((e.config.log_gen > 0) && mod(e.gen, e.config.log_gen) == 0)
        log_gen(e)
    end
end

function run!(e::AbstractEvolution,levels_dict)
    levels = levels_dict["levels_string"]
    objectives = levels_dict["nb_objectives"]
    n_gen = length(levels)*life_envs
    current_idx = 1
    current_lvl = levels[current_idx]
    current_objectives = objectives[current_idx]
    for i in tqdm((e.gen+1):n_gen-1)
        if i%life_envs == 0
            current_idx += 1
            current_lvl = levels[current_idx]
            current_objectives = objectives[current_idx]
        end
        step!(e,current_lvl,current_objectives)
    end
end
#-----------------------------Main---------------------------------------------#
agents = sNES{SokoAgent}(agent_model,cfg_agent,fitness_lvl;logfile=string("logs/","$expe_name/$expe_name", ".csv"))
run!(agents,levels_dict)
