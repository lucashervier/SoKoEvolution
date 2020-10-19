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

levels_path = "cfg/set_known_envs.json"
levels_string = read(levels_path,String)
levels_dict = JSON.Parser.parse(levels_string)
# println(levels_dict["levels_string"][8])

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

#-------------------------Initialize Report File-------------------------------#
results_df = DataFrame(Level = Int[], Max_gen_nb = Int[], Individual_per_gen = Int[],
                        Nb_trial = Int[], Nb_success = Int[], Best_gen_to_solve1 = Any[],
                        Max_fitness1 = Float64[], Best_gen_to_solve2 = Any[], Max_fitness2 = Float64[],
                        Best_gen_to_solve3 = Any[], Max_fitness3 = Float64[], Best_gen_to_solve4 = Any[],
                        Max_fitness4 = Float64[], Best_gen_to_solve5 = Any[], Max_fitness5 = Float64[],
                        Mean_nb_gen_to_solve = Any[], Std_nb_gen_to_solve = Any[])
println(results_df)
# Those would be fixed parameters
global max_gen = 5000
global nb_trial = 5

# utils to add rows
function add_row!(df::DataFrame,lvl_nb::Int,ind_per_gen::Int,nb_success::Int,nb_gens,max_fits::Array{Float64,1})
        if length(nb_gens) != nb_trial
                throw(DimensionMismatch("length of nb_gens is not equal to nb_trial"))
        elseif length(max_fits) != nb_trial
                throw(DimensionMismatch("length of max_fits is not equal to nb_trial"))
        else
                mean_nb_gen = 0
                std_nb_gen = 0
                if nb_success == nb_trial
                        mean_nb_gen = mean(nb_gens)
                        std_nb_gen = std(nb_gens)
                else
                        mean_nb_gen = ""
                        std_nb_gen = ""
                end
                push!(df,[lvl_nb,max_gen,ind_per_gen,nb_trial,nb_success,nb_gens[1],max_fits[1],
                nb_gens[2],max_fits[2],nb_gens[3],max_fits[3],nb_gens[4],max_fits[4],nb_gens[5],
                max_fits[5],mean_nb_gen,std_nb_gen])
        end
end

# add_row!(results_df,1,13,5,[2,4,8,9,4],[1.0,1.0,1.0,1.0,1.0])
# println(results_df)
# add_row!(results_df,1,13,4,[2,4,8,9,""],[1.0,1.0,1.0,1.0,0.5])
# println(results_df)

#-----------------------------Evaluate Function--------------------------------#
# function fitness_lvl(agent::SokoAgent,lvl_string::String,nb_objectives::Int)
#     transcript_sokoagent_genes!(agent)
#     Griddly.load_level_string!(grid,lvl_string)
#     Griddly.reset!(game)
#     total_reward = 0
#     observation = Griddly.observe(game)
#     observation = Griddly.get_data(observation)
#     for step in 1:200
#         dir = choose_action(observation,agent)
#         reward, done = Griddly.step_player!(player1,"move", [dir])
#         total_reward += reward
#         if done==1
#             break
#         end
#         observation = Griddly.observe(game)
#         observation = Griddly.get_data(observation)
#     end
#     return [total_reward/nb_objectives]
# end
function fitness_lvl(agent::SokoAgent,lvl_string::String,nb_objectives::Int)
    transcript_sokoagent_genes!(agent)
    Griddly.load_level_string!(grid,lvl_string)
    Griddly.reset!(game)
    total_reward = 0
    nb_step = 0
    no_boxes_moved = 1 # true
    first_observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
    nb_box = count_items(1,first_observation)
    observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
    for step in 1:200
        dir = choose_action(observation,agent)
        observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
        reward, done = Griddly.step_player!(player1,"move", [dir])
        nb_step += 1
        total_reward += reward
        if done==1
            break
        end
    end
    if has_the_box_moved(first_observation,observation,1)
        no_boxes_moved = 0 # false
    end
    nb_box_blocked = count_blocked_box(observation)
    fitness = total_reward/nb_objectives + (200-nb_step)/200 -0.5*(nb_box_blocked/nb_box) -2*no_boxes_moved
    return [fitness]
end

function evaluate(e::AbstractEvolution,lvl_string::String,nb_objectives::Int)
    for i in eachindex(e.population)
        e.population[i].fitness[:] = e.fitness(e.population[i],lvl_string,nb_objectives)
    end
end

function save_gen(e::AbstractEvolution,id::String)
    path = Formatting.format("gens/agent_set_known_envs_fitness3/{1}/{2:04d}",id, e.gen)
    mkpath(path)
    sort!(e.population)
    for i in eachindex(e.population)
        path_ind = Formatting.format("{1}/{2:04d}.dna", path, i)
        save_ind(e.population[i],path_ind)
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

function run!(e::AbstractEvolution,lvl_string::String,nb_objectives::Int,id::String)
    success = false
    best_gen_fitness = 0
    e_best = deepcopy(e)
    for i in tqdm((e.gen+1):e.config.n_gen)
        step!(e,lvl_string,nb_objectives)
        best_agent = sort(e.population)[end]
        if best_agent.fitness[1]>=1
            save_gen(e,id)
            success = true
            return [e.gen,best_agent.fitness[1],1]
        elseif best_agent.fitness[1] >= best_gen_fitness
            best_gen_fitness = best_agent.fitness[1]
            e_best = deepcopy(e)
        end
    end

    save_gen(e_best,id)
    return [e_best.gen,best_gen_fitness,0]
end

#-----------------------------Main---------------------------------------------#
for k in 1:length(levels_dict["levels_string"])
    nb_success = 0
    nb_gens = []
    max_fits = zeros(nb_trial)
    lvl_nb = k
    lvl_str = levels_dict["levels_string"][k]
    nb_objectives= levels_dict["nb_objectives"][k]
    ind_per_gen = 0
    for trial in 1:nb_trial
        agents = sNES{SokoAgent}(agent_model,cfg_agent,fitness_lvl;logfile=string("logs/","agent_set_known_envs_fitness3/lvl_$lvl_nb/trial_$trial", ".csv"))
        ind_per_gen = length(agents.population)
        id_gens = "lvl_$lvl_nb/trial_$trial"
        results = run!(agents,lvl_str,nb_objectives,id_gens)
        push!(nb_gens,results[1])
        max_fits[trial]= results[2]
        nb_success += results[3]
    end
    add_row!(results_df,lvl_nb,ind_per_gen,Int(nb_success),nb_gens,max_fits)
end
CSV.write("gens/agent_set_known_envs_fitness3/analysis", results_df)
