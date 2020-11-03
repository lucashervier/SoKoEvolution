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
global levels_dict = JSON.Parser.parse(levels_string)
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
results_df = DataFrame(Level = Int[], Nb_objectives = Int[], Trial_nb = Int[],
                        Max_gen_nb = Int[], Individual_per_gen = Int[],
                        Best_gen_to_solve_this_lvl = Any[], Max_fitness_on_this_lvl = Float64[],
                        Overall_best_gen = Any[], Overall_best_on_this_lvl_fitness = Float64[])
println(results_df)
# Those would be fixed parameters
global max_gen = 5000
global nb_trial = 5

# utils to add rows
function add_row!(df::DataFrame,lvl_nb::Int,trial_nb::Int,max_gen_nb::Int,
                  ind_per_gen::Int,gen_nb_ownbest::Int,max_fit_own_lvl::Float64,
                  overall_best_gen::Int,overall_best_fit_this_lvl::Float64)
    nb_objectives = levels_dict["nb_objectives"][lvl_nb]
    push!(df,[lvl_nb,nb_objectives,trial_nb,max_gen_nb,ind_per_gen,gen_nb_ownbest,
    max_fit_own_lvl,overall_best_gen,overall_best_fit_this_lvl])
end

# add_row!(results_df,1,13,5,[2,4,8,9,4],[1.0,1.0,1.0,1.0,1.0])
# println(results_df)
# add_row!(results_df,1,13,4,[2,4,8,9,""],[1.0,1.0,1.0,1.0,0.5])
# println(results_df)

#-----------------------------Evaluate Function--------------------------------#
# Helpers for the evaluate function
# function play_lvl(agent::SokoAgent,lvl_string::String,nb_objectives::Int)
#     transcript_sokoagent_genes!(agent)
#     Griddly.load_level_string!(grid,lvl_string)
#     Griddly.reset!(game)
#     total_reward = 0
#     nb_step = 0
#     for step in 1:200
#         observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
#         dir = choose_action(observation,agent)
#         reward, done = Griddly.step_player!(player1,"move", [dir])
#         nb_step += 1
#         total_reward += reward
#         if done==1
#             break
#         end
#     end
#     return total_reward/nb_objectives + (200-nb_step)/200
# end

function play_lvl(agent::SokoAgent,lvl_string::String,nb_objectives::Int)
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
    return total_reward/nb_objectives + (200-nb_step)/200 -0.5*(nb_box_blocked/nb_box) -2*no_boxes_moved
end

function get_local_evaluation(agents::sNES{SokoAgent},levels_dict::Dict{String,Any})
    nb_levels = length(levels_dict["levels_string"])
    agents_size = length(agents.population)
    local_eval = zeros(nb_levels,agents_size)
    for i in 1:nb_levels
        for j in 1:agents_size
            lvl_str = levels_dict["levels_string"][i]
            nb_objectives = levels_dict["nb_objectives"][i]
            local_eval[i,j] = play_lvl(agents.population[j],lvl_str,nb_objectives)
        end
    end
    return local_eval
end

function overall_fitness(idx_agent::Int64,local_eval::Array{Float64})
    return [mean(local_eval[:,idx_agent])]
end

function evaluate(e::AbstractEvolution,levels_dict::Dict{String,Any})
    eval_matrix = get_local_evaluation(e,levels_dict)
    for i in eachindex(e.population)
        e.population[i].fitness[:] = e.fitness(i,eval_matrix)
    end
    return eval_matrix
end

function save_gen(e::AbstractEvolution,id::String)
    path = Formatting.format("gens/sokoevolution_existinglvl_sokoagent_fitness3/{1}/{2:04d}",id, e.gen)
    mkpath(path)
    sort!(e.population)
    for i in eachindex(e.population)
        path_ind = Formatting.format("{1}/{2:04d}.dna", path, i)
        save_ind(e.population[i],path_ind)
    end
end

# add a step! and run function for this evolution strategies
function step!(e::AbstractEvolution,levels_dict::Dict{String,Any})
    e.gen += 1
    if e.gen > 1
        populate(e)
    end

    eval_matrix = evaluate(e,levels_dict::Dict{String,Any})
    generation(e)
    if ((e.config.log_gen > 0) && mod(e.gen, e.config.log_gen) == 0)
        log_gen(e)
    end
    return eval_matrix
end

function run!(e::AbstractEvolution,levels_dict::Dict{String,Any})

    nb_level = length(levels_dict["levels_string"])
    best_gen_per_level = [deepcopy(e) for i in 1:nb_level]
    best_fitness_per_level = zeros(nb_level)
    current_best_overrall_gen = deepcopy(e)
    current_best_overrall_fit = 0
    fitness_overrall_per_lvl = zeros(nb_level)
    nb_ind = length(e.population)

    for i in tqdm((e.gen+1):e.config.n_gen)
        eval_matrix = step!(e,levels_dict)
        best_agent = sort(e.population)[end]

        if best_agent.fitness[1] > current_best_overrall_fit
            current_best_overrall_gen = deepcopy(e)
            current_best_overrall_fit = best_agent.fitness[1]
            index_best_in_eval_mat = argmax([mean(eval_matrix[:,i]) for i in 1:nb_ind])
            fitness_overrall_per_lvl = eval_matrix[:,index_best_in_eval_mat]
        end
        for k in 1:nb_level
            if maximum(eval_matrix[k,:])>best_fitness_per_level[k]
                best_gen_per_level[k] = deepcopy(e)
                best_fitness_per_level[k] = maximum(eval_matrix[k,:])
            end
        end
    end
    return [best_gen_per_level,best_fitness_per_level,current_best_overrall_gen,current_best_overrall_fit,fitness_overrall_per_lvl]
end

#-----------------------------Main---------------------------------------------#
for trial in 1:nb_trial

    agents = sNES{SokoAgent}(agent_model,cfg_agent,overall_fitness;logfile=string("logs/","sokoevolution_existinglvl_sokoagent_fitness3/trial_$trial", ".csv"))
    ind_per_gen = length(agents.population)
    results = run!(agents,levels_dict)
    overall_best_gen = results[3]
    save_gen(overall_best_gen,"overall/trial_$trial")
    overall_best_fitness = results[4]
    for lvl_nb in 1:length(levels_dict["levels_string"])
        best_gen_this_lvl = results[1][lvl_nb]
        save_gen(best_gen_this_lvl,"level_$lvl_nb/trial_$trial")

        best_fitness_this_lvl = results[2][lvl_nb]
        fitness_overall_this_lvl = results[5][lvl_nb]
        add_row!(results_df,lvl_nb,trial,max_gen,ind_per_gen
                ,best_gen_this_lvl.gen,best_fitness_this_lvl
                ,overall_best_gen.gen,fitness_overall_this_lvl)
    end
end
CSV.write("gens/sokoevolution_existinglvl_sokoagent_fitness3/analysis", results_df)
