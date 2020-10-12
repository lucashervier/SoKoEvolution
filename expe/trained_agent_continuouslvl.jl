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
import Cambrian: mutate, crossover, populate, save_gen, generation, ga_populate, step!
import EvolutionaryStrategies: snes_populate, snes_generation

include("../src/utils.jl")
include("../src/continuous_sokolvl.jl")
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

agent_path = "../results/gens/sokoevo_rnnagents_directenv_sokoban3/best/agents/7978/0013.dna"
agent_model = Chain(
                    Conv((3,3),4=>1,pad=(1,1),relu),
                    Flux.flatten,
                    RNN(144,144),
                    Dense(144,4),
                    softmax
                    )
agent_str = read(agent_path,String)
agent = SokoAgent(agent_str,agent_model)

cfg_envs = Cambrian.get_config("cfg/sokoevo_continuous_envs.yaml")

envs_model = Chain(
                    Dense(2,16),
                    Dense(16,32),
                    Dense(32,16),
                    Dense(16,5),
                    softmax
                    )
println("nb_params:$(get_params_count(envs_model))")
# Overrides of the mutate function
mutate(i::ContinuousSokoLvl) = mutate(i, cfg_envs.p_mutation)

selection(pop::Array{<:Individual}) = Cambrian.tournament_selection(pop, cfg_envs.tournament_size)
#-----------------------------sNES Helpers-----------------------------#
populate(e::sNES{ContinuousSokoLvl}) = snes_populate(e)
generation(e::sNES{ContinuousSokoLvl}) = snes_generation(e)

# Helpers for the evaluate function
function fitness_env(env::ContinuousSokoLvl)
    apply_continuoussokolvl_genes!(env)
    lvl_str = write_map!(env)

    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)
    total_reward = 0
    for step in 1:100
        observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
        dir = choose_action(observation,agent)
        reward, done = Griddly.step_player!(player1,"move", [dir])
        total_reward += reward
        if done==1
            break
        end
    end
    return [total_reward]
end

function evaluate(e::AbstractEvolution)
    for i in eachindex(e.population)
        e.population[i].fitness[:] = e.fitness(e.population[i])
    end
end

# overrides save_gen for 2 evolution
function save_gen(e::sNES{ContinuousSokoLvl})
    path = Formatting.format("gens/trained_agent_continuouslvl/{1:04d}", e.gen)
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

    evaluate(e)
    generation(e)

    if ((e.config.log_gen > 0) && mod(e.gen, e.config.log_gen) == 0)
        log_gen(e)
    end
    if ((e.config.save_gen > 0) && mod(e.gen, e.config.save_gen) == 0)
        save_gen(e)
    end
end
# overrides run! to add saving options
function run!(e::sNES{ContinuousSokoLvl})
    best_fitness = 0
    for i in tqdm((e.gen+1):e.config.n_gen)
        step!(e)
        best_env = sort(e.population)[end]
        if best_env.fitness[1]>best_fitness
            println("Gen:$(e.gen)")
            println("Fit_env:$(best_env.fitness[1])")
            best_fitness = best_env.fitness[1]
            if i%e.config.save_gen != 0
                save_gen(e)
            end
        end
    end
end

#------------------------------------Main------------------------------------#
envs = sNES{ContinuousSokoLvl}(envs_model,cfg_envs,fitness_env;logfile=string("logs/","trained_agent_continuouslvl/trained_agent_continuouslvl", ".csv"))

run!(envs)

best_env = sort(envs.population)[end]
println("Final fitness envs: ", best_env.fitness[1])
