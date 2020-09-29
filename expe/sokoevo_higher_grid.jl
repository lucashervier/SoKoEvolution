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

include("../src/sokolvl_individual.jl")
include("../src/utils.jl")
include("../src/soko_agent.jl")

#-----------------------------Configuration-----------------------------#
image_path = joinpath(@__DIR__,"..","resources","images")
shader_path = joinpath(@__DIR__,"..","resources","shaders")
gdy_path = joinpath(@__DIR__,"..","resources","games")
gdy_reader = Griddly.GDYReader(image_path,shader_path)

grid = Griddly.load!(gdy_reader,joinpath(gdy_path,"Single-Player/GVGAI/sokoban.yaml"))
game = Griddly.create_game(grid,Griddly.VECTOR)
player1 = Griddly.register_player!(game,"Tux", Griddly.BLOCK_2D)
Griddly.init!(game)

cfg_envs = Cambrian.get_config("cfg/sokoevo_higher_envs.yaml")
cfg_agent = Cambrian.get_config("cfg/sokoevo_higher_agents.yaml")

agent_model = Chain(
                    Conv((3,3),4=>1,pad=(1,1),relu),
                    Flux.flatten,
                    RNN(144,144),
                    Dense(144,4),
                    softmax
                    )
println("nb_params:$(get_params_count(agent_model))")

# Overrides of the mutate function
mutate(i::SokoAgent) = mutate(i, cfg_agent.p_mutation)
mutate(i::SokoLvlIndividual) = mutate(i, cfg_envs.p_mutation)

selection(pop::Array{<:Individual}) = Cambrian.tournament_selection(pop, cfg_envs.tournament_size)
#-----------------------------sNES Helpers-----------------------------#
# For agents
mutable struct sNES_soko <: Cambrian.AbstractEvolution
    config::NamedTuple
    logger::CambrianLogger
    population::Array{SokoAgent}
    elites::Array{SokoAgent}
    state::EvolutionaryStrategies.sNESState
    fitness::Function
    gen::Int
end

function snes_init(model, cfg::NamedTuple, state::EvolutionaryStrategies.ESState)
    population = Array{SokoAgent}(undef, cfg.n_population)
    for i in 1:cfg.n_population
        genes = state.μ .+ state.σ .* view(state.s, :, i)
        population[i] = SokoAgent(genes,model,cfg)
    end
    population
end

function sNES_soko(model, cfg::NamedTuple, fitness::Function, state::EvolutionaryStrategies.sNESState; logfile=string("logs/", cfg.id, ".csv"))
    logger = CambrianLogger(logfile)
    population = snes_init(model, cfg, state)
    elites = deepcopy([population[i] for i in 1:cfg.n_elite])
    sNES_soko(cfg, logger, population, elites, state, fitness, 0)
end

function sNES_soko(model, cfg::NamedTuple, fitness::Function; logfile=string("logs/", cfg.id, ".csv"))
    logger = CambrianLogger(logfile)
    cfg = merge(EvolutionaryStrategies.snes_config(cfg.n_genes), cfg)
    state = EvolutionaryStrategies.sNESState(cfg.n_genes, cfg.n_population)
    sNES_soko(model, cfg, fitness, state; logfile=logfile)
end

function snes_populate(e::sNES_soko)
    for i in eachindex(e.population)
        e.population[i].genes .= e.state.μ .+ e.state.σ .* view(e.state.s, :, i)
        e.population[i].fitness .= -Inf
    end
end

"update NES state, called after populate and evaluate"
function snes_generation(e::sNES_soko)
    d = e.config.n_genes
    n = e.config.n_population

    # copy population information
    F = zeros(n)
    for i in eachindex(e.population)
        F[i] = -e.population[i].fitness[1]
    end
    idx = sortperm(F)

    # compute gradients
    ∇μ = zeros(d)
    ∇σ = zeros(d)
    for i in 1:n
        j = idx[i]
        ∇μ .+= e.state.u[i] .* e.state.s[:, j]
        ∇σ .+= e.state.u[i] .* (e.state.s[:, j].^2 .- 1.0)
    end

    # update state variables
    e.state.μ .+= e.config.ημ .* e.state.σ .* ∇μ
    e.state.σ .*= exp.(e.config.ησ/2 .* ∇σ)
    randn!(e.state.s)
    Cambrian.elites_generation(e)
end

populate(e::sNES_soko) = snes_populate(e)
generation(e::sNES_soko) = snes_generation(e)

# Helpers for the evaluate function
function play_env(env::SokoLvlIndividual,agent::SokoAgent;nb_step = 200)
    lvl_str = transcript_sokolvl_genes(env)
    transcript_sokoagent_genes!(agent)
    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)
    total_reward = 0
    for step in 1:nb_step
        observation = convert(Array{Int8,3},Griddly.get_data(Griddly.observe(game)))
        dir = choose_action(observation,agent)
        reward, done = Griddly.step_player!(player1,"move", [dir])
        total_reward += reward
        if done==1
            total_reward += 100
            Griddly.reset!(game)
        end
    end
    return total_reward
end

function get_local_evaluation(envs::GAEvo{SokoLvlIndividual},agents::sNES_soko)
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
    path = "localfit/sokoevo_higher_box_constraints"
    mkpath(path)
    CSV.write("$path/gen-$(e1.gen).csv",  DataFrame(local_eval), header=false)
    for i in eachindex(e1.population)
        e1.population[i].fitness[:] = e1.fitness(i,local_eval)
    end

    for i in eachindex(e2.population)
        e2.population[i].fitness[:] = e2.fitness(i,local_eval)
    end

end

# overrides save_gen for 2 evolution
function save_gen(e1::AbstractEvolution,e2::AbstractEvolution;id1="envs",id2="agents")
    path1 = Formatting.format("gens/sokoevo_higher_box_constraints/{1}/{2:04d}",id1, e1.gen)
    mkpath(path1)
    sort!(e1.population)
    for i in eachindex(e1.population)
        f = open(Formatting.format("{1}/{2:04d}.dna", path1, i), "w+")
        write(f, string(e1.population[i]))
        close(f)
    end
    path2 = Formatting.format("gens/sokoevo_higher_box_constraints/{1}/{2:04d}",id2, e2.gen)
    mkpath(path2)
    sort!(e2.population)
    for i in eachindex(e2.population)
        f = open(Formatting.format("{1}/{2:04d}.dna", path2, i), "w+")
        write(f,"""{"genes":""")
        write(f, string(e2.population[i].genes))
        write(f,""","fitness":""")
        write(f, string(e2.population[i].fitness))
        write(f,""","width":""")
        write(f, string(e2.population[i].width))
        write(f,""","height":""")
        write(f, string(e2.population[i].height))
        write(f,""","nb_object":""")
        write(f, string(e2.population[i].nb_object))
        write(f,"""}""")
        close(f)
    end
end

# add a step! and run function for 2 evolution
function step!(e1::AbstractEvolution,e2::AbstractEvolution)
    e1.gen += 1
    e2.gen += 1
    if e1.gen > 1
        populate(e1)
        populate(e2)
    end
    # for the population of environment we need to apply constraint
    for i in eachindex(e1.population)
        apply_sokolvl_constraint!(e1.population[i])
        apply_box_holes_constraint!(e1.population[i],7,1,3)
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

"Call step!(e1,e2) e1.config.n_gen times consecutively"
function run!(e1::AbstractEvolution,e2::AbstractEvolution)
    for i in tqdm((e1.gen+1):e1.config.n_gen)
        step!(e1,e2)
        best_agent = sort(e2.population)[end]
        best_env  = sort(e1.population)[end]
        if best_agent.fitness[1]>=15
            println("Gen:$(e1.gen)")
            println("Fit_agent:$(best_agent.fitness[1])")
            println("Fit_env:$(best_env.fitness[1])")
            save_gen(e1,e2;id1="envs/bests",id2="agents/bests")
        elseif best_env.fitness[1]>=14
            println("Gen:$(e1.gen)")
            println("Fit_agent:$(best_agent.fitness[1])")
            println("Fit_env:$(best_env.fitness[1])")
            save_gen(e1,e2;id1="envs/bests",id2="agents/bests")
        end
    end
end

#------------------------------------Main------------------------------------#
envs = GAEvo{SokoLvlIndividual}(cfg_envs,fitness_env;logfile=string("logs/","sokoevo_higher_box_constraints/envs", ".csv"))
agents = sNES_soko(agent_model,cfg_agent,fitness_agent;logfile=string("logs/","sokoevo_higher_box_constraints/agents", ".csv"))

run!(envs,agents)

best_agent = sort(agents.population)[end]
println("Final fitness agent: ", best_agent.fitness[1])
best_env = sort(envs.population)[end]
println("Final fitness envs: ", best_env.fitness[1])
