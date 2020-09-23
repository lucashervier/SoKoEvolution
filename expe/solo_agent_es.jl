using Griddly
using Cambrian
using EvolutionaryStrategies
using Flux
using Statistics
using Formatting
using CSV
using DataFrames
using ProgressBars
using Random
import Cambrian: mutate, populate, save_gen
import EvolutionaryStrategies: snes_populate, snes_generation
include("../src/utils.jl")
include("../src/soko_agent.jl")

#-----------------------------Configuration-----------------------------#
image_path = joinpath(@__DIR__,"..","resources","images")
shader_path = joinpath(@__DIR__,"..","resources","shaders")
gdy_path = joinpath(@__DIR__,"..","resources","games")
gdy_reader = Griddly.GDYReader(image_path,shader_path)

grid = Griddly.load!(gdy_reader,joinpath(gdy_path,"Single-Player/GVGAI/sokoban.yaml"))
game = Griddly.create_game(grid,Griddly.SPRITE_2D)
player1 = Griddly.register_player!(game,"Tux", Griddly.BLOCK_2D)
Griddly.init!(game)

cfg_agent = Cambrian.get_config("cfg/solo_agent_es.yaml")

agent_model = Chain(
                    Conv((3,3),4=>4,pad=(1,1),relu),
                    Conv((3,3),4=>1,pad=(1,1),relu),
                    Flux.flatten,
                    Dense(64,32,relu),
                    Dense(32,4),
                    softmax
                    )
println("nb_params:$(get_params_count(agent_model))")
# Overrides of the mutate function
mutate(i::SokoAgent) = mutate(i, cfg_agent.p_mutation)

lvl_str = """
wwwwwwww
w......w
w.h....w
w..b...w
w......w
w....A.w
w......w
wwwwwwww
"""

#-----------------------------Helpers-----------------------------#
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

# Helper for the evaluate function
function has_the_box_moved(old_observation,new_observation)::Bool
    old_observation = convert(Array{Int8,3},old_observation)
    new_observation = convert(Array{Int8,3},new_observation)

    old_box = old_observation[1,:,:]
    new_box = new_observation[1,:,:]

    if sum(old_box-new_box)!=0
        return true
    end
    return false
end

function fitness_solo(agent::SokoAgent)
    # render_window = RenderWindow(700,700)
    # display(render_window.scene)
    transcript_sokoagent_genes!(agent)
    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)
    total_reward = 0
    old_observation = Griddly.vector_obs(grid)
    old_observation = Griddly.get_data(old_observation)
    for step in 1:200
        new_observation = Griddly.vector_obs(grid)
        new_observation = Griddly.get_data(new_observation)
        # sprite = Griddly.observe(game)
        # sprite = Griddly.get_data(sprite)
        # render(render_window,sprite)
        if has_the_box_moved(old_observation,new_observation)
            total_reward += 1
        end
        dir = choose_action(new_observation,agent)
        # println("dir:$dir")
        reward, done = Griddly.step_player!(player1,"move", [dir])
        total_reward += reward*100
        old_observation = deepcopy(new_observation)
        if done==1
            break
        end
    end
    return [total_reward]
end

# overrides evolution function
populate(e::sNES_soko) = snes_populate(e)
function evaluate(e::AbstractEvolution)
    for i in eachindex(e.population)
        e.population[i].fitness[:] = e.fitness(e.population[i])
    end
end
generation(e::sNES_soko) = snes_generation(e)

# overrides save_gen for 2 evolution
function save_gen(e::AbstractEvolution,id::String)
    path = Formatting.format("gens/{1}/{2:04d}",id, e.gen)
    mkpath(path)
    sort!(e.population)
    for i in eachindex(e.population)
        f = open(Formatting.format("{1}/{2:04d}.dna", path, i), "w+")
        write(f,"""{"genes":""")
        write(f, string(e.population[i].genes))
        write(f,""","fitness":""")
        write(f, string(e.population[i].fitness))
        write(f,""","width":""")
        write(f, string(e.population[i].width))
        write(f,""","height":""")
        write(f, string(e.population[i].height))
        write(f,""","nb_object":""")
        write(f, string(e.population[i].nb_object))
        write(f,"""}""")
        close(f)
    end
end

# add a step! and run function for this evolution strategies
function step!(e::AbstractEvolution,id::String)
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
        save_gen(e,id)
    end

end

function run!(e::AbstractEvolution,id::String)
    for i in tqdm((e.gen+1):e.config.n_gen)
        step!(e,id)
        best_agent = sort(e.population)[end]
        if best_agent.fitness[1]>=100
            println("Gen:$(e.gen)")
            println("Fit:$(best_agent.fitness[1])")
            save_gen(e,"ES/solo_agent_es/bests")
            break
        end
    end
end

#------------------------------------Main------------------------------------#
agents = sNES_soko(agent_model,cfg_agent,fitness_solo;logfile=string("logs/","ES/solo_agent_es", ".csv"))
# render_window = RenderWindow(700,700)
# display(render_window.scene)
run!(agents,"ES/solo_agent_es")

best_agent = sort(agents.population)[end]
println("Final fitness agent: ", best_agent.fitness[1])
