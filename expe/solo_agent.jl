using Griddly
using Cambrian
using Flux
using Statistics
using Formatting
using CSV
using DataFrames
using ProgressBars
import Cambrian: mutate
import Cambrian: populate
import Cambrian: save_gen

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

cfg_agent = Cambrian.get_config("cfg/solo_agent.yaml")

agent_model = Chain(
                    Conv((3,3),4=>1,pad=(1,1),relu),
                    Flux.flatten,
                    Dense(64,32,relu),
                    Dense(32,4),
                    softmax
                    )

# Overrides of the mutate function
mutate(i::SokoAgent) = mutate(i, cfg_agent.p_mutation)

selection(pop::Array{<:Individual}) = Cambrian.tournament_selection(pop, cfg_agent.tournament_size)

lvl_str = """
wwwwwwww
w......w
w.h....w
w.b....w
w......w
w....A.w
w......w
wwwwwwww
"""

#-----------------------------Helpers-----------------------------#
# Create a generic GAEvo for agent (because of their model)
function initialize(itype::Type, model, cfg::NamedTuple)
    population = Array{itype}(undef, cfg.n_population)
    for i in 1:cfg.n_population
        population[i] = itype(model,cfg)
    end
    population
end

function GAEvo{T}(model, cfg::NamedTuple, fitness::Function;
                      logfile=string("logs/", cfg.id, ".csv")) where T
    logger = CambrianLogger(logfile)
    population = initialize(T, model, cfg)
    GAEvo(cfg, logger, population, fitness, 0)
end

# Helper for the evaluate function
function has_the_box_moved(old_observation,new_observation)::Bool
    old_box = old_observation[1:64]
    new_box = new_observation[1:64]
    if sum(old_box-new_box)!=0
        return true
    end
    return false
end

function fitness_solo(agent::SokoAgent)
    transcript_sokoagent_genes!(agent)
    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)
    total_reward = 0
    old_observation = convert(Array{Int8,3},Griddly.get_data(Griddly.observe(game)))
    for step in 1:200
        new_observation = convert(Array{Int8,3},Griddly.get_data(Griddly.observe(game)))
        if has_the_box_moved(old_observation,new_observation)
            total_reward += 1
        end
        dir = choose_action(new_observation,agent)
        reward, done = Griddly.step_player!(player1,"move", [dir])
        total_reward += reward*100
        old_observation = deepcopy(new_observation)
        if done==1
            break
        end
    end
    return [total_reward]
end

# overrides evaluate function
function evaluate(e::AbstractEvolution)
    for i in eachindex(e.population)
        e.population[i].fitness[:] = e.fitness(e.population[i])
    end
end

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

# add a step! and run function for 2 evolution
function step!(e::AbstractEvolution,id::String)
    e.gen += 1
    if e.gen > 1
        populate(e)
    end

    evaluate(e)

    if ((e.config.log_gen > 0) && mod(e.gen, e.config.log_gen) == 0)
        log_gen(e)
    end
    if ((e.config.save_gen > 0) && mod(e.gen, e.config.save_gen) == 0)
        save_gen(e,id)
    end

end

"Call step!(e1,e2) e1.config.n_gen times consecutively"
function run!(e::AbstractEvolution,id::String)
    for i in tqdm((e.gen+1):e.config.n_gen)
        step!(e,id)
        best_agent = sort(e.population)[end]
        if best_agent.fitness[1]>=100
            println("Gen:$(e.gen)")
            println("Fit:$(best_agent.fitness[1])")
            break
        end
    end
end

#------------------------------------Main------------------------------------#
agents = GAEvo{SokoAgent}(agent_model,cfg_agent,fitness_solo;logfile=string("logs/","solo_agent", ".csv"))

run!(agents,"solo_agent")

best_agent = sort(agents.population)[end]
println("Final fitness agent: ", best_agent.fitness[1])
