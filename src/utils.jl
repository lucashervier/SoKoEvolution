using Flux
using EvolutionaryStrategies
import EvolutionaryStrategies: snes_populate, snes_generation, sNESState

function get_params_count(model)
    size = 0
    ps = Flux.params(model)
    for layer in ps
        size += length(layer)
    end
    return size
end

function load_weights_from_array!(model,weights)
    nb_params = get_params_count(model)
    nb_weight = length(weights)
    if nb_params > nb_weight
        throw("Your weight vector is not long enough")
    elseif nb_params < nb_weight
        @warn("Your weight vector have more element than you have parameters to change")
    end
    ps = Flux.params(model)
    layer_idx = 1
    curr_idx = 1
    for layer in ps
        for i in eachindex(layer)
            ps[layer_idx][i] = weights[curr_idx]
            curr_idx += 1
        end
        layer_idx +=1
    end
end

#-------------Utils to create GA of SokoAgent or ContinuousSokoLvl-------------#
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

#------------Utils to create sNES of SokoAgent or ContinuousSokoLvl------------#
mutable struct sNES{T} <: Cambrian.AbstractEvolution
    config::NamedTuple
    logger::CambrianLogger
    population::Array{T}
    elites::Array{T}
    state::sNESState
    fitness::Function
    gen::Int
end

function snes_init(itype::Type, model, cfg::NamedTuple, state::EvolutionaryStrategies.ESState)
    population = Array{itype}(undef, cfg.n_population)
    for i in 1:cfg.n_population
        genes = state.μ .+ state.σ .* view(state.s, :, i)
        population[i] = itype(genes,model,cfg)
    end
    population
end

function sNES{T}(model, cfg::NamedTuple, fitness::Function, state::EvolutionaryStrategies.sNESState;
     logfile=string("logs/", cfg.id, ".csv")) where T
    logger = CambrianLogger(logfile)
    population = snes_init(T, model, cfg, state)
    elites = deepcopy([population[i] for i in 1:cfg.n_elite])
    sNES{T}(cfg, logger, population, elites, state, fitness, 0)
end

function sNES{T}(model, cfg::NamedTuple, fitness::Function;
    logfile=string("logs/", cfg.id, ".csv")) where T
    logger = CambrianLogger(logfile)
    cfg = merge(EvolutionaryStrategies.snes_config(cfg.n_genes), cfg)
    state = sNESState(cfg.n_genes, cfg.n_population)
    sNES{T}(model, cfg, fitness, state; logfile=logfile)
end

function snes_populate(e::sNES{T}) where T
    for i in eachindex(e.population)
        e.population[i].genes .= e.state.μ .+ e.state.σ .* view(e.state.s, :, i)
        e.population[i].fitness .= -Inf
    end
end

function snes_generation(e::sNES{T}) where T
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

#-------------------------Utils to replay experiments--------------------------#

# function to replay one agent on one lvl_str, you have to configure Griddly
# first
function replay(agent,lvl_str::String;render=false)
    transcript_sokoagent_genes!(agent)

    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)
    total_reward = 0

    if render
        render_window = RenderWindow(700,700)

        observation = Griddly.vector_obs(grid)
        observation = Griddly.get_data(observation)

        sprites = Griddly.observe(game)
        sprites = Griddly.get_data(sprites)
        render(render_window,sprites;nice_render=false)

        for step in 1:200
            dir = choose_action(observation,agent)
            reward, done = Griddly.step_player!(player1,"move", [dir])
            observation = Griddly.vector_obs(grid)
            observation = Griddly.get_data(observation)
            total_reward += reward
            sprites = Griddly.observe(game)
            sprites = Griddly.get_data(sprites)
            render(render_window,sprites;nice_render=false)
        end
    else
        observation = Griddly.vector_obs(grid)
        observation = Griddly.get_data(observation)

        for step in 1:200
            dir = choose_action(observation,agent)
            reward, done = Griddly.step_player!(player1,"move", [dir])
            observation = Griddly.vector_obs(grid)
            observation = Griddly.get_data(observation)
            total_reward += reward
        end
    end
    return total_reward
end

# function to replay one agent on all level in envs_path being SokoLvlIndividual
# you have to configure Griddly first
function replay_sokolvl(agent,envs_path;render=false)
    rewards = []
    individualLvlNameList = readdir(envs_path)
    for i in eachindex(individualLvlNameList)
        env_str = read("$envs_path/$(individualLvlNameList[i])", String)
        lvl = SokoLvlIndividual(env_str)
        lvl_str = transcript_sokolvl_genes(lvl)

        reward = replay(agent,lvl_str;render=render)
        push!(rewards,reward)
    end
    return rewards
end

# function to replay one agent on all level in envs_path being ContinuousSokoLvl
# you have to configure Griddly first
function replay_continuoussokolvl(agent,envs_path,env_model;render=false)
    rewards = []
    individualLvlNameList = readdir(envs_path)
    for i in eachindex(individualLvlNameList)
        env_str = read("$envs_path/$(individualLvlNameList[i])", String)
        lvl = ContinuousSokoLvl(env_str,env_model)
        lvl_str = lvl.output_map

        reward = replay(agent,lvl_str;render=render)
        push!(rewards,reward)
    end
    return rewards
end

# function to replay all agents in agents_path on all level in envs_path being
# SokoLvlIndividual, you have to configure Griddly first
function replay_sokolvl(agents_path::String,agent_model,envs_path::String;render=false)
    agentNameList = readdir(agents_path)
    individualLvlNameList = readdir(envs_path)
    rewards = zeros(length(agentNameList), length(individualLvlNameList))
    for i in eachindex(agentNameList)
        agent_str = read("$agents_path/$(agentNameList[i])", String)
        agent = SokoAgent(agent_str,agent_model)
        rewards[:,i] = replay_sokolvl(agent,envs_path;render=render)
    end
    return rewards
end

# function to replay all agents in agents_path on all level in envs_path being
# ContinuousSokoLvl, you have to configure Griddly first
function replay_continuoussokolvl(agents_path::String,agent_model,envs_path::String,env_model;render=false)
    agentNameList = readdir(agents_path)
    individualLvlNameList = readdir(envs_path)
    rewards = zeros(length(agentNameList), length(individualLvlNameList))
    for i in eachindex(agentNameList)
        agent_str = read("$agents_path/$(agentNameList[i])", String)
        agent = SokoAgent(agent_str,agent_model)
        rewards[:,i] = replay_continuoussokolvl(agent,envs_path,env_model;render=render)
    end
    return rewards
end
