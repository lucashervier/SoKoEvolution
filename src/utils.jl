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

function load_weights_from_array!(model::Chain, params::AbstractArray)
    p = 1
    layers = Flux.params(model)
    for li in 1:length(layers)
        copyto!(layers[li], 1, params, p, length(layers[li]))
        p += length(layers[li])
    end
end

# function load_weights_from_array!(model,weights)
#     nb_params = get_params_count(model)
#     nb_weight = length(weights)
#     if nb_params > nb_weight
#         throw("Your weight vector is not long enough")
#     elseif nb_params < nb_weight
#         @warn("Your weight vector have more element than you have parameters to change")
#     end
#     ps = Flux.params(model)
#     layer_idx = 1
#     curr_idx = 1
#     for layer in ps
#         for i in eachindex(layer)
#             ps[layer_idx][i] = weights[curr_idx]
#             curr_idx += 1
#         end
#         layer_idx +=1
#     end
# end
#-------------Utils for some fitness function----------------------------------#
function has_the_box_moved(old_observation,new_observation,box_idx)::Bool
    old_box = old_observation[box_idx,:,:]
    new_box = new_observation[box_idx,:,:]

    if old_box-new_box==zeros(8,8)
        return false
    end
    return true
end

function count_items(item_idx::Int,observation)
    item_obs = observation[item_idx,:,:]
    return sum(item_obs)
end

function count_blocked_box(observation)
    nb_box = count_items(1,observation)
    box_see = 0
    box_blocked = 0
    box_obs = observation[1,:,:]
    wall_obs = observation[3,:,:]
    box_on_holes = observation[2,:,:]
    width,height = size(box_obs)
    for i in 1:height,j in 1:width
        # tell if there is a box at poss (i,j)
        box_in_i_j = (box_obs[i,j] == 1)
        if box_in_i_j
            box_see += 1

            # handle upper border cases
            if i == 1
                # special corner cases
                if j == 1
                    box_blocked += 1
                elseif j == width
                    box_blocked += 1
                # other border
                else
                    # if obstacles on a side of the box its blocked
                    if (box_obs[i,j-1] == 1 || wall_obs[i,j-1] == 1 || box_on_holes[i,j-1] == 1 || box_obs[i,j+1] == 1 || wall_obs[i,j+1] == 1 || box_on_holes[i,j+1] == 1)
                        box_blocked += 1
                    end
                end
            # handle lower border cases
            elseif i == height
                # special corner cases
                if j == 1
                    box_blocked += 1
                elseif j == width
                    box_blocked += 1
                # other border
                else
                    # if obstacles on a side of the box its blocked
                    if (box_obs[i,j-1] == 1 || wall_obs[i,j-1] == 1 || box_on_holes[i,j-1] == 1 || box_obs[i,j+1] == 1 || wall_obs[i,j+1] == 1 || box_on_holes[i,j+1] == 1)
                        box_blocked += 1
                    end
                end
            # handle right border case scenario
            elseif j == 1
                # if obstacles on a up/downside of the box its blocked
                if (box_obs[i-1,j] == 1 || wall_obs[i-1,j] == 1 || box_on_holes[i-1,j] == 1 || box_obs[i+1,j] == 1 || wall_obs[i+1,j] == 1 || box_on_holes[i+1,j] == 1)
                    box_blocked += 1
                end
            # handle left border case scenario
            elseif j == width
                # if obstacles on a up/downside of the box its blocked
                if (box_obs[i-1,j] == 1 || wall_obs[i-1,j] == 1 || box_on_holes[i-1,j] == 1 || box_obs[i+1,j] == 1 || wall_obs[i+1,j] == 1 || box_on_holes[i+1,j] == 1)
                    box_blocked += 1
                end
            # all non border case
            else
                up_or_down_obstacles = (box_obs[i-1,j] == 1 || wall_obs[i-1,j] == 1 || box_on_holes[i-1,j] == 1 || box_obs[i+1,j] == 1 || wall_obs[i+1,j] == 1 || box_on_holes[i+1,j] == 1)
                left_or_right_obstacles = (box_obs[i,j-1] == 1 || wall_obs[i,j-1] == 1 || box_on_holes[i,j-1] == 1 || box_obs[i,j+1] == 1 || wall_obs[i,j+1] == 1 || box_on_holes[i,j+1] == 1)
                if (up_or_down_obstacles&&left_or_right_obstacles)
                    box_blocked +=1
                end
            end
        end
        # if we check all boxes we quit
        if box_see == nb_box
            return box_blocked
        end
    end
    return box_blocked
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
function replay(agent,lvl_str::String;rendering=false)
    transcript_sokoagent_genes!(agent)

    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)
    total_reward = 0

    if rendering
        render_window = RenderWindow(700,700)

        observation = Griddly.vector_obs(grid)
        observation = Griddly.get_data(observation)

        sprites = Griddly.observe(game)
        sprites = Griddly.get_data(sprites)
        render(render_window,sprites;nice_render=true)

        for step in 1:200
            dir = choose_action(observation,agent)
            reward, done = Griddly.step_player!(player1,"move", [dir])
            observation = Griddly.vector_obs(grid)
            observation = Griddly.get_data(observation)
            total_reward += reward
            sprites = Griddly.observe(game)
            sprites = Griddly.get_data(sprites)
            render(render_window,sprites;nice_render=false)
            if done==1
                break
            end
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
            if done==1
                break
            end
        end
    end
    return total_reward
end

# function to replay one agent on all level in envs_path being SokoLvlIndividual
# you have to configure Griddly first
function replay_sokolvl(agent,envs_path;rendering=false)
    rewards = []
    individualLvlNameList = readdir(envs_path)
    for i in eachindex(individualLvlNameList)
        env_str = read("$envs_path/$(individualLvlNameList[i])", String)
        lvl = SokoLvlIndividual(env_str)
        lvl_str = transcript_sokolvl_genes(lvl)

        reward = replay(agent,lvl_str;rendering=rendering)
        push!(rewards,reward)
    end
    return rewards
end

# function to replay one agent on all level in envs_path being ContinuousSokoLvl
# you have to configure Griddly first
function replay_continuoussokolvl(agent,envs_path,env_model;rendering=false)
    rewards = []
    individualLvlNameList = readdir(envs_path)
    for i in eachindex(individualLvlNameList)
        env_str = read("$envs_path/$(individualLvlNameList[i])", String)
        lvl = ContinuousSokoLvl(env_str,env_model)
        lvl_str = lvl.output_map

        reward = replay(agent,lvl_str;rendering=rendering)
        push!(rewards,reward)
    end
    return rewards
end

# function to replay all agents in agents_path on all level in envs_path being
# SokoLvlIndividual, you have to configure Griddly first
function replay_sokolvl(agents_path::String,agent_model,envs_path::String;rendering=false)
    agentNameList = readdir(agents_path)
    individualLvlNameList = readdir(envs_path)
    rewards = zeros(length(agentNameList), length(individualLvlNameList))
    for i in eachindex(agentNameList)
        agent_str = read("$agents_path/$(agentNameList[i])", String)
        agent = SokoAgent(agent_str,agent_model)
        rewards[:,i] = replay_sokolvl(agent,envs_path;rendering=rendering)
    end
    return rewards
end

# function to replay all agents in agents_path on all level in envs_path being
# ContinuousSokoLvl, you have to configure Griddly first
function replay_continuoussokolvl(agents_path::String,agent_model,envs_path::String,env_model;rendering=false)
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

# function to replay one agent on all level in envs_path being SokoLvlIndividual
# you have to configure Griddly first and to save the video
function replay_video_sokolvl(agent,envs_path,saving_path,video_name)
    individualLvlNameList = readdir(envs_path)
    video = VideoRecorder((700,700),video_name;saving_path=saving_path)
    io = start_video(video)
    for i in eachindex(individualLvlNameList)
        env_str = read("$envs_path/$(individualLvlNameList[i])", String)
        lvl = SokoLvlIndividual(env_str)
        lvl_str = transcript_sokolvl_genes(lvl)

        Griddly.load_level_string!(grid,lvl_str)
        Griddly.reset!(game)

        observation = Griddly.vector_obs(grid)
        observation = Griddly.get_data(observation)

        sprites = Griddly.observe(game)
        sprites = Griddly.get_data(sprites)
        add_frame!(video,io,sprites;speed=1/5,fast_display=true)

        for step in 1:200
            dir = choose_action(observation,agent)
            reward, done = Griddly.step_player!(player1,"move", [dir])

            observation = Griddly.vector_obs(grid)
            observation = Griddly.get_data(observation)

            sprites = Griddly.observe(game)
            sprites = Griddly.get_data(sprites)
            add_frame!(video,io,sprites;speed=1/5,fast_display=true)

            if done==1
                break
            end
        end
    end
    save_video(video,io)
end

# function to replay one agent on all level in envs_path being ContinuousSokoLvl
# you have to configure Griddly first and to save the video
function replay_video_continuoussokolvl(agent,envs_path,envs_model,saving_path,video_name)
    individualLvlNameList = readdir(envs_path)
    video = VideoRecorder((700,700),video_name;saving_path=saving_path)
    io = start_video(video)
    for i in eachindex(individualLvlNameList)
        env_str = read("$envs_path/$(individualLvlNameList[i])", String)
        lvl = ContinuousSokoLvl(env_str,env_model)
        lvl_str = lvl.output_map

        Griddly.load_level_string!(grid,lvl_str)
        Griddly.reset!(game)

        observation = Griddly.vector_obs(grid)
        observation = Griddly.get_data(observation)

        sprites = Griddly.observe(game)
        sprites = Griddly.get_data(sprites)
        add_frame!(video,io,sprites;speed=1/5,fast_display=true)

        for step in 1:200
            dir = choose_action(observation,agent)
            reward, done = Griddly.step_player!(player1,"move", [dir])

            observation = Griddly.vector_obs(grid)
            observation = Griddly.get_data(observation)

            sprites = Griddly.observe(game)
            sprites = Griddly.get_data(sprites)
            add_frame!(video,io,sprites;speed=1/5,fast_display=true)

            if done==1
                break
            end
        end
    end
    save_video(video,io)
end

function replay_video_list(agent,levels_list,saving_path,video_name)
    video = VideoRecorder((700,700),video_name;saving_path=saving_path)
    io = start_video(video)
    for i in eachindex(levels_list)
        lvl_str = levels_list[i]

        Griddly.load_level_string!(grid,lvl_str)
        Griddly.reset!(game)

        observation = Griddly.vector_obs(grid)
        observation = Griddly.get_data(observation)

        sprites = Griddly.observe(game)
        sprites = Griddly.get_data(sprites)
        add_frame!(video,io,sprites;speed=1/5,fast_display=true)

        for step in 1:200
            dir = choose_action(observation,agent)
            reward, done = Griddly.step_player!(player1,"move", [dir])

            observation = Griddly.vector_obs(grid)
            observation = Griddly.get_data(observation)

            sprites = Griddly.observe(game)
            sprites = Griddly.get_data(sprites)
            add_frame!(video,io,sprites;speed=1/5,fast_display=true)

            if done==1
                break
            end
        end
    end
    save_video(video,io)
end
