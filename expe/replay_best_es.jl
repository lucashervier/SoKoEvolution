using Griddly
using Cambrian
using Flux

include("../src/utils.jl")
include("../src/soko_agent.jl")

#-----------------------------Griddly Configuration----------------------------#
image_path = joinpath(@__DIR__,"..","resources","images")
shader_path = joinpath(@__DIR__,"..","resources","shaders")
gdy_path = joinpath(@__DIR__,"..","resources","games")
gdy_reader = Griddly.GDYReader(image_path,shader_path)

grid = Griddly.load!(gdy_reader,joinpath(gdy_path,"Single-Player/GVGAI/sokoban.yaml"))
game = Griddly.create_game(grid,Griddly.SPRITE_2D)
player1 = Griddly.register_player!(game,"Tux", Griddly.BLOCK_2D)
Griddly.init!(game)

#-----------------------------Agent Configuration------------------------------#
agent_model = Chain(
                    Conv((3,3),4=>4,pad=(1,1),relu),
                    Conv((3,3),4=>1,pad=(1,1),relu),
                    Flux.flatten,
                    Dense(64,32,relu),
                    Dense(32,4),
                    softmax
                    )
agent_path = "gens//ES//solo_agent_es//bests//0750//0013.dna"

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

function has_the_box_moved(old_observation,new_observation)::Bool
    old_observation = convert(Array{Int,3},old_observation)
    new_observation = convert(Array{Int,3},new_observation)

    old_box = old_observation[1,:,:]
    new_box = new_observation[1,:,:]

    if old_box-new_box==zeros(8,8)
        return false
    end
    return true
end

agent_string = read("$agent_path", String)
agent = SokoAgent(agent_string,agent_model)

transcript_sokoagent_genes!(agent)
Griddly.load_level_string!(grid,lvl_str)
Griddly.reset!(game)

function run_best()
    # render_window = RenderWindow(700,700)
    video = VideoRecorder((700,700),"video_best_agent_es2";saving_path="videos/")
    io = start_video(video)
    for j in 1:10
        total_reward = 0
        old_observation = Griddly.vector_obs(grid)
        old_observation = Griddly.get_data(old_observation)
        for step in 1:200
            new_observation = Griddly.vector_obs(grid)
            new_observation = Griddly.get_data(new_observation)
            if has_the_box_moved(old_observation,new_observation)
                total_reward += 1
            end
            dir = choose_action(new_observation,agent)
            # println("dir:$dir")
            reward, done = Griddly.step_player!(player1,"move", [dir])
            sprite = Griddly.observe(game)
            sprite = Griddly.get_data(sprite)
            # render(render_window,sprite;nice_render=true)
            # sleep(1)
            add_frame!(video,io,sprite;speed=1/5,fast_display=true)
            total_reward += reward*100
            if done == 1
                break
            end
            old_observation = deepcopy(new_observation)
        end
        println(total_reward)
    end
    save_video(video,io)
end
# run_best()
agents_path = "gens//ES//solo_agent_es//bests//0750"

function video_best_gen()
    individualNameList = readdir("$agents_path")
    video = VideoRecorder((700,700),"video_best_gen_es";saving_path="videos/")
    io = start_video(video)
    for i in eachindex(individualNameList)
        agent_string = read("$agents_path/$(individualNameList[i])", String)
        agent = SokoAgent(agent_string,agent_model)
        transcript_sokoagent_genes!(agent)
        Griddly.reset!(game)
        total_reward = 0
        old_observation = Griddly.vector_obs(grid)
        old_observation = Griddly.get_data(old_observation)
        for step in 1:200
            new_observation = Griddly.vector_obs(grid)
            new_observation = Griddly.get_data(new_observation)
            if has_the_box_moved(old_observation,new_observation)
                total_reward += 1
            end
            dir = choose_action(new_observation,agent)
            reward, done = Griddly.step_player!(player1,"move", [dir])
            sprite = Griddly.observe(game)
            sprite = Griddly.get_data(sprite)
            add_frame!(video,io,sprite;speed=1/5,fast_display=true)
            total_reward += reward*100
            if done == 1
                break
            end
            old_observation = deepcopy(new_observation)
        end
        println(total_reward)
    end
    save_video(video,io)
end
# video_best_gen()
