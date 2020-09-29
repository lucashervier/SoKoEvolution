using Griddly
using Cambrian
using Flux

include("../src/sokolvl_individual.jl")
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
                    Conv((3,3),4=>1,pad=(1,1),relu),
                    Flux.flatten,
                    RNN(64,64),
                    Dense(64,4),
                    softmax
                    )
agent_path = "gens//sokoevo_rnn_agents//agents//0500//0001.dna"
envs_path = "gens//sokoevo_rnn_agents//envs//0500"

agent_string = read("$agent_path", String)
agent = SokoAgent(agent_string,agent_model)

transcript_sokoagent_genes!(agent)

function video_best_gen()
    individualLvlNameList = readdir("$envs_path")
    video = VideoRecorder((700,700),"video_best_gen_es_sokoevo_worst";saving_path="videos/")
    io = start_video(video)
    for i in eachindex(individualLvlNameList)
        env_str = read("$envs_path/$(individualLvlNameList[i])", String)
        lvl = SokoLvlIndividual(env_str)
        lvl_str = transcript_sokolvl_genes(lvl)
        Griddly.load_level_string!(grid,lvl_str)
        Griddly.reset!(game)
        total_reward = 0
        observation = Griddly.vector_obs(grid)
        observation = Griddly.get_data(observation)
        for step in 1:200
            dir = choose_action(observation,agent)
            reward, done = Griddly.step_player!(player1,"move", [dir])
            observation = Griddly.vector_obs(grid)
            observation = Griddly.get_data(observation)
            sprite = Griddly.observe(game)
            sprite = Griddly.get_data(sprite)
            add_frame!(video,io,sprite;speed=1/5,fast_display=true)
            total_reward += reward
        end
        println(total_reward)
    end
    save_video(video,io)
end
video_best_gen()
