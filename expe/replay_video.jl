using Griddly
using Cambrian
using Flux
using JSON
include("../src/utils.jl")
include("../src/soko_agent.jl")
# include("../src/sokolvl_individual.jl")
# include("../src/continuous_sokolvl.jl")
#-----------------------------Griddly Configuration----------------------------#
image_path = joinpath(@__DIR__,"..","resources","images")
shader_path = joinpath(@__DIR__,"..","resources","shaders")
gdy_path = joinpath(@__DIR__,"..","resources","games")
gdy_reader = Griddly.GDYReader(image_path,shader_path)

game_name = "sokoban3.yaml"
grid = Griddly.load!(gdy_reader,joinpath(gdy_path,"Single-Player/GVGAI/$game_name"))
game = Griddly.create_game(grid,Griddly.SPRITE_2D)
player1 = Griddly.register_player!(game,"Tux", Griddly.BLOCK_2D)
Griddly.init!(game)
#-----------------------------Agent Configuration------------------------------#
agent_model = Chain(
                    Conv((3,3),5=>1,pad=(1,1),relu),
                    Flux.flatten,
                    RNN(64,64),
                    Dense(64,4),
                    softmax
                    )
agent_path = "..//Buboresults//Buboresults//gens//sokoevolution_existinglvl_sokoagent//overall//trial_1//4135//0015.dna"

agent_string = read("$agent_path", String)
agent = SokoAgent(agent_string,agent_model)

transcript_sokoagent_genes!(agent)
#-----------------------------Envs Configuration------------------------------#
# envs_model = Chain(
#                     Dense(2,16),
#                     Dense(16,32),
#                     Dense(32,16),
#                     Dense(16,5),
#                     softmax
#                     )

# envs_path = "gens//sokoevo_rnn_agents//envs//0500"

levels_path = "cfg/set_known_envs.json"
levels_string = read(levels_path,String)
levels_dict = JSON.Parser.parse(levels_string)
levels_list = levels_dict["levels_string"]
#-------------------------------Main-------------------------------------------#
# replay_video_sokolvl(agent,envs_path,"videos/","sokoevo_rnn_agents")
# replay_video_continuoussokolvl(agent,envs_path,envs_model,"videos/","sokoevo_continuouslvl")
replay_video_list(agent,levels_list,"videos/","evolution_on_known_levels")
