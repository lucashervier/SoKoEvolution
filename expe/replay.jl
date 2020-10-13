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

grid = Griddly.load!(gdy_reader,joinpath(gdy_path,"Single-Player/GVGAI/sokoban3.yaml"))
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
agent_path = "..//Buboresults//Buboresults//gens//sokoevo_rnnagents_directenv_sokoban3//best//agents//7978//0013.dna"
envs_path = "..//Buboresults//Buboresults//gens//sokoevo_rnnagents_directenv_sokoban3//best//envs//0064"

agent_string = read("$agent_path", String)
agent = SokoAgent(agent_string,agent_model)

replay_sokolvl(agent,envs_path;rendering=true)
