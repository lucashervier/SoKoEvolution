using Griddly
using Cambrian
using Flux
using JSON

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
agent_path = "..//Buboresults//Buboresults//gens//sokoevolution_existinglvl_sokoagent_fitness2//overall//trial_1//3151//0015.dna"
agent_string = read("$agent_path", String)
agent = SokoAgent(agent_string,agent_model)
# envs_path = "..//Buboresults//Buboresults//gens//sokoevo_rnnagents_directenv_sokoban3//best//envs//0064"

levels_path = "cfg/set_known_envs_2.json"
levels_string = read(levels_path,String)
levels_dict = JSON.Parser.parse(levels_string)
levels_list = levels_dict["levels_string"]

# replay_sokolvl(agent,envs_path;rendering=true)
for i in eachindex(levels_list)
    reward = replay(agent,levels_list[i];rendering=true)
    println(reward)
end
