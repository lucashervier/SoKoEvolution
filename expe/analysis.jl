using Plots
using DelimitedFiles
using Statistics
using Flux
using Cambrian
using Griddly
include("../src/sokolvl_individual.jl")
include("../src/utils.jl")
include("../src/soko_agent.jl")
include("../src/continuous_sokolvl.jl")
#-------------------------------Expe Configuration-----------------------------#
agent_logs_path = "..//results//logs//sokoevo_rnnagents_directenv_sokoban3//agents.csv"
env_logs_path = "..//results//logs//sokoevo_rnnagents_directenv_sokoban3//envs.csv"

agent_mat_logs = readdlm(agent_logs_path,',')
env_mat_logs = readdlm(env_logs_path,',')

agents_gen_path = "..//results//gens//sokoevo_rnnagents_directenv_sokoban3"
envs_gen_path = "..//results//gens//sokoevo_rnnagents_directenv_sokoban3"

game_name = "sokoban3.yaml"

agent_model = Chain(
                    Conv((3,3),4=>1,pad=(1,1),relu),
                    Flux.flatten,
                    RNN(144,144),
                    Dense(144,4),
                    softmax
                    )
# if continuous envs
envs_model = Chain(
                    Dense(2,16),
                    Dense(16,32),
                    Dense(32,16),
                    Dense(16,5),
                    softmax
                    )
# from which gen you want to start analysis to which gen you want to stop and
# gen_step to get all the saved gen (if you save 1 out of 10 it would be 10)
gen_start = 1
gen_step =  1
gen_end = 9000
# for saving figures
figure_saving_path = "..//results//gens//sokoevo_rnnagents_directenv_sokoban3//fitness_plot.png"
#-----------------------------Griddly Configuration----------------------------#
image_path = joinpath(@__DIR__,"..","resources","images")
shader_path = joinpath(@__DIR__,"..","resources","shaders")
gdy_path = joinpath(@__DIR__,"..","resources","games")
gdy_reader = Griddly.GDYReader(image_path,shader_path)

grid = Griddly.load!(gdy_reader,joinpath(gdy_path,"Single-Player/GVGAI/$game_name"))
game = Griddly.create_game(grid,Griddly.SPRITE_2D)
player1 = Griddly.register_player!(game,"Tux", Griddly.BLOCK_2D)
Griddly.init!(game)
#-------------------------------Fitness Analysis-------------------------------#
fitness_mean_envs = env_mat_logs[:,6]
fitness_std_envs = env_mat_logs[:,7]

fitness_mean_agents = agent_mat_logs[:,6]
fitness_std_agents = agent_mat_logs[:,7]

title_plot = "Fitness of a CoEvolution between Sokoban level and Sokoban players"
function plot_results(title_plot,gen_start,gen_step,gen_end,figure_saving_path)
    xs = [i for i in gen_start:gen_step:gen_end]
    plot(xs,background=:lightgray,xaxis="Generation number",yaxis="Fitness",title=title_plot,fitness_mean_envs[Int(gen_start/gen_step):Int(gen_end/gen_step)],ribbon=fitness_std_envs[Int(gen_start/gen_step):Int(gen_end/gen_step)],fillalpha=.5,label="Envs fitness");
    plot!(xs,fitness_mean_agents[Int(gen_start/gen_step):Int(gen_end/gen_step)],ribbon=fitness_std_agents[Int(gen_start/gen_step):Int(gen_end/gen_step)],fillalpha=.5,label="Agents fitness");
    savefig(figure_saving_path)
end
# #---------------------Best Fitness Agent Overall-------------------------------#
# best_agent_gen = agent_mat_logs[argmax(agent_mat_logs[:,5]),4]
# best_agent_path = Formatting.format("$agents_gen_path//{1:04d}//0013.dna",best_agent_gen)
# best_agent_overall_string = read("$best_agent_path", String)
# best_agent = SokoAgent(best_agent_overall_string,agent_model)
#
# rendering = false
#
# # replay the best agents on its own envs
# best_agent_envs_path = Formatting.format("$envs_gen_path//{1:04d}",best_agent_gen)
# # if your envs are SokoLvlIndividual
# best_own_rewards = replay_sokolvl(best_agent,best_agent_envs_path;render=rendering)
# # # if your envs are ContinuousSokoLvl
# # best_own_rewards = replay_continuoussokolvl(best_agent,best_agent_envs_path,envs_model;render=rendering)
#
# # comparison with the last agent to see if its get better generall results
# last_gen_agent_path = Formatting.format("$agents_gen_path//{1:04d}//0013.dna",gen_end)
# last_gen_agent_str = read(last_gen_agent_str,String)
# last_gen_agent = SokoAgent(last_gen_agent_str,agent_model)
#
# # if your envs are SokoLvlIndividual
# last_bestenvs_rewards = replay_sokolvl(last_gen_agent,best_agent_envs_path;render=rendering)
# # # if your envs are ContinuousSokoLvl
# # last_bestenvs_rewards = replay_continuoussokolvl(last_gen_agent,best_agent_envs_path,envs_model;render=rendering)
#
# last_gen_envs_path = Formatting.format("$envs_gen_path//{1:04d}",gen_end)
#
# # if your envs are SokoLvlIndividual
# last_own_rewards = replay_sokolvl(last_gen_agent,last_gen_envs_path;render=rendering)
# # # if your envs are ContinuousSokoLvl
# # last_own_rewards = replay_continuoussokolvl(last_gen_agent,last_gen_envs_path,envs_model;render=rendering)
#
# # if your envs are SokoLvlIndividual
# best_lastenvs_rewards = replay_sokolvl(best_agent,last_gen_envs_path;render=rendering)
# # # if your envs are ContinuousSokoLvl
# # best_lastenvs_rewards = replay_continuoussokolvl(best_agent,last_gen_envs_path,envs_model;render=rendering)
#
# println("best_agent_gen:$best_agent_gen")
# println("best agent on its own envs:\n$best_own_rewards")
# println("last agent on the best agent envs:\n$last_bestenvs_rewards")
# println("best agent on last envs:\n$best_lastenvs_rewards")
# println("last agent on its own envs:\n$last_own_rewards")
#--------------------------------Main------------------------------------------#
plot_results(title_plot,gen_start,gen_step,gen_end,figure_saving_path)
