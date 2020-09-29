using Plots
using DelimitedFiles
using Statistics
using Flux
using Cambrian
using Griddly
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
#-------------------------------Fitness Analysis-------------------------------#
agent_logs_path = "logs//sokoevo_higher_box_constraints//agents.csv"
env_logs_path = "logs//sokoevo_higher_box_constraints//envs.csv"

agent_mat_logs = readdlm(agent_logs_path,',')
env_mat_logs = readdlm(env_logs_path,',')

fitness_mean_envs = env_mat_logs[:,6]
fitness_std_envs = env_mat_logs[:,7]
fitness_mean_agents = agent_mat_logs[:,6]
fitness_std_agents = agent_mat_logs[:,7]

#-------------------------Best Agent Overall-----------------------------------#
agent_model = Chain(
                    Conv((3,3),4=>1,pad=(1,1),relu),
                    Flux.flatten,
                    RNN(144,144),
                    Dense(144,4),
                    softmax
                    )

best_agent_gen = agent_mat_logs[argmax(agent_mat_logs[:,5]),4]
best_agent_path = "gens//sokoevo_higher_box_constraints//agents//bests//0$best_agent_gen//0013.dna"
best_agent_overall_string = read("$best_agent_path", String)
best_agent = SokoAgent(best_agent_overall_string,agent_model)

transcript_sokoagent_genes!(best_agent)

last_agent_path = "gens//sokoevo_higher_box_constraints//agents//bests//0900//0013.dna"
last_agent_string = read("$last_agent_path", String)
last_agent = SokoAgent(last_agent_string,agent_model)

transcript_sokoagent_genes!(last_agent)

envs_path = "gens//sokoevo_higher_box_constraints//envs//bests//0$best_agent_gen"

best_agent_scores1 = []
last_agent_scores1 = []

individualLvlNameList = readdir("$envs_path")
for i in eachindex(individualLvlNameList)
    env_str = read("$envs_path/$(individualLvlNameList[i])", String)
    lvl = SokoLvlIndividual(env_str)
    lvl_str = transcript_sokolvl_genes(lvl)
    Griddly.load_level_string!(grid,lvl_str)
    # best agent
    Griddly.reset!(game)
    total_reward = 0
    observation = Griddly.vector_obs(grid)
    observation = Griddly.get_data(observation)
    for step in 1:200
        dir = choose_action(observation,best_agent)
        reward, done = Griddly.step_player!(player1,"move", [dir])
        observation = Griddly.vector_obs(grid)
        observation = Griddly.get_data(observation)
        total_reward += reward
    end
    push!(best_agent_scores1,total_reward)
    # last agent
    Griddly.reset!(game)
    total_reward = 0
    observation = Griddly.vector_obs(grid)
    observation = Griddly.get_data(observation)
    for step in 1:200
        dir = choose_action(observation,last_agent)
        reward, done = Griddly.step_player!(player1,"move", [dir])
        observation = Griddly.vector_obs(grid)
        observation = Griddly.get_data(observation)
        total_reward += reward
    end
    push!(last_agent_scores1,total_reward)
end

last_envs_path = "gens//sokoevo_higher_box_constraints//envs//0900"

best_agent_scores2 = []
last_agent_scores2 = []

individualLvlNameList = readdir("$last_envs_path")
for i in eachindex(individualLvlNameList)
    env_str = read("$last_envs_path/$(individualLvlNameList[i])", String)
    lvl = SokoLvlIndividual(env_str)
    lvl_str = transcript_sokolvl_genes(lvl)
    Griddly.load_level_string!(grid,lvl_str)
    # best agent
    Griddly.reset!(game)
    total_reward = 0
    observation = Griddly.vector_obs(grid)
    observation = Griddly.get_data(observation)
    for step in 1:200
        dir = choose_action(observation,best_agent)
        reward, done = Griddly.step_player!(player1,"move", [dir])
        observation = Griddly.vector_obs(grid)
        observation = Griddly.get_data(observation)
        total_reward += reward
    end
    push!(best_agent_scores2,total_reward)
    # last agent
    Griddly.reset!(game)
    total_reward = 0
    observation = Griddly.vector_obs(grid)
    observation = Griddly.get_data(observation)
    for step in 1:200
        dir = choose_action(observation,last_agent)
        reward, done = Griddly.step_player!(player1,"move", [dir])
        observation = Griddly.vector_obs(grid)
        observation = Griddly.get_data(observation)
        total_reward += reward
    end
    push!(last_agent_scores2,total_reward)
end

println("best_agent_gen:$best_agent_gen")
println("best_overall_own_envs:\n$best_agent_scores1")
println("last_agent_overall_envs:\n$last_agent_scores1")

println("best_overall_last_envs:\n$best_agent_scores2")
println("last_agent_own_envs:\n$last_agent_scores2")


xs = [i for i in 400:900]
plot(xs,background=:lightgray,xaxis="Generation number",yaxis="Fitness",title="CoEvolution between Sokoban level and agents with box constraint",fitness_mean_envs[400:900],ribbon=fitness_std_envs[400:900],fillalpha=.5,label="Envs fitness")
plot!(xs,fitness_mean_agents[400:900],ribbon=fitness_std_agents[400:900],fillalpha=.5,label="Agents fitness")
