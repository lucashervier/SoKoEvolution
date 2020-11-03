using Plots
using DelimitedFiles
using Statistics
#-------------------------------Expe Configuration-----------------------------#
expe_name = "your_expe_name"
# if either envs or agents only experiment
expe_logs_path = "..//Buboresults//logs//$expe_name//$expe_name.csv"
# # if logs for a coevolution
# env_logs_path = "..//Buboresults//logs//$expe_name//envs.csv"
# agent_logs_path = "..//Buboresults//logs//$expe_name//agents.csv"

# from which gen you want to start analysis to which gen you want to stop and
# gen_step to get all the saved gen (if you save 1 out of 10 it would be 10)
gen_start = 1
gen_step = 1
gen_end = 50000
xs = [i for i in gen_start:gen_step:gen_end]

title_plot = "your_title_plot"
#---------------------------Either Agent or Envs-------------------------------#
mat_logs = readdlm(expe_logs_path,',')
# basic logs info
fitness_max = mat_logs[gen_start:gen_step:gen_end,5]
fitness_mean = mat_logs[gen_start:gen_step:gen_end,6]
fitness_std_agents = mat_logs[gen_start:gen_step:gen_end,7]
# # if you changed the logs function you might have more information
# fitness_agent = mat_logs[gen_start:gen_step:gen_end,8]
# fitness_random = mat_logs[gen_start:gen_step:gen_end,9]
# nb_objectives = mat_logs[gen_start:gen_step:gen_end,10]
# connectivity = mat_logs[gen_start:gen_step:gen_end,11]
# no_boxes_moved = mat_logs[gen_start:gen_step:gen_end,12]

# basic plot
plot(xaxis="Generation number",yaxis="Fitness",title=title_plot, size=(700,700),legend=:bottomright)
plot!(xs,fitness_max,label="Fitness maximum");
plot!(xs,fitness_mean,label="Fitness mean")
# plot!(xs,fitness_mean,ribbon=fitness_std_agents,fillalpha=.5,label="Fitness mean")

# # if you have more things to plot
# plot!(xs,fitness_agent./nb_objectives,label="Agent reward")
# plot!(xs,fitness_random./nb_objectives,label="Random reward")
# plot!(xs,connectivity./10,label="Connectivity reward")
# plot!(xs,no_boxes_moved.*(-2),label="No boxes pushed penalty")
# plot!(xs,nb_objectives)

#------------------------------CoEvolution-------------------------------------#
# title_plot = "your_title_plot"
# agent_mat_logs = readdlm(agent_logs_path,',')
# env_mat_logs = readdlm(env_logs_path,',')
#
# fitness_max_envs = env_mat_logs[gen_start:gen_step:gen_end,5]
# fitness_mean_envs = env_mat_logs[gen_start:gen_step:gen_end,6]
# fitness_std_envs = env_mat_logs[gen_start:gen_step:gen_end,7]
#
# fitness_max_agents = agent_mat_logs[gen_start:gen_step:gen_end,5]
# fitness_mean_agents = agent_mat_logs[gen_start:gen_step:gen_end,6]
# fitness_std_agents = agent_mat_logs[gen_start:gen_step:gen_end,7]
#
# # basic plot
# plot(xaxis="Generation number",yaxis="Fitness",title=title_plot, size=(700,700),legend=:bottomright)
# plot!(xs,fitness_max_envs,label="Fitness maximum Envs");
# plot!(xs,fitness_mean_envs,label="Fitness mean Envs")
# # plot!(xs,fitness_mean_envs,ribbon=fitness_std_envs,fillalpha=.5,label="Fitness mean Envs")
#
# plot!(xs,fitness_max_agents,label="Fitness maximum Agents");
# plot!(xs,fitness_mean_agents,label="Fitness mean Agents")
# # plot!(xs,fitness_mean_agents,ribbon=fitness_std_agents,fillalpha=.5,label="Fitness mean Agents")
