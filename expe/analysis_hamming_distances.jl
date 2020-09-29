using Distances
using DelimitedFiles
using Statistics
using Formatting
using JSON
using CSV
using Plots

function ind_parse(st::String)
    d = JSON.Parser.parse(st)
    for i in 1:length(d["fitness"])
        if d["fitness"][i] == nothing
            d["fitness"][i] = -Inf
        end
    end
    d
end

mean_hamming_envs = []
std_hamming_envs = []
envs_path = "gens//sokoevo_higher//envs"
for k in 1:600
    envs_genes = []
    path = Formatting.format("$envs_path//{1:04d}",k*5)
    individualLvlNameList = readdir("$path")
    for i in 1:13
        genes_str = read("$path//$(individualLvlNameList[i])", String)
        genes_dict = ind_parse(genes_str)
        genes_array = BitArray(genes_dict["genes"])
        push!(envs_genes,genes_array)
    end
    R = []
    for i in 1:12
        for j in i+1:13
            push!(R,hamming(envs_genes[i],envs_genes[j]))
        end
    end
    push!(mean_hamming_envs,mean(R))
    push!(std_hamming_envs,std(R))
end
println(length(mean_hamming_envs))
println(length(std_hamming_envs))
println(mean_hamming_envs[1])
xs = [i*5 for i in 401:600]
plot(xs,xaxis="Generation number",yaxis="Mean Hamming distange",title="Hamming distance inside a generation: gen 2001 to 3000",mean_hamming_envs[401:600],ribbon=std_hamming_envs[401:600],fillalpha=.3,label="")
