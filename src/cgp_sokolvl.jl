import Cambrian: ind_parse
import Base: isless

struct CGPSokoLvl
    width::Int64
    height::Int64
    objects_char_list::Array{String}
    agent_idx::Int64
    output_map::Array{String}
    cgp::CGPInd
end

function CGPSokoLvl(cfg::NamedTuple)::CGPSokoLvl
    width = cfg.width
    height = cfg.height
    objects_char_list = cfg.objects_char_list
    agent_idx = cfg.agent_idx
    output_map = [""]
    cgp = CGPInd(cfg)
    CGPSokoLvl(width,height,objects_char_list,agent_idx,output_map,cgp)
end

function CGPSokoLvl(cfg::NamedTuple,st::String)::CGPSokoLvl
    dict = ind_parse(st)
    width = cfg.width
    height = cfg.height
    objects_char_list = cfg.objects_char_list
    agent_idx = cfg.agent_idx
    output_map = dict["output_map"]
    cgp = CGPInd(cfg, Array{Float64}(dict["chromosome"]))
    CGPSokoLvl(width,height,objects_char_list,agent_idx,output_map,cgp)
end

function isless(i1::CGPSokoLvl, i2::CGPSokoLvl)
    all(i1.cgp.fitness .< i2.cgp.fitness)
end

function save_ind(ind::CGPSokoLvl,path::String)
    f = open(path, "w+")
    write(f,"""{"width":""")
    write(f, string(ind.width))
    write(f,""","height":""")
    write(f, string(ind.height))
    write(f,""","objects_char_list":""")
    write(f, string(ind.objects_char_list))
    write(f,""","agent_idx":""")
    write(f, string(ind.agent_idx))
    write(f,""","output_map":""")
    write(f, string(ind.output_map))
    write(f,""","chromosome":""")
    write(f, string(ind.cgp.chromosome))
    write(f,""","fitness":""")
    write(f, string(ind.cgp.fitness))
    write(f,"""}""")
    close(f)
end

function write_map!(cgp_lvl::CGPSokoLvl)
    # first get the size of the level
    width = cgp_lvl.width
    height = cgp_lvl.height
    # get the list of objects the level may include
    objects_char_list = cgp_lvl.objects_char_list
    # to know if the agent was selected
    agent_idx = cgp_lvl.agent_idx
    agent_in_place = false

    lvl_str = """"""

    if width%2 == 0
        x_center_grid = width/2 + 1/2
    else
        x_center_grid = width/2
    end
    if height%2 == 0
        y_center_grid = height/2 + 1/2
    else
        y_center_grid = height/2
    end

    for y_pos in 1:height
        for x_pos in 1:width
            inputs = []
            if (cgp_lvl.cgp.n_in==4)
                x = x_pos - x_center_grid
                y = y_pos - y_center_grid
                r = sqrt(x^2 + y^2)
                phi = 0
                if x==0
                    if y > 0
                        phi = pi/2
                    elseif y < 0
                        phi = -pi/2
                    else
                        phi = 0
                    end
                else
                    phi = atan(y,x)
                end
                inputs = [x,y,r,phi]
            elseif (cgp_lvl.cgp.n_in==6)
                x = x_pos - x_center_grid
                y = y_pos - y_center_grid
                r = sqrt(x^2 + y^2)
                phi = 0
                if x==0
                    if y > 0
                        phi = pi/2
                    elseif y < 0
                        phi = -pi/2
                    else
                        phi = 0
                    end
                else
                    phi = atan(y,x)
                end
                inputs = [x,y,r,phi,x_pos,y_pos]
            else
                inputs = [x_pos,y_pos]
            end
            # println("inputs:$inputs")
            # our model got 5 output: box,wall,holes,agent and floor
            object_at_x_y = ""
            output = process(cgp_lvl.cgp,inputs)
            # println("output:$output")
            idx_object = argmax(output)[1]
            # println("idx_object:$idx_object")
            if !(agent_in_place)&&(idx_object==agent_idx)
                agent_in_place = true
            elseif (agent_in_place)&&(idx_object==agent_idx)
                output[agent_idx] = 0
                idx_object = argmax(output)[1]
            end
            object_at_x_y = objects_char_list[idx_object]
            lvl_str = string(lvl_str,object_at_x_y)
        end
        lvl_str = string(lvl_str,"\n")
    end
    # check if there is at least one agent otherwise put one in the middle
    if !(agent_in_place)
        x_agent = Int(round(width/2))
        y_agent = Int(round(height/2))
        lvl_vec = Vector{Char}(lvl_str)
        lvl_vec[x_agent+(y_agent-1)*5] = 'A'
        lvl_str = String(lvl_vec)
    end
    cgp_lvl.output_map[1] = lvl_str
    return lvl_str
end

import Cambrian.populate, Cambrian.evaluate

mutable struct CGPSokoLvlEvolution <: Cambrian.AbstractEvolution
    config::NamedTuple
    logger::CambrianLogger
    population::Array{CGPSokoLvl}
    fitness::Function
    gen::Int
end

function max_selection(pop::Array{CGPSokoLvl,1})
    sort(pop)[end]
end

function oneplus_populate(e::CGPSokoLvlEvolution)
    p1 = max_selection(e.population)
    e.population[1] = p1
    for i in 2:e.config.n_population
        e.population[i] = mutate(p1)
    end
end

function nelitesplus_populate(e::CGPSokoLvlEvolution)
    n_elites = e.config.n_elite
    elites = sort(e.population)[end-(n_elites-1):end]
    e.population[1:n_elites] = elites
    for i in (n_elites+1):e.config.n_population
        r = rand(1:n_elites)
        e.population[i] = mutate(elites[r])
    end
end

function populate(e::CGPSokoLvlEvolution)
     # oneplus_populate(e)
     nelitesplus_populate(e)
end

function evaluate(e::CGPSokoLvlEvolution)
    for i in eachindex(e.population)
        e.population[i].cgp.fitness[:] = e.fitness(e.population[i])
    end
end

function CGPSokoLvlEvolution(cfg::NamedTuple, fitness::Function;
                      logfile=string("logs/", cfg.id, ".csv"))
    logger = CambrianLogger(logfile)
    population = Cambrian.initialize(CGPSokoLvl, cfg)
    CGPSokoLvlEvolution(cfg, logger, population, fitness, 0)
end
