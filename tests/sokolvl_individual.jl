#--------------------Set-up for tests--------------------
cfg = Cambrian.get_config("tests/test.yaml")

width = cfg.width
height = cfg.height
agent_idx = cfg.agent_idx
objects_char_list = cfg.objects_char_list
nb_object = length(objects_char_list)

image_path = joinpath(@__DIR__,"..","resources","images")
shader_path = joinpath(@__DIR__,"..","resources","shaders")
gdy_path = joinpath(@__DIR__,"..","resources","games")
gdy_reader = Griddly.GDYReader(image_path,shader_path)

grid = Griddly.load!(gdy_reader,joinpath(gdy_path,"Single-Player/GVGAI/sokoban.yaml"))
game = Griddly.create_game(grid,Griddly.VECTOR)
player1 = Griddly.register_player!(game,"Tux", Griddly.BLOCK_2D)
Griddly.init!(game)

#----------Basic Test on our SokoLvlIndividual and its function----------
@testset "SokoLvlIndividual" begin
    # test the construction of our SokoLvlIndividual from config
    ind = SokoLvlIndividual(cfg)
    @test ind.width == 4
    @test ind.height == 4
    @test ind.objects_char_list == ["b","w","h","A"]
    @test length(ind.genes) == ind.width*ind.height*length(ind.objects_char_list)
    @test ind.agent_idx == 4
    # test the transcript function
    level_genes = BitArray([0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0 0 1 1 0 0 1 1 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0])
    sokolvl_ind = SokoLvlIndividual(level_genes,cfg)
    lvl_str = transcript_sokolvl_genes(sokolvl_ind)
    @test lvl_str == """
    wwww
    wh.w
    wbAw
    wwww
    """
    # test if we can load this level
    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)
    observation = Griddly.observe(game)
    observation = convert(Array{Int8,3},Griddly.get_data(observation))
    println(observation)
    println(observation[1])
    # test that after constraint are applied we get exactly 1 agent
    multiple_agent_genes = BitArray([0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0 0 1 1 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0])
    multiple_agent_ind = SokoLvlIndividual(multiple_agent_genes,cfg)
    apply_sokolvl_constraint!(multiple_agent_ind)
    nb_agent = sum(multiple_agent_ind.genes[49:64])
    @test nb_agent == 1
    # test that after constraint we do not get two object at the same position
    redundant_tiles_genes = BitArray([1 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0 0 1 1 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0])
    redundant_tiles_ind = SokoLvlIndividual(redundant_tiles_genes,cfg)
    apply_sokolvl_constraint!(redundant_tiles_ind)
    for pos in 1:width*height
        nb_object_in_pos = 0
        for i in 1:nb_object
            if redundant_tiles_ind.genes[pos + height*width*(i-1)] == 1
                nb_object_in_pos += 1
            end
        end
        @test nb_object_in_pos <= 1
    end
end

#-----------------Check if it is well integrated with Cambrian----------------
# mutate must be overriden in the global scope (or use eval)
mutate(i::SokoLvlIndividual) = mutate(i, cfg.m_rate)

function count_box(ind::Individual)
    width = ind.width
    height = ind.height
    start_box_idx = (1-1)*width*height+1
    stop_box_idx = 1*width*height
    nb_box = sum(ind.genes[start_box_idx:stop_box_idx])
    return [nb_box]
end

function modified_one_plus(e::AbstractEvolution)
    p1 = sort(e.population)[end]
    e.population[1] = p1
    for i in 2:e.config.n_population
        e.population[i] = mutate(p1)
        apply_sokolvl_constraint!(e.population[i])
    end
end
populate(e::OnePlusEvo) = modified_one_plus(e)
function test_one_plus_evo()
    e = OnePlusEvo{SokoLvlIndividual}(cfg,count_box;logfile=string("../logs/","test_logs", ".csv"))

    @test length(e.population) == cfg.n_population

    for ind in e.population
        apply_sokolvl_constraint!(ind)
    end

    evaluate(e)
    fits = [i.fitness[1] for i in e.population]
    evaluate(e)
    # no values should change between two evaluation
    for i in eachindex(e.population)
        @test fits[i] == e.population[i].fitness[1]
    end

    run!(e)

    best = sort(e.population)[end]
    println("Final fitness: ", best.fitness[1])
    println(best.genes)
    println(length(best.genes))
    lvl_str = transcript_sokolvl_genes(best)
    println(lvl_str)
    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)
    observation = Griddly.observe(game)
    observation = convert(Array{Int8,3},Griddly.get_data(observation))
    println(observation)
end
@testset "Nb box fitness" begin
    test_one_plus_evo()
end
