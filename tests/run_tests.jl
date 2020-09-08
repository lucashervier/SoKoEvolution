using Griddly
using Cambrian
using Test
include("../src/sokoban_level_individual.jl")

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

@testset "Random fitness" begin
    e = Evolution{SokoLvlIndividual}(cfg;logfile=string("../logs/"," cfg.id", ".csv"))

    evaluate(e::Evolution) = random_evaluate(e)
    populate(e::Evolution) = oneplus_populate(e)
    for ind in e.population
        apply_sokolvl_constraint!(ind)
    end

    @test length(e.population) == cfg.n_population
    for i in e.population
        @test all(i.fitness .== -Inf)
    end

    evaluate(e)
    fits = [i.fitness[1] for i in e.population]
    evaluate(e)
    # random fitness, all values should change
    for i in eachindex(e.population)
        @test fits[i] != e.population[i].fitness[1]
    end

    fits = copy([i.fitness[:] for i in e.population])
    step!(e)
    # random fitness, all values should change
    for i in eachindex(e.population)
        @test fits[i] != e.population[i].fitness[1]
    end
end
