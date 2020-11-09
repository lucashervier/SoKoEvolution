#--------------------Set-up for tests--------------------
cfg = Cambrian.get_config("tests/test_cgp.yaml")

width = cfg.width
height = cfg.height
agent_idx = cfg.agent_idx
objects_char_list = cfg.objects_char_list
nb_object = length(objects_char_list)

mutate(i::CGPSokoLvl) = CGPSokoLvl(i.width,i.height,i.objects_char_list,i.agent_idx,i.output_map,goldman_mutate(cfg,i.cgp))

@testset "CGPSokoLvl" begin
    cgp_lvl = CGPSokoLvl(cfg)
    @test cgp_lvl.width == width
    @test cgp_lvl.height == height
    @test cgp_lvl.agent_idx == agent_idx
    @test cgp_lvl.objects_char_list == objects_char_list
    @test typeof(cgp_lvl.cgp) == CartesianGeneticProgramming.CGPInd

    # expected_output_map = """
    # bbbb
    # bAbb
    # bbbb
    # bbbb
    # """

    lvl_str = write_map!(cgp_lvl)
    # @test lvl_str == expected_output_map
    # @test cgp_lvl.output_map[1] == expected_output_map
    @test lvl_str == cgp_lvl.output_map[1]
    println(lvl_str)

    new_cgp = mutate(cgp_lvl)
    @test new_cgp.cgp.chromosome != cgp_lvl.cgp.chromosome
    cgp_lvl.cgp.fitness[1] = 0
    save_ind(cgp_lvl,"tests//save_cgp.dna")
    load_str = read("tests//save_cgp.dna", String)
    load_ind = CGPSokoLvl(cfg,load_str)
    @test load_ind.cgp.genes == cgp_lvl.cgp.genes
    @test load_ind.width == cgp_lvl.width
    @test load_ind.height == cgp_lvl.height
    @test load_ind.objects_char_list == cgp_lvl.objects_char_list
    @test load_ind.output_map == cgp_lvl.output_map
end
