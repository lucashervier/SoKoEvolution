#--------------------Set-up for tests--------------------
cfg = Cambrian.get_config("tests/test.yaml")

width = cfg.width
height = cfg.height
agent_idx = cfg.agent_idx
objects_char_list = cfg.objects_char_list
nb_object = length(objects_char_list)

model = Chain(
Dense(2,16),
Dense(16,5)
)

mutate(i::ContinuousSokoLvl) = mutate(i, cfg.m_rate)
#----------Basic Test on our ContinuousSokoLvl and its function----------#
@testset "ContinousSokoLvl" begin
    # from cfg with random genes
    ind = ContinuousSokoLvl(model,cfg)
    @test ind.width == 4
    @test ind.height == 4
    @test ind.objects_char_list == ["b","w","h","A"]
    @test ind.agent_idx == 4
    # from config with a chosen genes
    genes = [0.5 for i in 1:133]
    ind = ContinuousSokoLvl(genes, model, cfg)
    @test ind.width == 4
    @test ind.height == 4
    @test ind.objects_char_list == ["b","w","h","A"]
    @test ind.agent_idx == 4
    # test the apply function
    apply_continuoussokolvl_genes!(ind)
    @test ind.model([1,1]) == [12.5, 12.5, 12.5, 12.5, 12.5]
    # test the map string
    expected_str = """
    bbbb
    bAbb
    bbbb
    bbbb
    """
    lvl_str = write_map!(ind)
    @test lvl_str == expected_str
    @test ind.output_map[1] == expected_str
    new_ind = mutate(ind)
    @test new_ind.genes != genes
    ind.fitness[1] = 0
    save_ind(ind,"tests//save_continuous.dna")
    load_str = read("tests//save_continuous.dna", String)
    load_ind = ContinuousSokoLvl(load_str,model)
    @test load_ind.genes == ind.genes
    @test load_ind.width == ind.width
    @test load_ind.height == ind.height
    @test load_ind.objects_char_list == ind.objects_char_list
    @test load_ind.output_map == ind.output_map
end
