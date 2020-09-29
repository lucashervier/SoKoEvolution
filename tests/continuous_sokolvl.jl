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
    bbbb
    bbbb
    bbbb
    """
    lvl_str = write_map(ind)
    @test lvl_str == expected_str
end
