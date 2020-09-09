#--------------------Set-up for tests--------------------
cfg = Cambrian.get_config("tests/test_agent.yaml")

width = cfg.width
height = cfg.height
nb_object = cfg.nb_object

image_path = joinpath(@__DIR__,"..","resources","images")
shader_path = joinpath(@__DIR__,"..","resources","shaders")
gdy_path = joinpath(@__DIR__,"..","resources","games")
gdy_reader = Griddly.GDYReader(image_path,shader_path)

grid = Griddly.load!(gdy_reader,joinpath(gdy_path,"Single-Player/GVGAI/sokoban.yaml"))
game = Griddly.create_game(grid,Griddly.VECTOR)
player1 = Griddly.register_player!(game,"Tux", Griddly.BLOCK_2D)
Griddly.init!(game)

#----------Basic Test on our SokoAgent and its function----------
# test the construction of our SokoAgent from config
@testset "SokoAgent" begin
    model = Chain(
    Conv((3,3),4=>16,pad=(1,1),relu),
    MaxPool((2,2)),
    flatten,
    Dense(64,4)
    )
    nb_params = get_params_count(model)
    sokoagent = SokoAgent(model, cfg)
    @test sokoagent.width == 5
    @test sokoagent.height == 5
    @test sokoagent.nb_object == 4
    @test length(sokoagent.genes) == nb_params
    # test the transcript function
    weight_genes = rand(length(sokoagent.genes))
    sokoagent = SokoAgent(weight_genes,model,cfg)
    transcript_sokoagent_genes!(sokoagent)
    lvl_str = """
    wwwww
    wh..w
    wbA.w
    w...w
    wwwww
    """
    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)
    observation = Griddly.observe(game)
    observation = convert(Array{Int8,3},Griddly.get_data(observation))
    total_reward = 0
    for i in 1:10
        dir = choose_action(observation,sokoagent)
        println(dir)
        reward, done = Griddly.step_player!(player1,"move", [dir])
        total_reward += reward
        observation = convert(Array{Int8,3},Griddly.get_data(Griddly.observe(game)))
        println(observation)
    end
    # @test fitness(sokoagent,10) == total_reward
end
