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
game = Griddly.create_game(grid,Griddly.SPRITE_2D)
player1 = Griddly.register_player!(game,"Tux", Griddly.BLOCK_2D)
Griddly.init!(game)

model = Chain(
Conv((3,3),4=>16,pad=(1,1),relu),
MaxPool((2,2)),
flatten,
Dense(64,4)
)
#----------Basic Test on our SokoAgent and its function----------
# test the construction of our SokoAgent from config
@testset "SokoAgent" begin
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
    observation = convert(Array{Int,3},Griddly.get_data(observation))
    vector_obs = Griddly.vector_obs(grid)
    vector_obs = convert(Array{Int8,3},Griddly.get_data(vector_obs))
    total_reward = 0
    for i in 1:10
        dir = choose_action(vector_obs,sokoagent)
        println(dir)
        reward, done = Griddly.step_player!(player1,"move", [dir])
        total_reward += reward
        observation = convert(Array{Int,3},Griddly.get_data(Griddly.observe(game)))
        vector_obs = convert(Array{Int8,3},Griddly.get_data(Griddly.vector_obs(grid)))
        println(vector_obs)
    end
end

# test if we can load an agent from file
file_model = Chain(
                    Dense(2,4)
                    )
file_path = "tests//test_agent_file.dna"
@testset "SokoAgent From File" begin
    indString = read("$file_path", String)
    agent_file = SokoAgent(indString,file_model)
    @test agent_file.genes == [0.0, 1.0, 0.0, 0.5, 0.7, 0.4, 0.3, 0.0]
    @test agent_file.fitness == [100]
    @test agent_file.width == 8
    @test agent_file.height == 8
    @test agent_file.nb_object == 4
end
#-----------------Check if it is well integrated with Cambrian----------------
# mutate must be overriden in the global scope (or use eval)
mutate(i::SokoAgent) = mutate(i, cfg.m_rate)

"create all members of the first generation"
function initialize(itype::Type, model, cfg::NamedTuple)
    population = Array{itype}(undef, cfg.n_population)
    for i in 1:cfg.n_population
        population[i] = itype(model,cfg)
    end
    population
end

function OnePlusEvo{T}(model, cfg::NamedTuple, fitness::Function;
                      logfile=string("logs/", cfg.id, ".csv")) where T
    logger = CambrianLogger(logfile)
    population = initialize(T, model, cfg)
    OnePlusEvo(cfg, logger, population, fitness, 0)
end

function fitness_agent(ind::Individual)
    lvl_str = """
    wwwww
    wh..w
    wbA.w
    w...w
    wwwww
    """
    Griddly.load_level_string!(grid,lvl_str)
    Griddly.reset!(game)
    total_reward = 0
    observation = convert(Array{Int8,3},Griddly.get_data(Griddly.observe(game)))
    for i in 1:100
        dir = choose_action(observation,ind)
        println("action:$dir")
        reward, done = Griddly.step_player!(player1,"move", [dir])
        total_reward += reward
        observation = convert(Array{Int8,3},Griddly.get_data(Griddly.observe(game)))
    end
    return [total_reward]
end

function agent_one_plus(e::AbstractEvolution)
    p1 = sort(e.population)[end]
    e.population[1] = p1
    for i in 2:e.config.n_population
        e.population[i] = mutate(p1)
        transcript_sokoagent_genes!(e.population[i])
    end
end

populate(e::OnePlusEvo) = agent_one_plus(e)

function test_one_plus_evo_agent()
    e = OnePlusEvo{SokoAgent}(model,cfg,fitness_agent;logfile=string("../logs/","test_logs_agent", ".csv"))

    @test length(e.population) == cfg.n_population

    for ind in e.population
        transcript_sokoagent_genes!(ind)
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
end
# @testset "OnePlus evo with a basic fitness" begin
#     test_one_plus_evo_agent()
# end
