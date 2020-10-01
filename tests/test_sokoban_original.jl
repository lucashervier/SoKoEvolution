using Griddly

image_path = joinpath(@__DIR__,"..","resources","images")
shader_path = joinpath(@__DIR__,"..","resources","shaders")
gdy_path = joinpath(@__DIR__,"..","resources","games")
gdy_reader = Griddly.GDYReader(image_path,shader_path)

grid = Griddly.load!(gdy_reader,joinpath(gdy_path,"Single-Player/GVGAI/sokoban3.yaml"))
game = Griddly.create_game(grid,Griddly.SPRITE_2D)
player1 = Griddly.register_player!(game,"Tux", Griddly.BLOCK_2D)
Griddly.init!(game)

lvl_str = """
wwwww
w...w
wh.hw
wh.bw
wb.Aw
w...w
wwwww
"""
total_reward = 0

Griddly.load_level_string!(grid,lvl_str)
Griddly.reset!(game)
render_window = RenderWindow(700,700)

reward, done = Griddly.step_player!(player1,"move", [4]);
sprite = Griddly.observe(game);
sprite = Griddly.get_data(sprite);
render(render_window,sprite;nice_render=true)
sleep(1)
total_reward += reward
println(total_reward)
reward, done = Griddly.step_player!(player1,"move", [1]);
sprite = Griddly.observe(game);
sprite = Griddly.get_data(sprite);
render(render_window,sprite;nice_render=true)
sleep(1)
total_reward += reward
println(total_reward)
reward, done = Griddly.step_player!(player1,"move", [1]);
sprite = Griddly.observe(game);
sprite = Griddly.get_data(sprite);
render(render_window,sprite;nice_render=true)
sleep(1)
total_reward += reward
println(total_reward)
reward, done = Griddly.step_player!(player1,"move", [2]);
sprite = Griddly.observe(game);
sprite = Griddly.get_data(sprite);
render(render_window,sprite;nice_render=true)
sleep(1)
total_reward += reward
println(total_reward)
reward, done = Griddly.step_player!(player1,"move", [2]);
sprite = Griddly.observe(game);
sprite = Griddly.get_data(sprite);
render(render_window,sprite;nice_render=true)
sleep(1)
total_reward += reward
println(total_reward)
reward, done = Griddly.step_player!(player1,"move", [2]);
sprite = Griddly.observe(game);
sprite = Griddly.get_data(sprite);
render(render_window,sprite;nice_render=true)
sleep(1)
total_reward += reward
println(total_reward)
reward, done = Griddly.step_player!(player1,"move", [3]);
sprite = Griddly.observe(game);
sprite = Griddly.get_data(sprite);
render(render_window,sprite;nice_render=true)
sleep(1)
total_reward += reward
println(total_reward)
