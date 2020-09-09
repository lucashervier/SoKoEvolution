@testset "Utils on Flux function" begin
    # a NN with two fully connected layer
    m = Chain(Dense(4,4),Dense(4,2))
    nb_params = 4*4 + 4 + 4*2 + 2
    @test get_params_count(m) == nb_params
    input = ones(4)
    # we want to set all the weights to one
    weights = [1.0 for i in 1:nb_params]
    expected_result = [21.0, 21.0]
    load_weights_from_array!(m,weights)
    @test m(input) == expected_result

    # now we try with a more sophisticated neural network
    m2 = Chain(
    # a 3*3 filter for a 28*28 image
    Conv((3,3), 1=>16,pad=(1,1), relu),
    MaxPool((2,2)),
    # at this point we get a 14*14 img on 16 channels
    flatten,
    Dense(16*14*14,10)
    )
    nb_params_conv = (3*3*1+1)*16
    nb_params_dense = 16*14*14*10 + 10
    nb_params = nb_params_conv + nb_params_dense
    @test get_params_count(m2) == nb_params
    # we want to set all the weights to 1
    weights = [1 for i in 1:nb_params]
    img = ones(28,28)
    img = reshape(img,(28,28,1,1))

    out_value_from_conv = 10 # for each cell
    out_value_from_dense = out_value_from_conv*16*14*14 + 1
    load_weights_from_array!(m2,weights)
    for i in 1:10
        @test m2(img)[i] == out_value_from_dense
    end
end
