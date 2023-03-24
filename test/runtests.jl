using Test
using Random
using Statistics
using JuliaNN

relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(a)), maximum(abs.(b)), 1e-12)

@testset "JuliaNN" begin
    @testset "dense forward and shapes" begin
        rng = MersenneTwister(1)
        d = Dense(3 => 2; rng = rng)
        x = rand(rng, 3, 5)
        y = forward(d, x)
        @test size(y) == (2, 5)
        @test y ≈ d.W * x .+ d.b
        dx = backward(d, ones(2, 5))
        @test size(dx) == (3, 5)
        @test d.db ≈ fill(5.0, 2)
        @test d.dW ≈ ones(2, 5) * transpose(x)
    end

    @testset "activations" begin
        a = Activation(:relu)
        @test forward(a, [-1.0 2.0; 0.5 -3.0]) == [0.0 2.0; 0.5 0.0]
        @test backward(a, ones(2, 2)) == [0.0 1.0; 1.0 0.0]
        s = Activation(:sigmoid)
        @test forward(s, zeros(1, 1)) == fill(0.5, 1, 1)
        @test backward(s, ones(1, 1)) ≈ fill(0.25, 1, 1)
        p = softmax([1.0 3.0; 2.0 3.0; 3.0 3.0])
        @test all(sum(p, dims = 1) .≈ 1.0)
        @test p[:, 2] ≈ fill(1 / 3, 3)
        @test p[3, 1] > p[2, 1] > p[1, 1]
        @test softmax([1000.0; 1000.0;;]) ≈ [0.5; 0.5;;]  # no overflow
        @test_throws ArgumentError Activation(:nope)
    end

    @testset "losses" begin
        y = onehot([1, 3], 3)
        @test y == [1.0 0.0; 0.0 0.0; 0.0 1.0]
        ŷ = [0.7 0.1; 0.2 0.2; 0.1 0.7]
        @test loss(CrossEntropy(), ŷ, y) ≈ -(log(0.7) + log(0.7)) / 2
        z = [2.0 -1.0; 0.0 0.0; -1.0 2.0]
        @test loss(SoftmaxCrossEntropy(), z, y) ≈ loss(CrossEntropy(), softmax(z), y)
        @test gradient(SoftmaxCrossEntropy(), z, y) ≈ (softmax(z) .- y) ./ 2
        @test loss(MSE(), [1.0 2.0], [1.0 4.0]) ≈ 1.0
    end

    @testset "backward matches numerical gradients" begin
        rng = MersenneTwister(7)
        X = randn(rng, 4, 6)
        for (model, lossfn, Y) in [
            (Sequential(Dense(4 => 5; rng = rng), Activation(:tanh), Dense(5 => 3; rng = rng)), MSE(), randn(rng, 3, 6)),
            (Sequential(Dense(4 => 5; rng = rng), Activation(:relu), Dense(5 => 3; rng = rng), Activation(:sigmoid)), MSE(), rand(rng, 3, 6)),
            (Sequential(Dense(4 => 6; rng = rng), Activation(:sigmoid), Dense(6 => 3; rng = rng)), SoftmaxCrossEntropy(), onehot(rand(rng, 1:3, 6), 3)),
            (Sequential(Dense(4 => 6; rng = rng), Activation(:tanh), Dense(6 => 3; rng = rng), Activation(:softmax)), CrossEntropy(), onehot(rand(rng, 1:3, 6), 3)),
        ]
            ŷ = forward(model, X)
            backward(model, gradient(lossfn, ŷ, Y))
            analytic = [copy(g) for (_, g) in parameters(model)]
            numeric = numerical_gradient(model, lossfn, X, Y)
            for (a, n) in zip(analytic, numeric)
                @test relerr(reshape(a, size(n)), n) < 1e-6
            end
        end
    end

    @testset "XOR with sigmoid + MSE + Adam" begin
        rng = MersenneTwister(3)
        X = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]
        Y = [0.0 1.0 1.0 0.0]
        model = Sequential(Dense(2 => 8; rng = rng), Activation(:tanh), Dense(8 => 1; rng = rng), Activation(:sigmoid))
        history = train!(model, MSE(), Adam(0.05), X, Y; epochs = 500, batchsize = 4, rng = rng)
        @test history[end] < history[1]
        @test history[end] < 0.01
        @test round.(Int, predict(model, X)) == [0 1 1 0]
    end

    @testset "XOR with momentum SGD" begin
        rng = MersenneTwister(4)
        X = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]
        Y = [0.0 1.0 1.0 0.0]
        model = Sequential(Dense(2 => 8; rng = rng), Activation(:tanh), Dense(8 => 1; rng = rng), Activation(:sigmoid))
        train!(model, MSE(), SGD(0.5; momentum = 0.9), X, Y; epochs = 800, batchsize = 4, rng = rng)
        @test round.(Int, predict(model, X)) == [0 1 1 0]
    end

    @testset "spiral classification" begin
        rng = MersenneTwister(11)
        X, labels = spiral_data(120, 3; rng = rng)
        Y = onehot(labels, 3)
        model = Sequential(Dense(2 => 32; rng = rng), Activation(:relu), Dense(32 => 32; rng = rng), Activation(:relu), Dense(32 => 3; rng = rng))
        history = train!(model, SoftmaxCrossEntropy(), Adam(0.01), X, Y; epochs = 200, batchsize = 32, rng = rng)
        @test history[end] < history[1] / 3
        @test accuracy(model, X, Y) > 0.9
        @test length(classify(model, X)) == 360
        probs = softmax(predict(model, X))
        @test all(isapprox.(sum(probs, dims = 1), 1.0))
    end

    @testset "regression of a sine wave" begin
        rng = MersenneTwister(5)
        X = reshape(range(-π, π; length = 200), 1, :)
        Y = sin.(X)
        model = Sequential(Dense(1 => 16; rng = rng), Activation(:tanh), Dense(16 => 16; rng = rng), Activation(:tanh), Dense(16 => 1; rng = rng))
        train!(model, MSE(), Adam(0.01), X, Y; epochs = 300, batchsize = 20, rng = rng)
        @test mean((predict(model, X) .- Y) .^ 2) < 0.01
    end

    @testset "save and load" begin
        rng = MersenneTwister(2)
        model = Sequential(Dense(3 => 4; rng = rng), Activation(:relu), Dense(4 => 2; rng = rng))
        X = randn(rng, 3, 5)
        path = tempname()
        save_model(path, model)
        loaded = load_model(path)
        @test predict(loaded, X) ≈ predict(model, X)
        rm(path)
    end

    @testset "errors" begin
        model = Sequential(Dense(2 => 1))
        @test_throws DimensionMismatch train!(model, MSE(), SGD(), zeros(2, 3), zeros(1, 4))
    end
end
