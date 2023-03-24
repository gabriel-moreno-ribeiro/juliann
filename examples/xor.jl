# The smallest possible non-linear problem.  Run with:  julia --project examples/xor.jl
using JuliaNN
using Random

rng = MersenneTwister(1)
X = [0.0 0.0 1.0 1.0;
     0.0 1.0 0.0 1.0]
Y = [0.0 1.0 1.0 0.0]

model = Sequential(Dense(2 => 8; rng = rng), Activation(:tanh), Dense(8 => 1; rng = rng), Activation(:sigmoid))
history = train!(model, MSE(), Adam(0.05), X, Y; epochs = 500, batchsize = 4, rng = rng)

println("loss: ", round(history[1]; digits = 4), " -> ", round(history[end]; digits = 6))
for j in 1:4
    println(Int.(X[:, j]), " => ", round(predict(model, X[:, j:j])[1]; digits = 3))
end
