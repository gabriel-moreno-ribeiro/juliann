# Trains a classifier on the three-arm spiral dataset and draws the decision
# boundary as ASCII art.  Run with:  julia --project examples/spiral.jl
using JuliaNN
using Random

rng = MersenneTwister(42)
X, labels = spiral_data(150, 3; rng = rng)
Y = onehot(labels, 3)

model = Sequential(
    Dense(2 => 32; rng = rng), Activation(:relu),
    Dense(32 => 32; rng = rng), Activation(:relu),
    Dense(32 => 3; rng = rng),
)
history = train!(model, SoftmaxCrossEntropy(), Adam(0.01), X, Y; epochs = 300, batchsize = 32, rng = rng, verbose = true)
println("training accuracy: ", round(accuracy(model, X, Y) * 100; digits = 1), "%")

# decision boundary: classify a grid of points and print one character per cell
chars = ('.', '+', '#')
for row in 20:-1:-20
    line = IOBuffer()
    for col in -40:40
        x = col / 40
        y = row / 20
        c = classify(model, reshape([x, y], 2, 1))[1]
        print(line, chars[c])
    end
    println(String(take!(line)))
end
