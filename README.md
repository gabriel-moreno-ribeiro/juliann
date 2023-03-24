# JuliaNN

A neural network library written from scratch in Julia: layers with
hand-written backpropagation, losses, SGD with momentum and Adam, a
mini-batch training loop, and gradient checking to prove the derivatives
are right. Only the standard library is used (no Flux, no autodiff).

```julia
using JuliaNN, Random

rng = MersenneTwister(42)
X, labels = spiral_data(150, 3; rng = rng)      # 2 x 450 points, three spiral arms
Y = onehot(labels, 3)

model = Sequential(
    Dense(2 => 32; rng = rng), Activation(:relu),
    Dense(32 => 32; rng = rng), Activation(:relu),
    Dense(32 => 3; rng = rng),
)
train!(model, SoftmaxCrossEntropy(), Adam(0.01), X, Y; epochs = 300, batchsize = 32, rng = rng)
accuracy(model, X, Y)                            # ~0.99
classify(model, [0.3; -0.5;;])                   # predicted class of one point
```

```sh
julia --project examples/xor.jl
julia --project examples/spiral.jl               # prints the decision boundary as ASCII art
```

## What is inside

| Piece | Details |
| --- | --- |
| `Dense(in => out)` | `y = W x + b`, He initialisation, caches `x` for the backward pass |
| `Activation(:relu \| :sigmoid \| :tanh \| :softmax)` | element-wise, with exact derivatives (softmax uses the full Jacobian) |
| `Sequential(layers...)` | forward in order, backward in reverse |
| `MSE()`, `CrossEntropy()`, `SoftmaxCrossEntropy()` | scalar loss and its gradient with respect to the prediction; the fused version works on logits and is numerically stable |
| `SGD(lr; momentum)`, `Adam(lr)` | optimisers keyed per parameter array |
| `train!` | shuffles, batches, forward, loss, backward, update; returns the loss history |
| `numerical_gradient` | central finite differences over every weight, used by the tests |
| `spiral_data`, `onehot`, `accuracy`, `classify`, `save_model` / `load_model` | helpers |

Data is laid out as `features x batch`, so a layer's weight is `out x in` and
a batch is one matrix multiplication.

## How backpropagation is implemented

Every layer stores what it needs during `forward` and, given the gradient of
the loss with respect to its output, returns the gradient with respect to its
input while recording the gradients of its own parameters:

```julia
forward(l::Dense, x)   = (l.x = x; l.W * x .+ l.b)
backward(l::Dense, dy) = (l.dW = dy * x'; l.db = sum(dy, dims = 2); l.W' * dy)
```

`Sequential.backward` just chains these from the last layer to the first.
The test-suite compares every analytic gradient with finite differences on
four different architectures and losses, to a relative error below 1e-6.

## Tests

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

## License

MIT
