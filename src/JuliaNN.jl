"""
JuliaNN - a neural network library written from scratch in Julia.

Layers cache what they need during `forward` and implement `backward` by
hand (no automatic differentiation), so the whole chain rule is visible:
Dense, activations (relu, sigmoid, tanh, softmax), losses (MSE,
cross-entropy, softmax + cross-entropy on logits), optimizers (SGD with
momentum, Adam) and a mini-batch training loop.

Data layout: every matrix is `features x batch`, so a batch of 64 inputs
with 2 features is a 2x64 matrix and a Dense(2 => 8) layer has an 8x2 weight.
"""
module JuliaNN

using LinearAlgebra
using Random
using Statistics
using Serialization

export Layer, Dense, Activation, Sequential, forward, backward, parameters
export MSE, CrossEntropy, SoftmaxCrossEntropy, loss, gradient
export SGD, Adam, update!, train!, predict, accuracy, classify
export softmax, onehot, spiral_data, numerical_gradient, save_model, load_model

# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------

abstract type Layer end

"""Fully connected layer: y = W x + b, with He initialisation."""
mutable struct Dense <: Layer
    W::Matrix{Float64}
    b::Vector{Float64}
    x::Matrix{Float64}   # cached input for backward
    dW::Matrix{Float64}
    db::Vector{Float64}
end

function Dense(pair::Pair{Int,Int}; rng::AbstractRNG = Random.default_rng())
    nin, nout = pair
    W = randn(rng, nout, nin) .* sqrt(2.0 / nin)
    Dense(W, zeros(nout), zeros(nin, 0), zeros(nout, nin), zeros(nout))
end

function forward(l::Dense, x::AbstractMatrix)
    l.x = Matrix{Float64}(x)
    return l.W * l.x .+ l.b
end

function backward(l::Dense, dy::AbstractMatrix)
    l.dW = dy * transpose(l.x)
    l.db = vec(sum(dy, dims = 2))
    return transpose(l.W) * dy
end

"""Element-wise non-linearity; `kind` is :relu, :sigmoid, :tanh or :softmax."""
mutable struct Activation <: Layer
    kind::Symbol
    y::Matrix{Float64}
    function Activation(kind::Symbol)
        kind in (:relu, :sigmoid, :tanh, :softmax) || throw(ArgumentError("unknown activation $kind"))
        new(kind, zeros(0, 0))
    end
end

sigmoid(x) = 1.0 / (1.0 + exp(-x))

"""Column-wise softmax, computed stably by subtracting the column maximum."""
function softmax(z::AbstractMatrix)
    shifted = z .- maximum(z, dims = 1)
    e = exp.(shifted)
    return e ./ sum(e, dims = 1)
end

function forward(l::Activation, x::AbstractMatrix)
    l.y = if l.kind == :relu
        max.(x, 0.0)
    elseif l.kind == :sigmoid
        sigmoid.(x)
    elseif l.kind == :tanh
        tanh.(x)
    else
        softmax(x)
    end
    return l.y
end

function backward(l::Activation, dy::AbstractMatrix)
    y = l.y
    if l.kind == :relu
        return dy .* (y .> 0)
    elseif l.kind == :sigmoid
        return dy .* y .* (1 .- y)
    elseif l.kind == :tanh
        return dy .* (1 .- y .^ 2)
    else
        # full softmax Jacobian applied column by column: dx = y .* (dy - sum(dy .* y))
        return y .* (dy .- sum(dy .* y, dims = 1))
    end
end

"""A stack of layers applied in order."""
struct Sequential <: Layer
    layers::Vector{Layer}
end
Sequential(layers::Layer...) = Sequential(collect(Layer, layers))

function forward(m::Sequential, x::AbstractMatrix)
    for l in m.layers
        x = forward(l, x)
    end
    return x
end

function backward(m::Sequential, dy::AbstractMatrix)
    for l in Iterators.reverse(m.layers)
        dy = backward(l, dy)
    end
    return dy
end

"""All (parameter, gradient) pairs of a model, in a stable order."""
parameters(l::Dense) = [(l.W, l.dW), (l.b, l.db)]
parameters(::Activation) = Tuple{AbstractArray,AbstractArray}[]
parameters(m::Sequential) = reduce(vcat, [parameters(l) for l in m.layers]; init = Tuple{AbstractArray,AbstractArray}[])

(m::Sequential)(x) = forward(m, x)

# ---------------------------------------------------------------------------
# losses: `loss(L, ŷ, y)` is the scalar, `gradient(L, ŷ, y)` is ∂loss/∂ŷ
# ---------------------------------------------------------------------------

abstract type Loss end

"""Mean squared error, averaged over the batch: sum((ŷ-y)^2) / (2 batch)."""
struct MSE <: Loss end
loss(::MSE, ŷ, y) = sum((ŷ .- y) .^ 2) / (2 * size(y, 2))
gradient(::MSE, ŷ, y) = (ŷ .- y) ./ size(y, 2)

"""Cross-entropy on probabilities (use after a :softmax activation)."""
struct CrossEntropy <: Loss end
const ϵ = 1e-12
loss(::CrossEntropy, ŷ, y) = -sum(y .* log.(ŷ .+ ϵ)) / size(y, 2)
gradient(::CrossEntropy, ŷ, y) = -(y ./ (ŷ .+ ϵ)) ./ size(y, 2)

"""Softmax and cross-entropy fused on raw logits: numerically stable and the gradient is simply (softmax(z) - y)."""
struct SoftmaxCrossEntropy <: Loss end
function loss(::SoftmaxCrossEntropy, z, y)
    logp = z .- maximum(z, dims = 1)
    logp = logp .- log.(sum(exp.(logp), dims = 1))
    return -sum(y .* logp) / size(y, 2)
end
gradient(::SoftmaxCrossEntropy, z, y) = (softmax(z) .- y) ./ size(y, 2)

# ---------------------------------------------------------------------------
# optimizers
# ---------------------------------------------------------------------------

abstract type Optimizer end

"""Stochastic gradient descent with optional momentum."""
mutable struct SGD <: Optimizer
    lr::Float64
    momentum::Float64
    velocity::IdDict{Any,Any}
end
SGD(lr = 0.01; momentum = 0.0) = SGD(lr, momentum, IdDict())

function update!(opt::SGD, model::Layer)
    for (p, g) in parameters(model)
        if opt.momentum == 0
            p .-= opt.lr .* g
        else
            v = get!(opt.velocity, p, zeros(size(p)))
            v .= opt.momentum .* v .- opt.lr .* g
            p .+= v
        end
    end
end

"""Adam (Kingma & Ba): per-parameter adaptive learning rates with bias correction."""
mutable struct Adam <: Optimizer
    lr::Float64
    β1::Float64
    β2::Float64
    eps::Float64
    t::Int
    m::IdDict{Any,Any}
    v::IdDict{Any,Any}
end
Adam(lr = 0.001; β1 = 0.9, β2 = 0.999, eps = 1e-8) = Adam(lr, β1, β2, eps, 0, IdDict(), IdDict())

function update!(opt::Adam, model::Layer)
    opt.t += 1
    for (p, g) in parameters(model)
        m = get!(opt.m, p, zeros(size(p)))
        v = get!(opt.v, p, zeros(size(p)))
        m .= opt.β1 .* m .+ (1 - opt.β1) .* g
        v .= opt.β2 .* v .+ (1 - opt.β2) .* g .^ 2
        m̂ = m ./ (1 - opt.β1^opt.t)
        v̂ = v ./ (1 - opt.β2^opt.t)
        p .-= opt.lr .* m̂ ./ (sqrt.(v̂) .+ opt.eps)
    end
end

# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

"""
    train!(model, lossfn, opt, X, Y; epochs, batchsize, rng, verbose)

Mini-batch training. Returns the mean loss per epoch.
"""
function train!(model::Layer, lossfn::Loss, opt::Optimizer, X::AbstractMatrix, Y::AbstractMatrix;
                epochs::Int = 100, batchsize::Int = 32, rng::AbstractRNG = Random.default_rng(), verbose::Bool = false)
    n = size(X, 2)
    size(Y, 2) == n || throw(DimensionMismatch("X has $n columns but Y has $(size(Y, 2))"))
    history = Float64[]
    for epoch in 1:epochs
        order = randperm(rng, n)
        total = 0.0
        for start in 1:batchsize:n
            idx = order[start:min(start + batchsize - 1, n)]
            xb = X[:, idx]
            yb = Y[:, idx]
            ŷ = forward(model, xb)
            total += loss(lossfn, ŷ, yb) * length(idx)
            backward(model, gradient(lossfn, ŷ, yb))
            update!(opt, model)
        end
        push!(history, total / n)
        verbose && epoch % max(1, epochs ÷ 10) == 0 && println("epoch $epoch  loss $(round(history[end]; digits = 5))")
    end
    return history
end

"""Forward pass without training side effects (returns probabilities when the last layer is not softmax but the loss was SoftmaxCrossEntropy: use `softmax(predict(...))`)."""
predict(model::Layer, X::AbstractMatrix) = forward(model, X)

"""Predicted class index (1-based) per column."""
classify(model::Layer, X::AbstractMatrix) = vec(map(i -> i[1], argmax(predict(model, X), dims = 1)))

"""Fraction of columns whose argmax matches the one-hot target."""
function accuracy(model::Layer, X::AbstractMatrix, Y::AbstractMatrix)
    ŷ = classify(model, X)
    y = vec(map(i -> i[1], argmax(Y, dims = 1)))
    return mean(ŷ .== y)
end

# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

"""One-hot encode integer labels (1-based) into a `classes x n` matrix."""
function onehot(labels::AbstractVector{<:Integer}, classes::Int)
    Y = zeros(classes, length(labels))
    for (j, c) in enumerate(labels)
        Y[c, j] = 1.0
    end
    return Y
end

"""The classic spiral toy dataset: `points` per class, `classes` arms. Returns (X 2xN, labels)."""
function spiral_data(points::Int, classes::Int; rng::AbstractRNG = Random.default_rng(), noise = 0.2)
    X = zeros(2, points * classes)
    labels = zeros(Int, points * classes)
    for c in 1:classes
        for i in 1:points
            j = (c - 1) * points + i
            r = i / points
            t = (c - 1) * 4.0 + r * 4.0 + randn(rng) * noise
            X[1, j] = r * sin(t)
            X[2, j] = r * cos(t)
            labels[j] = c
        end
    end
    return X, labels
end

"""
    numerical_gradient(model, lossfn, X, Y)

Finite-difference gradients for every parameter (slow; for checking backward).
"""
function numerical_gradient(model::Layer, lossfn::Loss, X, Y; h = 1e-5)
    grads = Matrix{Float64}[]
    for (p, _) in parameters(model)
        g = zeros(size(p))
        for i in eachindex(p)
            old = p[i]
            p[i] = old + h
            lp = loss(lossfn, forward(model, X), Y)
            p[i] = old - h
            lm = loss(lossfn, forward(model, X), Y)
            p[i] = old
            g[i] = (lp - lm) / (2h)
        end
        push!(grads, reshape(g, size(p, 1), :))
    end
    return grads
end

save_model(path::AbstractString, model::Layer) = serialize(path, model)
load_model(path::AbstractString) = deserialize(path)::Layer

end # module
