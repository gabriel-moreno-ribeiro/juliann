# JuliaNN

Uma biblioteca de redes neurais em Julia, sem Flux e sem autodiff: camadas com backpropagation escrita à mão, funções de perda, SGD com momentum e Adam, loop de treino em mini-batches, e gradient checking pra provar que as derivadas estão certas.

Fiz depois de anos usando PyTorch sem nunca ter derivado um softmax na mão. O objetivo não era performance, era não ter mais nenhuma parte da rede que eu não soubesse explicar.

```julia
using JuliaNN, Random

rng = MersenneTwister(42)
X, labels = spiral_data(150, 3; rng = rng)      # 2 x 450 pontos, três braços de espiral
Y = onehot(labels, 3)

model = Sequential(
    Dense(2 => 32; rng = rng), Activation(:relu),
    Dense(32 => 32; rng = rng), Activation(:relu),
    Dense(32 => 3; rng = rng),
)
train!(model, SoftmaxCrossEntropy(), Adam(0.01), X, Y; epochs = 300, batchsize = 32, rng = rng)
accuracy(model, X, Y)                            # ~0.99
```

```sh
julia --project examples/xor.jl
julia --project examples/spiral.jl               # imprime a fronteira de decisão em ASCII
```

| Peça | Detalhe |
| --- | --- |
| `Dense(in => out)` | `y = W x + b`, inicialização de He, guarda `x` pro backward |
| `Activation(:relu \| :sigmoid \| :tanh \| :softmax)` | derivadas exatas (softmax usa a Jacobiana inteira) |
| `Sequential` | forward em ordem, backward ao contrário |
| `MSE`, `CrossEntropy`, `SoftmaxCrossEntropy` | a versão fundida trabalha em logits e é numericamente estável |
| `SGD(lr; momentum)`, `Adam(lr)` | estado por array de parâmetros |
| `train!` | embaralha, faz batches, forward, loss, backward, update; devolve o histórico |
| `numerical_gradient` | diferenças finitas centrais sobre todos os pesos |

Os dados ficam como `features x batch`, então o peso de uma camada é `out x in` e um batch inteiro é uma multiplicação de matriz só. Julia com BLAS por baixo faz isso rápido o suficiente pra treinar o exemplo da espiral em segundos.

## Backprop

Cada camada guarda o que precisa no `forward` e, dado o gradiente da loss em relação à saída, devolve o gradiente em relação à entrada e anota os gradientes dos próprios parâmetros:

```julia
forward(l::Dense, x)   = (l.x = x; l.W * x .+ l.b)
backward(l::Dense, dy) = (l.dW = dy * x'; l.db = sum(dy, dims = 2); l.W' * dy)
```

`Sequential.backward` só encadeia isso de trás pra frente. A suíte compara todo gradiente analítico com diferenças finitas em quatro arquiteturas e losses, com erro relativo abaixo de 1e-6. Quando esse teste passou pela primeira vez eu entendi por que gradient checking é a primeira coisa que qualquer curso sério manda fazer.

Testes: `julia --project -e 'using Pkg; Pkg.test()'`.

---

**EN:** a neural network library in plain Julia: dense layers with hand-derived backpropagation, activations with exact derivatives (softmax with the full Jacobian), MSE and cross-entropy losses, SGD with momentum and Adam, a mini-batch training loop, and finite-difference gradient checking that the test-suite runs on several architectures (relative error < 1e-6). MIT.
