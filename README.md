# Scalar Autograd Engine + MLP — Built From Scratch

A minimal automatic differentiation engine and a small neural network library, implemented from first principles in pure Python — no PyTorch, no NumPy, no external dependencies. Built as a deep dive into how backpropagation actually works under the hood, in the spirit of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

Every arithmetic operation on a `Value` object silently builds a computational graph. Calling `.backward()` on the output walks that graph in reverse and computes exact gradients for every parameter that touched it — the same core mechanism that powers `loss.backward()` in PyTorch, just written out in ~80 lines instead of hidden behind a framework.

## Why build this

Frameworks like PyTorch make `loss.backward()` feel like magic. Reimplementing it by hand — one operation, one gradient rule at a time — is the fastest way to actually understand:
- What a computational graph is and how it's constructed dynamically at runtime
- Why topological sort matters for correct gradient propagation
- How the chain rule composes across `+`, `*`, `tanh`, `exp`, `pow`, etc.
- How a "neural network" is really just a specific pattern of scalar operations

## Repository structure

```
.
├── value.py        # Core autograd engine — the Value class
├── neural_net.py   # Neuron, Layer, and MLP built on top of Value
├── test.py         # Training loop demo on a toy dataset
├── README.md
├── LICENSE
└── .gitignore
```

### `value.py`
The heart of the project. `Value` wraps a single scalar and tracks:
- `.data` — the actual number
- `.grad` — the gradient accumulated during backpropagation
- `._prev` — the set of `Value`s that produced this one
- `._backward` — a closure defining the local gradient rule for this operation

Supported operations: `+`, `-`, `*`, `/`, `**` (int/float powers), unary negation, `tanh()`, `exp()`. Each one defines its own `_backward` function implementing that operation's specific derivative.

`.backward()` performs a topological sort of the graph rooted at the calling node, then walks it in reverse order applying the chain rule — accumulating gradients into every `Value` that contributed to the output.

### `neural_net.py`
Three classes, each building on the last, loosely mirroring PyTorch's `nn.Module` pattern (`__call__` for forward pass, `.parameters()` for optimization):

| Class | Description |
|---|---|
| `Neuron` | Holds `nin` weights + 1 bias as `Value`s. Forward pass: weighted sum → `tanh` activation. |
| `Layer` | A list of `Neuron`s operating on the same input in parallel. |
| `MLP` | A stack of `Layer`s — a full feedforward multi-layer perceptron. |

Because everything is composed of `Value` operations, the entire network is automatically differentiable — no manual backprop derivation needed at the network level, only at the operation level in `value.py`.

### `test.py`
Trains a `3 → 4 → 4 → 1` MLP on 4 toy examples using mean squared error and vanilla gradient descent (no optimizer, no batching — just the raw update rule) to demonstrate the whole pipeline end to end.

## Running it

No dependencies beyond the Python standard library (`math`, `random`).

```bash
python test.py
```

Expected output — decreasing loss over training steps, followed by final predictions vs. targets:

```
step 0:  loss = 6.8421
step 1:  loss = 4.1153
...
step 19: loss = 0.0842

Final predictions:
target: +1.0  predicted: +0.9123
target: -1.0  predicted: -0.8871
target: -1.0  predicted: -0.9045
target: +1.0  predicted: +0.8932
```

## Example usage

```python
from value import Value
from neural_net import MLP

# 3 inputs -> two hidden layers of 4 -> 1 output
n = MLP(3, [4, 4, 1])

x = [2.0, 3.0, -1.0]
y = n(x)          # forward pass
y.backward()      # backward pass, populates .grad on every parameter
```

## Concepts demonstrated

- Reverse-mode automatic differentiation
- Dynamic (define-by-run) computational graph construction
- Manual implementation of the chain rule per operation
- Topological sort for correct gradient propagation order
- Gradient accumulation and the necessity of zeroing gradients between steps
- Gradient descent optimization implemented from scratch

## Roadmap / possible extensions

- [ ] Additional activation functions (ReLU, sigmoid, GELU)
- [ ] Vectorized `Value` operations for basic tensor support
- [ ] Simple optimizers (SGD with momentum, Adam)
- [ ] Visualization of the computational graph (Graphviz)
- [ ] Numerical gradient checking for regression testing

## Acknowledgments

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and his ["Neural Networks: Zero to Hero"](https://karpathy.ai/zero-to-hero.html) series.

## License

MIT — see [LICENSE](./LICENSE).