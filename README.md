# Autograd_py_ripoff

### A simple implementation of PyTorch-style autograd using Python and NumPy, from scratch.

It helped in understanding the core underlying structure of autograd, and the graph-like structure it keeps track of as Tensors flow through operations.

## What it does

At the center is a `Tensor` class that wraps a NumPy array and remembers how it was created. Every operation (`+`, matmul, `**`, `exp`, `transpose`, `softmax`, `relu`, ...) builds a node in a computation graph: each output Tensor stores its parent Tensors (`_prev`) and a `_backward` closure containing the local gradient rule for that specific operation.

Calling `.backward()` on the final output:
1. Builds a topological ordering of the graph (so every node's gradient is fully accumulated *before* it propagates further back).
2. Seeds the output gradient as `1`.
3. Walks the graph in reverse, calling each node's `_backward()` to push gradients to its parents via `+=` (so nodes used in multiple places correctly sum contributions from every path — this is the multivariable chain rule in code).

## Implemented operations

| Op | Behavior | Backward rule shape |
|---|---|---|
| `__add__` | elementwise add | gradient passes through unchanged to both operands |
| `__mul__` | matrix multiply (`@`) | local gradient via transpose + `@` (mirrors the forward contraction) |
| `elementwise_mul` | elementwise multiply | gradient = other's data × upstream, elementwise |
| `__pow__` | elementwise power | gradient = `n·x^(n-1)` × upstream, elementwise |
| `exp` | elementwise exponential | gradient = `e^x` × upstream, elementwise |
| `transpose` | returns a new transposed Tensor | upstream gradient transposed back |
| `relu` | elementwise ReLU | gradient passes through only where output > 0 |
| `softmax` | row-wise softmax | full row-wise Jacobian: `s·(upstream − Σ(s·upstream))` |

Each operation's backward rule mirrors its *forward dependency structure* — elementwise ops need elementwise backward, matmul needs a transposed matmul backward, and softmax (where every output in a row depends on every input in that row) needs the full row Jacobian rather than a simple elementwise rule.

## Example: self-attention built on top of it

As a stress test for the autograd engine, a minimal self-attention mechanism is implemented on top of `Tensor`:

- Sinusoidal positional encoding added to word embeddings
- Separate Q/K/V projection weights (kept as independent Tensors so gradients stay connected through the graph, rather than sliced out of one stacked weight tensor)
- Scaled dot-product attention: `softmax((Q @ K.T) / sqrt(d_k)) @ V`

## Why this exists

This isn't meant to replace or compete with PyTorch — it's a from-scratch rebuild (inspired by [micrograd](https://github.com/karpathy/micrograd) and [CS231n](https://cs231n.github.io/)) to internalize *why* backprop works the way it does: local gradients, the chain rule as a sum over paths, and why the backward pass needs a specific execution order.

## Usage

```python
from backprop import Tensor

a = Tensor([[1, 2, 3]])
b = Tensor([[4], [5], [6]])

c = a * b        # matmul
d = c.exp()
d.backward()

print(a.grad)
```

## Status

Work in progress — built incrementally while debugging shape mismatches and gradient-rule bugs along the way.
