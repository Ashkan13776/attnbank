# Quickstart

```bash
pip install attnbank            # torch + numpy, nothing else
```

Optional extras, none of them needed to start:

| extra | what it adds |
|---|---|
| `attnbank[viz]` | matplotlib and the figure style used by the monograph |
| `attnbank[fast]` | `flash-attn`, `triton`, `xformers` — CUDA-only kernels |
| `attnbank[studies]` | `transformers` + `datasets`, for capturing real activations |
| `attnbank[dev]` | pytest, ruff and the docs toolchain |

## Tensors

```
query  [B, H, m, d]
key    [B, H, n, d]
value  [B, H, n, dv]
```

Causal masking aligns the **last** query with the **last** key, so `m <= n` and query
`i` sees keys `0 … i + (n - m)`. That is the decode convention, and it means the same
code path serves prefill (`m == n`) and a single decode step (`m == 1`).

!!! note "Why `query, key, value` and not `q, k, v`"
    The positional names follow the transformers convention, which frees `k=` to mean
    what a reader expects — `ab.attend("topk", q, k, v, k=64)` is top-k's budget. The
    tensors are positional, so nothing about `attn(q, k, v)` changed.

## Three functions

### `attend` — the one-liner

```python
import attnbank as ab

y = ab.attend("topk", query, key, value, causal=True, k=64)
```

Builds the method, runs it, returns `y`. Nothing else.

### `build` — the object

```python
attn = ab.build("topk", k=64)            # a torch.nn.Module
out = attn(query, key, value, causal=True, return_weights=True)

out.y                                     # [B, H, m, dv]
out.weights                               # [B, H, m, n] — the implied measure
out.v_hat                                 # [B, H, n, dv] — if values were compressed
out.extras                                # method-specific diagnostics
```

`return_weights` materialises an `m × n` tensor, so it is off by default.

### `compare` — the numbers

```python
rows = ab.compare(["softmax", "topk", "performer"], query, key, value, causal=True,
                  configs={"topk": {"k": 64}, "performer": {"r": 256}})
```

One JSON-serialisable dict per method: `output_error`, `relative_error`, the two terms
of the [error decomposition](invariant.md#why-it-matters), total variation, and the
matrix norms of §4.2 — all reduced to a median over queries, batch and heads. Pass
`raw=True` to keep the per-query tensors.

## Finding a method

```python
ab.list_methods()                 # the whole bank, grouped by category
ab.list_methods("restrict")       # one category
ab.list_methods(causal=True).filter(row_stochastic=True)
ab.available(objective="modify")  # just the names
ab.describe("nystrom")            # one method, in full
ab.categories()                   # what each category means
```

`ab.describe` prints the declaration attributes, the constructor signature with its
defaults, the correctness anchor, the backends, the paper section and the citations:

```text
nystrom  --  interaction / approximate
=====================================
Landmark Nystrom factorization of the softmax matrix; signed weights.

changes      the interaction g(q, x) is replaced by a cheaper surrogate
objective    targets fidelity to dense softmax; judged by output error
causal       no
weights      yes
stochastic   no
heads        independent
trainable    no
anchor       limit_config = {'r': None}
backends     reference   (reference only: no equivalent kernel exists for this operator)

build        ab.build('nystrom', r=16)
section      sec:kernel
references   xiong2021nystromformer, williams2001using
```

## Choosing an execution path

```python
print(ab.backends())                      # what runs here, and why anything does not
ab.build("softmax", backend="sdpa")       # PyTorch's fused kernel
ab.backend_matrix()                       # method -> declared backends
```

The default is `reference`: plain PyTorch, always available, and the path every other
backend is tested against. See [Backends](backends.md).

## Checking the contract

```python
ab.check_invariant(out, value)            # y == weights @ (v_hat or value)
ab.check_conformance("topk")              # a printable pass/fail report
```

## Errors that tell you what to do

```pycon
>>> ab.build("topkk")
UnknownMethodError: unknown method 'topkk'. Did you mean: 'topk'? ab.available() lists
all 32 registered methods; ab.list_methods() prints them grouped by category.

>>> ab.attend("topk", torch.randn(8, 8), key, value)
ShapeError: query (in ab.attend(...)) must be [B, H, m, d]; got [8, 8] (2 dims, expected 4)

>>> ab.build("sinkhorn")(q, k, v, causal=True)
CausalNotSupportedError: 'sinkhorn' does not support causal attention: it declares
supports_causal = False, which is a statement about the operator, not a missing
feature. No method in the coupling category supports it; 21 others do, e.g.
'blocktopk', 'cosformer', … ab.list_methods(causal=True) lists them all.
```

## Next

- [The taxonomy](taxonomy.md) — what the seven categories mean.
- [The invariant](invariant.md) — why `weights` and `v_hat` are part of the interface.
- [Examples gallery](examples/index.md) — short scripts you can run now.
