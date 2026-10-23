# Backends

A method in this bank is two things that are easy to confuse: an **operator** — the
mathematics, one readable file — and an **execution path**. The operator never changes
with the backend. The execution path does, and you choose it.

```python
print(ab.backends())                    # what runs here, and why anything does not
ab.build("softmax", backend="sdpa")     # ask for one explicitly
ab.backend_matrix()                     # which method declares which
```

## The four backends

| backend | what it is | needs |
|---|---|---|
| `reference` | plain PyTorch; **the correctness anchor and the default** | nothing |
| `sdpa` | `torch.nn.functional.scaled_dot_product_attention`; on CUDA this *is* FlashAttention-2 or the memory-efficient kernel | torch ≥ 2.2 |
| `flash_attn` | the `flash-attn` package | CUDA, fp16/bf16, `pip install attnbank[flash]` |
| `xformers`, `triton` | **detected and reported, declared by nothing** | — |

`xformers` and `triton` are listed because silence would be worse. `ab.backends()`
says why neither is used: torch ≥ 2.2 already routes `sdpa` to the same
memory-efficient kernel, so a second path would add a dependency and a second thing to
keep equal to the reference; and the bank ships no hand-written Triton kernel, because
these implementations exist to be read against the monograph.

## The matrix

```pycon
>>> ab.backend_matrix("sdpa")
{'flash': ('reference', 'sdpa'),
 'gqa': ('reference', 'sdpa'),
 'mla': ('reference', 'sdpa'),
 'softmax': ('reference', 'sdpa', 'flash_attn')}
>>> sum(len(b) == 1 for b in ab.backend_matrix().values())
28
```

**Seventy-four of seventy-eight methods declare `reference` only, and that is the correct
answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$.
Four methods in the bank are that, once their own preprocessing is done: `softmax`,
`flash`, and — after pooling or reconstructing the cache — `gqa` and `mla`. The rest
are not:

- `topk`, `blocktopk`, `lsh_sample` and the evictions report the omitted mass
  $\varepsilon_q$ in `extras`, which needs the full score matrix anyway. A kernel that
  skips materialising it cannot produce it.
- `nmf_oracle` and `svd_oracle` are factorisations; `merge_kv` is a clustering step;
  `sinkhorn` is a fixed-point iteration. None of them is a kernel call.
- `lowprec` quantises *inside* the kernel and returns the perturbed logits.

Declaring `sdpa` for those would be pretending. `ab.describe` says so explicitly:

```text
backends     reference   (reference only: no equivalent kernel exists for this operator)
```

## Detection explains itself

```text
attnbank backends on this machine (torch 2.10.0+cu128, cuda yes)
----------------------------------------------------------------
reference   available    plain PyTorch; always available; the correctness anchor
            version: 2.10.0+cu128
sdpa        available    torch SDPA; CUDA dispatches to FlashAttention-2 or mem-efficient
            version: 2.10.0+cu128
flash_attn  unavailable  the flash-attn package; CUDA, fp16/bf16 only
            reason: the flash_attn package is not installed
            fix:    pip install 'attnbank[flash]'   # needs CUDA and a C++ toolchain
…
default: 'reference'.
```

Every unavailable backend carries a reason and, where one exists, the exact command
that fixes it.

## Selection, and why fallback is loud

Three rules:

1. **No backend named.** Use the process default if the method supports it and it is
   installed, otherwise `reference`. Quiet — a default is a preference.
2. **A backend the method does not declare.** Raise `BackendNotSupportedError`,
   listing what it does declare. An explicit request is an instruction; silently doing
   something else would make a benchmark meaningless.
3. **A declared backend that is not installed here.** Warn with
   `BackendFallbackWarning` and use `reference`. Announced, never silent.

There is a fourth case inside a call: no fused kernel materialises $\widehat\pi_q$, so
`return_weights=True` falls back to `reference` for that call and says so. The result
is identical; only the speed is not. (`transformers` does the same for
`output_attentions` under `attn_implementation="sdpa"`.)

```python
with ab.default_backend("sdpa"):        # process-wide, scoped
    ...
ab.set_default_backend("reference")     # or permanently
```

## Pinning the sub-backend

"SDPA" is three kernels with three numerics profiles. For a benchmark, pin one:

```python
ab.build("softmax", backend="sdpa", sdpa_kernel="flash")          # FlashAttention-2
ab.build("softmax", backend="sdpa", sdpa_kernel="mem_efficient")
ab.build("softmax", backend="sdpa", sdpa_kernel="math")
```

An unavailable sub-backend raises rather than quietly choosing another.

## One detail worth knowing

`is_causal=True` in SDPA aligns the mask at the **top left**. This library aligns the
last query with the last key, so query `i` sees `0 … i + (n - m)`. They agree only when
`m == n`. `attnbank.components.sdpa_attend` therefore always passes an explicit boolean
mask, which makes decode (`m < n`) correct rather than accidentally right — and
`tests/test_backends.py` asserts that the naive `is_causal=True` call would in fact
disagree, so the test keeps proving something.

## Equality is tested, not asserted

Declaring a backend is a claim, and
`tests/test_conformance.py::test_backend_equivalence` is where it is checked: every
method, every declared backend, both causal settings, float32 and float64, on five
shapes including `m < n`, `n == 1` and logits scaled by 10. Tolerances are `1e-5`
(fp32) and `1e-10` (fp64), scaled by $\lVert y\rVert_\infty$. On a CUDA machine the
same comparison runs again on device, where `sdpa` dispatches to a genuinely fused
kernel.

## The studies are pinned

Every number in the monograph came out of `attnbank/studies/`. Importing that package
pins the default backend to `reference`:

```python
# attnbank/studies/__init__.py
from ..backends import REFERENCE, set_default_backend

set_default_backend(REFERENCE)
```

A published figure must not be able to move because the library grew a faster path.
`tests/test_backends.py` asserts the pin in a subprocess, and separately greps the
eight study modules for a backend override.
