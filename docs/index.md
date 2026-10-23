# attnbank

**Every efficient attention method changes one object in one formula.**
`attnbank` implements them side by side, organised by *which* object they change, and
holds every one of them to a single invariant.

```bash
pip install attnbank
```

```python
import attnbank as ab

y = ab.attend("topk", query, key, value, causal=True, k=64)
```

That is the whole first lesson. The second is that `attnbank` will tell you *what the
approximation cost you*, and in which of the two ways an approximation can be wrong.

```python
rows = ab.compare(["softmax", "topk", "performer", "quantkv"], query, key, value,
                  configs={"topk": {"k": 16}, "performer": {"r": 64},
                           "quantkv": {"bits_k": 16, "bits_v": 2}})
```

| | output error | distributional | value |
|---|---|---|---|
| `topk` | 1.03 | **1.03** | 0.00 |
| `performer` | 2.76 | **2.76** | 0.00 |
| `quantkv` | 0.72 | 0.00 | **0.72** |

Restriction moves mass to the wrong places. Quantisation aggregates the wrong values.
Both show up as "error" in a single number; only the split says which mechanism is at
fault — and the split is exact, not a heuristic.

---

## The idea in one formula

Attention is a query-conditioned expectation:

$$y(q) \;=\; \int v \,\mathrm{d}\pi_q, \qquad \pi_q \;=\; \frac{g(q,\cdot)\,\mu}{Z(q)} .$$

Efficiency has to come from somewhere, and there are only so many places it can come
from. A method either evaluates this expression differently, or it changes one of the
objects in it. That is the taxonomy, and it is also the directory structure.

| Category | What it changes |
|---|---|
| [`exact`](taxonomy.md#exact) | nothing — only the evaluation |
| [`interaction`](taxonomy.md#interaction) | the interaction $g$ |
| [`restrict`](taxonomy.md#restrict) | the support of $\pi_q$ |
| [`compress`](taxonomy.md#compress) | the stored context $(\mu, k, v)$ |
| [`factorize`](taxonomy.md#factorize) | the factorisation of $\pi_q$ through latents |
| [`coupling`](taxonomy.md#coupling) | normalisation, which stops being per-query |
| [`recurrent`](taxonomy.md#recurrent) | the operator itself (finite state) |

A third axis matters as much as the first: a method that **approximates** the reference
operator is judged by fidelity to it, while a method that **modifies** it is a different
operator and fidelity to softmax is not its goal. Conflating the two is the most
persistent confusion in this literature, so every method here declares which it is.

## The one invariant

```python
y == weights @ (v_hat if v_hat is not None else value)
```

Every method realises, explicitly or implicitly, a measure $\widehat\pi_q$ over context
positions and a value field $\widehat v$ — and must be able to show you both.
[Why that matters](invariant.md), and how it is enforced for all 32 methods in float32
and float64, is the subject of its own page.

## Where to go next

<div class="grid cards" markdown>

- :material-rocket-launch: **[Quickstart](quickstart.md)** — five minutes, three functions.
- :material-sitemap: **[The taxonomy](taxonomy.md)** — what each category means and which methods are in it.
- :material-check-decagram: **[The invariant](invariant.md)** — the property everything else rests on.
- :material-test-tube: **[Conformance](conformance.md)** — the eleven checks every method passes.
- :material-rocket: **[Backends](backends.md)** — `reference`, `sdpa`, `flash_attn`, and why most methods declare only the first.
- :material-image-multiple: **[Examples gallery](examples/index.md)** — short runnable scripts.
- :material-api: **[API reference](api/index.md)** — every public function, with examples.

</div>

## Citing

`attnbank` is the companion code for *What Does Efficient Attention Approximate? A
Unified View of Interactions, Memory, and Error*. If the library or its taxonomy is
useful in your work, please cite the monograph:

```bibtex
@article{shahbazi2026attention,
  title  = {What Does Efficient Attention Approximate?
            A Unified View of Interactions, Memory, and Error},
  author = {Shahbazi, Ashkan},
  year   = {2026},
}
```

Released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
