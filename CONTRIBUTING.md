# Contributing to attnbank

The library is small on purpose. The useful contribution is almost always **a new
method**, and the bar for one is specific: it must declare what it is, and it must pass
the conformance suite on the strength of that declaration. Nothing is merged with a
`skip`.

```bash
git clone https://github.com/attnbank/attnbank
cd attnbank
pip install -e ".[dev]"
pytest -q          # ~1000 tests, about ten seconds on a laptop
ruff check .
```

---

## Adding a method

### 1. Decide what it changes

Every efficient attention method changes exactly one object in

$$y(q) \;=\; \int v \,\mathrm{d}\pi_q, \qquad \pi_q \;=\; \frac{g(q,\cdot)\,\mu}{Z(q)} .$$

That choice is the directory it goes in. `ab.categories()` prints the seven options;
`DESIGN.md` §3 lists the methods already in each. **Do not invent a new category** — if
a method genuinely does not fit, that is a finding about the taxonomy and belongs in an
issue before it belongs in a pull request.

One method per file: `attnbank/methods/<category>/<name>.py`.

### 2. Declare what it is

```python
@ab.register
class MyAttention(ab.AttentionMethod):
    name = "my_method"                    # unique, lowercase, no spaces
    category = ab.Category.RESTRICT       # which object it changes
    objective = ab.Objective.APPROXIMATE  # accelerate | approximate | modify
    supports_causal = True
    expresses_weights = True              # can it name pi_hat on the original n?
    row_stochastic = True                 # are those weights a distribution?
    head_independent = True               # False only if it shares K/V across heads
    trainable = False
    references = ("author2024paper",)     # keys that resolve in references.bib
    paper_section = "sec:sparse"
```

Those attributes are **commitments, not documentation**. The conformance suite reads
them and tests the method against what it claims, which is why there is no list of
exemptions anywhere in the suite: a method that cannot be row-stochastic says
`row_stochastic = False` and the check becomes inapplicable rather than skipped.

Two of them deserve care.

- **`objective`.** `ACCELERATE` means exact up to floating point. `APPROXIMATE` means
  the method targets dense softmax and is judged by fidelity to it. `MODIFY` means it
  defines a *different operator* and fidelity to softmax is not its goal. Mislabelling
  an operator modification as an approximation is the most common mistake in this
  literature; the library exists partly to make the distinction unavoidable.
- **`expresses_weights`.** Set it to `False` only if the method genuinely cannot write
  its measure on the original $n$ positions. Setting it to `False` to dodge Invariant W
  is the one thing that will get a pull request closed: it silently removes the method
  from every error-decomposition study.

### 3. Implement three things

```python
def _forward(self, q, k, v, causal, return_weights) -> AttentionOutput: ...
def cost(self, m, n, d, dv, bytes_per_el=2) -> dict:   # flops, state_bytes, bytes_moved
@classmethod
def limit_config(cls) -> dict | None: ...
```

Inside a method file the maths is written in the paper's `q, k, v` notation; the public
argument names (`query, key, value`) are the base class's business.

`_forward` must return an `AttentionOutput` satisfying

```python
y == weights @ (v_hat if v_hat is not None else v)
```

in float64 to `1e-10`. This is **Invariant W** and it is the whole point of the
library: `attnbank.metrics` splits output error into a distributional and a value term,
and that split is a measurement only if the reported weights and values actually
reproduce the output.

`limit_config()` returns the configuration under which the method *provably* reduces to
dense softmax — `{"k": None}` for top-k, `{"bits": 16}` for quantisation — and the
suite uses it as a correctness anchor. A Monte-Carlo estimator with no exact limit at
finite resource returns `None` and supplies `convergence_configs()` instead, a sequence
along which the mean error must decrease. Only methods whose `objective` is `MODIFY`
are exempt, and the suite asserts the exemption rather than skipping it.

Four rules that prevent the bugs we actually hit:

1. Compute softmax in log space; never materialise `exp` of an unshifted score.
2. A fully masked row must produce a documented, finite result — never `nan`. Window
   methods guarantee at least one visible key by including the diagonal.
3. Renormalise explicitly, and floor the denominator: `pi / pi.sum(-1, keepdim=True).clamp_min(TINY)`.
4. Seeded randomness goes through `torch.Generator`, never the global RNG, so studies
   reproduce and two builds agree bit for bit.

Never mutate `q`, `k` or `v`. That is tested.

### 4. Wire it up and check

Import the module in the category's `__init__.py`. That is the only registration step —
the conformance suite is parametrized over the registry, so it picks the method up
automatically.

```python
import attnbank as ab
print(ab.check_conformance("my_method"))     # the fast half, immediately
```

```bash
pytest tests/test_conformance.py -k my_method    # the full eleven points
pytest -q                                        # and everything else
```

The eleven checks, in order: shapes · **Invariant W** · row stochasticity · causal
non-leakage · batch and head independence · the exact limit · Monte-Carlo convergence ·
float32/float64/CUDA agreement · the cost model · no NaN at degenerate supports or
large logits · every citation resolves in `references.bib`. `DESIGN.md` §4 is the
normative description.

### 5. Add the citation

Put the BibTeX entry in `attnbank/references.bib`. Check 11 fails if a key in
`references` does not resolve there, which is deliberate: an implementation without a
source is not reviewable.

### 6. Write the method's own tests

Conformance proves the method is *well-formed*. It does not prove the method is
*correct*. Add `tests/test_<category>.py` cases for the behaviour that makes your
method what it is — the support it selects, the bound it satisfies, the limit it
approaches — the way the existing files do.

---

## Other kinds of contribution

**A metric.** Add it to `attnbank/metrics.py` with a docstring giving the equation in
the paper's notation and a runnable example. A function named `*_bound` must return an
upper bound on something, and `tests/test_metrics.py` must contain a randomised test
that it really bounds it.

**Documentation.** The site is MkDocs Material; `pip install -e ".[docs]"` then
`mkdocs serve`. Pages live in `docs/`, the gallery in `examples/`. Every example script
is executed in CI, so an example that no longer runs is a failing build, not a stale
page.

**A bug.** A failing test first, in the file where the behaviour belongs, then the fix.

---

## House style

Reference implementations favour clarity over speed. They are read by people checking
the mathematics against the paper with the relevant section open, so: no
micro-optimisation, no fused tricks, and no `einsum` string longer than the equation it
implements. Every method file opens with a docstring giving the operator in the paper's
notation, the section label and the citation.

`ruff check .` and `ruff format --diff .` must be clean. Line length is 100.

## The contract wins

`DESIGN.md` is the binding specification. If an implementation and that file disagree,
the file wins and the implementation is the bug. If you believe the file is wrong, say
so in an issue — changing it is a legitimate outcome, changing it quietly in a pull
request that also adds a method is not.

## Licence

By contributing you agree that your contribution is licensed under the Apache License
2.0, as in [`LICENSE`](LICENSE).
