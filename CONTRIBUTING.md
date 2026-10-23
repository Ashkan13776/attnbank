# Contributing to attnbank

The library is small on purpose. The useful contribution is almost always **a new
method**, and the bar for one is specific: it must declare what it is, and it must pass
the conformance suite on the strength of that declaration. Nothing is merged with a
`skip`.

```bash
git clone https://github.com/Ashkan13776/attnbank
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
    backends = ("reference",)             # see step 5; this is usually correct
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

(Steps 5 and 6 below are about backends and shared components; step 4 is the only
wiring you need to be picked up by the suite.)

```bash
pytest tests/test_conformance.py -k my_method    # the full eleven points
pytest -q                                        # and everything else
```

The eleven checks, in order: shapes · **Invariant W** · row stochasticity · causal
non-leakage · batch and head independence · the exact limit · Monte-Carlo convergence ·
float32/float64/CUDA agreement · the cost model · no NaN at degenerate supports or
large logits · every citation resolves in `references.bib`. `DESIGN.md` §4 is the
normative description.

### 5. Declare a backend, or declare `reference` only

```python
backends = ("reference",)      # the default, and usually the right answer
```

`reference` is plain PyTorch: always available, every dtype and device, and the path
every other backend is tested against. A method may additionally declare `sdpa` or
`flash_attn` — **but only if it can prove equality**.
`tests/test_conformance.py::test_backend_equivalence` compares every declared backend
against `reference` for both causal settings, float32 and float64, on `m < n`,
`m == n`, `n == 1`, a single decode step and logits scaled by 10. Tolerances are `1e-5`
(fp32) and `1e-10` (fp64), scaled by the output magnitude.

Twenty-eight of the thirty-two methods declare `reference` only, and they are right to.
`sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$;
if your operator is not that — or if its `extras` need the full score matrix anyway, as
every score-based selector's omitted mass does — there is nothing to dispatch to.
**Declaring `reference` only is a correct answer and far better than pretending
otherwise.**

If you do declare one, implement `_forward_<backend>` and route through
`attnbank.components.sdpa.sdpa_attend` rather than calling SDPA directly: it carries
the end-alignment convention that `is_causal=True` gets wrong when `m < n`. Delegate to
`self._fallback_to_reference(reason)` for anything the kernel cannot serve —
`return_weights=True`, most often — so the fallback is announced rather than silent.

### 6. Reuse the shared components

Before writing a helper, check [`attnbank/components/`](attnbank/components/):

| module | what lives there |
|---|---|
| `components.restriction` | the renormalised restriction of `prop:sparse-identity`, and `topk_support` |
| `components.online_softmax` | the state / merge / readout monoid of `prop:exact-monoid` |
| `components.quantization` | symmetric INT, asymmetric affine with grouping, FP8, Hadamard |
| `components.recurrence` | feature maps and the readout convention |
| `components.sdpa` | the one SDPA call every `_forward_sdpa` goes through |
| `attnbank.core` | masking, log-space softmax, renormalisation, seeded generators |

A new selector in `restrict/` should be three lines plus the support rule. If yours is
not, either you are reinventing something, or the shared piece needs widening — which
is a fine pull request on its own.

### 7. Add the citation

Put the BibTeX entry in `attnbank/references.bib`. Check 11 fails if a key in
`references` does not resolve there, which is deliberate: an implementation without a
source is not reviewable.

### 8. Write the method's own tests

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
`mkdocs serve`. Pages live in `docs/`, the gallery in `examples/`.

Two pages are generated and should not be hand-edited:

```bash
python docs/gen_taxonomy.py    # docs/taxonomy.md, from the live registry
python docs/gen_gallery.py     # docs/examples/*.md, from examples/*.py + captured output
```

The gallery pages *include* each script rather than copying it, so the code can never
drift; only the captured output can, which is why the generator re-runs everything.
`tests/test_examples.py` executes every script in CI, so a broken example is a failing
build rather than a stale page. `mkdocs build --strict` runs in CI too: a dangling link
is an error.

**A docstring.** Every public function carries a runnable `Examples` block, and
`tests/test_docs.py` executes all of them and fails the build for a public name without
one. Add the example, run `pytest tests/test_docs.py`, and paste in the output it
tells you it got.

**A bug.** A failing test first, in the file where the behaviour belongs, then the fix.

---

## House style

Reference implementations favour clarity over speed. They are read by people checking
the mathematics against the paper with the relevant section open, so: no
micro-optimisation, no fused tricks, and no `einsum` string longer than the equation it
implements. Every method file opens with a docstring giving the operator in the paper's
notation, the section label and the citation.

`ruff check .` must be clean; CI runs it on Python 3.10–3.13. Line length is 100.

`ruff format` is deliberately **not** enforced. The method files align equations by hand
so they can be read next to the monograph, and the formatter would undo that. Match the
surrounding file.

## The contract wins

`DESIGN.md` is the binding specification. If an implementation and that file disagree,
the file wins and the implementation is the bug. If you believe the file is wrong, say
so in an issue — changing it is a legitimate outcome, changing it quietly in a pull
request that also adds a method is not.

## Licence

By contributing you agree that your contribution is licensed under the Apache License
2.0, as in [`LICENSE`](LICENSE).
