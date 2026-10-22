import pytest
import torch
import attnbank as ab
from attnbank import metrics


def _qkv(B=2, H=2, m=16, n=32, d=8, dv=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, H, m, d, generator=g), torch.randn(B, H, n, d, generator=g),
            torch.randn(B, H, n, dv, generator=g))


def test_softmax_matches_torch():
    q, k, v = _qkv()
    for causal in (False, True):
        out = ab.build("softmax")(q, k, v, causal=causal, return_weights=True)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=ab.causal_mask(q.shape[-2], k.shape[-2]) if causal else None)
        assert torch.allclose(out.y, ref, atol=1e-5)
        assert torch.allclose(out.weights.sum(-1), torch.ones(()), atol=1e-6)


def test_error_decomposition_is_exact():
    q, k, v = _qkv()
    pi = ab.build("softmax")(q, k, v, return_weights=True).weights
    pi_hat = torch.softmax(torch.randn_like(pi), -1)
    v_hat = v + 0.1 * torch.randn_like(v)
    d = metrics.error_decomposition(pi, pi_hat, v, v_hat)
    direct = (pi @ v - pi_hat @ v_hat).norm(dim=-1)
    assert torch.allclose(d["total"], direct, atol=1e-5)


def test_restriction_identity():
    q, k, v = _qkv()
    pi = ab.build("softmax")(q, k, v, return_weights=True).weights
    idx = pi.topk(8, dim=-1).indices
    S = torch.zeros_like(pi, dtype=torch.bool).scatter_(-1, idx, True)
    pi_hat = pi * S / (pi * S).sum(-1, keepdim=True)
    r = metrics.restriction_identity(pi, S, v)
    assert torch.allclose(pi @ v - pi_hat @ v, r["predicted_error"], atol=1e-5)


def test_tv_osc_bound_holds():
    q, k, v = _qkv()
    pi = ab.build("softmax")(q, k, v, return_weights=True).weights
    pi_hat = torch.softmax(torch.randn_like(pi), -1)
    err = ((pi - pi_hat) @ v).norm(dim=-1)
    assert (err <= metrics.tv_osc_bound(pi, pi_hat, v) + 1e-6).all()


# ---------------------------------------------------------------------------
# The convenience API of DESIGN.md Sec. 2
# ---------------------------------------------------------------------------
def test_attend_returns_only_y_and_renames_colliding_kwargs():
    q, k, v = _qkv()
    y = ab.attend("topk", q, k, v, causal=True, k_=4)
    assert y.shape == (2, 2, 16, 8)
    assert torch.allclose(y, ab.build("topk", k=4)(q, k, v, causal=True).y)
    assert torch.allclose(ab.attend("softmax", q, k, v), ab.build("softmax")(q, k, v).y)


def test_compare_scores_every_method_against_the_reference():
    q, k, v = _qkv()
    rows = ab.compare(["softmax", "topk", "performer", "linformer"], q, k, v,
                      configs={"topk": {"k": 4}, "performer": {"r": 32}})
    assert [r["name"] for r in rows] == ["softmax", "topk", "performer", "linformer"]
    by = {r["name"]: r for r in rows}
    assert by["softmax"]["output_error"] < 1e-6
    assert by["topk"]["output_error"] > 0
    for key in ("distributional_error", "value_error", "tv", "matrix_fro", "matrix_AV_fro"):
        assert key in by["topk"]
    # the decomposition must reconstruct the output error
    assert by["topk"]["distributional_error"] == pytest.approx(by["topk"]["output_error"],
                                                               rel=1e-5)
    assert "_raw" not in by["topk"]                 # the default row is JSON-friendly
    import json
    json.dumps(rows)
    deep = {r["name"]: r for r in ab.compare(["topk"], q, k, v, configs={"topk": {"k": 4}},
                                             raw=True)}
    assert deep["topk"]["_raw"]["weights"] is not None


def test_method_table_has_the_declaration_columns():
    rows = {r["name"]: r for r in ab.method_table()}
    assert rows["nystrom"]["row_stochastic"] is False
    assert rows["topk"]["row_stochastic"] is True
    assert rows["gqa"]["head_independent"] is False
    assert rows["topk"]["limit_config"] == {"k": None}
    assert rows["performer"]["limit_config"] is None
    assert rows["performer"]["n_convergence_configs"] > 0
    assert all(r["paper_section"].startswith("sec:") for r in rows.values())


def test_available_filters_by_category():
    for cat in ("exact", "interaction", "restrict", "compress", "factorize", "coupling",
                "recurrent"):
        names = ab.available(cat)
        assert names, cat
        assert all(ab.get_class(n).category.value == cat for n in names)
    assert sorted(sum((ab.available(c) for c in
                       ("exact", "interaction", "restrict", "compress", "factorize",
                        "coupling", "recurrent")), [])) == ab.available()
