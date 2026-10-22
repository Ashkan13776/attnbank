"""Tests for the study infrastructure: ``attnbank.data`` and ``attnbank.studies._harness``.

The study results are only as good as these two, so this file checks:

* the synthetic control's shapes, conventions and determinism;
* the activation cache round-trips (save -> load -> identical tensors and provenance);
* the sweep runner's resumability (a crashed sweep continues, finished jobs are not
  recomputed) and its result files;
* the summariser's median/IQR against hand-computed values;
* **that captured real activations reproduce the model's own attention output** —
  the test that catches a wrong scale, a missing RoPE, or a transposed/mis-grouped
  head axis. It is skipped (not silently passed) when the model cannot be obtained.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
import torch

from attnbank import data
from attnbank.studies import _harness as H


# ======================================================================================
# synthetic control
# ======================================================================================


def test_synthetic_shapes_and_convention():
    b = data.synthetic(n=64, d=8, dv=6, heads=4, passages=3, seed=0)
    assert b.q.shape == (3, 4, 64, 8)       # [B, H, m, d]
    assert b.k.shape == (3, 4, 64, 8)       # [B, H, n, d]
    assert b.v.shape == (3, 4, 64, 6)       # [B, H, n, dv]
    assert b.q.dtype == torch.float64       # error measurements are float64
    assert b.scale == pytest.approx(8 ** -0.5)
    assert b.causal and b.heads == (0, 1, 2, 3) and b.passages == (0, 1, 2)
    assert b.shape == {"B": 3, "H": 4, "n_kv_heads": 4, "m": 64, "n": 64, "d": 8, "dv": 6}
    assert b.meta["seed"] == 0 and b.source == "synthetic"


def test_synthetic_is_seeded_and_gaussian():
    a = data.synthetic(n=32, d=8, heads=2, passages=2, seed=7)
    b = data.synthetic(n=32, d=8, heads=2, passages=2, seed=7)
    c = data.synthetic(n=32, d=8, heads=2, passages=2, seed=8)
    assert torch.equal(a.q, b.q) and torch.equal(a.v, b.v)
    assert not torch.equal(a.q, c.q)
    big = data.synthetic(n=4096, d=16, heads=4, passages=4, seed=1)
    assert abs(float(big.q.mean())) < 0.02
    assert abs(float(big.q.std()) - 1.0) < 0.02


def test_synthetic_q_scale_grows_logits():
    a = data.synthetic(n=32, d=8, heads=1, passages=1, seed=3, q_scale=1.0)
    b = data.synthetic(n=32, d=8, heads=1, passages=1, seed=3, q_scale=5.0)
    assert torch.allclose(b.q, a.q * 5.0)
    assert torch.equal(a.k, b.k)            # only q is scaled: the S3 logit-scale knob


def test_synthetic_gqa_shape_and_expand():
    b = data.synthetic(n=16, d=8, heads=6, passages=1, seed=0, kv_groups=3)
    assert b.H == 6 and b.n_kv_heads == 2 and b.kv_groups == 3
    e = b.expand_kv()
    assert e.k.shape == (1, 6, 16, 8) and e.kv_groups == 1
    for h in range(6):                       # query head h must read KV head h // 3
        assert torch.equal(e.k[0, h], b.k[0, h // 3])


def test_batch_units_and_select():
    b = data.synthetic(n=16, d=8, heads=4, passages=2, seed=0, kv_groups=2)
    tags = [t for t, _ in b.units()]
    assert len(tags) == 8
    assert tags[0] == {"source": "synthetic", "layer": 0, "head": 0, "kv_head": 0, "passage": 0}
    for tag, u in b.units():
        assert u.q.shape == (1, 1, 16, 8) and u.k.shape == (1, 1, 16, 8)
        assert torch.equal(u.k[0, 0], b.k[b.passages.index(tag["passage"]), tag["kv_head"]])
    s = b.select(heads=[1, 2], passages=[1])
    assert s.q.shape == (1, 2, 16, 8) and s.heads == (1, 2) and s.passages == (1,)
    assert torch.equal(s.q[0, 0], b.q[1, 1])
    assert b.select(n_tokens=4).n == 4


def test_batch_provenance_is_json_serializable():
    b = data.synthetic(n=8, d=4, heads=2, passages=1, seed=0)
    p = b.provenance()
    json.dumps(p)                            # results files must be able to hold it
    assert p["source"] == "synthetic" and p["scale"] == pytest.approx(0.5)
    assert p["causal"] is True and p["seed"] == 0


# ======================================================================================
# sampling helpers
# ======================================================================================


def test_stratified_layers_spans_depth():
    assert data.stratified_layers(12, 4) == (0, 4, 7, 11)
    assert data.stratified_layers(28, 4) == (0, 9, 18, 27)
    assert data.stratified_layers(5, 1) == (2,)
    assert data.stratified_layers(3, 10) == (0, 1, 2)


def test_sample_heads_is_seeded_and_spread():
    assert data.sample_heads(12, None) == tuple(range(12))
    assert data.sample_heads(12, 20) == tuple(range(12))
    a = data.sample_heads(28, 8, seed=0)
    assert a == data.sample_heads(28, 8, seed=0)
    assert a != data.sample_heads(28, 8, seed=1)
    assert len(a) <= 8 and a == tuple(sorted(a)) and max(a) < 28
    assert min(a) < 7 and max(a) > 20        # stratified: spans the head range


# ======================================================================================
# activation cache round-trip
# ======================================================================================


def _write_layer(d, layer, b, model_key="fake", dataset_key="fake-data"):
    """Write one layer file the way ``_run_capture`` does, from synthetic tensors."""
    info = {"model_key": model_key, "dataset_key": dataset_key, "layer": layer,
            "heads": list(b.heads), "kv_heads": list(b.kv_heads), "kv_groups": b.kv_groups,
            "lib_scale": b.scale, "model_scale": b.scale, "scale_fold": 1.0,
            "n_kv_heads": b.n_kv_heads, "n_q_heads": b.H, "n_tokens": b.n,
            "n_passages": b.B, "model_repo": "fake/repo", "dataset_path": "fake",
            "dataset_name": None, "dataset_split": "test",
            "captured_at": "2026-01-01T00:00:00", "store_dtype": "float32",
            "model_dtype": "float32", "rope_applied": True, "head_sample_seed": 0,
            "verification": {"rel_output_error": 1e-7}}
    d.mkdir(parents=True, exist_ok=True)
    np.savez(data._layer_file(d, layer),
             q=b.q.float().numpy(), k=b.k.float().numpy(), v=b.v.float().numpy(),
             info=np.frombuffer(json.dumps(info).encode(), dtype=np.uint8))
    (d / "meta.json").write_text(json.dumps(info))
    return d


def _fake_capture(tmp_path, n_kv=2, groups=3):
    b = data.synthetic(n=12, d=8, dv=8, heads=n_kv * groups, passages=2, seed=5,
                       kv_groups=groups)
    return _write_layer(tmp_path / "capture", 3, b), b


def test_activation_cache_round_trip(tmp_path):
    d, b = _fake_capture(tmp_path)
    loaded = data.load_activations(d, 3, dtype=torch.float64)
    assert loaded.q.shape == b.q.shape and loaded.k.shape == b.k.shape
    assert torch.allclose(loaded.q, b.q, atol=1e-6)
    assert torch.allclose(loaded.v, b.v, atol=1e-6)
    assert loaded.layer == 3 and loaded.kv_groups == 3 and loaded.n_kv_heads == 2
    assert loaded.source == "fake/fake-data"
    assert loaded.scale == pytest.approx(b.scale)
    assert loaded.meta["rope_applied"] is True
    assert loaded.meta["model_repo"] == "fake/repo"
    assert loaded.meta["capture_dir"] == str(d)
    sub = data.load_activations(d, 3, heads=[0, 4], dtype=torch.float64)
    assert sub.heads == (0, 4) and sub.q.shape[1] == 2
    with pytest.raises(FileNotFoundError):
        data.load_activations(d, 99)


def test_get_activations_serves_from_cache_without_running_the_model(tmp_path, monkeypatch):
    """"Capture once, never re-run a model inside a study": a warm cache loads no model."""
    monkeypatch.setattr(data, "cache_dir", lambda: tmp_path)
    mspec = data.MODELS["gpt2"]
    heads = data._resolve_heads(None, mspec, 0)
    want = data.stratified_layers(mspec.n_layers, mspec.n_layer_sample)
    key, _ = data._capture_key("gpt2", "wikitext103", 32, 2, heads, "float32", "float32", 0)
    for li in want:
        b = data.synthetic(n=32, d=8, heads=len(heads), passages=2, seed=li, layer=li)
        _write_layer(tmp_path / key, li, b, model_key="gpt2", dataset_key="wikitext103")

    def boom(*a, **kw):
        raise AssertionError("a study must not re-run the model when the cache is warm")
    monkeypatch.setattr(data, "_load_model", boom)
    monkeypatch.setattr(data, "_load_tokenizer", boom)

    got = list(data.get_activations("gpt2", n_tokens=32, n_passages=2,
                                    dataset="wikitext103", dtype=torch.float64))
    assert [g.layer for g in got] == list(want)
    assert got[0].q.shape == (2, len(heads), 32, 8)
    assert got[0].source == "gpt2/wikitext103"
    rows = data.list_captures()
    assert len(rows) == 1 and rows[0]["model"] == "gpt2"
    assert rows[0]["layers"] == list(want)
    json.dumps(rows)


def test_cache_key_is_stable_and_discriminating():
    k1, s1 = data._capture_key("gpt2", "wikitext103", 1024, 64, (0, 1), "float32", "float32", 0)
    k2, _ = data._capture_key("gpt2", "wikitext103", 1024, 64, (0, 1), "float32", "float32", 0)
    k3, _ = data._capture_key("gpt2", "wikitext103", 1024, 64, (0, 2), "float32", "float32", 0)
    k4, _ = data._capture_key("gpt2", "pg19", 1024, 64, (0, 1), "float32", "float32", 0)
    assert k1 == k2 and k1 != k3 and k1 != k4
    assert s1["n_tokens"] == 1024 and s1["model"] == "gpt2"


def test_unknown_sources_raise_clear_errors():
    with pytest.raises(KeyError):
        next(data.get_activations(model="not-a-model"))
    with pytest.raises(KeyError):
        next(data.get_activations(model="gpt2", dataset="not-a-dataset"))


def test_offline_falls_back_to_synthetic(monkeypatch):
    """A study must still run when the model cannot be downloaded."""
    def boom(*a, **kw):
        raise data.DataUnavailableError("no network (simulated)")
    monkeypatch.setattr(data, "_load_tokenizer", boom)
    with pytest.warns(RuntimeWarning):
        batches = list(data.get_activations_or_synthetic(
            "gpt2", n_tokens=32, n_passages=2, fallback={"n": 32, "heads": 2, "passages": 2}))
    assert len(batches) == 1
    b = batches[0]
    assert b.source == "synthetic" and "fallback" in b.meta
    assert b.q.shape == (2, 2, 32, 64)


def test_describe_sources_lists_the_three_models():
    s = data.describe_sources()
    for key in ("gpt2", "qwen2.5-0.5b", "qwen2.5-7b-instruct"):
        assert key in s["models"]
    assert s["models"]["qwen2.5-7b-instruct"]["n_tokens"] >= 8192
    assert {"wikitext103", "pg19"} <= set(s["datasets"])
    json.dumps(s)


# ======================================================================================
# the capture is correct: it reproduces the model
# ======================================================================================


def _capture_available(model: str) -> str | None:
    """Return a skip reason, or None if the model can be captured here."""
    if os.environ.get("ATTNBANK_SKIP_MODEL_TESTS"):
        return "ATTNBANK_SKIP_MODEL_TESTS is set"
    try:
        import transformers  # noqa: F401
        import datasets      # noqa: F401
    except Exception as e:
        return f"transformers/datasets unavailable: {e}"
    return None


@pytest.mark.parametrize("model,dataset,n_tokens,tol", [
    ("gpt2", "wikitext103", 256, 1e-5),
    ("qwen2.5-0.5b", "wikitext103", 256, 1e-4),
])
def test_captured_activations_reproduce_the_model(model, dataset, n_tokens, tol):
    """softmax(QK^T/sqrt(d) + causal) V from the captured tensors == the model's own output.

    This is the test that catches a wrong scale, a missing RoPE, a transposed head
    axis, or a bad GQA group mapping. If it fails, every study built on the capture
    is wrong.
    """
    reason = _capture_available(model)
    if reason:
        pytest.skip(reason)
    try:
        report = data.verify_capture(model, n_tokens=n_tokens, n_passages=1, layers=2,
                                     dataset=dataset, tol=tol)
    except data.DataUnavailableError as e:
        pytest.skip(f"{model} not downloadable: {e}")
    assert report, "no layers were captured"
    for layer, r in report.items():
        assert r["rel_output_error"] < tol, (model, layer, r)
        assert r["scale"] == pytest.approx(r["shape"][-1] ** -0.5)
    if model == "qwen2.5-0.5b":
        some = next(iter(report.values()))
        assert some["kv_groups"] == 7 and some["n_kv_heads"] == 2   # GQA mapping exercised


def test_wrong_scale_would_be_caught():
    """The verification is sharp: a 2x scale error must blow past the tolerance.

    Guards the guard — it is why the model-match test above is meaningful.
    """
    b = data.synthetic(n=128, d=16, heads=2, passages=1, seed=0)
    mask = data._causal_mask(b.n, b.n)
    ref = torch.softmax(((b.q @ b.k.transpose(-1, -2)) * b.scale)
                        .masked_fill(~mask, float("-inf")), -1) @ b.v
    bad = torch.softmax(((b.q @ b.k.transpose(-1, -2)) * (2 * b.scale))
                        .masked_fill(~mask, float("-inf")), -1) @ b.v
    rel = float((bad - ref).norm() / ref.norm())
    assert rel > 0.1


# ======================================================================================
# the sweep runner
# ======================================================================================


def test_grid_and_zip_grid():
    g = H.grid(method=["topk", "performer"], k=[4, 8])
    assert g == [{"method": "topk", "k": 4}, {"method": "topk", "k": 8},
                 {"method": "performer", "k": 4}, {"method": "performer", "k": 8}]
    z = H.zip_grid(a=[1, 2], b=["x", "y"])
    assert z == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    with pytest.raises(ValueError):
        H.zip_grid(a=[1, 2], b=["x"])


def test_sweep_runs_every_job_once(tmp_path):
    calls = []
    sweep = H.Sweep("t_basic", cache_dir=tmp_path, progress=False, device="cpu")
    jobs = H.grid(a=[1, 2, 3])
    recs = sweep.run(jobs, lambda job: (calls.append(job["a"]), {"y": job["a"] ** 2})[1])
    assert calls == [1, 2, 3]
    assert [r["metrics"]["y"] for r in recs] == [1, 4, 9]
    assert all(r["_status"] == "ok" for r in recs)
    # re-running the same jobs hits the in-memory cache: no recomputation
    recs2 = sweep.run(jobs, lambda job: (calls.append(job["a"]), {"y": -1})[1])
    assert calls == [1, 2, 3]
    assert [r["metrics"]["y"] for r in recs2] == [1, 4, 9]


def test_sweep_is_resumable_after_a_crash(tmp_path):
    """A sweep that dies mid-way continues; finished jobs are not recomputed."""
    jobs = H.grid(a=[1, 2, 3, 4, 5])
    calls_1 = []

    def crashy(job):
        if job["a"] == 4:
            raise KeyboardInterrupt("simulated kill")
        calls_1.append(job["a"])
        return {"y": job["a"] * 10}

    s1 = H.Sweep("t_resume", cache_dir=tmp_path, progress=False, device="cpu")
    with pytest.raises(KeyboardInterrupt):
        s1.run(jobs, crashy)
    assert calls_1 == [1, 2, 3]
    assert H.Sweep("t_resume", cache_dir=tmp_path, progress=False,
                   device="cpu")._loaded_from_cache == 3

    calls_2 = []
    s2 = H.Sweep("t_resume", cache_dir=tmp_path, progress=False, device="cpu")
    recs = s2.run(jobs, lambda job: (calls_2.append(job["a"]), {"y": job["a"] * 10})[1])
    assert calls_2 == [4, 5]                       # only the unfinished tail re-ran
    assert [r["metrics"]["y"] for r in recs] == [10, 20, 30, 40, 50]
    assert len(recs) == 5

    # resume=False ignores the cache
    s3 = H.Sweep("t_resume", cache_dir=tmp_path, progress=False, device="cpu", resume=False)
    assert s3._loaded_from_cache == 0


def test_sweep_cache_survives_a_torn_line(tmp_path):
    s = H.Sweep("t_torn", cache_dir=tmp_path, progress=False, device="cpu")
    s.run(H.grid(a=[1, 2]), lambda job: {"y": job["a"]})
    with open(s.cache_file, "a") as f:
        f.write('{"job": {"a": 3}, "metri')      # killed mid-write
    s2 = H.Sweep("t_torn", cache_dir=tmp_path, progress=False, device="cpu")
    assert s2._loaded_from_cache == 2
    calls = []
    s2.run(H.grid(a=[1, 2, 3]), lambda job: (calls.append(job["a"]), {"y": job["a"]})[1])
    assert calls == [3]


def test_sweep_records_failures_without_stopping(tmp_path):
    def sometimes(job):
        if job["a"] == 2:
            raise ValueError("boom")
        return {"y": job["a"]}

    s = H.Sweep("t_fail", cache_dir=tmp_path, progress=False, device="cpu")
    recs = s.run(H.grid(a=[1, 2, 3]), sometimes)
    assert [r["_status"] for r in recs] == ["ok", "error", "ok"]
    assert "ValueError: boom" in recs[1]["_error"]
    assert len(s.failures()) == 1
    s2 = H.Sweep("t_fail2", cache_dir=tmp_path, progress=False, device="cpu",
                 on_error="raise")
    with pytest.raises(ValueError):
        s2.run(H.grid(a=[2]), sometimes)


def test_sweep_passes_a_context_when_the_function_wants_one(tmp_path):
    seen = {}

    def fn(job, ctx):
        seen["device"] = ctx.device
        seen["dtype"] = ctx.dtype
        seen["total"] = ctx.total
        g = ctx.generator(job["a"])
        return {"y": float(torch.rand(1, generator=g))}

    s = H.Sweep("t_ctx", cache_dir=tmp_path, progress=False, device="cpu", seed=11)
    recs = s.run(H.grid(a=[0, 1]), fn)
    assert seen["dtype"] is torch.float64          # float64 error measurement by default
    assert str(seen["device"]) == "cpu" and seen["total"] == 2
    # same seed -> same numbers
    s2 = H.Sweep("t_ctx2", cache_dir=tmp_path, progress=False, device="cpu", seed=11)
    recs2 = s2.run(H.grid(a=[0, 1]), fn)
    assert [r["metrics"]["y"] for r in recs] == [r["metrics"]["y"] for r in recs2]


def test_sweep_requires_a_dict_from_the_measurement(tmp_path):
    s = H.Sweep("t_type", cache_dir=tmp_path, progress=False, device="cpu")
    with pytest.raises(TypeError):
        s.run(H.grid(a=[1]), lambda job: 1.0)


def test_job_key_ignores_dict_ordering():
    assert H.job_key({"a": 1, "b": 2}) == H.job_key({"b": 2, "a": 1})
    assert H.job_key({"a": 1}) != H.job_key({"a": 2})


# ======================================================================================
# result files
# ======================================================================================


def test_save_and_load_results_round_trip(tmp_path):
    s = H.Sweep("t_save", cache_dir=tmp_path, progress=False, device="cpu", seed=3,
                extra_meta={"study": "S0", "data": {"source": "synthetic"}})
    recs = s.run(H.grid(method=["topk"], k=[4, 8]), lambda job: {"err": 1.0 / job["k"]})
    path = H.save_results("t_save", recs, s.meta(), out_dir=tmp_path)
    assert path.exists()
    got = H.load_results("t_save", out_dir=tmp_path)
    assert got["name"] == "t_save"
    assert len(got["records"]) == 2 == got["meta"]["n_records"]
    assert got["records"][0]["job"] == {"method": "topk", "k": 4}
    assert got["records"][0]["metrics"]["err"] == pytest.approx(0.25)
    m = got["meta"]
    for required in ("git_commit", "seed", "timestamp", "device", "torch", "dtype", "study"):
        assert required in m, required
    assert m["seed"] == 3 and m["data"]["source"] == "synthetic"
    assert m["dtype"] == "float64"
    with pytest.raises(FileNotFoundError):
        H.load_results("nope", out_dir=tmp_path)


def test_results_hold_the_full_sweep_not_just_plotted_points(tmp_path):
    jobs = H.grid(method=["a", "b"], k=[1, 2, 4, 8])
    s = H.Sweep("t_full", cache_dir=tmp_path, progress=False, device="cpu")
    recs = s.run(jobs, lambda job: {"err": job["k"] * 1.0})
    H.save_results("t_full", recs, s.meta(), out_dir=tmp_path)
    got = H.load_results("t_full", out_dir=tmp_path)
    assert len(got["records"]) == len(jobs) == 8


def test_provenance_block():
    p = H.provenance(seed=1)
    assert set(p) >= {"git_commit", "timestamp", "device", "torch", "platform", "dtype"}
    assert isinstance(p["git_commit"], str) and len(p["git_commit"]) >= 7
    json.dumps(p)


def test_to_rows_flattens():
    recs = [{"job": {"m": "topk", "k": 4}, "metrics": {"err": 0.5}, "_status": "ok"}]
    rows = H.to_rows(recs)
    assert rows == [{"m": "topk", "k": 4, "err": 0.5, "_status": "ok"}]


# ======================================================================================
# the summariser: median + IQR against hand-computed values
# ======================================================================================


def test_summarize_median_and_iqr_known_input():
    # values 1,2,3,4 -> median 2.5; linear-interpolated quartiles q1=1.75, q3=3.25
    recs = [{"job": {"g": "A"}, "metrics": {"e": x}, "_status": "ok"} for x in (1, 2, 3, 4)]
    (row,) = H.summarize(recs, by="g", value="e")
    assert row["g"] == "A" and row["n"] == 4
    assert row["median"] == pytest.approx(2.5)
    assert row["q1"] == pytest.approx(1.75)
    assert row["q3"] == pytest.approx(3.25)
    assert row["iqr"] == pytest.approx(1.5)
    assert row["mean"] == pytest.approx(2.5)
    assert row["min"] == 1.0 and row["max"] == 4.0
    assert row["std"] == pytest.approx(math.sqrt(5.0 / 3.0))


def test_summarize_odd_count_and_grouping():
    # A: 1,2,3,4,5 -> median 3, q1 2, q3 4 ; B: 10,20,30 -> median 20, q1 15, q3 25
    recs = ([{"job": {"g": "A"}, "metrics": {"e": x}, "_status": "ok"} for x in (1, 2, 3, 4, 5)]
            + [{"job": {"g": "B"}, "metrics": {"e": x}, "_status": "ok"} for x in (10, 20, 30)])
    rows = H.summarize(recs, by="g", value="e")
    assert [r["g"] for r in rows] == ["A", "B"]          # first-seen order
    assert rows[0]["median"] == 3 and rows[0]["q1"] == 2 and rows[0]["q3"] == 4
    assert rows[1]["median"] == 20 and rows[1]["q1"] == 15 and rows[1]["q3"] == 25


def test_summarize_multiple_keys_and_values():
    recs = [{"job": {"m": m, "k": k}, "metrics": {"e": float(k), "t": float(k * 2)},
             "_status": "ok"}
            for m in ("a", "b") for k in (1, 2)]
    rows = H.summarize(recs, by=("m", "k"), value=("e", "t"))
    assert len(rows) == 4
    r = rows[0]
    assert r["m"] == "a" and r["k"] == 1
    assert r["median_e"] == 1.0 and r["median_t"] == 2.0
    assert r["n_e"] == 1


def test_summarize_drops_failures_and_nonfinite():
    recs = [{"job": {"g": "A"}, "metrics": {"e": 1.0}, "_status": "ok"},
            {"job": {"g": "A"}, "metrics": {"e": 3.0}, "_status": "ok"},
            {"job": {"g": "A"}, "metrics": {"e": float("nan")}, "_status": "ok"},
            {"job": {"g": "A"}, "metrics": {"e": 99.0}, "_status": "error"}]
    (row,) = H.summarize(recs, by="g", value="e")
    assert row["n"] == 2 and row["n_dropped"] == 1 and row["median"] == pytest.approx(2.0)
    (row2,) = H.summarize(recs, by="g", value="e", ok_only=False)
    assert row2["n"] == 3 and row2["max"] == 99.0


def test_summarize_ungrouped_and_dotted_lookup():
    recs = [{"job": {"g": "A"}, "metrics": {"e": x}, "_status": "ok"} for x in (2, 4, 6)]
    (row,) = H.summarize(recs, value="metrics.e")
    assert row["median"] == pytest.approx(4.0) and row["n"] == 3
    (row2,) = H.summarize(recs, by="job.g", value="e")
    assert row2["job.g"] == "A"


def test_summarize_requires_a_value():
    with pytest.raises(ValueError):
        H.summarize([], by="g", value="")


# ======================================================================================
# GPU / memory plumbing
# ======================================================================================


def test_device_helpers():
    dev = H.default_device()
    assert isinstance(dev, torch.device)
    info = H.device_info(dev)
    assert info["device"] == str(dev) and "torch" in info
    json.dumps(info)
    assert H.free_memory(dev) > 0
    assert 0 < H.memory_budget(dev, 0.1) <= H.free_memory(dev)


def test_chunk_sizes_bounds_memory():
    budget = H.memory_budget("cpu", 1.0)
    assert H.chunk_sizes(1000, 1, "cpu", 1.0) == 1000          # everything fits
    assert H.chunk_sizes(1000, budget, "cpu", 1.0) == 1        # one at a time
    assert H.chunk_sizes(1000, 10 * budget, "cpu", 1.0) == 1   # never zero
    assert H.chunk_sizes(0, 10, "cpu") == 0
    items = list(range(10))
    assert [list(c) for c in H.iter_chunks(items, 4)] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_generator_and_set_seed_are_deterministic():
    a = torch.randn(4, generator=H.generator(5))
    b = torch.randn(4, generator=H.generator(5))
    assert torch.equal(a, b)
    H.set_seed(0)
    x = torch.randn(3)
    H.set_seed(0)
    assert torch.equal(x, torch.randn(3))


# ======================================================================================
# end-to-end: a miniature study on synthetic data
# ======================================================================================


def test_end_to_end_mini_study(tmp_path):
    """A study-shaped use of the harness: data -> sweep -> results -> summary."""
    from attnbank import metrics

    batch = data.synthetic(n=64, d=16, heads=4, passages=2, seed=0)
    units = list(batch.units(max_units=4))
    jobs = [{**tag, "k": k} for tag, _ in units for k in (4, 16)]

    def measure(job, ctx):
        tag_keys = ("source", "layer", "head", "kv_head", "passage")
        tag = {k: job[k] for k in tag_keys}
        unit = next(u for t, u in units if t == tag).to(ctx.device, ctx.dtype)
        s = (unit.q @ unit.k.transpose(-1, -2)) * unit.scale
        s = s.masked_fill(~data._causal_mask(unit.m, unit.n, s.device), float("-inf"))
        pi_ref = torch.softmax(s, dim=-1)
        y_ref = pi_ref @ unit.v
        topv, topi = s.topk(min(job["k"], unit.n), dim=-1)
        pi = torch.zeros_like(s).scatter_(-1, topi, torch.softmax(topv, -1))
        y = pi @ unit.v
        return {"output_error": float(metrics.output_error(y_ref, y).median()),
                "tv": float(metrics.total_variation(pi_ref, pi).median())}

    sweep = H.Sweep("t_mini", cache_dir=tmp_path, progress=False, seed=0,
                    extra_meta={"data": batch.provenance()})
    recs = sweep.run(jobs, measure)
    assert all(r["_status"] == "ok" for r in recs), [r["_error"] for r in recs
                                                     if r["_status"] != "ok"]
    rows = H.summarize(recs, by="k", value="output_error")
    assert len(rows) == 2
    assert rows[1]["median"] < rows[0]["median"]      # more budget, less error
    assert all(r["q1"] <= r["median"] <= r["q3"] for r in rows)
    path = sweep.save("t_mini")
    try:
        got = H.load_results("t_mini")
        assert got["meta"]["data"]["source"] == "synthetic"
        assert len(got["records"]) == len(jobs)
    finally:
        path.unlink(missing_ok=True)
