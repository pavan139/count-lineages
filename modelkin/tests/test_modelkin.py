import csv
import os
import numpy as np
import modelkin as mk
from modelkin import cli

HERE = os.path.dirname(os.path.abspath(__file__))


def sim(M, J, seed, fam=None, sigma=0.0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=M); a = rng.lognormal(0, 0.35, J); b = rng.normal(0, 1, J)
    eta = a[None, :] * (theta[:, None] - b[None, :])
    if fam is not None and sigma:
        eta = eta + sigma * rng.normal(size=(fam.max() + 1, J))[fam]
    return (rng.random((M, J)) < 1 / (1 + np.exp(-eta))).astype(np.int8)


def test_effective_clusters():
    assert mk.effective_clusters(["a"] * 10) == 1.0
    assert mk.effective_clusters(list("abcdefghij")) == 10.0
    assert abs(mk.effective_clusters(["a"] * 6 + ["b"] * 2 + [None, None]) - 100 / 42) < 1e-9


def test_panel_card_counts_unassigned_as_singletons():
    c = mk.panel_card(["a", "a", "b", None, ""])
    assert (c["models"], c["families"], c["unassigned"], c["largest_family"]) == (5, 2, 2, 2)


def test_independent_models_show_no_inflation():
    X = sim(300, 600, 0)
    fam = np.repeat(np.arange(60), 5)              # arbitrary groups of independent models
    infl = mk.variance_inflation(X, fam)
    assert 0.8 < np.nanmedian(infl["difficulty"]) < 1.2
    assert 0.8 < np.nanmedian(infl["discrimination"]) < 1.2


def test_exact_copies_recover_kish_design_effect():
    base = sim(100, 600, 1)
    mult = np.random.default_rng(2).integers(1, 8, size=100)
    X = np.repeat(base, mult, axis=0); fam = np.repeat(np.arange(100), mult)
    kish = (mult ** 2).sum() / mult.sum()
    assert abs(mk.panel_card(fam)["kish_design_effect"] - kish) < 1e-9
    med = np.nanmedian(mk.variance_inflation(X, fam)["difficulty"])
    assert 0.7 * kish < med < 1.3 * kish


def test_family_bootstrap_is_invariant_to_duplication():
    base = sim(150, 300, 3)
    one = mk.family_bootstrap(base, np.arange(150), "item_rest", B=150, seed=0)["se"]
    four = mk.family_bootstrap(np.repeat(base, 4, axis=0), np.repeat(np.arange(150), 4), "item_rest", B=150, seed=0)["se"]
    assert 0.85 < np.nanmedian(four / one) < 1.15


def test_family_bootstrap_widens_under_families():
    fam = np.repeat(np.arange(20), 15)
    X = sim(300, 300, 4, fam=fam, sigma=1.5)
    naive = mk.family_bootstrap(X, np.arange(300), "accuracy", B=150, seed=0)["se"]
    clustered = mk.family_bootstrap(X, fam, "accuracy", B=150, seed=0)["se"]
    assert np.median(clustered / naive) > 1.5


def test_similarity_detects_families():
    fam = np.repeat(np.arange(30), 10)
    s = mk.similarity_by_family(sim(300, 800, 5, fam=fam, sigma=1.5), fam)
    assert s["same_family"] > 0.05 > s["different_family"]


def test_screen_flags_fewer_items_with_families():
    fam = np.repeat(np.arange(15), 20)
    rng = np.random.default_rng(6)
    X = sim(300, 400, 6, fam=fam, sigma=1.5)
    sc = mk.screen_items(X, fam, B=150)
    assert sc["flag_family"].sum() <= sc["flag_model"].sum()
    assert np.nanmedian(sc["se_ratio"]) > 1.0


def test_screen_matches_model_bootstrap_without_families():
    X = sim(300, 300, 9)
    sc = mk.screen_items(X, np.arange(300), B=150)
    assert 0.9 < np.nanmedian(sc["se_ratio"]) < 1.1


def test_leave_family_out_gap_is_positive_with_families():
    fam = np.repeat(np.arange(30), 10)
    r = mk.leave_family_out(sim(300, 300, 7, fam=fam, sigma=1.5), fam, seeds=6, controls=2)
    assert r["gap"] > 0


def test_config_and_name_families():
    assert mk.config_family({"cfg": {"architectures": ["MistralForCausalLM"], "hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000}}) == "mistral-7b"
    assert mk.config_family({"cfg": {"architectures": ["LlamaForCausalLM"], "hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 128256}}) == "llama3-8b"
    assert mk.config_family({"cfg": {"architectures": ["LlamaForCausalLM"], "hidden_size": 5120, "num_hidden_layers": 40, "vocab_size": 32000}}) == "llama-13b"
    assert mk.config_family({}) is None
    assert mk.name_family("someone/zephyr-7b-beta") == "mistral-7b"
    assert mk.name_family("someone/WizardLM-13B-V1.2") == "llama-13b"
    assert mk.name_family("someone/mystery-model") is None
    same_shape = {"architectures": ["LlamaForCausalLM"], "hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000}
    assert mk.config_family({"cfg": dict(same_shape, intermediate_size=14336)}) == "mistral-7b"
    assert mk.config_family({"cfg": dict(same_shape, intermediate_size=11008)}) == "llama-7b"
    assert mk.config_family({"cfg": dict(same_shape, vocab_size=151936, intermediate_size=11008)}) == "qwen-7b"


def test_assign_families_follows_declared_root():
    meta = {"u/child": {"base_model": ["u/parent"]},
            "u/parent": {"cfg": {"architectures": ["MistralForCausalLM"], "hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000}}}
    out = mk.assign_families(["u/child", "u/unknown-thing"], meta)
    assert out["u/child"] == ("mistral-7b", "declared-root")
    assert out["u/unknown-thing"] == (None, "unknown")


def test_labels_roundtrip_and_shipped_files(tmp_path=None):
    import tempfile
    d = tmp_path or tempfile.mkdtemp()
    path = os.path.join(str(d), "labels.csv")
    mk.write_labels(path, {"a/x": ("mistral-7b", "signature"), "b/y": (None, "unknown")})
    assert mk.read_labels(path, ["b/y", "a/x", "c/z"]) == [None, "mistral-7b", None]
    shipped = os.path.join(HERE, "..", "data", "open_llm_leaderboard_v1_families.csv")
    rows = list(csv.DictReader(open(shipped)))
    assert len(rows) == 1151 and sum(r["family"] == "mistral-7b" for r in rows) == 332


def test_cli_card_and_audit(tmp_path=None, capsys=None):
    import tempfile
    d = str(tmp_path or tempfile.mkdtemp())
    fam = np.repeat(np.arange(20), 5)
    X = sim(100, 200, 8, fam=fam, sigma=1.0)
    models = np.array([f"org/m{i}" for i in range(100)], dtype=object)
    np.savez(os.path.join(d, "m.npz"), X=X, models=models)
    mk.write_labels(os.path.join(d, "l.csv"), {m: (f"fam{f}", "signature") for m, f in zip(models, fam)})
    assert cli.main(["card", "--matrix", os.path.join(d, "m.npz"), "--labels", os.path.join(d, "l.csv")]) == 0
    assert cli.main(["audit", "--matrix", os.path.join(d, "m.npz"), "--labels", os.path.join(d, "l.csv"), "--bootstrap", "50", "--json"]) == 0
