"""Fetch only config.json for the newer leaderboard's models (enough for a configuration signature).
Resumable; reuses records already fetched by lineage_fetch.py."""
import json, os, ssl, time, urllib.error, urllib.request
import numpy as np
CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or ("/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else None))
KEEP = ["architectures", "model_type", "hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size", "n_embd", "n_layer", "d_model", "num_local_experts"]
models = [str(m) for m in np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external_routereval", "v2_models.npz"), allow_pickle=True)["models"]]
out = "v2_configs.json"
done = json.load(open(out)) if os.path.exists(out) else {}
if os.path.exists("lineage_raw_v2.json"):
    for r in json.load(open("lineage_raw_v2.json")):
        if r["model"] in set(models) and r["model"] not in done and r.get("api_status") in (200, 401, 404):
            done[r["model"]] = {"status": r.get("cfg_status") or r["api_status"], "cfg": r.get("cfg")}
todo = [m for m in models if m not in done]
print("already have", len(done), "| to fetch", len(todo), flush=True)
for i, m in enumerate(todo, 1):
    rec = {"status": -1, "cfg": None}
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/{m}/resolve/main/config.json", headers={"User-Agent": "python-research-script"}), timeout=30, context=CTX) as r:
                cfg = json.loads(r.read()); rec = {"status": 200, "cfg": {k_: cfg[k_] for k_ in KEEP if k_ in cfg}}; break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(45); continue
            rec = {"status": e.code, "cfg": None}; break
        except Exception:
            time.sleep(3 * (k + 1))
    done[m] = rec
    time.sleep(0.25)
    if i % 200 == 0:
        json.dump(done, open(out, "w")); print(i, "fetched", flush=True)
json.dump(done, open(out, "w"))
print("DONE", len(done), "| readable", sum(v["status"] == 200 for v in done.values()), flush=True)
