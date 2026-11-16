"""Fetch public Hugging Face metadata that could identify model lineage.

Per model: API record (declared base model, tags, architecture, parameter count,
file list, creation date) and config.json (_name_or_path, layer dimensions).
Then follows declared parents that are outside the panel, up to 4 hops.
Polite pacing; anonymous; resumable.
"""
import os
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
import numpy as np

UA = {"User-Agent": "python-research-script"}
CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or ("/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else None))   # system CA bundle; verification stays on
KEEP_CFG = ["_name_or_path", "architectures", "model_type", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "intermediate_size", "vocab_size", "n_embd", "n_layer",
            "n_head", "d_model", "num_local_experts", "max_position_embeddings"]


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30, context=CTX) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(60 * (k + 1))
                continue
            return e.code, b""
        except Exception:
            time.sleep(3 * (k + 1))
    return -1, b""


def probe(m):
    rec = {"model": m}
    code, body = get(f"https://huggingface.co/api/models/{m}")
    rec["api_status"] = code
    if code == 200:
        d = json.loads(body)
        cd = d.get("cardData") or {}
        bm = cd.get("base_model")
        rec["base_model"] = [bm] if isinstance(bm, str) else (bm if isinstance(bm, list) else None)
        rec["tags"] = [t for t in d.get("tags", []) if t.startswith("base_model") or t in ("merge", "mergekit", "moe", "peft", "lora")]
        rec["arch"] = ((d.get("config") or {}).get("architectures") or [None])[0]
        rec["model_type"] = (d.get("config") or {}).get("model_type")
        rec["params"] = (d.get("safetensors") or {}).get("total")
        rec["created"] = d.get("createdAt")
        files = [s.get("rfilename", "") for s in d.get("siblings", [])]
        rec["has_adapter"] = "adapter_config.json" in files
        rec["has_mergekit"] = any("mergekit" in f.lower() for f in files)
        rec["id"] = d.get("id")
        c2, b2 = get(f"https://huggingface.co/{m}/resolve/main/config.json")
        rec["cfg_status"] = c2
        if c2 == 200:
            try:
                cfg = json.loads(b2)
                rec["cfg"] = {k: cfg[k] for k in KEEP_CFG if k in cfg}
            except Exception:
                pass
        if rec["has_adapter"]:
            c3, b3 = get(f"https://huggingface.co/{m}/resolve/main/adapter_config.json")
            if c3 == 200:
                try:
                    rec["adapter_base"] = json.loads(b3).get("base_model_name_or_path")
                except Exception:
                    pass
    return rec


def main(matrix, out, hops=4):
    out = Path(out)
    done = {r["model"]: r for r in json.loads(out.read_text())} if out.exists() else {}
    panel = [str(m) for m in np.load(matrix, allow_pickle=True)["models"]]
    queue = [(m, 0) for m in panel]
    seen = set(panel)
    n = 0
    while queue:
        m, hop = queue.pop(0)
        if m not in done:
            r = probe(m)
            r["in_panel"] = m in set(panel)
            r["hop"] = hop
            done[m] = r
            n += 1
            time.sleep(0.62)
            if n % 50 == 0:
                out.write_text(json.dumps(list(done.values())))
                print(n, "fetched;", len(queue), "queued", flush=True)
        r = done[m]
        if hop < hops:
            parents = list(r.get("base_model") or [])
            if r.get("adapter_base"):
                parents.append(r["adapter_base"])
            for p in parents:
                if isinstance(p, str) and "/" in p and not p.startswith(("/", ".")) and p not in seen:
                    seen.add(p)
                    queue.append((p, hop + 1))
    out.write_text(json.dumps(list(done.values())))
    print("DONE", len(done), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
