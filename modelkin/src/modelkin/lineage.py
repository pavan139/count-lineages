"""Assign models to base-model families from repository metadata (never from responses)."""
import csv
import json
import re
import ssl
import time
import urllib.error
import urllib.request

_CFG_KEYS = ["_name_or_path", "architectures", "model_type", "hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size",
             "n_embd", "n_layer", "d_model", "num_local_experts"]
_LLAMA_SHAPES = {(4096, 32): "llama-7b", (5120, 40): "llama-13b", (6656, 60): "llama-30b", (8192, 80): "llama-70b",
                 (4096, 48): "solar-10.7b", (7168, 60): "yi-34b", (8192, 48): "codellama-34b", (2048, 22): "tinyllama-1.1b"}


def config_family(record):
    """Family from architecture class, hidden size, layer count and vocabulary bucket."""
    cfg = record.get("cfg") or {}
    archs = cfg.get("architectures") or ([record["arch"]] if record.get("arch") else None)
    h = cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model")
    L = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    v = cfg.get("vocab_size")
    if not archs or not h or not L:
        return None
    arch = archs[0]
    inter = cfg.get("intermediate_size")
    if arch in ("MistralForCausalLM", "LlamaForCausalLM") and (h, L) == (4096, 32):
        # four base models share this shape; vocabulary and feed-forward width tell them apart
        if v and v >= 150000:
            return "qwen-7b"
        if v and v >= 120000:
            return "llama3-8b"
        if v and 60000 <= v < 70000 and inter != 14336 and arch == "LlamaForCausalLM":
            return "yi-6b"
        if inter == 14336:
            return "mistral-7b"
        if inter == 11008:
            return "llama-7b"
        return "mistral-7b" if arch == "MistralForCausalLM" else "llama-7b"
    if arch == "MixtralForCausalLM":
        return "mixtral-8x7b" if cfg.get("num_local_experts", 8) == 8 and (h, L) == (4096, 32) else f"mistral-moe-{cfg.get('num_local_experts')}x"
    if arch == "LlamaForCausalLM":
        lab = _LLAMA_SHAPES.get((h, L))
        if lab == "llama-70b" and v and v >= 120000:
            lab = "llama3-70b"
        if lab and lab != "llama-7b":
            return lab
    return f"{arch}|h{h}|L{L}|v{(v or 0) // 1000}k"


def name_family(model_id):
    """Family guessed from tokens in the repository name. Lowest-confidence signal."""
    s = model_id.lower()
    if "mixtral" in s or "8x7b" in s: return "mixtral-8x7b"
    if "solar" in s or "10.7b" in s or re.search(r"\b11b\b", s): return "solar-10.7b"
    if "tinyllama" in s or "1.1b" in s: return "tinyllama-1.1b"
    if "codellama" in s and "34b" in s: return "codellama-34b"
    if "34b" in s and "yi" in s: return "yi-34b"
    if re.search(r"70b|65b", s): return "llama-70b"
    if re.search(r"30b|33b", s): return "llama-30b"
    if "13b" in s: return "llama-13b"
    if re.search(r"mistral|zephyr|openhermes-2|neural-chat-7b|openchat[-_]3|starling|dolphin-2\.\d-mistral", s): return "mistral-7b"
    if "7b" in s and re.search(r"llama|vicuna|alpaca|wizardlm|guanaco|orca|platypus|airoboros-l2|beluga", s): return "llama-7b"
    return None


def _parents(record, model):
    p = list(record.get("base_model") or [])
    if record.get("adapter_base"):
        p.append(record["adapter_base"])
    return [q for q in p if isinstance(q, str) and "/" in q and q != model]


def assign_families(models, metadata):
    """Return {model: (family or None, source)}; source is signature, declared-root, name or unknown.

    `metadata` maps repository id to a record produced by fetch_metadata."""
    def root(m, depth=0):
        ps = _parents(metadata.get(m, {}), m)
        return m if (not ps or depth > 6) else root(ps[0], depth + 1)
    out = {}
    for m in models:
        s = config_family(metadata.get(m, {}))
        if s:
            out[m] = (s, "signature"); continue
        rt = root(m)
        lab = (config_family(metadata.get(rt, {})) or name_family(rt)) if rt != m else None
        if lab:
            out[m] = (lab, "declared-root"); continue
        nm = name_family(m)
        out[m] = (nm, "name") if nm else (None, "unknown")
    return out


def _get(url, ctx, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "modelkin"})
            with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(60 * (k + 1)); continue
            return e.code, b""
        except Exception:
            time.sleep(3 * (k + 1))
    return -1, b""


def fetch_metadata(models, pause=0.6, hops=4, cafile=None, progress=None):
    """Query public Hugging Face metadata for each model and its declared ancestors.

    Anonymous and rate-limited by `pause` seconds per model. Returns {repository id: record}."""
    ctx = ssl.create_default_context(cafile=cafile)
    done, queue, seen = {}, [(m, 0) for m in models], set(models)
    while queue:
        m, hop = queue.pop(0)
        rec = {"model": m}
        code, body = _get(f"https://huggingface.co/api/models/{m}", ctx)
        rec["api_status"] = code
        if code == 200:
            d = json.loads(body)
            bm = (d.get("cardData") or {}).get("base_model")
            rec["base_model"] = [bm] if isinstance(bm, str) else (bm if isinstance(bm, list) else None)
            rec["arch"] = ((d.get("config") or {}).get("architectures") or [None])[0]
            rec["created"] = d.get("createdAt")
            files = [s.get("rfilename", "") for s in d.get("siblings", [])]
            c2, b2 = _get(f"https://huggingface.co/{m}/resolve/main/config.json", ctx)
            if c2 == 200:
                try:
                    cfg = json.loads(b2); rec["cfg"] = {k: cfg[k] for k in _CFG_KEYS if k in cfg}
                except ValueError:
                    pass
            if "adapter_config.json" in files:
                c3, b3 = _get(f"https://huggingface.co/{m}/resolve/main/adapter_config.json", ctx)
                if c3 == 200:
                    try:
                        rec["adapter_base"] = json.loads(b3).get("base_model_name_or_path")
                    except ValueError:
                        pass
        done[m] = rec
        if progress:
            progress(len(done), len(queue))
        if hop < hops:
            for p in _parents(rec, m):
                if not p.startswith(("/", ".")) and p not in seen:
                    seen.add(p); queue.append((p, hop + 1))
        time.sleep(pause)
    return done


def write_labels(path, assignment):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "family", "source"])
        for m, (fam, src) in assignment.items():
            w.writerow([m, fam or "", src])


def read_labels(path, models=None):
    """Read a labels CSV. With `models`, return the family of each model in that order (None if absent)."""
    with open(path, newline="") as f:
        d = {r["model"]: (r["family"] or None) for r in csv.DictReader(f)}
    return d if models is None else [d.get(str(m)) for m in models]
