"""Read published label-error annotations through the public datasets-server API (JSON pages)."""
import os
import json, ssl, sys, time, urllib.error, urllib.parse, urllib.request
CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or ("/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else None))


def get(url):
    for k in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "python-research-script"}), timeout=40, context=CTX) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            err = e; time.sleep(75 if e.code == 429 else 4 * (k + 1))
        except Exception as e:
            err = e; time.sleep(4 * (k + 1))
    raise err


def splits(ds):
    return get("https://datasets-server.huggingface.co/splits?dataset=" + urllib.parse.quote(ds, safe=""))["splits"]


def rows(ds, config, split, keep=None):
    out, off = [], 0
    while True:
        d = get(f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(ds, safe='')}&config={urllib.parse.quote(config)}&split={split}&offset={off}&length=100")
        if not d.get("rows"):
            break
        out += [{k: v for k, v in x["row"].items() if keep is None or k in keep} for x in d["rows"]]
        off += 100
        time.sleep(1.3)
        if off >= d.get("num_rows_total", 0):
            break
    return out


if __name__ == "__main__":
    ds = sys.argv[1]
    sp = splits(ds)
    print("splits:", [(s["config"], s["split"]) for s in sp][:8], "... total", len(sp))
    if len(sys.argv) > 2:
        cfg, split, out = sys.argv[2], sys.argv[3], sys.argv[4]
        r = rows(ds, cfg, split)
        json.dump(r, open(out, "w"))
        print("rows", len(r), "| fields", list(r[0]))
        print({k: str(v)[:90] for k, v in r[0].items()})
