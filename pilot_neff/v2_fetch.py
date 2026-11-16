"""Read the Open LLM Leaderboard v2 results table (submission-time metadata, no responses)
through the public datasets-server API. One small JSON page per request."""
import os
import json, ssl, time, urllib.request
CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or ("/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else None))
KEEP = ["fullname", "Model", "Base Model", "Architecture", "#Params (B)", "Type", "Merged", "MoE", "Flagged",
        "Submission Date", "Upload To Hub Date", "Generation", "Available on the hub", "Official Providers", "Average ⬆️"]
if __name__ == "__main__":
    rows, off = [], 0
    while True:
        url = f"https://datasets-server.huggingface.co/rows?dataset=open-llm-leaderboard%2Fcontents&config=default&split=train&offset={off}&length=100"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "python-research-script"}), timeout=40, context=CTX) as r:
                    d = json.loads(r.read()); break
            except Exception as e:
                time.sleep(5 * (attempt + 1)); d = None
        if not d or not d.get("rows"):
            break
        if off == 0:
            print("columns:", [f["name"] for f in d["features"]]); print("total rows:", d.get("num_rows_total"))
        rows += [{k: x["row"].get(k) for k in KEEP} for x in d["rows"]]
        off += 100
        time.sleep(0.4)
        if off >= d.get("num_rows_total", 0):
            break
    json.dump(rows, open("v2_contents.json", "w"))
    print("fetched", len(rows)); print(rows[0])
    # keep going until every row is present, then drop the HTML link column
    while len(rows) < 4576:
        url = f"https://datasets-server.huggingface.co/rows?dataset=open-llm-leaderboard%2Fcontents&config=default&split=train&offset={len(rows)}&length=100"
        d = None
        for attempt in range(8):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "python-research-script"}), timeout=40, context=CTX) as r:
                    d = json.loads(r.read()); break
            except Exception:
                time.sleep(10 * (attempt + 1))
        if not d or not d.get("rows"):
            break
        rows += [{k: x["row"].get(k) for k in KEEP} for x in d["rows"]]
        time.sleep(0.8)
    for r_ in rows:
        r_.pop("Model", None)
    json.dump(rows, open("v2_contents.json", "w"))
    print("total rows saved", len(rows))
