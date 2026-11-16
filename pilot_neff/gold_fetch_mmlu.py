import json, os, gold_fetch as g
cfgs = sorted(json.load(open("gold_mmlu_redux.json")))
path = "gold_mmlu_original_questions.json"
orig = json.load(open(path)) if os.path.exists(path) else {}
for c in cfgs:
    if c in orig:
        continue
    orig[c] = [r["question"] for r in g.rows("cais/mmlu", c, "test", keep={"question"})]
    json.dump(orig, open(path, "w"))
    print(c, len(orig[c]), flush=True)
print("DONE", sum(len(v) for v in orig.values()))
