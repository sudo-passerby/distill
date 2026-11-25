"""Score pplx-decider-flash-v1.3 (not trained on Banking77) with one 77-option `choice` question per message,
on the same 500 test messages as baseline.py. Run: python -u decider_baseline.py"""
import csv, json, random, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download

ck = Path(snapshot_download("perplexity-ai/pplx-decider-flash-v1.3"))
sys.path.insert(0, str(ck / "source" / "src"))
from pplx_decider import Decider
import torch; torch.backends.cudnn.enabled = False  # MIOpen finds no conv algo for long prompts on gfx1151

test = [(r["text"], r["category"]) for r in csv.DictReader(open("data/test.csv"))]
labels = sorted({r["category"] for r in csv.DictReader(open("data/train.csv"))})
random.seed(0); sample = random.sample(test, 500)  # same sample as baseline.py

question = {"type": "choice", "instructions": "Which category does this bank customer's message belong to?",
            "criteria": {c: c.replace("_", " ") for c in labels}}
d = Decider.from_pretrained(ck, device="cuda")
ask = lambda text: d.decide({"state": text, "questions": {"category": question}})["answers"]["category"]

ask(sample[0][0])  # warm up
correct, t0, rows = 0, time.time(), []
for i, (text, gold) in enumerate(sample, 1):
    a = ask(text); correct += a["choice"] == gold
    rows.append({"text": text, "gold": gold, "choice": a["choice"], "confidence": a["confidence"]})
    if i % 50 == 0: print(f"{i}/500 acc {correct / i:.3f} ({(time.time() - t0) / i * 1000:.0f} ms each)", flush=True)

result = {"model": "pplx-decider-flash-v1.3", "accuracy": correct / 500, "n": 500,
          "latency_ms_single": round((time.time() - t0) / 500 * 1000)}
print(json.dumps(result, indent=2))
json.dump(result, open("results/baseline_decider.json", "w"), indent=2)
json.dump(rows, open("results/decider_preds.json", "w"))
