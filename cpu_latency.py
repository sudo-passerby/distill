"""Latency of the trained 0.6B classifier on CPU only, batch 1, same 200 messages as train.py's GPU number."""
import csv, json, time, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os; torch.set_num_threads(int(os.environ.get("THREADS", "8")))
tok = AutoTokenizer.from_pretrained("out/tiny")
model = AutoModelForSequenceClassification.from_pretrained("out/tiny", torch_dtype=torch.float32).eval()
test = [r["text"] for r in csv.DictReader(open("data/test.csv"))][:200]
with torch.no_grad():
    for t in test[:10]: model(**tok(t, return_tensors="pt"))  # warm up
    t0 = time.time()
    for t in test: model(**tok(t, return_tensors="pt"))
ms = (time.time() - t0) / len(test) * 1000
r = {"device": f"cpu ({torch.get_num_threads()} threads, Ryzen AI MAX+ 392)", "dtype": "float32", "latency_ms_single": round(ms, 1)}
print(json.dumps(r)); json.dump(r, open("results/tiny_cpu.json", "w"), indent=2)
