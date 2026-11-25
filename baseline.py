"""Score a general model (via Ollama), told the 77 categories but never trained on them, on the Banking77 test set.
Run: python3 baseline.py <ollama-model> [n]   e.g. python3 baseline.py gemma4:e4b 500"""
import csv, json, random, sys, time, urllib.request

model, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 500
test = [(r["text"], r["category"]) for r in csv.DictReader(open("data/test.csv"))]
labels = sorted({r["category"] for r in csv.DictReader(open("data/train.csv"))})
random.seed(0); sample = random.sample(test, n)  # same sample for every model

system = ("You sort bank customer messages into exactly one category. Categories:\n" + "\n".join(labels) +
          '\nReply as JSON: {"category": "<one category from the list>"}')
schema = {"type": "object", "properties": {"category": {"type": "string", "enum": labels}}, "required": ["category"]}

def ask(text):
    body = json.dumps({"model": model, "stream": False, "format": schema, "think": False, "options": {"temperature": 0},
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", body, {"Content-Type": "application/json"})
    return json.loads(json.load(urllib.request.urlopen(req, timeout=120))["message"]["content"])["category"]

ask(sample[0][0])  # warm up: load the model and cache the long system prompt
correct, t0, preds = 0, time.time(), []
for i, (text, gold) in enumerate(sample, 1):
    p = ask(text); preds.append(p); correct += p == gold
    if i % 50 == 0: print(f"{i}/{n} acc {correct / i:.3f} ({(time.time() - t0) / i * 1000:.0f} ms each)", flush=True)

result = {"model": model, "accuracy": correct / n, "n": n, "latency_ms_single": round((time.time() - t0) / n * 1000)}
print(json.dumps(result, indent=2))
json.dump(result, open(f"results/baseline_{model.replace(':', '_')}.json", "w"), indent=2)
json.dump(preds, open(f"results/preds_{model.replace(':', '_')}.json", "w"))
