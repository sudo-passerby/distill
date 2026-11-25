"""Fine-tune a tiny model (Qwen3-0.6B + classification head) on Banking77, then score it on the test set.
Run: python train.py  (needs torch with GPU support and transformers 5.x)"""
import csv, json, random, time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "Qwen/Qwen3-0.6B"
EPOCHS, BATCH, LR, MAXLEN = 3, 32, 3e-5, 64
random.seed(0); torch.manual_seed(0)

def load(split):
    return [(r["text"], r["category"]) for r in csv.DictReader(open(f"data/{split}.csv"))]

train, test = load("train"), load("test")
labels = sorted({c for _, c in train})
lid = {c: i for i, c in enumerate(labels)}

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(labels), torch_dtype=torch.bfloat16)
model.config.pad_token_id = tok.pad_token_id
model.cuda()

def batches(rows, shuffle):
    rows = rows[:]
    if shuffle: random.shuffle(rows)
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        enc = tok([t for t, _ in chunk], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to("cuda")
        yield enc, torch.tensor([lid[c] for _, c in chunk], device="cuda")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * ((len(train) + BATCH - 1) // BATCH)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)

t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    for i, (enc, y) in enumerate(batches(train, True)):
        loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if i % 50 == 0: print(f"epoch {ep} step {i} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
train_s = time.time() - t0

model.eval(); correct, preds = 0, []
with torch.no_grad():
    for enc, y in batches(test, False):
        p = model(**enc).logits.argmax(-1)
        correct += (p == y).sum().item(); preds += p.tolist()
    # Latency the way a customer calls it: one message at a time.
    sample = [t for t, _ in test[:200]]
    torch.cuda.synchronize(); t1 = time.time()
    for t in sample: model(**tok(t, return_tensors="pt").to("cuda"))
    torch.cuda.synchronize(); ms = (time.time() - t1) / len(sample) * 1000

result = {"model": BASE, "test_accuracy": correct / len(test), "test_n": len(test),
          "train_seconds": round(train_s), "latency_ms_single": round(ms, 1)}
print(json.dumps(result, indent=2))
json.dump(result, open("results/tiny.json", "w"), indent=2)
json.dump([labels[p] for p in preds], open("results/tiny_preds.json", "w"))
model.save_pretrained("out/tiny"); tok.save_pretrained("out/tiny")
json.dump(labels, open("out/tiny/labels.json", "w"))
