"""Recompute every model's accuracy from its saved answers and check it against the reported score.
Run: python3 verify.py   (stdlib only, no model or API key needed)"""
import csv, json, random

test = [(r["text"], r["category"]) for r in csv.DictReader(open("data/test.csv"))]
random.seed(0); idx = random.sample(range(len(test)), 500)  # the same 500 messages every model got
gold = [test[i][1] for i in idx]

def check(name, answers, reported):
    acc = sum(a == g for a, g in zip(answers, gold)) / 500
    assert len(answers) == 500 and abs(acc - reported) < 1e-9, f"{name}: {acc} != {reported}"
    print(f"{name:28} {acc:6.1%}  matches")

tiny = json.load(open("results/tiny_preds.json"))
full = sum(p == t[1] for p, t in zip(tiny, test)) / len(test)
assert abs(full - json.load(open("results/tiny.json"))["test_accuracy"]) < 1e-9
print(f"{'tiny, full test (3,080)':28} {full:6.1%}  matches")
check("tiny (Qwen3-0.6B trained)", [tiny[i] for i in idx], sum(tiny[i] == g for i, g in zip(idx, gold)) / 500)

d = json.load(open("results/decider_preds.json"))
assert [x["text"] for x in d] == [test[i][0] for i in idx], "decider answers are for different messages"
check("pplx-decider", [x["choice"] for x in d], json.load(open("results/baseline_decider.json"))["accuracy"])

for m in ["gpt-6-luna", "gemini-3.5-flash-lite", "decisions_gpt-6-luna", "gemma4_e4b", "qwen3.5_4b"]:
    check(m, json.load(open(f"results/preds_{m}.json")), json.load(open(f"results/baseline_{m}.json"))["accuracy"])
