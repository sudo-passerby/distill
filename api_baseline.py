"""Score a hosted API model (told the 77 categories, never trained on them) on the same 500 test messages.
Run: SECRETS=<json with OPENAI_API_KEY / GEMINI_API_KEY> python3 api_baseline.py <model> [n]
Prints the real token usage and cost; ask for n=20 first to check spend."""
import csv, json, os, random, sys, time, urllib.error, urllib.request

# USD per 1M tokens (input, output), from the official pricing pages on 2026-11-26. Gemini thinking tokens bill as output.
PRICES = {"gpt-6-luna": (0.10, 0.50), "decisions:gpt-6-luna": (0.10, 0.0), "gemini-3.8-flash": (1.50, 7.50), "gemini-3.5-flash-lite": (0.30, 2.50)}

model, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 500
keys = json.load(open(os.environ["SECRETS"]))
test = [(r["text"], r["category"]) for r in csv.DictReader(open("data/test.csv"))]
labels = sorted({r["category"] for r in csv.DictReader(open("data/train.csv"))})
random.seed(0); sample = random.sample(test, 500)[:n]  # same 500 as baseline.py; n<500 takes a prefix

system = ("You sort bank customer messages into exactly one category. Categories:\n" + "\n".join(labels) +
          '\nReply as JSON: {"category": "<one category from the list>"}')
schema = {"type": "object", "properties": {"category": {"type": "string", "enum": labels}},
          "required": ["category"], "additionalProperties": False}

def post(url, body, headers):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    for attempt in range(5):
        try: return json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 4: time.sleep(2 ** attempt); continue
            raise SystemExit(f"HTTP {e.code}: {e.read().decode()[:400]}")

def ask(text):
    """Returns (category, input_tokens, output_tokens incl. reasoning/thinking)."""
    if model.startswith("decisions:"):  # OpenAI Decisions API: one 77-option choice question, like the pplx-decider run
        r = post("https://api.openai.com/v1/decisions",
                 {"model": model.split(":", 1)[1], "input": text,
                  "questions": [{"type": "choice", "name": "category",
                                 "instructions": "Which category does this bank customer's message belong to?",
                                 "choices": [{"value": c, "description": c.replace("_", " ")} for c in labels]}]},
                 {"Authorization": f"Bearer {keys['OPENAI_API_KEY']}"})
        a, u = r["answers"][0], r["usage"]
        return a.get("choice"), u["input_tokens"], u["output_tokens"]  # a refusal has no choice: counted wrong
    if model.startswith("gpt"):
        r = post("https://api.openai.com/v1/chat/completions",
                 {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                  "response_format": {"type": "json_schema", "json_schema": {"name": "intent", "strict": True, "schema": schema}}},
                 {"Authorization": f"Bearer {keys['OPENAI_API_KEY']}"})
        u = r["usage"]
        return json.loads(r["choices"][0]["message"]["content"])["category"], u["prompt_tokens"], u["completion_tokens"]
    gschema = {k: v for k, v in schema.items() if k != "additionalProperties"}
    r = post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
             {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": text}]}],
              "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": gschema}},
             {"x-goog-api-key": keys["GEMINI_API_KEY"]})
    u = r["usageMetadata"]
    out = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    return json.loads(r["candidates"][0]["content"]["parts"][0]["text"])["category"], u["promptTokenCount"], out

correct, tin, tout, t0, preds = 0, 0, 0, time.time(), []
for i, (text, gold) in enumerate(sample, 1):
    cat, a, b = ask(text); preds.append(cat); correct += cat == gold; tin += a; tout += b
    if i % 50 == 0: print(f"{i}/{n} acc {correct / i:.3f}", flush=True)

pin, pout = PRICES[model]
cost = tin / 1e6 * pin + tout / 1e6 * pout
result = {"model": model, "accuracy": correct / n, "n": n, "latency_ms_single": round((time.time() - t0) / n * 1000),
          "input_tokens": tin, "output_tokens": tout, "cost_usd": round(cost, 4), "cost_usd_per_500": round(cost / n * 500, 4)}
print(json.dumps(result, indent=2))
if n == 500:
    json.dump(result, open(f"results/baseline_{model.replace(':', '_')}.json", "w"), indent=2)
    json.dump(preds, open(f"results/preds_{model.replace(':', '_')}.json", "w"))
