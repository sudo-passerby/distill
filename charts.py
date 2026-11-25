"""Builds the blog post from post.md: every <!-- chart:NAME --> marker becomes a chart and every <!-- val:NAME -->
a number, both drawn from results/*.json, so nothing in the post is typed by hand.
Run: python3 charts.py > ~/code/portfolio/content/blog/tiny-model-8-minutes.md"""
import json, math, re
from pathlib import Path

R = Path("results")
load = lambda f: json.loads((R / f).read_text())
tiny = load("tiny.json")
# (label, file, where it runs, accuracy key) — the tiny model's 500-sample score is computed from its saved answers below
MODELS = [
    ("Ours: Qwen3-0.6B, trained", None, "local"),
    ("pplx-decider 4B", "baseline_decider.json", "local"),
    ("GPT-6 Luna", "baseline_gpt-6-luna.json", "api"),
    ("Gemini 3.5 Flash-Lite", "baseline_gemini-3.5-flash-lite.json", "api"),
    ("OpenAI Decisions API", "baseline_decisions_gpt-6-luna.json", "api"),
    ("gemma4 4B", "baseline_gemma4_e4b.json", "local"),
    ("qwen3.5 4B", "baseline_qwen3.5_4b.json", "local"),
]

def tiny_on_sample():
    import csv, random
    test = [(r["text"], r["category"]) for r in csv.DictReader(open("data/test.csv"))]
    preds = load("tiny_preds.json")
    random.seed(0); idx = random.sample(range(len(test)), 500)  # same indices as random.sample(test, 500)
    return sum(preds[i] == test[i][1] for i in idx) / 500

rows = []
for name, f, where in MODELS:
    r = load(f) if f else {"accuracy": tiny_on_sample(), "latency_ms_single": tiny["latency_ms_single"], "cost_usd": 0}
    rows.append({"name": name, "where": where, "acc": r["accuracy"] * 100, "ms": r["latency_ms_single"],
                 "cost500": r.get("cost_usd", 0), "ours": f is None})

esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;")

SHORT = {"Ours: Qwen3-0.6B, trained": "Ours", "pplx-decider 4B": "pplx-decider", "GPT-6 Luna": "Luna",
         "Gemini 3.5 Flash-Lite": "Flash-Lite", "OpenAI Decisions API": "Decisions API", "gemma4 4B": "gemma4", "qwen3.5 4B": "qwen3.5"}

def scatter_svg(W, H, short):
    L, Rm, T, B = 48, 16, 16, 52
    lo, hi = math.log10(10), math.log10(3000)
    x = lambda ms: L + (math.log10(ms) - lo) / (hi - lo) * (W - L - Rm)
    y = lambda a: T + (96 - a) / (96 - 66) * (H - T - B)
    g = []
    for a in range(70, 96, 5):
        g.append(f'<line x1="{L}" x2="{W - Rm}" y1="{y(a):.1f}" y2="{y(a):.1f}" class="grid"/>'
                 f'<text x="{L - 10}" y="{y(a) + 5:.1f}" text-anchor="end">{a}%</text>')
    for ms, lab in [(10, "10 ms"), (100, "100 ms"), (1000, "1 s")]:
        g.append(f'<text x="{x(ms):.1f}" y="{H - B + 26}" text-anchor="middle">{lab}</text>')
    g.append(f'<text x="{(L + W - Rm) / 2:.0f}" y="{H - 6}" text-anchor="middle" class="axis">{"time per message →" if short else "time per message (log scale) → slower"}</text>')
    # label side per model so names never collide
    side = {"Ours: Qwen3-0.6B, trained": "r", "pplx-decider 4B": "l", "GPT-6 Luna": "l", "Gemini 3.5 Flash-Lite": "l",
            "OpenAI Decisions API": "l", "gemma4 4B": "r", "qwen3.5 4B": "l"}
    for r in rows:
        assert 10 <= r["ms"] <= 3000 and 66 <= r["acc"] <= 96, f'{r["name"]} falls outside the chart axes; widen them'
        cx, cy = x(r["ms"]), y(r["acc"])
        cls = "dot ours" if r["ours"] else ("dot api" if r["where"] == "api" else "dot")
        s = side[r["name"]]
        tx, anchor = (cx + 12, "start") if s == "r" else (cx - 12, "end")
        g.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{9 if r["ours"] else 7}" class="{cls}"/>'
                 f'<text x="{tx:.1f}" y="{cy + 5:.1f}" text-anchor="{anchor}" class="{"lbl ours" if r["ours"] else "lbl"}">{esc(SHORT[r["name"]] if short else r["name"])} · {r["acc"]:.1f}%</text>')
    return f'<svg class="{"phone" if short else "wide"}" viewBox="0 0 {W} {H}" role="img" aria-label="Accuracy against time per message">{"".join(g)}</svg>'

def scatter():
    return (f'<figure class="viz">{scatter_svg(640, 380, False)}{scatter_svg(360, 460, True)}'
            '<figcaption><span class="key ours"></span>ours <span class="key api"></span>paid API <span class="key"></span>local, not trained</figcaption></figure>')

def bars(title, items, fmt, note=""):
    top = max(v for _, v, _ in items) or 1
    out = [f'<figure class="viz bars"><figcaption class="viz-title">{title}</figcaption>']
    for name, v, kind in items:  # kind: True = ours (amber), False = paid API (blue), "plain" = neither
        cls = " ours" if kind is True else (" plain" if kind == "plain" else "")
        out.append(f'<div class="bar{cls}"><span class="bar-name">{esc(name)}</span>'
                   f'<span class="bar-track"><span class="bar-fill" style="width:{max(v / top * 100, 0.8):.1f}%"></span></span>'
                   f'<span class="bar-val">{fmt(v)}</span></div>')
    if note: out.append(f'<p class="viz-note">{note}</p>')
    return "".join(out) + "</figure>"

def time500():
    items = sorted(((r["name"], r["ms"] * 500 / 1000, r["ours"]) for r in rows if r["ours"] or r["where"] == "api"), key=lambda t: t[1])
    return bars("Time to sort 500 messages, one at a time", items, lambda v: f"{v:,.0f} s")

def bill():
    items = [(r["name"], r["cost500"] / 500 * 1_000_000, r["ours"]) for r in rows if r["ours"] or r["where"] == "api"]
    items.sort(key=lambda t: -t[1])
    return bars("API bill for 1 million messages", items, lambda v: "$0 per call" if v == 0 else f"${v:,.0f}",
                "Prices from OpenAI's and Google's pricing pages, 26 Nov 2026, using the real token counts from this run. "
                "Ours runs on a GPU you host: that box costs money, but the same amount at 1k or 1M messages.")

def confidence():
    preds = load("decider_preds.json")
    hi = [p for p in preds if p["confidence"] >= 0.5]; lo = [p for p in preds if p["confidence"] < 0.5]
    acc = lambda ps: sum(p["choice"] == p["gold"] for p in ps) / len(ps) * 100
    return bars("pplx-decider: right how often, by its own confidence",
                [(f"confident ({len(hi)} msgs)", acc(hi), "plain"), (f"unsure ({len(lo)} msgs)", acc(lo), "plain")], lambda v: f"{v:.0f}%")

def flow():
    luna = next(r for r in rows if r["name"] == "GPT-6 Luna")
    return ('<figure class="viz flow">'
            '<div class="flow-row"><span class="flow-tag">today</span><span class="flow-box">your app</span><span class="flow-arrow">every message →</span>'
            f'<span class="flow-box api">GPT / Gemini API</span><span class="flow-cost">{luna["ms"] / 1000:.1f} s · pay per call</span></div>'
            f'<div class="flow-row"><span class="flow-tag">once</span><span class="flow-box">your past answers</span><span class="flow-arrow">{VALS["train_min"]} min of training →</span>'
            '<span class="flow-box ours">your own small model</span><span class="flow-cost"></span></div>'
            '<div class="flow-row"><span class="flow-tag">after</span><span class="flow-box">your app</span><span class="flow-arrow">every message →</span>'
            f'<span class="flow-box ours">your own small model</span><span class="flow-cost">{rows[0]["ms"]:.0f} ms · $0 per call</span></div>'
            '</figure>')

VALS = {
    "tiny_acc": f'{rows[0]["acc"]:.1f}%', "tiny_full": f'{tiny["test_accuracy"] * 100:.1f}%',
    "tiny_ms": f'{tiny["latency_ms_single"]:.0f} ms', "train_min": f'{round(tiny["train_seconds"] / 60)}',
    "cpu_ms": f'{round(load("tiny_cpu.json")["latency_ms_single"], -1):.0f} ms',
    "api_cost": f'${sum(load(f)["cost_usd"] for _, f, w in MODELS if w == "api"):.2f}',
}

CHARTS = {"scatter": scatter, "time500": time500, "bill": bill, "confidence": confidence, "flow": flow}
post = Path("post.md").read_text()
post = re.sub(r"<!-- val:(\w+) -->", lambda m: VALS[m.group(1)], post)
print(re.sub(r"<!-- chart:(\w+) -->", lambda m: CHARTS[m.group(1)](), post), end="")
