# distill

A 0.6B model trained for one job, against paid API models doing the same job.

Write-up: [shipmaxxer.dev/blog/tiny-model-8-minutes](https://shipmaxxer.dev/blog/tiny-model-8-minutes)

**The job:** sort [Banking77](https://huggingface.co/datasets/PolyAI/banking77) bank-support messages into 77 categories.
Every model is scored on the same 500 test messages (`random.seed(0)`, `random.sample(test, 500)`).

## What's here

| Path | What |
|---|---|
| `data/` | Banking77 train (10,003) and test (3,080) splits, unchanged from [PolyAI's repo](https://github.com/PolyAI-LDN/task-specific-datasets) (CC-BY 4.0) |
| `train.py` | Fine-tunes `Qwen/Qwen3-0.6B` with a 77-way classification head, scores it on the full test set |
| `baseline.py` | Local models through Ollama (`gemma4:e4b`, `qwen3.5:4b`), told the 77 labels, output constrained to them |
| `decider_baseline.py` | `perplexity-ai/pplx-decider-flash-v1.3`, one 77-option `choice` question per message |
| `api_baseline.py` | GPT-6 Luna, Gemini 3.5 Flash-Lite and the OpenAI Decisions API. Prints real token usage and cost |
| `cpu_latency.py` | The trained model's latency on CPU |
| `verify.py` | Recomputes every model's score from its saved answers and checks it against the reported number |
| `charts.py` + `post.md` | Builds the blog post. Every chart is drawn from `results/`, nothing is typed in by hand |
| `results/` | Scores (`baseline_*.json`, `tiny.json`), **every answer each model gave** (`preds_*.json`, `tiny_preds.json`, `decider_preds.json`) and run logs |

`results/preds_<model>.json` is a list of 500 answers in the order of the sample above. `tiny_preds.json` covers the
full 3,080-message test set in file order; `decider_preds.json` holds the message, the right label, the decider's
choice and its confidence.

## Check the numbers

```bash
python3 verify.py   # recomputes every score from the saved answers; stdlib only
```

## Reproduce

```bash
# Fine-tune (about 8 minutes on a Radeon 8060S; needs torch + transformers 5.x)
python train.py

# Local baselines (Ollama running, models pulled)
python3 baseline.py gemma4:e4b 500
python3 baseline.py qwen3.5:4b 500

# The decision model (its checkpoint ships its own code)
python decider_baseline.py

# Paid APIs: SECRETS points at a JSON file with OPENAI_API_KEY and GEMINI_API_KEY. About $0.21 for all three.
SECRETS=keys.json python3 api_baseline.py gpt-6-luna 20        # check the spend on 20 first
SECRETS=keys.json python3 api_baseline.py gpt-6-luna 500
SECRETS=keys.json python3 api_baseline.py gemini-3.5-flash-lite 500
SECRETS=keys.json python3 api_baseline.py decisions:gpt-6-luna 500

python3 charts.py > post-with-charts.md
```

## Notes

- The trained model saw 10k labelled examples. The other models saw none, only the category names. That's the trade being measured.
- API latency includes the network round-trip from Bangalore, and the API runs went in parallel.
- `results/first_run/` holds the first API run's scores. The APIs were run again so every answer could be saved; the post uses the second run.
- The OpenAI Decisions API returned no choice for 3 of the 500 messages; those count as wrong.
- CPU latency (`results/tiny_cpu.json`) is 8 threads, float32, plain PyTorch, no CPU optimisation.
- A Gemini 3.8 Flash run was stopped at 50 messages for being slow (~8 s a call); its partial log is in `results/`.
- ROCm on gfx1151: MIOpen fails on the decider's long prompts; `torch.backends.cudnn.enabled = False` fixes it (set in `decider_baseline.py`).
- The trained weights (1.2 GB) aren't in the repo. `train.py` recreates them.

Code: MIT. Data: Banking77 by PolyAI, CC-BY 4.0.
