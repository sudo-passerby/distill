---
title: A 0.6B model I trained on my laptop beat GPT-6 Luna and Gemini Flash-Lite
date: 2026-11-26
summary: Bank support messages, 77 categories. <!-- val:train_min --> minutes of training, <!-- val:tiny_acc --> accuracy, <!-- val:tiny_ms --> a message, and $0 per call.
---

Lots of companies pay an API to do one boring job millions of times, like sorting support messages. I wanted to see if a tiny model trained for just that job does it better.

The job: sort [Banking77](https://huggingface.co/datasets/PolyAI/banking77) bank support messages into 77 categories ("My PIN is blocked" → `pin_blocked`). Every model got the same 500 test messages.

<!-- chart:scatter -->

Top left is where you want to be: right more often, and fast.

## What changes if you use it

<!-- chart:flow -->

<!-- chart:time500 -->

<!-- chart:bill -->

## The surprise

The decision model from Perplexity, not trained on this task, beat every paid API. It also knows when it's guessing:

<!-- chart:confidence -->

## Check my work

Anyone can make up a benchmark, so everything is public:

- **The data** is a public dataset, and I used its official test split. I didn't pick the messages.
- **The same 500 messages** for every model (`random.seed(0)`).
- **Every answer** every model gave, message by message, is in the repo. `python3 verify.py` recomputes every score from those answers.
- **The exact prompts and code** are in the repo. Re-running the APIs costs about <!-- val:api_cost -->.
- **Sanity check:** on the full test set (3,080 messages) the tiny model gets <!-- val:tiny_full -->. Published results on Banking77 are around 93-94%, so the setup isn't broken.

Code, data and results: [github.com/sudo-passerby/distill](https://github.com/sudo-passerby/distill)

## Fine print

- The tiny model learned from 10k human-labelled examples. The APIs saw none, just the list of 77 categories. That's the whole trade: you need examples of the job.
- Its 77 categories are fixed. A new category means retraining (<!-- val:train_min --> minutes).
- The <!-- val:tiny_ms --> is a raw forward pass on my laptop's GPU (Radeon 8060S), with no server in front. On my CPU, unoptimised, it's about <!-- val:cpu_ms -->.
- pplx-decider ran on slow fallback kernels on my AMD GPU, so its time is worse than it would be on a supported setup.
- API times include the network from India. 500 messages means each accuracy number is give or take ~3 points.

Next up: let the decision model write the labels instead of humans, so a company doesn't need labelled data at all.
