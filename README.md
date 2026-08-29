---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3-1.7B
tags:
  - forecasting
  - calibration
  - prediction-markets
  - probability-estimation
  - scalar-head
  - text-classification
library_name: peft
pipeline_tag: text-classification
model-index:
  - name: Cournot-Cold 1.7B
    results:
      - task: {type: text-classification, name: Binary event forecasting}
        dataset: {type: manifold, name: "Cournot published split (contamination-free, n=277)"}
        metrics:
          - {type: brier, value: 0.2004, name: Brier score}
          - {type: calibration, value: 0.0070, name: Murphy calibration}
          - {type: resolution, value: 0.0574, name: Murphy resolution}
---

# Cournot-Cold 1.7B

**The small end of the Cournot-Cold ladder.** It answers the same interface as
the 8B and 4B, on a base model 4.7× smaller than the flagship.

```
input:  question + resolution criteria + resolution date + as_of
output: p ∈ [0,1]
```

No evidence, no retrieval, no market price. A question and a date, and a number.

## Read this first: the trade this model offers

**It is worse than the 4B, measurably, and that is the point of it existing.**

| | dev Brier (n=3,000) | vs Cournot-Cold 4B |
|---|---:|---|
| Cournot-Cold 8B | 0.1674 | — |
| Cournot-Cold 4B | 0.1684 | — |
| **Cournot-Cold 1.7B** | **0.1752** | **+0.0068 [+0.0035, +0.0101], significant** |

Paired, question-clustered bootstrap, 10,000 resamples, seed 20260822.

**The offer is +0.0068 Brier for 57% fewer parameters.** If you are running on a
laptop, on CPU, or anywhere the 8B does not fit, that is the trade. If you are not
constrained, **use the 4B** — it is free to run by comparison and it ties the 8B.

We are telling you to prefer a different model in most cases. That is the honest
reading of our own numbers and it belongs at the top of the card, not in a
footnote.

## The headline number

**Brier 0.2004** on the contamination-free published split, n=277 — questions that
resolved *after* the training freeze, so no part of this model saw them.

| | value |
|---|---:|
| Brier | 0.2004 |
| Murphy calibration ↓ | 0.0070 |
| Murphy resolution ↑ | 0.0574 |
| base rate | 0.4946 |
| base-rate Brier | 0.2500 |

`uv run python verify.py` recomputes every number above from the forecasts and
eval split in this repository. It needs no GPU, no weights and no network.

## What the size ladder actually shows

This is the finding the three models exist to support, and it is more interesting
than any one of them.

**Calibration is size-invariant on this task. Discrimination is not.**

| dev, n=3,000 | calibration ↓ | resolution ↑ |
|---|---:|---:|
| Cournot-Cold 8B | 0.0010 | 0.0718 |
| Cournot-Cold 4B | 0.0007 | 0.0708 |
| Cournot-Cold 1.7B | 0.0008 | 0.0639 |

Across a **4.7× parameter range**, calibration is flat — the differences are well
inside the ±0.003 seed-noise floor we measured. Every point of the degradation is
**resolution**.

**The small model learns *how confident to be* as well as the large one. What it
loses is *which questions to be confident about*.**

On the published split (n=277) the calibration point estimates do differ —
0.0048 / 0.0034 / 0.0070 — but the 1.7B-minus-8B interval is
**+0.0022 [−0.0072, +0.0151], not significant**. That split is too small to
resolve a calibration difference this size, so the claim above rests on the dev
split, where there is power. Both are given.

## Calibration: none applied

The shipped probability is the head's raw `sigmoid(logit)`. **No post-hoc
calibration map is applied and none is distributed.** The calibration figures
above are what the model produces untuned.

## What this model cannot do

1. **It cannot beat the 4B.** See the top of this card.
2. **It cannot do mechanical threshold or counting questions.** Questions needing
   a time series and precise arithmetic are outside a judgment prior.
3. **Short horizons.** Under 7 days it never beats a market crowd at any point in
   a question's life.
4. **Calibration does not transfer to a new venue.** A new venue needs its own
   calibration mapping, fit on that venue's own development split.

<!-- PENDING batch10: transfer numbers measured on THIS model.
     Do not publish this card until this block is replaced with measured values
     and the contamination-free Kalshi subset (n=117) is stated as a limit, per
     docs/10 2026-08-28n. A card without them is the flattering omission. -->

## Training

LoRA r=32, α=64 on `Qwen/Qwen3-1.7B` with a scalar regression head
(`Qwen3ForSequenceClassification`, `num_labels=1`, `modules_to_save=["score"]`),
Brier loss against the terminal 0/1 outcome. 81,870 Manifold questions resolving
before the freeze. Seed 20260822, LR 1e-4 OneCycle, right padding, last-non-pad
pooling, chat template on **both** train and score.

**Identical corpus, seed, learning rate and footing to the 8B and 4B** — the only
variable across the three is parameter count. That is what makes the ladder above
a measurement rather than three separate results.

## A note on the base model

An earlier internal evaluation rejected Qwen3-1.7B as mode-collapsed: 226 of 500
answers exactly 0.5, 46.2% of probability mass within 0.05 of it, 3.02 bits of
entropy. **That was measured at Q4 quantisation on generated probability tokens.**

This model is a bf16 scalar regression head, which does not generate. Its dev
output distribution: **15.7% within 0.05 of 0.5, 6.34 bits of entropy, 1,251
distinct values across 3,000 questions, sd 0.2739.** Not collapsed.

The earlier finding was about a substrate for prompt-based iteration, and it does
not transfer to a regression head.

## Intended use

Base rates and cold-start estimates where a larger model will not fit. Research on
scale and calibration. **Not** for trading, and not as a substitute for the 4B
where the 4B will run.

## Licence and data

Adapter weights and evaluation metadata under CC BY-NC 4.0. Base model
`Qwen/Qwen3-1.7B` is Apache 2.0 and unaffected.

**No venue question text is redistributed here.** Eval splits carry question ids,
dates and outcomes only. Each `question_id` is the venue's own stable identifier,
so the text is retrievable from that venue directly under that venue's terms.

Training data derives from the Manifold Markets public API and is not
redistributed. Manifold's terms restrict bulk API data to personal and
non-commercial use and prohibit training ML models for commercial purposes without
a data licence; that licence is not ours to grant.
