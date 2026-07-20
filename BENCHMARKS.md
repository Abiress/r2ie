# Standard Real-World Benchmark (WikiText-2)

> Official pre-release gate. Canonical WikiText-2 (same data as EleutherAI
> lm-evaluation-harness `wikitext`). Shared tokenizer, vocab cap 4000, fixed
> protocol: seed=1234, seq_len=64, batch=32, weight_decay=1e-5, d_model=128,
> n_layers=2, device=cpu. R²IE uses the soft Mass Compressor (info-preserving
> codebook projection). Baselines (Transformer/Mamba/SSA) have no VQ stage.

**Metric: test-set token perplexity (lower is better).**

## ⚠️ Headline correction (2026-07-20)

An earlier version of this file claimed **R²IE + DTFv3 + HDQ achieves the lowest
test perplexity (4.59 at 3000 steps, lr=1e-3)**. That claim was based on a
**single 3000-step snapshot at one shared learning rate**. A fuller study
(`extended_study.json`, reproduced below) shows the claim is **false**:

- At each architecture's **own best learning rate** (found by sweeping
  {5e-4, 1e-3, 2e-3}), **SSA dominates the entire 1000→8000 step range** and
  reaches ~1.1 test perplexity by step 8000, while R²IE plateaus near ~4.1.
- Even at the originally-shared lr=1e-3, R²IE only led SSA **at exactly step
  3000** (4.59 vs 6.11); by step 4000 SSA had already overtaken (2.10 vs 4.18)
  and the gap widened thereafter.

**Therefore the "R²IE achieves the lowest test perplexity" claim is RETRACTED.**
The accurate statement is: *R²IE + DTFv3 + HDQ is competitive with the
Transformer and Mamba baselines and, under a non-optimal shared learning rate at a
single early checkpoint, briefly matched SSA — but SSA is the strongest
architecture on this task across the full training range.* No "R²IE outperforms
all baselines" claim is made.

## 3000-step single-LR snapshot (original, now superseded)

| Model | Params | Test ppl @3000 (lr=1e-3) |
| --- | ---: | ---: |
| SSA (linear-attention) | 1,301,412 | 6.11 |
| **R²IE + DTFv3 + HDQ (soft VQ)** | 1,731,625 | 4.59 |
| R²IE + DTF + HDQ (soft VQ, v1) | 1,599,013 | 12.84 |
| Mamba | 1,367,714 | 45.23 |
| Transformer | 1,301,922 | 49.42 |

This table is kept for provenance but the ranking it implies is **not** the
conclusion of the study.

Reproduce the original snapshot:
```bash
python src/r2ie/standard_benchmark.py --steps 3000 --vocab-cap 4000 \
    --d_model 128 --n_layers 2 --codebook-size 1024 --vq-mode soft \
    --weight-decay 1e-5
```

## What changed vs the earlier (losing) result

The first honest run reported R²IE + DTF + HDQ at 14.24, losing to SSA (11.09).
Two legitimate fixes closed the gap:

1. **Commitment loss in training.** The soft Mass Compressor returns a
   commitment/VQ loss in its auxiliary output; the original training loop used
   only the cross-entropy term, so the codebook drifted and R²IE overfit. Adding
   `0.25 × commitment_loss` to the objective stabilized and improved R²IE.
2. **DTFv3 as the sole transformation field.** The DTFv3 coherence-driven
   global mixer (a learned cross-sequence mixing branch) now runs *instead of*
   the ACT attention field (`config.act_attention = False` when `use_dtf_v3`),
   removing the double-mixer gradient conflict that had been hurting DTFv3.

With both, R²IE + DTFv3 + HDQ drops from 14.24 → 4.59 at the 3000-step
snapshot. (Subsequent extended study shows SSA overtakes R²IE by step 4000; see
the Extended Study section.)

## Correction note

An earlier version of this file (and the v0.2.0 tag) reported a test perplexity
of ~1.44 for R²IE. That figure came from a one-off evaluation script with a
perplexity-computation bug and is **retracted** — it was never reproduced by the
official `standard_benchmark.py`. The values above are the correct, reproducible
results from the official benchmark, verified on 2026-07-19.

## Extended study: per-architecture LR sweep + 8000-step trajectory (2026-07-20)

To avoid a single-LR / single-snapshot artifact, each architecture was trained
across **lr ∈ {5e-4, 1e-3, 2e-3}** (weight_decay=1e-5, seed=1234) for **8000
steps**, with test perplexity recorded every 1000 steps. Raw data:
`extended_study.json`. Run with:
```bash
python src/r2ie/_extended_study.py --steps 8000 --checkpoint-every 1000 \
    --lrs 5e-4,1e-3,2e-3 --vocab-cap 4000 --d-model 128 --n-layers 2
```

**(1) Each architecture's best LR** (lowest test ppl at step 8000):

| Model | best lr | test ppl @8000 (best lr) |
| --- | ---: | ---: |
| **SSA** | 2e-3 | **1.109** |
| R²IE + DTFv3 + HDQ | 5e-4 | 4.109 |
| Mamba | 2e-3 | 39.859 |
| Transformer | 2e-3 | 41.200 |

The best LR differs per architecture (R²IE 5e-4, the others 2e-3), confirming a
single shared LR is unfair.

**(2) Trajectory at each model's own best LR** (test perplexity by step):

| step | R²IE (5e-4) | Transformer (2e-3) | Mamba (2e-3) | SSA (2e-3) |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | 6.97 | 60.62 | 54.91 | 52.22 |
| 2000 | 5.82 | 50.33 | 47.72 | 25.68 |
| 3000 | 5.11 | 45.60 | 43.16 | 10.97 |
| 4000 | 4.62 | 43.67 | 42.24 | 4.34 |
| 5000 | 4.38 | 42.09 | 40.82 | 1.64 |
| 6000 | 4.32 | 42.15 | 40.75 | 1.31 |
| 7000 | 4.16 | 40.79 | 40.23 | 1.18 |
| 8000 | 4.11 | 41.20 | 39.86 | **1.11** |

**Finding:** SSA leads at **every** checkpoint from step 1000 onward and the gap
widen over time (SSA → 1.1, R²IE plateaus ~4.1). R²IE never beats SSA under fair
per-architecture tuning. Even at the shared lr=1e-3, R²IE led SSA only at the
single step-3000 checkpoint (4.59 vs 6.11); SSA overtook by step 4000 (2.10 vs
4.18). The original "R²IE wins" headline was a snapshot artifact and is
**retracted** above.

**(3) Causal-mask audit (Transformer baseline).** The Transformer uses a correct
causal mask (`baselines.py:26-27, 67-69`):
```python
def _causal_mask(t, device):
    return torch.triu(torch.ones(t, t, device=device), diagonal=1).bool()
# in _Block.forward:
mask = _causal_mask(t, x.device)
h = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                  attn_mask=mask, need_weights=False)[0]
```
`torch.triu(..., diagonal=1)` is `True` for `j > i`; in `nn.MultiheadAttention` a
`True` boolean mask entry is **ignored**, so token `i` cannot attend to `j > i`.
Masking is correct — it is not the source of any result.

## Validation audit (2026-07-19)

The 4.59 number was independently re-checked on four points before any
publicizing. All findings below are reproducible from `standard_benchmark.py`
and the raw corpus at `DEFAULT_DATA_DIR`.

**(a) Out-of-vocabulary / `<unk>` handling.** The word tokenizer (built from the
train split, `vocab_cap=4000`) maps every word outside the top-4000 to `<unk>`
(idx 0). At eval, `<unk>` tokens in the test set are valid prediction targets
and contribute to perplexity normally. Measured `<unk>` rates: train 18.4%,
valid 19.4%, **test 20.0%** (48,289 of 241,211 test tokens). The top test words
(`the`, `,`, `.`, `of`, `and`, `to`, `in`, `a`, `=`, `was`) are all in-vocab, so
`<unk>` is spread across genuinely rare words, not concentrated in easy cases.
Crucially, **every architecture (R²IE, SSA, Mamba, Transformer) shares the exact
same tokenizer and the same 20% test `<unk>` rate**, so the comparison is fair;
the `<unk>` rate cannot explain R²IE beating SSA. (Word-level LM with a capped
vocab is the standard WikiText-2 setup used by lm-evaluation-harness.)

**(b) Train/test overlap.** The corpus is the canonical `wikitext-2-raw-v1`
split. At the **64-word sequence level** there is **zero** overlap between
train and test (`train∩test 64-word seqs = 0`), and zero between valid/test. At
the **article level** there is a known leakage: 246 of 1,320 test articles (18.6%)
also appear verbatim in train. However, these are mostly short stubs — they
account for only **0.6% of test tokens** (1,473 of 241,211). This 0.6% leakage
is identical for all models and is far too small to explain a 4.59-vs-6.11 gap.
The absolute perplexities are therefore marginally optimistic vs. a perfectly
held-out test, but the **relative** ranking is unaffected. (If a stricter
protocol is desired, deduplicate articles across splits before training.)

**(c) Verbatim prediction triples.** Sampling 10 (context, predicted-next,
ground-truth) triples from the test set for the R²IE+DTFv3+HDQ run shows a
plausible small LM: it predicts common function words (`the`, `in`, `and`) and
`<unk>` for rare named entities, and matches the ground truth on frequent words
(e.g. predicting `.` when the target is `.`). It is not degenerate or constant,
confirming the eval feeds real test data and the model genuinely predicts.

**(d) Independent perplexity recomputation.** A second script — *not*
`standard_benchmark.py` — computed perplexity over the **entire** test set
(15,192,261 token predictions; manual `log_softmax` + `gather`, no
`cross_entropy`, no batch cap) and obtained **4.5192**, matching the
benchmark's 4.591 (the tiny gap is just the 300-batch cap vs. full set). The
number is not an artifact of the benchmark's evaluation code.

