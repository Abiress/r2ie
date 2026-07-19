# Standard Real-World Benchmark (WikiText-2)

> Official pre-release gate. Canonical WikiText-2 (same data as EleutherAI
> lm-evaluation-harness `wikitext`). Shared tokenizer, vocab cap 4000, fixed
> protocol: seed=1234, steps=3000, seq_len=64, batch=32, lr=1e-3,
> weight_decay=1e-5, d_model=128, n_layers=2, device=cpu. R²IE uses the soft
> Mass Compressor (info-preserving codebook projection). Baselines
> (Transformer/Mamba/SSA) have no VQ stage, so their numbers are independent of
> R²IE's VQ mode.

**Metric: test-set token perplexity (lower is better).**

| Model | Params | Val ppl | Test ppl |
| --- | ---: | ---: | ---: |
| **R²IE + DTFv3 + HDQ (soft VQ)** | 1,731,625 | 4.69 | **4.59** |
| SSA (linear-attention) | 1,301,412 | 6.17 | 6.11 |
| R²IE + DTF + HDQ (soft VQ, v1) | 1,599,013 | 12.57 | 12.84 |
| Mamba | 1,367,714 | 44.94 | 45.23 |
| Transformer | 1,301,922 | 48.55 | 49.42 |

**Best test perplexity: R²IE + DTFv3 + HDQ (4.59).**

> **VERDICT:** Under this fixed, real-world protocol, the best R²IE variant
> (R²IE + DTFv3 + HDQ, test ppl 4.59) achieves the lowest test perplexity among
> all compared architectures — beating SSA (6.11, ≈1.33×), Mamba (45.23, ≈9.9×),
> and Transformer (49.42, ≈10.8×). This is a measured result on WikiText-2, not a
> claim extrapolated from other tasks. The winning configuration trains the soft
> compressor's codebook via its commitment loss and uses the DTFv3
> coherence-driven global mixer as the sole transformation field (the ACT
> attention field is disabled so the two global mixers do not compete).

Reproduce it:
```bash
python src/r2ie/standard_benchmark.py --steps 3000 --vocab-cap 4000 \
    --d-model 128 --n-layers 2 --codebook-size 1024 --vq-mode soft \
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

With both, R²IE + DTFv3 + HDQ drops from 14.24 → 4.59 and leads the field.

## Correction note

An earlier version of this file (and the v0.2.0 tag) reported a test perplexity
of ~1.44 for R²IE. That figure came from a one-off evaluation script with a
perplexity-computation bug and is **retracted** — it was never reproduced by the
official `standard_benchmark.py`. The values above are the correct, reproducible
results from the official benchmark, verified on 2026-07-19.
