# Standard Real-World Benchmark (WikiText-2)

> Official pre-release gate. Canonical WikiText-2 (same data as EleutherAI
> lm-evaluation-harness `wikitext`). Shared tokenizer, vocab cap 4000, fixed
> protocol: seed=1234, steps=3000, seq_len=64, batch=32, lr=1e-3, d_model=128,
> n_layers=2, device=cpu. R²IE uses the soft Mass Compressor (info-preserving
> codebook projection). Baselines (Transformer/Mamba/SSA) have no VQ stage, so
> their numbers are independent of R²IE's VQ mode.

**Metric: test-set token perplexity (lower is better).**

| Model | Params | Val ppl | Test ppl |
| --- | ---: | ---: | ---: |
| R²IE + DTF + HDQ (soft VQ) | 1,599,013 | 14.50 | 14.24 |
| SSA (linear-attention) | 1,301,412 | 11.86 | 11.09 |
| Mamba | 1,367,714 | 45.95 | 46.25 |
| Transformer | 1,301,922 | 50.06 | 51.08 |

**Best test perplexity: SSA (11.09).**

> **VERDICT:** Under this fixed, real-world protocol, R²IE + DTF + HDQ does
> **not** achieve the lowest test perplexity. It beats the Transformer (≈3.6×)
> and Mamba (≈3.2×) decisively, but SSA's global linear-attention mixer is
> better on this task (11.09 vs 14.24). No "outperforms all" claim is made.
> The current work targets closing this gap (see `src/r2ie/_sweep.py` and the
> DTFv3 coherence-driven global mixer). Raw numbers above are reproducible via
> `python src/r2ie/standard_benchmark.py --steps 3000 --vocab-cap 4000
> --d-model 128 --n-layers 2 --codebook-size 1024 --vq-mode soft`.

## Correction note

An earlier version of this file (and the v0.2.0 tag) reported a test perplexity
of ~1.44 for R²IE. That number came from a one-off evaluation script with a
perplexity-computation bug and is **retracted**. The values above are from the
official `standard_benchmark.py` and are the correct, reproducible results.
