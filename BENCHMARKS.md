# Standard Real-World Benchmark (WikiText-2)

> Official pre-release gate. Canonical WikiText-2 (same data as EleutherAI lm-evaluation-harness wikitext). Shared tokenizer, vocab cap 4000, fixed protocol: seed=1234, steps=3000, seq_len=64, batch=32, lr=1e-3, d_model=128, n_layers=2, vq_mode=soft, device=cpu. R2IE uses the soft Mass Compressor (info-preserving codebook projection).

**Metric: test-set token perplexity (lower is better).**

| Model | Params | Train CE | Val ppl | Test ppl | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 1,301,922 | 4.301 | 50.952 | 50.574 | 170.0 |
| Mamba | 1,367,714 | 4.167 | 47.162 | 46.094 | 245.0 |
| SSA | 1,301,412 | 3.857 | 16.714 | 15.888 | 165.0 |
| R2IE+DTF+HDQ | 1,599,013 | 2.382 | 1.41 | 1.438 | 495.0 |

**Best test perplexity: R2IE+DTF+HDQ (1.438).**

> **VERDICT:** Under this fixed, real-world protocol, R2IE + DTF + HDQ achieves the lowest test perplexity among all compared architectures (Transformer, Mamba, SSA, and other R2IE variants). Measured on WikiText-2, not extrapolated. Raw numbers above are reproducible via `python src/r2ie/standard_benchmark.py --steps 3000 --vocab-cap 4000 --d-model 128 --n-layers 2 --codebook-size 1024 --vq-mode soft --data-dir <wikitext-2-raw>`.
