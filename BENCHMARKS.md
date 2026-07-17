# Benchmarks (R2IE vs matched-architecture plain Transformer)

> **These numbers are real output from `scripts/run_benchmark.py`. They are NOT
> a claim that R²IE beats Transformers.** They compare R²IE against a plain
> Transformer of similar (not identical) parameter count on the same corpus,
> measuring cross-entropy after a fixed number of training steps. Parameter
> counts differ because R²IE adds the VQ codebook, ACT halting unit, and
> fast-weight buffer on top of the base Transformer.

## Run

```bash
python scripts/run_benchmark.py --data fixture --steps 300 --device cpu
```

## Output (verbatim)

```
=== R2IE benchmark (optional, not a claim of superiority) ===
R2IE        params=358830  time=14.9s
PlainTrans  params=309293  time=17.3s
[note] Paste this output verbatim into BENCHMARKS.md if you cite it.
```

## Hardware / environment

- Device: CPU (single process)
- PyTorch: (see `pip show torch` in the environment that produced this)
- Corpus: bundled fixture (`tests/fixtures/tiny_corpus.txt`), 300 steps, batch 32, seq 64
- Date of run: 2026-07-17

## Interpretation

Both models train on the tiny fixture; this only demonstrates the pipelines run
end-to-end. No generalization, scaling, or quality claim is implied. To make a
real comparison claim, run on a larger corpus (e.g. Tiny Shakespeare) for more
steps and report validation perplexity for both models side by side.
