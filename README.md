# R²IE — Recursive Relativistic Information Engine (Prototype)

> **Status: experimental / research prototype, not benchmarked against production models.**

R²IE is a small, experimental architecture for research and education that combines
three ideas behind a loose mass–energy-equivalence metaphor:

1. a **vector-quantization (VQ) bottleneck** — the "Mass Compressor" — that compresses
   each token's embedding into a discrete codebook entry before the sequence is processed;
2. an **Adaptive Computation Time (ACT)** loop — the "Transformation Field" — that lets the
   model spend *more* transformer steps on harder tokens and *fewer* on easy ones;
3. a **bounded, decayed fast-weight (Hebbian) memory** — the "Condensation Loop" — that
   modulates the output head at inference/consolidation time without ever overwriting the
   trained parameters.

It is trainable end-to-end on a character-level language-modeling task and is meant to be
read, modified, and learned from — not presented as a competitor to production Transformers.

## Origin

The conceptual starting point is an original idea by Abir Maheshwari: a
mass–energy-equivalence metaphor (`E = m c²`) mapped onto an AI architecture —
the "Recursive Relativistic Information Engine". The prototype implements that
idea faithfully where it is tractable, and replaces the metaphor-only parts
with published, trainable techniques (see "What this is not" below). The three
recursive loops from the original concept are implemented as real, bounded
modules (off by default, opt-in via CLI flags):

- **Condensation Loop** (`condensation.py`) — a persistent, double-buffered
  "mass" buffer that accretes high-confidence output energy (`m += α·E/c²`)
  during an explicit consolidation phase. It is a *separate* buffer, never an
  in-place overwrite of the trained weights.
- **Velocity Governor** (`velocity_governor.py`) — estimates per-token coherence
  (negative softmax entropy) and, when coherence is low, injects bounded
  stochastic noise into the ACT halting logits so the field takes more ponder
  steps (higher `c²`) before halting — the "escape velocity" behavior. An
  Event-Horizon gate (`E > m·c²`) exposes the emission condition.
- **Fast-weight modulation** (`fast_weight_loop.py`) — bounds and clamps a
  Hebbian fast-weight matrix that modulates the output head at inference time.

These are composed by `engine.py` (`R2IEEngine`), which also owns input
validation (`errors.py`) and explicit buffer lifecycle so no non-trainable
state leaks across sequences.

---

## Install

```bash
pip install -e .
```

(For linting, also install the dev extra: `pip install -e ".[dev]"`.)

## Quickstart

Train on the bundled tiny corpus (no download needed):

```bash
python -m r2ie.train --data fixture --steps 500 --device cpu
```

Train on the full Tiny Shakespeare corpus (download it first):

```bash
bash scripts/download_data.sh
python -m r2ie.train --data tinyshakespeare --steps 2000 --device auto
```

Generate text from a trained checkpoint:

```bash
python -m r2ie.generate --checkpoint checkpoints/r2ie_latest.pt --prompt "THE " --max-tokens 200
```

Train or generate with the original recursive loops enabled (off by default):

```bash
python -m r2ie.train --data fixture --steps 500 --use-condensation --use-governor --use-fast-weights
python -m r2ie.generate --checkpoint checkpoints/r2ie_latest.pt --prompt "THE " --use-condensation --use-governor
```

**Expected runtime:** on CPU, 500 steps on the bundled fixture takes a few seconds; 2000
steps on Tiny Shakespeare takes a few minutes. On a CUDA GPU the same runs noticeably faster.

> **Scope of the result:** the bundled 500-step run on the tiny corpus only demonstrates
> that the training loop runs and the model's cross-entropy decreases — i.e. the prototype
> is wired up correctly and learns *something*. It is **not** evidence that R²IE generalizes,
> scales, or competes with production Transformers on any real benchmark. Treat any such claim
> as unverified unless a benchmark script in this repo was actually run and its output pasted
> verbatim into `BENCHMARKS.md`.

## Architecture

```
 token
   │  (nn.Embedding + positional)
   ▼
 [VQ Compressor]  ── "Mass Compressor"  (VQ-VAE codebook, straight-through)
   │
   ▼
 [ACT Field]  ── "Transformation Field"  (pre-norm Transformer block, looped)
   │            per-token halting prob; ponder cost added to the loss
   ▼
 [Fast-weight-modulated Head]  ── "CPL"  (linear → temperature softmax)
   │                            (Condensation Loop modulates here at inference)
   ▼
 output (logits over vocab)
```

The data flow is: `token → embedding → VQ → [ACT loop ×N] → fast-weight-modulated head → output`.

## What this is not

- **Not** a claim of beating Transformers. R²IE is a small research prototype; it makes no
  performance claims versus GPT, Llama, or any production model.
- **No** unverified benchmark numbers. If a benchmark was run, its literal output lives in
  `BENCHMARKS.md` (produced by `scripts/run_benchmark.py`) — never fabricated.
- The narrative names (Mass Compressor, Transformation Field, Condensation Loop, Coherence
  Projection Layer) are metaphors over **published techniques**:
  - Vector Quantization: van den Oord, Vinyals & Kavukcuoglu, *Neural Discrete Representation
    Learning* (VQ-VAE), 2017.
  - Adaptive Computation Time: Graves, *Adaptive Computation Time for Recurrent Neural
    Networks*, 2016.
  - Fast weights / Hebbian associative memory: Hinton & Plaut (1987); Schmidhuber (1991);
    Ba, Hinton & Mnih, *Using Fast Weights to Attend to the Recent Past*, 2016.
  - Transformer block: Vaswani et al., *Attention Is All You Need*, 2017.

## License / Citation / Contributing

- License: [MIT](LICENSE)
- Citation: see [CITATION.cff](CITATION.cff)
- Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md)

## Created by Abir Maheshwari.
