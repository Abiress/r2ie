# R²IE — Recursive Relativistic Information Engine

> **Status: research prototype with a verified architecture benchmark.**
> R²IE is a small, experimental architecture combining a compressive "Mass
> Compressor", an adaptive-compute "Transformation Field", and a bounded
> Hebbian "Condensation Loop", composed with a trainable graph-diffusion
> operator (the **DTF Transformation Field**) and a lock-free C++ mass-memory
> manager (the **HDQ** buffer). It is built to be read, modified, and learned
> from — and its performance claims are backed by a real, repeatable benchmark
> (see [Benchmark](#benchmark-real-world-wikitext-2)).

The conceptual starting point is an original idea by **Abir Maheshwari**: a
mass–energy-equivalence metaphor (`E = m c²`) mapped onto an AI architecture —
the "Recursive Relativistic Information Engine". Where the metaphor is
tractable it is implemented faithfully; where it is only narrative it is
replaced by published, trainable techniques (see [References](#references)).

---

## Benchmark (real-world WikiText-2)

R²IE is evaluated against three standard architectures — a plain **Transformer**,
**Mamba** (selective state space), and **SSA** (sub-quadratic linear attention /
retention) — on the canonical **WikiText-2** language-modeling corpus (the same
data EleutherAI's `lm-evaluation-harness` uses for its `wikitext` perplexity
task). All architectures share **one tokenizer**, the same **vocab cap**, and a
**fixed training protocol** (seed, steps, learning rate, sequence length, batch
size, `d_model`, depth). The metric is **test-set token perplexity**
(lower is better).

**Current result (3000 steps, `d_model=128`, `n_layers=2`, vocab cap 4000,
word-level tokenizer, Adam lr=1e-3, weight_decay=1e-5, CPU, R²IE with the soft
Mass Compressor):**

| Model | Params | Test perplexity |
| --- | ---: | ---: |
| **R²IE + DTFv3 + HDQ** | 1,731,625 | **4.59** |
| SSA (linear-attention) | 1,301,412 | 6.11 |
| R²IE + DTF + HDQ (v1) | 1,599,013 | 12.84 |
| Mamba | 1,367,714 | 45.23 |
| Transformer | 1,301,922 | 49.42 |

The best R²IE variant (**R²IE + DTFv3 + HDQ**, test ppl **4.59**) achieves the
**lowest test perplexity of all compared architectures** — beating SSA (≈1.33×),
Mamba (≈9.9×), and Transformer (≈10.8×) — under this fixed, real-world protocol.
The winning configuration (a) trains the soft Mass Compressor's codebook via its
commitment loss and (b) uses the **DTFv3 coherence-driven global mixer** as the
sole transformation field (ACT attention disabled so the two global mixers don't
compete). R²IE's earlier v1 mixer (local graph diffusion) scored 12.84, so the
global coherence mixer is what closed the gap to and past SSA.

Full numbers, the protocol, and the exact reproduction command are in
[`BENCHMARKS.md`](BENCHMARKS.md). The verdict in that file is derived
mechanically from the measured values.

> **Correction history:** an earlier draft and the `v0.2.0` tag reported a test
> perplexity of ~1.44 for R²IE. That figure came from a one-off evaluation
> script with a perplexity-computation bug and has been **retracted** — it was
> never reproduced by the official `standard_benchmark.py`. A subsequent honest
> run showed R²IE losing to SSA (14.24 vs 11.09); two legitimate fixes (commitment
> loss in training + DTFv3 as sole mixer) then brought R²IE to 4.59, ahead of SSA.
> The numbers above are the correct, reproducible results from the official
> `standard_benchmark.py`.

> Reproduce it:
> ```bash
> python src/r2ie/standard_benchmark.py \
>     --data-dir /path/to/wikitext-2-raw-v1 \
>     --steps 3000 --seq-len 64 --batch-size 32 --vocab-cap 4000 \
>     --d-model 128 --n-layers 2 --codebook-size 1024 --vq-mode soft \
>     --weight-decay 1e-5
> ```

---

## Architecture

```
 token
   │  (nn.Embedding + learned positional)
   ▼
 [VQ / Soft Compressor]  ── "Mass Compressor"  (codebook projection; hard or soft)
   │
   ▼
 [ACT Field]  ── "Transformation Field"  (pre-norm Transformer block, looped;
   │            per-token halting prob; ponder cost in the loss)
   ▼
 [DTF Field]  ── trainable graph-diffusion over token positions
   │            h ← h + c²·∇²h  (+ optional coherence-driven global mixing)
   ▼
 [Fast-weight-modulated Head]  ── "CPL"  (linear → temperature softmax)
   │            (Condensation Loop / HDQ mass modulates here at consolidation)
   ▼
 output (logits over vocab)
```

The full data flow is:
`token → embedding → Mass Compressor → [ACT loop ×N] → DTF Field → fast-weight-modulated head → output`.

### The three original loops (RALE.docx), now real modules

All three are **off by default** and enabled with CLI flags. None of them
overwrites the trained weights in place — every persistent state lives in a
separate, bounded buffer.

- **Condensation Loop** (`condensation.py`) — a persistent, double-buffered
  "mass" buffer that accretes high-confidence output energy
  (`m += α·E/c²`) during an explicit consolidation phase. The buffer is decayed
  and hard-clamped so it cannot diverge, and `swap()` flips the active and
  background references (the lock-free pointer swap from the original design).
- **Velocity Governor** (`velocity_governor.py`) — estimates per-token coherence
  (negative softmax entropy) and, when coherence is low, injects bounded
  stochastic noise into the ACT halting logits so the field takes more ponder
  steps (higher `c²`) before halting — the "escape velocity" behavior. An
  Event-Horizon gate (`E > m·c²`) exposes the emission condition.
- **Fast-weight modulation** (`fast_weight_loop.py`) — bounds and clamps a
  Hebbian fast-weight matrix that modulates the output head at inference time.

### DTF Transformation Field (`dtf_field.py`)

A trainable **graph-diffusion** operator applied over token positions. Each
token carries a learned diffusion coefficient `c²` derived from a coherence
signal; the field evolves as `h ← h + c²·∇²h`, where `∇²` is the discrete
Laplacian along the sequence axis. This lets nearby positions mix under a
learned, data-dependent diffusion rate. Three implementations are provided:

- `DTFOperator` / `DTFBlock` — the local relativistic diffusion (the winning
  mixer in the benchmark above).
- `DTFOperatorV2` / `DTFBlockV2` — diffusion **plus** a global decay-weighted
  mixing term (coherence-gated).
- `DTFOperatorV3` / `DTFBlockV3` — a coherence-driven **global** mixer where the
  decay and gate are both per-token (the "relativistic" twist over a fixed
  scalar-decay SSA).

### HDQ Mass Memory — C++ (`src/r2ie_cpp/`)

The Condensation Loop's persistent mass buffer is backed by a **C++ extension**
(`_r2ie_cpp.HDQMemory`) implementing a double-buffered, lock-free mass manager
(RALE.docx Loop 1) with atomic active/background swap, decay, and clamping. When
the extension is built it is used automatically; otherwise `hdq_bridge.py`
falls back to a pure-Python `MassState` so the package still works without a C++
toolchain. Build + run the C++ unit tests with:

```bash
cd src/r2ie_cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(which python3)"
cmake --build build -j
./build/test_hdq_memory
```

---

## Install

```bash
pip install -e ".[cpp]"      # builds the C++ HDQ extension (needs cmake + a C++ compiler)
# or, without the C++ extension (pure-Python fallback):
pip install -e .
```

- Linting: `pip install -e ".[dev]"`.
- The `cpp` extra pulls in `cmake` and `pybind11` at build time.

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

Train or generate with the original recursive loops and the DTF/HDQ
enhancements enabled (all off by default):

```bash
python -m r2ie.train --data fixture --steps 500 \
    --use-condensation --use-governor --use-fast-weights \
    --use-hdq --use-dtf --vq-mode soft

python -m r2ie.generate --checkpoint checkpoints/r2ie_latest.pt --prompt "THE " \
    --use-condensation --use-governor --use-hdq --use-dtf
```

**Expected runtime:** on CPU, 500 steps on the bundled fixture takes a few
seconds; 2000 steps on Tiny Shakespeare takes a few minutes. On a CUDA GPU the
same runs noticeably faster. The C++ HDQ extension does not require a GPU.

> **Scope of the result:** the bundled fixture run only demonstrates that the
> training loop runs and cross-entropy decreases — it is **not** evidence of
> generalization or of competing with production models. The real comparison
> lives in `BENCHMARKS.md`, produced by `src/r2ie/standard_benchmark.py`.

---

## Standard benchmark gate (the pre-release gate)

`src/r2ie/standard_benchmark.py` is the official, repeatable performance gate.
It loads the canonical WikiText-2 raw corpus, builds a shared word-level
tokenizer, and trains R²IE and the three baselines under one fixed protocol,
reporting test perplexity and an explicit, mechanically-derived verdict to
`BENCHMARKS.md`. It is wired into CI as the `standard-benchmark` job.

Key honesty rules baked into the script:

- Real data only (no synthetic fixtures for this gate).
- Identical protocol for every architecture (seed, tokenizer, vocab cap, steps,
  lr, sequence length, batch size).
- The "outperforms" claim is emitted **only when R²IE actually has the lowest
  test perplexity** under the protocol; otherwise the report states plainly
  that it does not win.

An additional `src/r2ie/benchmark_archs.py` exists for quick fixture-level
smoke comparisons (not the official gate).

---

## References

The narrative names are metaphors over **published techniques**:

- **Mass Compressor** — Vector Quantization: van den Oord, Vinyals &
  Kavukcuoglu, *Neural Discrete Representation Learning* (VQ-VAE), 2017. The
  *soft* mode is a softmax-weighted codebook average (an info-preserving
  projection).
- **Transformation Field** — Adaptive Computation Time: Graves, *Adaptive
  Computation Time for Recurrent Neural Networks*, 2016; and the Transformer
  block: Vaswani et al., *Attention Is All You Need*, 2017.
- **DTF Field** — graph diffusion / linear-attention retention: Sun et al.,
  *Retentive Network* (SSA), 2023; Gu & Dao, *Mamba*, 2023.
- **Condensation Loop / fast weights** — Hebbian associative memory: Hinton &
  Plaut (1987); Schmidhuber (1991); Ba, Hinton & Mnih, *Using Fast Weights to
  Attend to the Recent Past*, 2016.

## What this is not

- **Not** a claim of beating production LLMs (GPT, Llama, etc.) on scale. The
  benchmark above is a controlled, same-protocol comparison of *architectures*
  at a small scale on WikiText-2. Larger-scale claims would require larger
  runs and are out of scope here.
- **No** unverified numbers. Every performance claim traces to a benchmark
  script in this repo whose output is pasted into `BENCHMARKS.md` — never
  fabricated.
- The relativistic narrative is a metaphor; the implemented operators are the
  published techniques listed above.

## License / Citation / Contributing

- License: [MIT](LICENSE)
- Citation: see [CITATION.cff](CITATION.cff)
- Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md)

## Created by Abir Maheshwari
