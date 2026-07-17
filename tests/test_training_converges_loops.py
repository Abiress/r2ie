"""Convergence test with the RALE.docx loops (Condensation + Velocity Governor)
enabled, proving the merged modules don't break training. Uses the bundled
inline fixture and a fixed seed for determinism.
"""

import torch

from r2ie.config import ModelConfig
from r2ie.data import CharDataset, CharTokenizer, load_corpus_text
from r2ie.engine import R2IEEngine


def _train(use_condensation, use_governor, seed):
    torch.manual_seed(seed)
    text = load_corpus_text("fixture")
    tokenizer = CharTokenizer(text)
    dataset = CharDataset(text, seq_len=32, tokenizer=tokenizer)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=256,
        codebook_size=32,
        max_ponder_steps=3,
    )
    engine = R2IEEngine(
        cfg,
        use_condensation=use_condensation,
        use_governor=use_governor,
    )
    optimizer = torch.optim.Adam(engine.model.parameters(), lr=1e-3)
    engine.model.train()

    loader_iter = iter(loader)
    # Baseline CE of the untrained model.
    x0, y0 = next(iter(loader))
    with torch.no_grad():
        logits0, _ = engine.forward_step(x0)
        import torch.nn.functional as F

        initial_ce = F.cross_entropy(logits0.reshape(-1, logits0.size(-1)),
                                     y0.reshape(-1)).item()

    final_ce = None
    for _ in range(200):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        optimizer.zero_grad()
        logits, _ = engine.forward_step(x, consolidate=use_condensation)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        ce.backward()
        optimizer.step()
        final_ce = ce.item()

    return initial_ce, final_ce


def test_converges_with_loops_enabled():
    seed = 1234
    initial, final = _train(use_condensation=True, use_governor=True, seed=seed)
    print(f"[loops ON] initial_ce={initial:.4f} final_ce={final:.4f}")
    assert final < 0.9 * initial, (
        f"model did not learn with loops on: initial={initial:.4f}, final={final:.4f}"
    )


def test_converges_with_condensation_only():
    seed = 99
    initial, final = _train(use_condensation=True, use_governor=False, seed=seed)
    print(f"[condensation only] initial_ce={initial:.4f} final_ce={final:.4f}")
    assert final < 0.9 * initial
