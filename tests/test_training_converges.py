"""Training convergence test.

Trains the model for a small fixed number of steps on the bundled inline
corpus (tests/fixtures/tiny_corpus.txt) and asserts the final loss is
meaningfully lower than the initial loss. Uses a fixed seed so the result is
deterministic across runs. No network access required.
"""

import torch

from r2ie.config import ModelConfig
from r2ie.data import CharDataset, CharTokenizer, load_corpus_text
from r2ie.model import R2IEModel


def _train_one_epoch(steps: int, seq_len: int, batch_size: int, lr: float, seed: int):
    torch.manual_seed(seed)
    text = load_corpus_text("fixture")
    tokenizer = CharTokenizer(text)
    dataset = CharDataset(text, seq_len, tokenizer)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=max(seq_len, 256),
        codebook_size=32,
        commitment_cost=0.25,
        max_ponder_steps=3,
        act_epsilon=0.01,
        act_ponder_weight=0.01,
        fast_weight_decay=0.9,
        fast_weight_lr=0.1,
        fast_weight_clamp=5.0,
    )
    model = R2IEModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    loader_iter = iter(loader)
    # Baseline: cross-entropy of the untrained model on the first batch.
    x0, y0 = next(iter(loader))
    with torch.no_grad():
        logits0, _ = model(x0, use_fast_weights=False, return_aux=True)
        initial_loss = torch.nn.functional.cross_entropy(
            logits0.reshape(-1, logits0.size(-1)), y0.reshape(-1)
        ).item()
    final_loss = None
    # The convergence signal is the cross-entropy (language-modeling) loss,
    # which is the primary training objective logged by train.py. The VQ
    # commitment + ACT ponder terms are auxiliary regularizers; we backprop
    # the full loss but judge learning by CE, which must decrease.
    for step in range(1, steps + 1):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        optimizer.zero_grad()
        logits, aux = model(x, use_fast_weights=False, return_aux=True)
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                               y.reshape(-1))
        total = ce + aux["commitment_loss"] + aux["ponder_cost"]
        total.backward()
        optimizer.step()

        if initial_loss is None:
            initial_loss = ce.item()
        final_loss = ce.item()

    return initial_loss, final_loss


def test_training_converges():
    seed = 1234
    initial_loss, final_loss = _train_one_epoch(
        steps=200, seq_len=32, batch_size=16, lr=1e-3, seed=seed
    )
    print(f"initial_loss={initial_loss:.4f} final_loss={final_loss:.4f}")
    assert final_loss < 0.9 * initial_loss, (
        f"model did not learn: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
