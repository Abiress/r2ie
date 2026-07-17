"""Generation test: a trained checkpoint must produce non-empty, crash-free text.

Trains a tiny model on the bundled fixture, saves a checkpoint, then runs the
generate path and asserts the output is a non-empty string within the token
budget. Uses a fixed seed for determinism.
"""

import os
import tempfile

import torch

from r2ie.config import ModelConfig
from r2ie.data import CharDataset, CharTokenizer, load_corpus_text
from r2ie.generate import main as generate_main
from r2ie.model import R2IEModel


def _train_and_save(tmp_dir: str, seed: int) -> str:
    torch.manual_seed(seed)
    text = load_corpus_text("fixture")
    tokenizer = CharTokenizer(text)
    dataset = CharDataset(text, seq_len=32, tokenizer=tokenizer)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=256,
        codebook_size=32,
        max_ponder_steps=3,
    )
    model = R2IEModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    it = iter(loader)
    for _ in range(40):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        optimizer.zero_grad()
        logits, aux = model(x)
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                               y.reshape(-1))
        (ce + aux["commitment_loss"] + aux["ponder_cost"]).backward()
        optimizer.step()

    ckpt = os.path.join(tmp_dir, "r2ie_test.pt")
    from dataclasses import asdict

    torch.save({"model_state": model.state_dict(), "config": asdict(config),
                "tokenizer_chars": tokenizer.chars}, ckpt)
    return ckpt


def test_generate_produces_nonempty_text():
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _train_and_save(tmp, seed=7)
        # Redirect generate() stdout by capturing via main().
        import contextlib
        import io

        argv = [
            "--checkpoint", ckpt,
            "--prompt", "THE ",
            "--max-tokens", "30",
            "--device", "cpu",
            "--use-fast-weights",
        ]
        # We call main but capture stdout to avoid polluting test output.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = generate_main(argv)
        assert rc == 0
        text = buf.getvalue().strip()
        assert len(text) > 0, "generated output was empty"
        # max-tokens limits how many new ids are appended; output should be
        # within a reasonable length bound (prompt + generated characters).
        assert len(text) <= 50 + 256
