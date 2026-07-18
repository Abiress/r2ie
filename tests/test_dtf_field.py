import torch

from r2ie.dtf_field import DTFBlock, DTFOperator, build_dtf_stacks


def test_dtf_identity_when_c_zero():
    # c_max == 0 forces c^2 -> 0 regardless of the learned coherence map, so
    # the operator is the identity (no diffusion).
    op = DTFOperator(8, c_max=0.0)
    h = torch.randn(2, 6, 8)
    out = op(h)
    assert torch.allclose(out, h, atol=1e-5)


def test_dtf_output_shape_preserved():
    op = DTFOperator(16)
    h = torch.randn(3, 10, 16)
    out = op(h)
    assert out.shape == h.shape


def test_dtf_single_position_no_nan():
    op = DTFOperator(8)
    h = torch.randn(2, 1, 8)
    out = op(h)
    assert out.shape == (2, 1, 8)
    assert torch.isfinite(out).all()


def test_dtf_smooths_signal():
    # A high-frequency alternating signal should lose variance across positions
    # after diffusion (Laplacian opposes alternation). With c^2 = 0.25 the
    # checkerboard is mapped exactly to zero in one step (stable, since c^2<0.25).
    op = DTFOperator(4, c_max=0.5)
    with torch.no_grad():
        op.coherence.weight.zero_()
        op.coherence.bias.fill_(10.0)  # sigmoid(10) ~ 1 -> c^2 ~ c_max = 0.5? no: 0.5*1
    # Force c^2 exactly 0.25 via c_max=0.5 and sigmoid~1 -> c^2=0.5; instead use
    # c_max=0.25 to guarantee c^2<=0.25 (stable smoothing for checkerboard).
    op = DTFOperator(4, c_max=0.25)
    with torch.no_grad():
        op.coherence.weight.zero_()
        op.coherence.bias.fill_(10.0)
    base = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0]).unsqueeze(0).unsqueeze(-1)
    base = base.expand(1, 5, 4).clone()
    out = op(base)
    in_var = base[:, 1:-1, 0].var(dim=1).item()
    out_var = out[:, 1:-1, 0].var(dim=1).item()
    assert out_var < in_var


def test_dtf_backward_runs():
    op = DTFOperator(8)
    h = torch.randn(2, 5, 8, requires_grad=True)
    out = op(h).sum()
    out.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_dtf_block_forward():
    blk = DTFBlock(16, d_ff=32)
    x = torch.randn(2, 7, 16)
    y = blk(x)
    assert y.shape == x.shape


def test_build_dtf_stacks_runs():
    blocks = torch.nn.ModuleList([DTFBlock(8, 16) for _ in range(2)])
    x = torch.randn(2, 5, 8)
    y = build_dtf_stacks(x, blocks)
    assert y.shape == x.shape
