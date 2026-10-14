"""CPU checks for poziomka_kd_loss: run with `python3 test_poziomka_kd_loss.py`."""
import torch

from poziomka_kd_loss import _bucket_terms, topk_reverse_kl


def make(n=37, vocab=300, k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    teacher = torch.randn(n, vocab, generator=g, dtype=torch.float64) * 3
    logp_full = torch.log_softmax(teacher, -1)
    logp, ids = logp_full.topk(k, -1)
    z = torch.randn(n, vocab, generator=g, dtype=torch.float64) * 2
    labels = torch.randint(0, vocab, (n,), generator=g)
    valid = torch.rand(n, generator=g) > 0.2
    return z, ids, logp, logp_full, labels, valid


def test_gradient_matches_autograd():
    z, ids, logp, _, labels, valid = make()
    z1 = z.clone().requires_grad_()
    loss, _, _ = topk_reverse_kl(z1, ids, logp, labels, valid, chunk=5)
    w = torch.rand(len(z), dtype=torch.float64)
    (loss * w).sum().backward()
    z2 = z.clone().requires_grad_()
    _, _, _, ref, _ = _bucket_terms(z2, ids, logp)  # plain autograd through the same formula
    (torch.where(valid, ref, torch.zeros_like(ref)) * w).sum().backward()
    assert torch.allclose(loss, torch.where(valid, ref, torch.zeros_like(ref)).float(), atol=1e-5)
    assert torch.allclose(z1.grad, z2.grad, atol=1e-6), (z1.grad - z2.grad).abs().max()
    assert (z1.grad[~valid] == 0).all()


def test_bucketed_bounded_by_full_reverse_kl():
    z, ids, logp, logp_full, labels, valid = make()
    loss, _, _ = topk_reverse_kl(z, ids, logp, labels, torch.ones_like(valid))
    logq = torch.log_softmax(z, -1)
    full = (logq.exp() * (logq - logp_full)).sum(-1)
    assert (loss <= full.float() + 1e-5).all() and (loss >= -1e-6).all()


def test_zero_when_student_matches_teacher_and_ce():
    _, ids, logp, logp_full, labels, valid = make()
    loss, ce, q_tail = topk_reverse_kl(logp_full.clone(), ids, logp, labels, torch.ones_like(valid))
    assert loss.abs().max() < 1e-5
    assert torch.allclose(ce, -logp_full.gather(1, labels[:, None])[:, 0].float(), atol=1e-5)
    assert torch.allclose(q_tail, (1 - logp.exp().sum(-1)).float(), atol=1e-5)


def test_invalid_rows_with_garbage_teacher_stay_finite():
    z, ids, logp, _, labels, valid = make()
    ids[~valid], logp[~valid] = 0, 0.0  # BOS/padding rows are zeros on disk
    z = z.to(torch.bfloat16).requires_grad_()
    loss, ce, q_tail = topk_reverse_kl(z, ids, logp.half(), labels, valid)
    loss.sum().backward()
    for t in (loss, ce, q_tail, z.grad.float()):
        assert torch.isfinite(t).all()
    assert (loss[~valid] == 0).all() and (z.grad[~valid] == 0).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
