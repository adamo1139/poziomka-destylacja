"""Reverse KL to a top-k teacher with a tail bucket; no Megatron imports (CPU-testable).

The teacher is stored as its K most likely tokens per position (ids + log p) and nothing
else, so the full-vocabulary KL(q_student || p_teacher) is not computable: it needs log p
for every token the student puts mass on. Instead both distributions are coarsened to the
same K + 1 buckets -- the teacher's K tokens, plus one "tail" bucket holding all the other
tokens -- and the loss is the exact reverse KL between those bucket distributions:

    L = sum_{v in S} q_v (log q_v - log p_v) + q_tail (log q_tail - log p_tail)

with S the teacher's top-k ids, q_tail = 1 - sum_S q and p_tail = 1 - sum_S p. Merging
outcomes never increases a KL (data-processing inequality), so L <= the true reverse KL:
student mass on a token the teacher rates far below its tail average is under-penalised.
`q_tail` is returned so training can watch that mass.

Gradient. With bucket probabilities Q_b and g_b = log Q_b - log P_b, dL/dQ_b = g_b + 1 and
dQ_b/dz_u = q_u (1[u in b] - Q_b), so dL/dz_u = q_u (g_{b(u)} - L): the "+1" cancels because
sum_b Q_b = 1. The backward pass recomputes softmax chunk by chunk and applies this formula
directly, so no [positions, vocab] fp32 tensor outlives one chunk.
"""
import torch

# Floor for the teacher's tail mass: fp16 log-probs can sum to a hair above 1.
TAIL_EPS = 1e-6


def _bucket_terms(z, ids, teacher_logp):
    """Per position: student log-softmax, bucket log-ratios g, loss L, student tail mass.

    z: [n, V] logits (any float dtype); ids: [n, K] int64; teacher_logp: [n, K] float.
    """
    logq = torch.log_softmax(z.float(), dim=-1)
    logq_s = logq.gather(1, ids)
    logq_tail = torch.logsumexp(logq.scatter(1, ids, float("-inf")), dim=-1)
    logp_s = teacher_logp.float()
    logp_tail = torch.log((1.0 - logp_s.exp().sum(-1)).clamp_min(TAIL_EPS))
    g_s, g_tail = logq_s - logp_s, logq_tail - logp_tail
    q_tail = logq_tail.exp()
    loss = (logq_s.exp() * g_s).sum(-1) + q_tail * g_tail
    return logq, g_s, g_tail, loss, q_tail


class TopKReverseKL(torch.autograd.Function):
    """Per-position bucketed reverse KL; also returns student CE on labels and tail mass.

    Rows whose `valid` is False (padding, positions without a teacher row) get loss 0 and
    zero gradient; their teacher entries are never read as real data.
    """

    @staticmethod
    def forward(ctx, logits, ids, teacher_logp, labels, valid, chunk):
        n = logits.shape[0]
        loss = logits.new_zeros(n, dtype=torch.float32)
        ce = logits.new_zeros(n, dtype=torch.float32)
        q_tail = logits.new_zeros(n, dtype=torch.float32)
        with torch.no_grad():
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                logq, _, _, l, t = _bucket_terms(logits[s:e], ids[s:e], teacher_logp[s:e])
                ok = valid[s:e]
                loss[s:e] = torch.where(ok, l, torch.zeros_like(l))
                q_tail[s:e] = torch.where(ok, t, torch.zeros_like(t))
                c = -logq.gather(1, labels[s:e, None])[:, 0]
                ce[s:e] = torch.where(ok, c, torch.zeros_like(c))
        ctx.save_for_backward(logits, ids, teacher_logp, valid)
        ctx.chunk = chunk
        ctx.mark_non_differentiable(ce, q_tail)
        return loss, ce, q_tail

    @staticmethod
    def backward(ctx, grad_loss, _grad_ce, _grad_tail):
        logits, ids, teacher_logp, valid = ctx.saved_tensors
        grad = torch.empty_like(logits)
        n = logits.shape[0]
        for s in range(0, n, ctx.chunk):
            e = min(s + ctx.chunk, n)
            logq, g_s, g_tail, l, _ = _bucket_terms(logits[s:e], ids[s:e], teacher_logp[s:e])
            g = g_tail[:, None].expand_as(logq).clone().scatter_(1, ids[s:e], g_s)
            w = torch.where(valid[s:e], grad_loss[s:e], torch.zeros_like(grad_loss[s:e]))
            grad[s:e] = (logq.exp() * (g - l[:, None]) * w[:, None]).to(grad.dtype)
        return grad, None, None, None, None, None


def topk_reverse_kl(logits, ids, teacher_logp, labels, valid, chunk=1024):
    """logits [n, V]; ids/teacher_logp [n, K]; labels [n]; valid [n] bool -> (loss, ce, q_tail)."""
    return TopKReverseKL.apply(logits, ids.long(), teacher_logp, labels.long(), valid.bool(), chunk)


def kd_loss_sums(logits, ids, teacher_logp, labels, mask, chunk=1024, ce_token=None):
    """Loss KD do zsumowania przez Megatron (--calculate-per-token-loss) i metryki [suma, liczba].

    Reverse KL na pozycjach z maską. Z ce_token (np. ID <|im_end|>) pozycje z tym labelem dostają
    zamiast KL zwykłe CE do tokenu z danych: nauczyciel, który pisałby dłużej niż dane (Bielik v3
    Instruct daje na końcach tur v8 średnio p(<|im_end|>) = 0,55), nie uczy wtedy ucznia, żeby nie
    kończył. → (loss_sum z gradientem, {nazwa: (suma, liczba)} bez gradientu).
    """
    mask, labels = mask.bool(), labels.long()
    eos = mask & (labels == ce_token) if ce_token is not None else torch.zeros_like(mask)
    kd = mask & ~eos
    kl, ce, tail = topk_reverse_kl(logits, ids, teacher_logp, labels, kd, chunk)
    loss = kl.sum()
    n_kd = kd.sum().float()
    ce_sum = ce.sum()
    metrics = {"kd loss": (kl.sum().detach(), n_kd), "student tail": (tail.sum(), n_kd)}
    if ce_token is not None:
        idx = eos.nonzero(as_tuple=True)[0]
        eos_ce = -torch.log_softmax(logits[idx].float(), dim=-1).gather(1, labels[idx, None])[:, 0]
        loss = loss + eos_ce.sum()
        ce_sum = ce_sum + eos_ce.sum().detach()
        metrics["eos ce"] = (eos_ce.sum().detach(), eos.sum().float())
    metrics["student ce"] = (ce_sum, mask.sum().float())
    return loss, metrics
