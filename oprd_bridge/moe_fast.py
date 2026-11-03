"""Szybki trening MoE Poziomki: eksperci w jednym tensorze + grouped GEMM.

Remote code z checkpointu (modeling_bailing_moe_v2.py z Ling-V2) w trybie
train powiela kazdy token num_experts_per_tok razy (32x) i robi petle po 128
ekspertach z maskowaniem. Tu tokeny sortujemy po ekspercie i liczymy wszystkich
ekspertow jednym grouped_mm na projekcje (wzorzec z transformers 5
integrations/moe.py). Routing (bramka) zostaje z remote code bez zmian.

Wagi na czas treningu siedza w dwoch parametrach na blok:
  gate_up (E, 2I, H) = [gate_proj.weight; up_proj.weight] kazdego eksperta
  down    (E, H, I)  = down_proj.weight
expert_state_dict() oddaje je z powrotem w kluczach HF per ekspert, wiec zapis
checkpointu i synchronizacja z SGLangiem widza zwykly uklad Poziomki.
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def _grouped_mm(x, w, offs):
    """x (S, K) posortowane po ekspercie, w (E, K, N), offs = cumsum liczby tokenow (int32)."""
    if hasattr(F, "grouped_mm"):  # torch >= 2.10, SM80+
        return F.grouped_mm(x, w, offs=offs)
    if hasattr(torch, "_grouped_mm") and torch.cuda.get_device_capability(x.device) >= (8, 0):
        return torch._grouped_mm(x, w, offs=offs)
    # petla po segmentach - roznicowalna, jedna synchronizacja CPU na wywolanie
    outs, start = [], 0
    for e, end in enumerate(offs.tolist()):
        if end > start:
            outs.append(x[start:end] @ w[e])
        start = end
    return torch.cat(outs) if outs else x.new_zeros(0, w.shape[-1])


class FusedExperts(nn.Module):
    def __init__(self, experts: nn.ModuleList):
        super().__init__()
        self.num_experts = len(experts)
        self.inter = experts[0].gate_proj.weight.shape[0]
        self.act_fn = experts[0].act_fn
        self.gate_up = nn.Parameter(torch.stack([
            torch.cat([e.gate_proj.weight, e.up_proj.weight], dim=0) for e in experts
        ]).detach().clone())
        self.down = nn.Parameter(torch.stack([e.down_proj.weight for e in experts]).detach().clone())

    def forward(self, x, topk_idx, topk_weight):
        """x (T, H), topk_idx/topk_weight (T, K) -> (T, H) w dtype x."""
        n_tok, k = topk_idx.shape
        flat = topk_idx.reshape(-1)
        expert_sorted, perm = torch.sort(flat)
        tok = perm // k
        counts = torch.histc(expert_sorted.int(), bins=self.num_experts, min=0, max=self.num_experts - 1)
        offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
        xs = x[tok]
        gu = _grouped_mm(xs, self.gate_up.transpose(-2, -1), offs)
        g, u = gu.split(self.inter, dim=-1)
        out = _grouped_mm(self.act_fn(g) * u, self.down.transpose(-2, -1), offs)
        # jak w remote code: wyniki wracaja na miejsca (token, k), mnozone przez wagi
        # routingu w fp32 i sumowane po k w stalej kolejnosci (index_add_ sumowalby
        # atomowo, niedeterministycznie)
        unsorted = torch.empty_like(out)
        unsorted[perm] = out
        y = (unsorted.view(n_tok, k, -1).float() * topk_weight.float().unsqueeze(-1)).sum(dim=1)
        return y.to(x.dtype)


def _fast_block_forward(self, hidden_states):
    identity = hidden_states
    bsz, seq_len, h = hidden_states.shape
    topk_idx, topk_weight, router_logits = self.gate(hidden_states)
    y = self.fused_experts(hidden_states.view(-1, h), topk_idx, topk_weight).view(bsz, seq_len, h)
    if self.config.num_shared_experts is not None:
        y = y + self.shared_experts(identity)
    return y, (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))


def moe_blocks(model):
    return [(name, m) for name, m in model.named_modules() if type(m).__name__ == "BailingMoeV2SparseMoeBlock"]


def patch_model(model):
    """Podmienia ekspertow we wszystkich blokach MoE na FusedExperts (w miejscu)."""
    blocks = moe_blocks(model)
    for _, block in blocks:
        block.fused_experts = FusedExperts(block.experts)
        del block.experts
        block.forward = types.MethodType(_fast_block_forward, block)
    return len(blocks)


def expert_state_dict(model):
    """Wagi ekspertow w kluczach HF per ekspert (do zapisu / synchronizacji z SGLangiem)."""
    out = {}
    for name, block in moe_blocks(model):
        fe = block.fused_experts
        gu, down = fe.gate_up.detach(), fe.down.detach()
        for e in range(fe.num_experts):
            p = f"{name}.experts.{e}"
            out[f"{p}.gate_proj.weight"] = gu[e, : fe.inter]
            out[f"{p}.up_proj.weight"] = gu[e, fe.inter:]
            out[f"{p}.down_proj.weight"] = down[e]
    return out


def hf_state_dict(model):
    """Pelny state_dict w ukladzie oryginalnego checkpointu."""
    sd = {k: v for k, v in model.state_dict().items() if ".fused_experts." not in k}
    sd.update(expert_state_dict(model))
    return sd


def split_fused_state_dict(sd):
    """Pelny state_dict (np. z FSDP) z kluczami fused_experts -> klucze HF per ekspert."""
    out = {}
    for key, value in sd.items():
        if key.endswith(".fused_experts.gate_up"):
            base, inter = key[: -len(".fused_experts.gate_up")], value.shape[1] // 2
            for e in range(value.shape[0]):
                out[f"{base}.experts.{e}.gate_proj.weight"] = value[e, :inter].contiguous()
                out[f"{base}.experts.{e}.up_proj.weight"] = value[e, inter:].contiguous()
        elif key.endswith(".fused_experts.down"):
            base = key[: -len(".fused_experts.down")]
            for e in range(value.shape[0]):
                out[f"{base}.experts.{e}.down_proj.weight"] = value[e].contiguous()
        else:
            out[key] = value
    return out
