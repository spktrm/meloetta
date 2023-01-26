import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


def binary_enc_matrix(num_embeddings: int):
    bits = math.ceil(math.log2(num_embeddings))
    mask = 2 ** torch.arange(bits)
    x = torch.arange(mask.sum().item() + 1)
    embs = x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    return embs[..., :num_embeddings:, :]


def sqrt_one_hot_matrix(num_embeddings: int):
    x = torch.arange(num_embeddings)
    i = (x**0.5).to(torch.long)
    return F.one_hot(i, torch.max(i) + 1).to(torch.float)


def _legal_policy(logits: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
    """A soft-max policy that respects legal_actions."""
    # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
    l_min = logits.min(axis=-1, keepdim=True).values
    logits = torch.where(legal_actions, logits, l_min)
    logits -= logits.max(axis=-1, keepdim=True).values
    logits *= legal_actions
    exp_logits = torch.where(
        legal_actions, torch.exp(logits), 0
    )  # Illegal actions become 0.
    exp_logits_sum = torch.sum(exp_logits, axis=-1, keepdim=True)
    policy = exp_logits / exp_logits_sum
    return policy


def _log_policy(logits: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
    """Return the log of the policy on legal action, 0 on illegal action."""
    # logits_masked has illegal actions set to -inf.
    logits_masked = logits + torch.log(legal_actions)
    max_legal_logit = logits_masked.max(axis=-1, keepdim=True).values
    logits_masked = logits_masked - max_legal_logit
    # exp_logits_masked is 0 for illegal actions.
    exp_logits_masked = torch.exp(logits_masked)

    baseline = torch.log(torch.sum(exp_logits_masked, axis=-1, keepdim=True))
    # Subtract baseline from logits. We do not simply return
    #     logits_masked - baseline
    # because that has -inf for illegal actions, or
    #     legal_actions * (logits_masked - baseline)
    # because that leads to 0 * -inf == nan for illegal actions.
    log_policy = torch.multiply(legal_actions, (logits - max_legal_logit - baseline))
    return log_policy


def gather_along_rows(
    x: torch.Tensor, index: torch.Tensor, dim: int = 1, keepdim: bool = True
):
    """
    x: torch.Tensor -> [B, T, D]
    dim: int
    index: torch.Tensor -> [B,], values in range (0, T)
    """
    assert len(x.shape) == 3
    B, _, D = x.shape
    index = index.view(B, 1).repeat(1, D).view(B, 1, D)
    rows = torch.gather(x, dim, index)
    if keepdim:
        return rows
    else:
        return rows.squeeze(dim)


class GLU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, context_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(context_dim, input_dim)
        self.lin2 = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        gate = self.lin1(context)
        gate = torch.sigmoid(gate)
        x = gate * x
        x = self.lin2(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.lin1 = nn.Linear(features, features)
        self.lin2 = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = F.relu(x + res)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float, bias_value: float = -1e9):
        super().__init__()
        self.temperature = temperature
        self.biasval = bias_value

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        # q: (b, n, lq, dk)
        # k: (b, n, lk, dk)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # atten: (b, n, lq, lk),
        if mask is not None:
            attn = attn.masked_fill(mask, self.biasval)

        attn = F.softmax(attn, dim=-1)

        # v: (b, n, lv, dv)
        # r: (b, n, lq, dv)
        r = torch.matmul(attn, v)

        return r, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # pre-attention projection
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        # after-attention projection
        self.conv = nn.Conv1d(d_v, d_model, 1)

        # attention
        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # q: (b, lq, dm)
        # k: (b, lk, dm)
        # v: (b, lv, dm)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        size_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # pass through the pre-attention projection
        # separate different heads

        # after that q: (b, lq, n, dk)
        q = self.w_qs(q).view(size_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(size_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(size_b, len_v, n_head, d_v)

        # transpose for attention dot product: (b, n, lq, dk)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # merge key padding and attention masks
        if mask is not None:
            mask = mask.view(size_b, 1, 1, len_q).expand(-1, n_head, -1, -1)

        q, attn = self.attention(q, k, v, mask=mask)
        q = q.view(size_b * n_head, len_q, d_v).transpose(-2, -1)
        q = self.conv(q).transpose(-2, -1)
        q = q.view(size_b, n_head, len_q, -1).sum(1)

        return q


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in: int, d_hid: int):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise

    def forward(self, x):
        x = self.w_2(F.relu(self.w_1(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_inner: int = 1024,
        n_layers: int = 3,
        n_head: int = 2,
        d_k: int = 128,
        d_v: int = 128,
    ):

        super().__init__()

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for enc_layer in self.layer_stack:
            x = enc_layer(x, mask=mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_inner: int = 1024,
        n_head: int = 2,
        d_k: int = 128,
        d_v: int = 128,
    ):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner)

        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        att_out = self.slf_attn(x, x, x, mask=mask)

        out_1 = self.ln1(att_out + att_out)

        ffn_out = self.pos_ffn(out_1)

        out = self.ln2(out_1 + ffn_out)

        return out
