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


def power_one_hot_matrix(num_embeddings: int, power: float):
    x = torch.arange(num_embeddings)
    i = (x**power).to(torch.long)
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


class Resblock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        layers = []
        for _ in range(num_layers):
            hiddens = [nn.ReLU(), nn.Linear(hidden_size, hidden_size)]
            if use_layer_norm:
                hiddens = [nn.LayerNorm(hidden_size)] + hiddens
            layers.append(nn.Sequential(*hiddens))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        shortcut = x
        for layer in self.layers:
            x = layer(x)
        return x + shortcut


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        key_size: int,
        value_size: int = None,
        model_size: int = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.key_size = key_size
        value_size = value_size or key_size
        model_size = model_size or key_size * num_heads
        self.value_size = value_size
        self.model_size = model_size

        self.lin_q = nn.Linear(input_size, num_heads * key_size)
        self.lin_k = nn.Linear(input_size, num_heads * key_size)
        self.lin_v = nn.Linear(input_size, num_heads * value_size)
        self.lin_out = nn.Linear(num_heads * value_size, model_size)

        self.attn_denom = key_size**0.5

    def _linear_projection(
        self, x: torch.Tensor, layer: nn.Module, head_size: int
    ) -> torch.Tensor:
        y = layer(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_heads, head_size))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ):
        *leading_dims, sequence_length, _ = query.shape

        query_heads = self._linear_projection(query, self.lin_q, self.key_size)
        key_heads = self._linear_projection(key, self.lin_k, self.key_size)
        value_heads = self._linear_projection(value, self.lin_v, self.value_size)

        attn_logits = torch.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        attn_logits = attn_logits / self.attn_denom

        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim}."
                )
            attn_logits = torch.where(mask, attn_logits, float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [H, T', T]

        # Weight the values by the attention and flatten the head vectors.
        attn = torch.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = torch.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        return self.lin_out(attn)  # [T', D']


class TransformerLayer(nn.Module):
    def __init__(
        self,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_value_size: int,
        input_size: int,
        model_size: int,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self._use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln = nn.LayerNorm(model_size)
        self.mha = MultiHeadAttention(
            num_heads=transformer_num_heads,
            key_size=transformer_key_size,
            value_size=transformer_value_size,
            model_size=model_size,
            input_size=input_size,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x1 = x
        if self._use_layer_norm:
            x1 = self.ln(x1)

        x1 = F.relu(x1)
        # The logits mask has shape [num_heads, num_units, num_units]:
        x1 = self.mha(x1, x1, x1, mask[:, None, None, :])

        # Mask here mostly for safety:
        x1 = torch.where(mask.unsqueeze(-1), x1, 0)
        x = x + x1

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        model_size: int,
        num_layers: int,
        num_heads: int,
        key_size: int,
        value_size: int,
        resblocks_num_before: int,
        resblocks_num_after: int,
        resblocks_hidden_size: Optional[int] = None,
        use_layer_norm: bool = True,
    ):

        super().__init__()
        self._transformer_num_layers = num_layers
        self._transformer_num_heads = num_heads
        self._transformer_key_size = key_size
        self._transformer_value_size = value_size

        if resblocks_hidden_size is None:
            resblocks_hidden_size = model_size

        self._transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    num_heads,
                    key_size,
                    value_size,
                    input_size=resblocks_hidden_size,
                    model_size=model_size,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_layers)
            ]
        )

        self._resblocks_before = nn.ModuleList(
            [
                Resblock(
                    hidden_size=resblocks_hidden_size,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(resblocks_num_before)
            ]
        )
        self._resblocks_after = nn.ModuleList(
            [
                Resblock(
                    hidden_size=resblocks_hidden_size,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(resblocks_num_after)
            ]
        )

        self._resblocks_hidden_size = resblocks_hidden_size
        self._use_layer_norm = use_layer_norm

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for resblock_before in self._resblocks_before:
            x = resblock_before(x)

        for transformer_layer in self._transformer_layers:
            x = transformer_layer(x, mask)

        for resblock_after in self._resblocks_after:
            x = resblock_after(x)

        # Mask here mostly for safety:
        x = torch.where(mask.unsqueeze(-1), x, 0)
        return x


class AttentionPool(nn.Module):
    def __init__(self, embed_dim: int, nhead: int) -> None:
        super().__init__()

        self.nhead = nhead
        self.lin_qk = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        TB, S, C = x.shape

        q, k, v = torch.chunk(self.lin_qk(x), 3, 2)
        q = q.view(TB, S, self.nhead, C // self.nhead).transpose(1, 2)
        k = k.view(TB, S, self.nhead, C // self.nhead).transpose(1, 2)
        v = v.view(TB, S, self.nhead, C // self.nhead).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = (mask * mask.transpose(-2, -1)).unsqueeze(1)
        attn = torch.masked_fill(attn, mask, float("-inf"))
        score = F.softmax(attn, -1)

        y = score @ v
        y = y.transpose(1, 2).contiguous().view(TB, S, C)
        y = y.sum(1, keepdim=True)
        y = self.proj(y)

        return y


class ToVector(nn.Module):
    """Per-unit processing then average over the units dimension."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * input_dim

        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.lin1 = nn.Linear(input_dim, hidden_dim // 2)
        self.lin2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        x = F.relu(x)
        x = self.lin1(x)
        # x = x * mask

        x = torch.cat(
            (
                x[..., :1, :].max(1).values,
                x[..., 1:, :].max(1).values,
            ),
            dim=-1,
        )
        x = self.ln2(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x
