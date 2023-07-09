import math

import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Sequence, Mapping, Dict, List

BIAS = -1e4
USE_LAYER_NORM = False


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def linear_layer(
    in_features: int,
    out_features: int,
    bias: bool = True,
    mean: float = 0.0,
    std: float = None,
    bias_const: float = 0.0,
) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    if std is None:
        std = 1 / math.sqrt(in_features)
    nn.init.trunc_normal_(layer.weight, mean=mean, std=std)
    if bias:
        nn.init.constant_(layer.bias, bias_const)
    return layer


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


def _legal_policy(
    logits: torch.Tensor, legal_actions: torch.Tensor = None
) -> torch.Tensor:
    """A soft-max policy that respects legal_actions."""
    # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
    if legal_actions is not None:
        logits = torch.where(legal_actions, logits, float("-inf"))
    return logits.softmax(-1)


def _multinomial(dist: torch.Tensor) -> torch.Tensor:
    """
    We use this method instead of `torch.multinomial` for two reasons:
    see: https://github.com/pytorch/pytorch/blob/9e089db32e6274e8eb094829afc97f2ddcd6f1ef/aten/src/ATen/native/Distributions.cpp#L632

    - The current version of torch is bugged for multinomial and selects invalid indices with a very small prob.
    - numpy is significantly faster at the operation, even on the cpu
    """
    og_shape = dist.shape
    np_dist = dist.cpu().flatten(0, -2).numpy().astype(np.float64)
    np_dist = np_dist / np_dist.sum(-1, keepdims=True)
    outs = []
    for i in range(np_dist.shape[0]):
        index = np.random.multinomial(1, np_dist[i])
        index = index.argmax(-1)
        index = torch.tensor(index)
        outs.append(index)
    return torch.stack(outs).view(*og_shape[:-1])


def _log_policy(logits: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
    log_policy = torch.where(legal_actions, logits, float("-inf"))
    log_policy = log_policy.log_softmax(-1)
    log_policy = torch.where(legal_actions, log_policy, 0)
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


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes: Sequence[int],
        use_layer_norm: bool = USE_LAYER_NORM,
    ):
        super().__init__()
        layers = []
        for (
            in_features,
            out_features,
        ) in zip(layer_sizes, layer_sizes[1:]):
            lin = linear_layer(in_features, out_features)
            hiddens = [nn.ReLU(), lin]
            if use_layer_norm:
                hiddens.insert(0, nn.LayerNorm(in_features))
            layers += hiddens

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.network(x)


class VectorResblock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        num_layers: int = 2,
        use_layer_norm: bool = USE_LAYER_NORM,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm

        layers = []
        hidden_size = input_size
        for i in range(num_layers):
            if i < num_layers - 1:
                output_size = self.hidden_size or input_size
            else:
                output_size = input_size

            lin = nn.Linear(hidden_size, output_size)
            nn.init.normal_(lin.weight, std=5e-3)
            nn.init.constant_(lin.bias, val=0.0)

            hiddens = [nn.ReLU(), lin]
            if use_layer_norm:
                hiddens.insert(0, nn.LayerNorm(hidden_size))

            hidden_size = output_size

            mod = nn.Sequential(*hiddens)
            layers.append(mod)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        shortcut = x
        for layer in self.layers:
            x = layer(x)
        return x + shortcut


class Resnet(nn.Module):
    def __init__(self, input_size: int, num_layers: int):
        super().__init__()

        self.resblocks = nn.ModuleList(
            [VectorResblock(input_size) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resblock in self.resblocks:
            x = resblock(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
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

        denom = 0.87962566103423978
        qkv_std = (1 / math.sqrt(model_size)) / denom

        self.lin_q = linear_layer(model_size, num_heads * key_size, std=qkv_std)
        self.lin_k = linear_layer(model_size, num_heads * key_size, std=qkv_std)
        self.lin_v = linear_layer(model_size, num_heads * value_size, std=qkv_std)

        out_std = (1 / math.sqrt(num_heads * value_size)) / denom
        self.lin_out = linear_layer(num_heads * value_size, model_size, std=out_std)

        self.attn_denom = math.sqrt(key_size)

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
            if mask.ndim == 2:
                mask = mask.view(*leading_dims, 1, 1, sequence_length)
                mask = mask.expand(-1, self.num_heads, -1, -1)
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
        model_size: int,
        use_layer_norm: bool = USE_LAYER_NORM,
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
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x1 = x
        if self._use_layer_norm:
            x1 = self.ln(x1)

        x1 = F.relu(x1)
        # The logits mask has shape [num_heads, num_units, num_units]:
        x1 = self.mha(x1, x1, x1, mask)

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
        use_layer_norm: bool = USE_LAYER_NORM,
    ):
        super().__init__()
        self._transformer_num_layers = num_layers
        self._transformer_num_heads = num_heads
        self._transformer_key_size = key_size
        self._transformer_value_size = value_size

        if resblocks_hidden_size is None:
            resblocks_hidden_size = model_size

        self._resblocks_before = nn.ModuleList(
            [
                VectorResblock(
                    input_size=model_size,
                    hidden_size=resblocks_hidden_size,
                    use_layer_norm=use_layer_norm,
                )
                for i in range(resblocks_num_before)
            ]
        )

        self._transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    num_heads,
                    key_size,
                    value_size,
                    model_size=model_size,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_layers)
            ]
        )

        self._resblocks_after = nn.ModuleList(
            [
                VectorResblock(
                    input_size=model_size,
                    hidden_size=resblocks_hidden_size,
                    use_layer_norm=use_layer_norm,
                )
                for i in range(resblocks_num_after)
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


class ToVector(nn.Module):
    """Per-unit processing then average over the units dimension."""

    def __init__(self, input_channels: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(input_channels, 16, 2),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4),
            nn.MaxPool1d(2),
        )
        self.mlp = MLP([hidden_dim, output_dim])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), x, 0)
        x = self.out(x)
        x = self.mlp(x.flatten(1))
        return x


class VectorMerge(nn.Module):
    def __init__(
        self,
        input_sizes: Mapping[str, Optional[int]],
        output_size: int,
        gating_type: str = "none",
        use_layer_norm: bool = USE_LAYER_NORM,
    ):
        super().__init__()

        if not input_sizes:
            raise ValueError("input_names cannot be empty")

        self._input_sizes = input_sizes
        self._output_size = output_size
        self._gating_type = gating_type
        self._use_layer_norm = use_layer_norm

        def _get_pre_layer(value):
            layers = [nn.ReLU()]
            if use_layer_norm:
                layers.insert(0, nn.LayerNorm(value))
            return nn.Sequential(*layers)

        self.pre_layers = nn.ModuleDict(
            {key: _get_pre_layer(value) for key, value in input_sizes.items()}
        )
        self.encoding_layers = nn.ModuleDict(
            {
                key: linear_layer(value, output_size)
                for key, value in input_sizes.items()
            }
        )

        def _get_gating_layer(*args, **kwargs):
            layer = nn.Linear(*args, **kwargs)
            nn.init.normal_(layer.weight, std=5e-3)
            nn.init.constant_(layer.bias, val=0)
            return layer

        self.gating_layers = nn.ModuleList(
            [
                _get_gating_layer(value, len(input_sizes) * value)
                for _, value in input_sizes.items()
            ]
        )

    def _compute_gate(
        self,
        inputs_to_gate: List[torch.Tensor],
        init_gate: List[torch.Tensor],
    ):
        leading_dims = inputs_to_gate[0].shape[:2]
        gate = [self.gating_layers[i](y) for i, y in enumerate(init_gate)]
        gate = sum(gate)
        gate = gate.reshape(*leading_dims, len(inputs_to_gate), self._output_size)
        gate = torch.softmax(gate, dim=len(leading_dims))
        gate = gate.chunk(len(init_gate), len(leading_dims))
        return gate

    def _encode(self, inputs: Dict[str, torch.Tensor]):
        gate, outputs = [], []
        for name, _ in self._input_sizes.items():
            feature = inputs[name]
            feature = self.pre_layers[name](feature)
            gate.append(feature)
            outputs.append(self.encoding_layers[name](feature))
        return gate, outputs

    def forward(self, inputs: Dict[str, torch.Tensor]):
        gate, outputs = self._encode(inputs)
        if len(outputs) == 1:
            # Special case of 1-D inputs that do not need any gating.
            output = outputs[0]
        else:
            gate = self._compute_gate(outputs, gate)
            data = [g.squeeze(-2) * o for g, o in zip(gate, outputs)]
            output = sum(data)
        return output
