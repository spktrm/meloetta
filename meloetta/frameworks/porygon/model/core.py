import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List

from meloetta.frameworks.porygon.model.interfaces import EncoderOutput

from meloetta.frameworks.porygon.model import config
from meloetta.frameworks.porygon.model.utils import Resblock


def script_lnlstm(
    input_size,
    hidden_size,
    num_layers,
    bias=True,
    batch_first=False,
    dropout=False,
) -> "StackedLSTM":
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""

    # The following are not implemented.
    assert bias
    assert not batch_first
    assert not dropout

    stack_type = StackedLSTM
    layer_type = LSTMLayer
    dirs = 1

    return stack_type(
        num_layers,
        layer_type,
        first_layer_args=[
            LayerNormLSTMCell,
            input_size,
            hidden_size,
        ],
        other_layer_args=[
            LayerNormLSTMCell,
            hidden_size * dirs,
            hidden_size,
        ],
    )


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [
        layer(*other_layer_args) for _ in range(num_layers - 1)
    ]
    return nn.ModuleList(layers)


class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        # The layernorms provide learnable biases
        ln = nn.LayerNorm

        self.layernorm_i = ln(4 * hidden_size)
        self.layernorm_h = ln(4 * hidden_size)
        self.layernorm_c = ln(hidden_size)

    def forward(
        self, input: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(
        self, input: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class StackedLSTM(nn.Module):
    __constants__ = ["layers"]  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTM, self).__init__()
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

    def forward(
        self, input: torch.Tensor, states: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # List[LSTMState]: One state per layer
        h_out = []
        c_out = []
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer, hi, ci in zip(self.layers, *states):
            state = (hi, ci)
            output, (ho, co) = rnn_layer(output, state)
            h_out += [ho]
            c_out += [co]
            i += 1
        h_out = torch.stack(h_out)
        c_out = torch.stack(c_out)
        return output, (h_out, c_out)


class CoreResblock(nn.Module):
    def __init__(self, inplanes, outplanes, end_dim):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(inplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(end_dim),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(outplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(end_dim),
        )

    def forward(self, x):
        return self.conv2(self.conv2(x)) + x


class Core(nn.Module):
    def __init__(self, config: config.CoreConfig):
        super().__init__()
        self.config = config

        # self.project_in = nn.Linear(config.raw_embedding_dim, config.hidden_dim)

        layers = [10, 64, 64, 128, 256, config.hidden_dim]
        end_dims = [256, 128, 64, 32, 16]

        self.resblock_in = nn.ModuleList(
            [
                nn.Sequential(
                    *[CoreResblock(s1, s1, e) for _ in range(2)]
                    + [nn.Conv1d(s1, s2, kernel_size=1, stride=2)]
                )
                for s1, s2, e in zip(layers, layers[1:], end_dims)
            ]
        )

        # self.resblock_stack = nn.Sequential(
        #     *[Resblock(config.hidden_dim, use_layer_norm=True) for _ in range(4)]
        # )

        # self.rnn = script_lnlstm(
        #     config.raw_embedding_dim,
        #     config.hidden_dim,
        #     num_layers=config.num_layers,
        # )

        # self.rnn = nn.GRU(
        #     input_size=config.raw_embedding_dim,
        #     hidden_size=config.hidden_dim,
        #     num_layers=config.num_layers,
        # )

    def initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # def initial_state(self, batch_size: int) -> torch.Tensor:
        return tuple(
            torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim)
            for _ in range(2)
        )
        # return torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim)

    def input_resblocks(self, state_embedding: torch.Tensor):
        T, B, *_ = state_embedding.shape
        x = state_embedding.flatten(0, 1)
        for layer in self.resblock_in:
            x = layer(x)
        return x.mean(-1).view(T, B, -1)

    def forward(
        self,
        encoder_output: EncoderOutput,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
    ):
        state_embedding = torch.cat(
            encoder_output.side_embedding
            + [
                torch.cat(
                    (encoder_output.weather_emb, encoder_output.scalar_emb), dim=-1
                ).unsqueeze(-2)
            ],
            dim=-2,
        )

        state_embedding = self.input_resblocks(state_embedding)
        # state_embedding = self.resblock_stack(state_embedding)

        # state_embedding, hidden_state = self.rnn(state_embedding, hidden_state)

        return state_embedding, hidden_state