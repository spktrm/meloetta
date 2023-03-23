import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List

from meloetta.frameworks.mewzero.model.interfaces import EncoderOutput

from meloetta.frameworks.mewzero.model import config


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


class Core(nn.Module):
    def __init__(self, config: config.CoreConfig):
        super().__init__()
        self.config = config

        self.lstm = script_lnlstm(
            config.raw_embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
        )

    def initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim)
            for _ in range(2)
        )

    def forward(
        self,
        encoder_output: EncoderOutput,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
    ):
        state_embedding = torch.cat(
            (
                encoder_output.private_entity_emb,
                encoder_output.public_entity_emb,
                encoder_output.public_spatial,
                encoder_output.private_spatial,
                encoder_output.weather_emb,
                encoder_output.scalar_emb,
            ),
            dim=-1,
        )

        state_embedding, hidden_state = self.lstm(state_embedding, hidden_state)

        return state_embedding, hidden_state
