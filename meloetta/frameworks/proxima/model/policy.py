import torch
import torch.nn as nn

from typing import Union
from collections import OrderedDict

from meloetta.actors.types import TensorDict
from meloetta.frameworks.proxima.model import config
from meloetta.frameworks.proxima.model.utils import (
    MLP,
    Resnet,
    VectorMerge,
    _legal_policy,
    _multinomial,
)


class FunctionHead(nn.Module):
    def __init__(
        self,
        gen: int,
        gametype: str,
        name: str,
        config: Union[config.ActionTypeHeadConfig, config.FlagHeadConfig],
    ):
        super().__init__()

        self.name = name
        self.config = config
        self.gametype = gametype
        self.gen = gen

        self.resnet = Resnet(config.input_dim, config.num_layers_resnet)
        self.logits = MLP(
            [config.input_dim for _ in range(config.num_layers_mlp)]
            + [config.num_actions]
        )
        self.embedding = nn.Embedding(config.num_actions, config.input_dim)
        self.merge = VectorMerge(
            OrderedDict(state=config.input_dim, action=config.input_dim),
            config.input_dim,
        )

    def forward(
        self,
        state_embedding: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor = None,
    ) -> TensorDict:
        state_embedding_ = self.resnet(state_embedding)
        action_logits = self.logits(state_embedding_)
        action_policy = _legal_policy(action_logits, mask)
        if action is None:
            action = _multinomial(action_policy)
        action_embedding = self.embedding(action)
        state_action_embedding = self.merge(
            OrderedDict(state=state_embedding, action=action_embedding)
        )
        return OrderedDict(
            {
                f"{self.name}_logits": action_logits,
                f"{self.name}_policy": action_policy,
                f"{self.name}_index": action,
                "state_embedding": state_action_embedding,
            }
        )


class EmbeddingSelectHead(nn.Module):
    def __init__(
        self,
        gen: int,
        gametype: str,
        name: str,
        config: Union[config.MoveHeadConfig, config.SwitchHeadConfig],
    ):
        super().__init__()

        self.name = name
        self.config = config
        self.gametype = gametype
        self.gen = gen

        self.resnet = Resnet(config.input_dim, config.num_layers_resnet)
        self.query_mlp = MLP(
            [config.query_dim for _ in range(config.num_layers_query)]
            + [config.key_dim]
        )
        self.keys_mlp = MLP(
            [config.key_dim for _ in range(config.num_layers_key)] + [config.key_dim]
        )
        self.denom = config.key_dim**0.5

    def forward(
        self,
        state_embedding: torch.Tensor,
        entity_embeddings: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor = None,
    ) -> TensorDict:
        state_embedding_ = self.resnet(state_embedding)
        query = self.query_mlp(state_embedding_).unsqueeze(-2)
        keys = self.keys_mlp(entity_embeddings)
        action_logits = keys @ query.transpose(-2, -1)
        action_logits = (action_logits / self.denom).squeeze(-1)
        action_policy = _legal_policy(action_logits, mask)
        if action is None:
            action = _multinomial(action_policy)
        return OrderedDict(
            {
                f"{self.name}_logits": action_logits,
                f"{self.name}_policy": action_policy,
                f"{self.name}_index": action,
            }
        )


class PolicyHeads(nn.Module):
    def __init__(self, gen: int, gametype: str, config: config.PolicyHeadsConfig):
        super().__init__()

        self.config = config
        self.gametype = gametype
        self.gen = gen

        self.action_type_head = FunctionHead(
            gen, gametype, "action_type", config.action_type_head_config
        )
        self.flag_head = FunctionHead(gen, gametype, "flag", config.flag_head_config)
        self.move_head = EmbeddingSelectHead(
            gen, gametype, "move", config.move_head_config
        )
        self.switch_head = EmbeddingSelectHead(
            gen, gametype, "switch", config.switch_head_config
        )

    def forward(
        self,
        state_embedding: torch.Tensor,
        encoder_output: TensorDict,
        state: TensorDict,
    ) -> TensorDict:
        action_type = self.action_type_head(
            state_embedding,
            state.get("action_type_mask"),
            state.get("action_type_index"),
        )
        flag = self.flag_head(
            action_type["state_embedding"],
            state.get("flag_mask"),
            state.get("flag_index"),
        )
        move = self.move_head(
            flag["state_embedding"],
            encoder_output["active_move_embeddings"],
            state.get("move_mask"),
            state.get("move_index"),
        )
        switch = self.switch_head(
            flag["state_embedding"],
            encoder_output["switch_embeddings"],
            state.get("switch_mask"),
            state.get("switch_index"),
        )
        action_type.pop("state_embedding")
        flag.pop("state_embedding")
        return OrderedDict(**action_type, **flag, **move, **switch)
