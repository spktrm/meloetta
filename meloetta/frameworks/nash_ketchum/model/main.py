import torch.nn as nn

from copy import deepcopy
from typing import Optional, Union, Literal, Dict, Any

from meloetta.actors.types import State

from meloetta.frameworks.nash_ketchum.model.utils import _log_policy, MLP

from meloetta.frameworks.nash_ketchum.model import config
from meloetta.frameworks.nash_ketchum.model import (
    Encoder,
    PolicyHeads,
    Core,
    Policy,
    ValueHead,
    Indices,
    ModelOutput,
    PostProcess,
)

from meloetta.data import _STATE_FIELDS


class NAshKetchumModel(nn.Module):
    def __init__(
        self,
        gen: int = 9,
        gametype: Literal["singles", "doubles", "triples"] = "singles",
        config: config.NAshKetchumModelConfig = config.NAshKetchumModelConfig(),
    ) -> None:
        super().__init__()

        self.gen = gen
        self.gametype = gametype
        n_active = {"singles": 1, "doubles": 2, "triples": 3}[gametype]

        self.encoder = Encoder(gen=gen, n_active=n_active, config=config.encoder_config)
        self.policy_heads = PolicyHeads(
            gen=gen, gametype=gametype, config=config.policy_heads_config
        )

        self.state_fields = deepcopy(_STATE_FIELDS)

        if gen != 8:
            self.state_fields.remove("max_move_mask")

        if self.gametype == "singles":
            self.state_fields.remove("target_mask")
            self.state_fields.remove("prev_choices")

        self.core = Core(config.resnet_core_config)
        self.value_Head = ValueHead(config.value_head_config)

        hidden_dim = config.resnet_core_config.hidden_dim
        self.state_action_mlp = MLP([hidden_dim, hidden_dim, hidden_dim])
        self.state_mlp = MLP([hidden_dim, hidden_dim, hidden_dim])

    def get_learnable_params(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def clean(self, state: State) -> State:
        return {k: state[k] for k in self.state_fields}

    def forward(
        self, state: State, compute_value: bool = True, indices: Indices = None
    ) -> ModelOutput:
        encoder_output = self.encoder.forward(state)
        state_emb = self.core.forward(encoder_output)
        if compute_value:
            value = self.value_Head.forward(state_emb)
        else:
            value = None
        indices, logits, policy, _ = self.policy_heads.forward(
            state_emb,
            encoder_output,
            state,
            indices,
        )
        return ModelOutput(
            indices=indices,
            logit=logits,
            pi=policy,
            v=value,
            # state_action_emb=self.state_action_mlp(state_action_emb),
            # state_emb=self.state_mlp(state_emb),
        )

    def acting_forward(self, state: State) -> ModelOutput:
        return self.forward(state, compute_value=False)

    def learning_forward(self, state: State, indices: Indices = None) -> ModelOutput:
        model_output = self.forward(state, compute_value=True, indices=indices)
        return self.postprocess(state, model_output)

    def postprocess(
        self,
        state: State,
        model_output: ModelOutput,
        choices: Optional[Dict[str, Any]] = None,
    ) -> Union[ModelOutput, PostProcess]:
        """post process method"""

        if choices:
            targeting = bool(state.get("targeting"))
            output = self.index_to_action(targeting, model_output.indices, choices)

        else:
            output = ModelOutput(
                indices=model_output.indices,
                pi=model_output.pi,
                v=model_output.v,
                log_pi=self.get_log_policy(model_output.logit, state),
                logit=model_output.logit,
            )

        return output

    def get_log_policy(self, logits: Policy, state: State):
        if self.gametype != "singles":
            target_log_policy = _log_policy(
                logits.target,
                state["target_mask"],
            )
        else:
            target_log_policy = None

        if self.gen == 8:
            max_move_log_policy = _log_policy(
                logits.max_move,
                state["max_move_mask"],
            )
        else:
            max_move_log_policy = None

        if self.gen >= 6:
            flag_log_policy = _log_policy(
                logits.flag,
                state["flag_mask"],
            )
        else:
            flag_log_policy = None

        return Policy(
            action_type=_log_policy(logits.action_type, state["action_type_mask"]),
            move=_log_policy(logits.move, state["move_mask"]),
            switch=_log_policy(logits.switch, state["switch_mask"]),
            max_move=max_move_log_policy,
            flag=flag_log_policy,
            target=target_log_policy,
        )

    def index_to_action(
        self,
        targeting: bool,
        indices: Indices,
        choices: Optional[Dict[str, Any]] = None,
    ):
        action_type_index = indices.action_type_index.item()
        move_index = indices.move_index.item()
        switch_index = indices.switch_index.item()

        if not targeting:
            index = action_type_index

            if action_type_index == 0:
                flag_index = (indices.flag_index).item()
                if flag_index == 3:
                    max_move_index = (indices.max_move_index).item()
                    index = max_move_index
                else:
                    index = move_index

                if choices is not None:
                    if flag_index == 0:
                        data = choices["moves"]
                    elif flag_index == 1:
                        data = choices["mega_moves"]
                    elif flag_index == 2:
                        data = choices["zmoves"]
                    elif flag_index == 3:
                        data = choices["max_moves"]
                    elif flag_index == 4:
                        data = choices["tera_moves"]
                    else:
                        data = choices["moves"]
                else:
                    data = None

            elif action_type_index == 1:
                if choices is not None:
                    data = choices["switches"]
                else:
                    data = None
                index = switch_index

        else:
            if choices is not None:
                data = choices["targets"]
            else:
                data = None

            target_index = (indices.target_index).item()
            index = target_index

        try:
            return PostProcess(data, index)
        except:
            print()
