import torch
import torch.nn as nn

from collections import OrderedDict
from typing import Optional, Union, Literal, Dict, Any

from meloetta.actors.types import State, TensorDict

from meloetta.frameworks.proxima.model.utils import _log_policy

from meloetta.frameworks.proxima.model import config
from meloetta.frameworks.proxima.model import (
    Encoder,
    PolicyHeads,
    Core,
    ValueHead,
    PostProcess,
)


class ProximaModel(nn.Module):
    def __init__(
        self,
        gen: int = 9,
        gametype: Literal["singles", "doubles", "triples"] = "singles",
        config: config.ProximaModelConfig = config.ProximaModelConfig(),
    ) -> None:
        super().__init__()

        self.gen = gen
        self.gametype = gametype
        n_active = {"singles": 1, "doubles": 2, "triples": 3}[gametype]

        self.encoder = Encoder(gen=gen, n_active=n_active, config=config.encoder_config)
        self.policy_heads = PolicyHeads(
            gen=gen, gametype=gametype, config=config.policy_heads_config
        )

        self._STATE_FIELDS = {
            "sides",
            "boosts",
            "volatiles",
            "side_conditions",
            "pseudoweathers",
            "weather",
            "wisher",
            "scalars",
            # "action_log",
            "prev_choices",
            "choices_done",
            "action_type_mask",
            "move_mask",
            "max_move_mask",
            "switch_mask",
            "flag_mask",
            "target_mask",
        }

        if gen != 8:
            self._STATE_FIELDS.remove("max_move_mask")

        if self.gametype == "singles":
            self._STATE_FIELDS.remove("target_mask")
            self._STATE_FIELDS.remove("prev_choices")

        self.core = Core(config.resnet_core_config)
        self.value_head = ValueHead(config.value_head_config)

    def get_learnable_params(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def clean(self, state: State) -> State:
        return {k: state[k] for k in self._STATE_FIELDS}

    def forward(self, state: State, compute_value: bool = True) -> TensorDict:
        encoder_output = self.encoder.forward(state)
        state_emb = self.core.forward(encoder_output)
        if compute_value:
            value = self.value_head.forward(state_emb)
        else:
            value = None
        action_policy_logits = self.policy_heads.forward(
            state_emb, encoder_output, state
        )
        return OrderedDict(
            **action_policy_logits,
            value=value,
        )

    def acting_forward(self, state: State) -> TensorDict:
        return self.forward(state, compute_value=False)

    def learning_forward(self, state: State, compute_value: bool = True) -> TensorDict:
        model_output = self.forward(state, compute_value=compute_value)
        return self.postprocess(state, model_output)

    def postprocess(
        self,
        state: State,
        model_output: TensorDict,
        choices: Optional[Dict[str, Any]] = None,
    ) -> Union[TensorDict, PostProcess]:
        """post process method"""

        if choices:
            targeting = bool(state.get("targeting"))
            actions = {
                k: v.squeeze().item()
                for k, v in model_output.items()
                if k.endswith("_index") and v is not None
            }
            output = self.index_to_action(targeting, choices, **actions)

        else:
            output = OrderedDict(
                **model_output,
                **{
                    k.replace("_logits", "_log_policy"): _log_policy(
                        v, state[k.replace("_logits", "_mask")]
                    )
                    for k, v in model_output.items()
                    if k.endswith("_logits") and v is not None
                },
            )

        return output

    def index_to_action(
        self,
        targeting: bool,
        choices: Optional[Dict[str, Any]] = None,
        action_type_index: int = None,
        move_index: int = None,
        max_move_index: int = None,
        switch_index: int = None,
        flag_index: int = None,
        target_index: int = None,
    ):
        if not targeting:
            index = action_type_index

            if action_type_index == 0:
                if flag_index == 3:
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

            index = target_index

        return PostProcess(data, index)
