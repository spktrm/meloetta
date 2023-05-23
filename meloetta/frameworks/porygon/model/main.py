import math
import json
import torch
import random
import traceback

import torch.nn as nn

from copy import deepcopy
from typing import Optional, Union, Literal, Tuple, Dict, Any, List

from meloetta.actors.types import State, Choices

from meloetta.frameworks.porygon.model.utils import _log_policy

from meloetta.frameworks.porygon.model import config
from meloetta.frameworks.porygon.model import (
    Encoder,
    PolicyHeads,
    Core,
    ValueHead,
    Indices,
    Logits,
    Policy,
    LogPolicy,
    TrainingOutput,
    EncoderOutput,
    ModelOutput,
    PostProcess,
    Batch,
)

from meloetta.data import _STATE_FIELDS


class PorygonModel(nn.Module):
    def __init__(
        self,
        gen: int = 9,
        gametype: Literal["singles", "doubles", "triples"] = "singles",
        config: config.PorygonModelConfig = config.PorygonModelConfig(),
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

    def get_learnable_params(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def clean(self, state: State) -> State:
        return {k: state[k] for k in self.state_fields}

    def forward(
        self,
        state: State,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] = None,
        choices: Choices = None,
        need_log_policy: bool = True,
        battle_tag: str = None,
    ):
        encoder_output = self.encoder.forward(state)
        state_emb, hidden_state = self.core.forward(encoder_output, hidden_state)
        value = self.value_Head.forward(state_emb)
        while True:
            indices, logits, policy = self.policy_heads.forward(
                state_emb,
                encoder_output,
                state,
            )
            try:
                post_process = self._postprocess(
                    state=state,
                    encoder_output=encoder_output,
                    hidden_state=hidden_state,
                    indices=indices,
                    logits=logits,
                    policy=policy,
                    value=value,
                    choices=choices,
                    need_log_policy=need_log_policy,
                )
            except Exception:
                traceback.print_exc()

                with open(
                    f"{battle_tag}_model_error"
                    + str(random.randint(0, 9999999))
                    + ".json",
                    "w",
                ) as f:
                    json.dump(
                        {
                            "logits": {
                                k: torch.squeeze(v).tolist()
                                for k, v in logits._asdict().items()
                                if v is not None
                            },
                            "policy": {
                                k: torch.squeeze(v).tolist()
                                for k, v in policy._asdict().items()
                                if v is not None
                            },
                            "indices": {
                                k: torch.squeeze(v).tolist()
                                for k, v in indices._asdict().items()
                                if v is not None
                            },
                        },
                        f,
                    )
            else:
                return post_process

    def learning_forward(
        self,
        batch: Batch,
    ):
        T, B = batch.trajectory_length, batch.batch_size

        outputs: List[TrainingOutput] = []

        minibatch_size = min(128, (4096 // batch.trajectory_length) + 1)

        for batch_index in range(math.ceil(batch.batch_size / minibatch_size)):
            batch_start = minibatch_size * batch_index
            batch_end = minibatch_size * (batch_index + 1)

            minibatch = {
                key: value[:, batch_start:batch_end].contiguous()
                for key, value in batch._asdict().items()
                if value is not None
            }
            output, _, decoder = self.forward(minibatch, None)
            outputs.append(output)

        pi_dict: Dict[str, List[torch.Tensor]] = {}
        log_pi_dict: Dict[str, List[torch.Tensor]] = {}
        logit_dict: Dict[str, List[torch.Tensor]] = {}
        vs: List[torch.Tensor] = []

        for o in outputs:
            for key in Policy._fields:
                mbv = getattr(o.pi, key)
                if getattr(o.pi, key) is not None:
                    if key not in pi_dict:
                        pi_dict[key] = []
                    pi_dict[key].append(mbv)

            for key in LogPolicy._fields:
                mbv = getattr(o.log_pi, key)
                if getattr(o.log_pi, key) is not None:
                    if key not in log_pi_dict:
                        log_pi_dict[key] = []
                    log_pi_dict[key].append(mbv)

            for key in Logits._fields:
                mbv = getattr(o.logit, key)
                if getattr(o.logit, key) is not None:
                    if key not in logit_dict:
                        logit_dict[key] = []
                    logit_dict[key].append(mbv)

            vs.append(o.v)

        pi_dict = {
            key: torch.cat(value, dim=1)
            .contiguous()
            .view(T, B, *value[0].shape[2:])
            .squeeze()
            for key, value in pi_dict.items()
        }
        log_pi_dict = {
            key: torch.cat(value, dim=1)
            .contiguous()
            .view(T, B, *value[0].shape[2:])
            .squeeze()
            for key, value in log_pi_dict.items()
        }
        logit_dict = {
            key: torch.cat(value, dim=1)
            .contiguous()
            .view(T, B, *value[0].shape[2:])
            .squeeze()
            for key, value in logit_dict.items()
        }
        vs = torch.cat(vs, dim=1).contiguous().view(T, B, 1)

        return TrainingOutput(
            pi=Policy(**pi_dict),
            v=vs,
            logit=Logits(**logit_dict),
            log_pi=LogPolicy(**log_pi_dict),
        )

    def _postprocess(
        self,
        state: State,
        encoder_output: EncoderOutput,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        indices: Indices,
        logits: Logits,
        policy: Policy,
        value: torch.Tensor,
        choices: Optional[Dict[str, Any]] = None,
        need_log_policy: bool = True,
    ) -> Union[
        Tuple[TrainingOutput, Tuple[torch.Tensor, torch.Tensor], EncoderOutput],
        Tuple[ModelOutput, PostProcess, Tuple[torch.Tensor, torch.Tensor]],
    ]:
        if not self.training:
            targeting = bool(state.get("targeting"))
            post_process = self.index_to_action(targeting, indices, choices)
            env_step = ModelOutput(
                indices=indices,
                policy=policy,
                value=value,
                logits=logits,
            )
            output = (env_step, post_process, hidden_state)
        else:
            if need_log_policy:
                log_policy = self.get_log_policy(logits, state)
            else:
                log_policy = None
            output = (
                TrainingOutput(
                    pi=policy,
                    v=value,
                    log_pi=log_policy,
                    logit=logits,
                ),
                hidden_state,
                encoder_output,
            )
        return output

    def get_log_policy(self, logits: Logits, state: State):
        if self.gametype != "singles":
            target_log_policy = _log_policy(
                logits.target_logits,
                state["target_mask"],
            )
        else:
            target_log_policy = None

        if self.gen == 8:
            max_move_log_policy = _log_policy(
                logits.max_move_logits,
                state["max_move_mask"],
            )
        else:
            max_move_log_policy = None

        if self.gen >= 6:
            flag_log_policy = _log_policy(
                logits.flag_logits,
                state["flag_mask"],
            )
        else:
            flag_log_policy = None

        return LogPolicy(
            action_type_log_policy=_log_policy(
                logits.action_type_logits, state["action_type_mask"]
            ),
            move_log_policy=_log_policy(logits.move_logits, state["move_mask"]),
            switch_log_policy=_log_policy(logits.switch_logits, state["switch_mask"]),
            max_move_log_policy=max_move_log_policy,
            flag_log_policy=flag_log_policy,
            target_log_policy=target_log_policy,
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

        return PostProcess(data, index)
