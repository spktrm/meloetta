import math
import time
import wandb
import traceback
import threading

import torch
import torch.optim as optim
import torch.nn.functional as F

from typing import Tuple

from meloetta.frameworks.porygon import utils
from meloetta.frameworks.porygon.buffer import ReplayBuffer
from meloetta.frameworks.porygon.config import PorygonConfig
from meloetta.frameworks.porygon.entropy import EntropySchedule
from meloetta.frameworks.porygon.model import (
    PorygonModel,
    TrainingOutput,
    Batch,
    Targets,
    Loss,
    Policy,
)
from meloetta.frameworks.porygon import vtrace


def compute_baseline_loss(advantages: torch.Tensor, valid: torch.Tensor):
    return (advantages**2) * valid


def compute_policy_gradient_loss(
    log_probs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    valid: torch.Tensor,
):
    cross_entropy = F.nll_loss(
        input=torch.flatten(log_probs, 0, 1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return cross_entropy * advantages.detach() * valid


class PorygonLearner:
    def __init__(
        self,
        learner_model: PorygonModel,
        actor_model: PorygonModel,
        replay_buffer: ReplayBuffer,
        config: PorygonConfig = PorygonConfig(),
    ):
        self.learner_model = learner_model
        self.actor_model = actor_model

        self.replay_buffer = replay_buffer

        self.config = config

        self.optimizer = optim.Adam(
            self.learner_model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam.b1, config.adam.b2),
            eps=config.adam.eps,
        )
        self.learner_steps = 0

    def get_config(self):
        return {
            "parameters": self.learner_model.get_learnable_params(),
            "learner_steps": self.learner_steps,
            **self.config.__dict__,
        }

    def save(self):
        torch.save(
            {
                "learner_model": self.learner_model.state_dict(),
                "actor_model": self.actor_model.state_dict(),
                "config": self.config,
                "optimizer": self.optimizer.state_dict(),
                "learner_steps": self.learner_steps,
            },
            f"cpkts/cpkt-{self.learner_steps:05}.tar",
        )

    @classmethod
    def from_config(self, config: PorygonConfig = None):
        return PorygonLearner.from_pretrained(config_override=config)

    @classmethod
    def from_pretrained(self, fpath: str = None, config_override: PorygonConfig = None):
        assert not (
            fpath is None and config_override is None
        ), "please set one of `fpath` or `config`"

        if fpath is not None:
            datum = torch.load(fpath)
        else:
            datum = {}

        if datum.get("config"):
            print("Found config!")
            config: PorygonConfig = datum["config"]

        if config_override is not None:
            config = config_override

        gen, gametype = utils.get_gen_and_gametype(config.battle_format)
        learner_model = PorygonModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        # learner_model.encoder.private_encoder.load_state_dict(torch.load("meloetta/frameworks/nash_ketchum/model/encoders/private_encoder.pt"))
        # for param in learner_model.encoder.private_encoder.parameters():
        #     param.requires_grad = False
        learner_model.train()

        print(f"learnabled params: {learner_model.get_learnable_params():,}")

        actor_model = PorygonModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        actor_model.load_state_dict(learner_model.state_dict())
        actor_model.share_memory()
        actor_model.eval()

        target_model = PorygonModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        target_model.load_state_dict(learner_model.state_dict())
        target_model.eval()

        model_prev = PorygonModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        model_prev.load_state_dict(learner_model.state_dict())
        model_prev.eval()

        model_prev_ = PorygonModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        model_prev_.load_state_dict(learner_model.state_dict())
        model_prev_.eval()

        if datum.get("learner_model"):
            try:
                learner_model.load_state_dict(datum["learner_model"], strict=False)
                actor_model.load_state_dict(datum["actor_model"], strict=False)
            except Exception as e:
                traceback.print_exc()

        learner_model = learner_model.to(config.learner_device, non_blocking=True)
        actor_model = actor_model.to(config.actor_device, non_blocking=True)

        learner = PorygonLearner(
            learner_model,
            actor_model,
            replay_buffer=ReplayBuffer(
                config.trajectory_length,
                gen,
                gametype,
                config.num_buffers,
                config.learner_device,
            ),
            config=config,
        )

        if datum.get("optimizer"):
            learner.optimizer.load_state_dict(datum["optimizer"])
        else:
            print("Optimizer not loaded")

        if datum.get("learner_steps"):
            learner.learner_steps = datum["learner_steps"]
        else:
            print("Learner steps not loaded")

        return learner

    def collect_batch_trajectory(self):
        hidden_state = self.learner_model.core.initial_state(self.config.batch_size)
        hidden_state = (
            hidden_state[0].to(self.config.learner_device, non_blocking=True),
            hidden_state[1].to(self.config.learner_device, non_blocking=True),
        )
        batch = None
        while batch is None:
            time.sleep(1)
            batch = self.replay_buffer.get_batch(self.config.batch_size)
        return batch, hidden_state

    def step(self):
        batch, hidden_state = self.collect_batch_trajectory()
        self.update_paramters(batch, hidden_state)
        self.learner_steps += 1
        if self.learner_steps % 500 == 0:
            self.save()

    def update_paramters(
        self,
        batch: Batch,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        lock=threading.Lock(),
    ):
        with lock:
            self.optimizer.zero_grad(set_to_none=True)

            targets = self._get_targets(batch, hidden_state)
            losses = self._backpropagate(batch, hidden_state, targets)

            static_loss_dict = losses.to_log(batch)
            static_loss_dict["s"] = self.learner_steps
            wandb.log(static_loss_dict)

            torch.nn.utils.clip_grad_value_(
                self.learner_model.parameters(), self.config.clip_gradient
            )

            self.optimizer.step()
            self.actor_model.load_state_dict(self.learner_model.state_dict())

    def _get_targets(
        self,
        batch: Batch,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Targets:
        bidx = torch.arange(
            batch.batch_size, device=next(self.learner_model.parameters()).device
        )
        lengths = batch.valid.sum(0)
        final_reward = batch.rewards[lengths.squeeze() - 1, bidx]
        # assert torch.all(final_reward.sum(-1) == 0).item(), "Env should be zero-sum!"

        learner: TrainingOutput
        with torch.no_grad():
            learner = self.learner_model.learning_forward(batch, hidden_state)

        targets_dict = {}

        for policy_field, pi in learner.pi._asdict().items():

            pi: torch.Tensor
            policy_field: str
            if pi is None:
                continue

            reward = batch.rewards
            bootstrap_value = final_reward

            action_index = batch.get_index_from_policy(policy_field)

            vtrace_returns = vtrace.from_logits(
                behavior_policy_logits=pi,
                target_policy_logits=getattr(batch, policy_field),
                actions=action_index,
                discounts=torch.ones_like(action_index),
                rewards=reward,
                values=learner.v.squeeze(),
                bootstrap_value=bootstrap_value,
            )

            targets_dict[
                policy_field.replace("_policy", "_value_target")
            ] = vtrace_returns.pg_advantages
            targets_dict[
                policy_field.replace("_policy", "_policy_target")
            ] = vtrace_returns.vs

        return Targets(**targets_dict)

    def _backpropagate(
        self,
        batch: Batch,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        targets: Targets,
    ) -> Loss:

        loss_dict = {key + "_loss": 0 for key in Policy._fields}
        loss_dict.update({key + "_ent": 0 for key in Policy._fields})
        loss_dict.update(
            {key.replace("policy", "value") + "_loss": 0 for key in Policy._fields}
        )

        action_type_index = batch.action_type_index

        total_valid = batch.valid.sum().item()
        total_move_valid = (batch.valid * (action_type_index == 0)).sum().item()
        total_switch_valid = (batch.valid * (action_type_index == 1)).sum().item()

        minibatch_size = (4096 // batch.trajectory_length) + 1

        for batch_index in range(math.ceil(batch.batch_size / minibatch_size)):
            mini_state_h, mini_state_c = hidden_state

            batch_start = minibatch_size * batch_index
            batch_end = minibatch_size * (batch_index + 1)

            mini_state_h = mini_state_h[:, batch_start:batch_end].contiguous()
            mini_state_c = mini_state_c[:, batch_start:batch_end].contiguous()

            loss = 0

            max_length = batch.valid[:, batch_start:batch_end].sum(0).max()
            minibatch = batch.slice(batch_start, batch_end, max_length=max_length)
            targ = targets.slice(batch_start, batch_end, max_length=max_length)

            output, (mini_state_c, mini_state_h) = self.learner_model.forward(
                minibatch._asdict(),
                (mini_state_c, mini_state_h),
                training=True,
                need_log_policy=True,
            )

            value_loss = 0
            policy_loss = 0

            valid = minibatch.valid
            action_type_index = minibatch.action_type_index

            minibatch_loss_dict = {key + "_loss": 0 for key in Policy._fields}
            minibatch_loss_dict.update({key + "_ent": 0 for key in Policy._fields})
            minibatch_loss_dict.update(
                {key.replace("policy", "value") + "_loss": 0 for key in Policy._fields}
            )

            for policy_field in (
                "action_type_policy",
                "move_policy",
                "switch_policy",
                # "max_move_policy",
                # "flag_policy",
                # "target_policy",
            ):
                pi = getattr(output.pi, policy_field)
                log_pi = getattr(
                    output.log_pi, policy_field.replace("policy", "log_policy")
                )

                if pi is None:
                    continue

                if policy_field == "switch_policy":
                    policy_valid = action_type_index == 1
                elif policy_field == "target_policy":
                    policy_valid = action_type_index == 2
                else:
                    policy_valid = action_type_index == 0

                policy_valid *= valid

                value_loss = 0.5 * compute_baseline_loss(
                    targ.get_value_target(policy_field) - output.v.squeeze(),
                    policy_valid,
                )

                policy_loss = compute_policy_gradient_loss(
                    log_pi,
                    minibatch.get_index_from_policy(policy_field),
                    targ.get_policy_target(policy_field),
                    policy_valid,
                )

                entropy_loss = 6e-4 * (pi * log_pi * valid.unsqueeze(-1))

                loss_field = policy_field + "_loss"
                value_field = policy_field.replace("policy", "value") + "_loss"
                ent_field = policy_field + "_ent"

                minibatch_loss_dict[loss_field] += policy_loss.sum()
                minibatch_loss_dict[value_field] += value_loss.sum()
                minibatch_loss_dict[ent_field] += entropy_loss.sum()

                loss_dict[loss_field] += minibatch_loss_dict[loss_field].item()
                loss_dict[value_field] += minibatch_loss_dict[value_field].item()
                loss_dict[ent_field] += minibatch_loss_dict[ent_field].item()

            loss = (
                (minibatch_loss_dict["action_type_value_loss"] / total_valid)
                + (minibatch_loss_dict["action_type_policy_loss"] / total_valid)
                + (minibatch_loss_dict["action_type_policy_ent"] / total_valid)
                + (minibatch_loss_dict["move_value_loss"] / total_move_valid)
                + (minibatch_loss_dict["move_policy_loss"] / total_move_valid)
                + (minibatch_loss_dict["move_policy_ent"] / total_move_valid)
                + (minibatch_loss_dict["switch_value_loss"] / total_switch_valid)
                + (minibatch_loss_dict["switch_policy_loss"] / total_switch_valid)
                + (minibatch_loss_dict["switch_policy_ent"] / total_switch_valid)
            )
            loss.backward()

        loss_dict["action_type_value_loss"] /= total_valid
        loss_dict["action_type_policy_loss"] /= total_valid
        loss_dict["action_type_policy_ent"] /= total_valid
        loss_dict["move_value_loss"] /= total_move_valid
        loss_dict["move_policy_loss"] /= total_move_valid
        loss_dict["move_policy_ent"] /= total_move_valid
        loss_dict["switch_value_loss"] /= total_switch_valid
        loss_dict["switch_policy_loss"] /= total_switch_valid
        loss_dict["switch_policy_ent"] /= total_switch_valid

        return Loss(**loss_dict)

    def run(self):
        while True:
            self.step()
