import math
import time
import wandb
import traceback
import threading

import torch
import torch.optim as optim
import torch.nn.functional as F

from meloetta.frameworks.porygon import utils
from meloetta.frameworks.porygon.buffer import ReplayBuffer
from meloetta.frameworks.porygon.config import PorygonConfig
from meloetta.frameworks.porygon.model import (
    PorygonModel,
    TrainingOutput,
    Batch,
    Targets,
    Loss,
    Policy,
)
from meloetta.frameworks.porygon.utils import from_logits as vtrace_from_logits


def compute_baseline_loss(advantages, valid):
    return torch.sum((advantages * valid) ** 2)


def compute_entropy_loss(policy, log_policy, valid):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    entropy = (policy * log_policy).sum(-1)
    entropy = entropy / (policy > 0).sum(-1).clamp(min=1)
    return torch.sum(entropy * valid)


def compute_policy_gradient_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    policy_mask: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    logits = torch.flatten(logits, 0, 1)
    policy_mask = torch.flatten(policy_mask, 0, 1)
    masked_logits = torch.masked_fill(logits, ~policy_mask, float("-inf"))
    actions = torch.flatten(actions, 0, 1)
    cross_entropy = F.cross_entropy(
        masked_logits,
        target=actions,
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach() * valid)


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
        # opt_learner_model = torch.compile(learner_model)
        learner_model.train()

        print(f"learnabled params: {learner_model.get_learnable_params():,}")

        actor_model = PorygonModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        actor_model.load_state_dict(learner_model.state_dict())
        actor_model.share_memory()
        actor_model.eval()

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
            try:
                learner.optimizer.load_state_dict(datum["optimizer"])
            except:
                print("Optimizer not loaded")
        else:
            print("Optimizer not loaded")

        if datum.get("learner_steps"):
            learner.learner_steps = datum["learner_steps"]
        else:
            print("Learner steps not loaded")

        return learner

    def collect_batch_trajectory(self):
        # hidden_state = hidden_state.to(self.config.learner_device, non_blocking=True)
        batch = None
        while batch is None:
            time.sleep(1)
            batch = self.replay_buffer.get_batch(self.config.batch_size)
        return batch

    def step(self):
        batch = self.collect_batch_trajectory()
        self.update_paramters(batch)
        self.learner_steps += 1
        if self.learner_steps % 500 == 0:
            self.save()

    def update_paramters(
        self,
        batch: Batch,
        lock=threading.Lock(),
    ):
        with lock:
            targets = self._get_targets(batch)

            self.optimizer.zero_grad(set_to_none=True)
            losses = self._backpropagate(batch, targets)

            static_loss_dict = losses.to_log(batch)
            static_loss_dict["s"] = self.learner_steps
            wandb.log(static_loss_dict)

            # torch.nn.utils.clip_grad_value_(
            #     self.learner_model.parameters(), self.config.clip_gradient
            # )

            # torch.nn.utils.clip_grad_norm_(
            #     self.learner_model.parameters(), self.config.clip_gradient_norm
            # )

            self.optimizer.step()

            # if self.learner_steps > 10:
            #     if self.learner_steps % 100 == 0:
            #         self.actor_model.load_state_dict(self.learner_model.state_dict())
            # else:

            if self.learner_steps % 20 == 0:
                self.actor_model.load_state_dict(self.learner_model.state_dict())

    def _get_targets(
        self,
        batch: Batch,
    ) -> Targets:
        bidx = torch.arange(
            batch.batch_size, device=next(self.learner_model.parameters()).device
        )
        lengths = batch.valid.sum(0)

        learner: TrainingOutput

        with torch.no_grad():
            learner = self.learner_model.learning_forward(batch)

        bootstrap_value = learner.v.squeeze(-1)[lengths.squeeze() - 1, bidx]

        dones = torch.cat(
            (batch.valid[1:], torch.zeros_like(batch.valid[0, None])), dim=0
        )
        discounts = dones * self.config.gamma

        targets_dict = {}

        for pi_field, pi in learner.pi._asdict().items():
            pi_field: str
            pi: torch.Tensor

            if pi is None:
                continue

            if pi_field == "action_policy":
                policy_mask = torch.cat((batch.move_mask, batch.switch_mask), dim=-1)
            else:
                policy_mask = batch.get_mask_from_policy(pi_field)

            vtrace_returns = vtrace_from_logits(
                behavior_policy_logits=getattr(batch, pi_field),
                target_policy_logits=pi,
                policy_mask=policy_mask,
                actions=batch.get_index_from_policy(pi_field),
                discounts=discounts,
                rewards=batch.rewards,
                values=learner.v.squeeze(-1),
                bootstrap_value=bootstrap_value,
            )

            targets_dict[
                pi_field.replace("_policy", "") + "_vtrace_returns"
            ] = vtrace_returns

        return Targets(**targets_dict)
        # return Targets(
        #     **{
        #         k: [vt.flatten(0, 1).unsqueeze(0) for vt in v]
        #         for k, v in targets_dict.items()
        #     }
        # )

    def _backpropagate(
        self,
        batch: Batch,
        targets: Targets,
    ) -> Loss:

        loss_dict = {
            key + "_loss": 0
            for key, value in batch._asdict().items()
            if "_policy" in key and value is not None
        }
        fields = set([field.replace("_policy_loss", "") for field in loss_dict])

        loss_dict.update({key.replace("policy", "value"): 0 for key in loss_dict})
        loss_dict.update({key.replace("policy", "entropy"): 0 for key in loss_dict})

        total_valid = (batch.valid).sum().clamp(min=1).item()
        total_move_valid = (
            (batch.valid * (batch.action_type_index == 0)).sum().clamp(min=1).item()
        )
        total_switch_valid = (
            (batch.valid * (batch.action_type_index == 1)).sum().clamp(min=1).item()
        )

        minibatch_size = 8

        for batch_index in range(math.ceil(batch.batch_size / minibatch_size)):
            batch_start = minibatch_size * batch_index
            batch_end = minibatch_size * (batch_index + 1)

            loss = 0

            minibatch = batch.slice(batch_start, batch_end)

            if minibatch.valid.sum() == 0:
                continue

            targ = targets.slice(batch_start, batch_end)

            output, _, _ = self.learner_model.forward(
                minibatch._asdict(),
                need_log_policy=True,
            )

            for field in fields:
                pi_field = field + "_policy"
                log_pi_field = field + "_log_policy"
                index_field = field + "_index"
                logit_field = field + "_logits"
                vtrace_field = field + "_vtrace_returns"

                pi = getattr(output.pi, pi_field)
                # logit = getattr(output.logit, logit_field)

                if pi is None:
                    continue

                valid = minibatch.valid
                policy_valid = total_valid

                policy_mask = minibatch.get_mask_from_policy(pi_field)

                if field == "move" or field == "flag":
                    valid = valid * (minibatch.action_type_index == 0)
                    policy_valid = total_move_valid

                elif field == "switch":
                    valid = valid * (minibatch.action_type_index == 1)
                    policy_valid = total_switch_valid

                vtrace_returns = getattr(targ, vtrace_field)

                value_loss = 0.25 * compute_baseline_loss(
                    vtrace_returns.vs - output.v.squeeze(-1), valid
                )

                policy_loss = compute_policy_gradient_loss(
                    getattr(output.logit, logit_field),
                    getattr(minibatch, index_field),
                    vtrace_returns.pg_advantages,
                    policy_mask,
                    valid,
                )

                entropy_loss = 5e-1 * compute_entropy_loss(
                    getattr(output.pi, pi_field),
                    getattr(output.log_pi, log_pi_field),
                    valid,
                )

                policy_loss_field = field + "_policy_loss"
                value_loss_field = field + "_value_loss"
                entropy_loss_field = field + "_entropy_loss"

                loss_dict[policy_loss_field] += policy_loss.item()
                loss_dict[value_loss_field] += value_loss.item()
                loss_dict[entropy_loss_field] += entropy_loss.item()

                loss += (value_loss + policy_loss + entropy_loss) / policy_valid

            loss.backward()

        for key in loss_dict:
            if "action_type" in key:
                loss_dict[key] /= total_valid
            elif "move" in key or "flag" in key:
                loss_dict[key] /= total_move_valid
            elif "switch" in key:
                loss_dict[key] /= total_switch_valid

        return Loss(**loss_dict)

    def run(self):
        while True:
            self.step()
