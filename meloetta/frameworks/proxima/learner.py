import os
import wandb
import traceback
import threading

import torch
import torch.optim as optim
import torch.nn.functional as F

from typing import Dict

from meloetta.actors.types import TensorDict

from meloetta.frameworks.proxima.buffer import ReplayBuffer
from meloetta.frameworks.proxima.config import ProximaConfig
from meloetta.frameworks.proxima.entropy import EntropySchedule
from meloetta.frameworks.proxima.model import ProximaModel
from meloetta.frameworks.proxima.utils import (
    FineTuning,
    _policy_ratio,
    _get_leading_dims,
)

_FIELDS = ["action_type", "flag", "move", "switch"]


class ProximaLearner:
    def __init__(
        self,
        learner_model: ProximaModel = None,
        actor_model: ProximaModel = None,
        replay_buffer: ReplayBuffer = None,
        load_devices: bool = True,
        init_optimizers: bool = True,
        config: ProximaConfig = ProximaConfig(),
        from_pretrained: bool = False,
    ):
        self.config = config

        self.learner_model = (
            learner_model if learner_model is not None else self._init_model(train=True)
        )

        if from_pretrained:
            self._init_from_pretrained()

        self.actor_model = (
            actor_model if actor_model is not None else self._init_model()
        )
        self.actor_model.load_state_dict(self.learner_model.state_dict())
        self.actor_model.share_memory()

        if load_devices:
            self._to_devices()

        if init_optimizers:
            self._init_optimizers()

        self.replay_buffer = replay_buffer if replay_buffer else self._init_buffer()

        self._entropy_schedule = EntropySchedule(
            sizes=self.config.entropy_schedule_size,
            repeats=self.config.entropy_schedule_repeats,
        )

        self.learner_steps = 0
        self.finetune = FineTuning()

        print(f"learnabled params: {self.learner_model.get_learnable_params():,}")

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
            f"cpkts/cpkt-{self.learner_steps:08}.tar",
        )

    def _init_from_pretrained(self):
        self.learner_model.encoder.load_state_dict(torch.load("supervised/encoder.pt"))
        self.learner_model.core.load_state_dict(torch.load("supervised/core.pt"))
        self.learner_model.value_head.load_state_dict(
            torch.load("supervised/value_head.pt")
        )
        self.learner_model.policy_heads.action_type_head.load_state_dict(
            torch.load("supervised/action_head.pt")
        )

        self.learner_model.encoder.requires_grad_(False)
        self.learner_model.core.requires_grad_(False)
        self.learner_model.value_head.requires_grad_(False)
        self.learner_model.policy_heads.action_type_head.requires_grad_(False)

    def _init_model(self, train: bool = False):
        model = ProximaModel(
            gen=self.config.gen,
            gametype=self.config.gametype,
            config=self.config.model_config,
        ).float()
        if train:
            model.train()
        else:
            model.eval()
        return model

    def _init_buffer(self):
        return ReplayBuffer(
            self.config.trajectory_length,
            self.config.gen,
            self.config.gametype,
            self.config.num_buffers,
            self.config.learner_device,
        )

    def _init_optimizers(self):
        self.optimizer = optim.Adam(
            self.learner_model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam.b1, self.config.adam.b2),
            eps=self.config.adam.eps,
        )

    def _to_devices(self):
        self.learner_model = self.learner_model.to(self.config.learner_device)
        self.actor_model = self.actor_model.to(self.config.actor_device)

    @classmethod
    def from_config(self, config: ProximaConfig = None):
        return ProximaLearner.from_fpath(config_override=config)

    @classmethod
    def from_latest(self):
        cpkts = list(
            sorted(
                os.listdir("cpkts"),
                key=lambda f: int(f.split("-")[-1].split(".")[0]),
            )
        )
        fpath = os.path.join("cpkts", cpkts[-1])
        return self.from_fpath(fpath, ProximaConfig())

    @classmethod
    def from_fpath(self, fpath: str, config: ProximaConfig = None):
        try:
            print(f"Using fpath: {fpath}")
            datum: dict = torch.load(fpath)
        except Exception as e:
            raise ValueError(f"cpkt at {fpath} not found")
        else:
            if config:
                print("Using new config!")
            else:
                print("Found config!")
                config: ProximaConfig = datum["config"]

        learner = ProximaLearner(
            config=config, load_devices=False, init_optimizers=False
        )

        for obj, model in [
            (learner.learner_model, "learner_model"),
            (learner.actor_model, "actor_model"),
        ]:
            try:
                obj.load_state_dict(datum[model], strict=False)
            except Exception as e:
                print(f"Error loading `{model}`")
                traceback.print_exc()
            else:
                print(f"Sucessfully loaded `{model}`")

        learner._to_devices()
        learner._init_optimizers()

        try:
            learner.optimizer.load_state_dict(datum["optimizer"])
        except Exception as e:
            print("Optimizer not loaded")
            traceback.print_exc()
        else:
            print(f"Optimizer loaded sucessfully!")

        if datum.get("learner_steps"):
            learner.learner_steps = datum["learner_steps"]
        else:
            print("Learner steps not loaded")

        return learner

    def collect_batch_trajectory(self):
        return self.replay_buffer.get_batch(self.config.batch_size)

    def step(self, lock=threading.Lock()):
        indices, batch = self.collect_batch_trajectory()

        self.optimizer.zero_grad()

        returns, advantages = self._get_targets(batch)
        loss_dict = self._update_params(batch, returns, advantages)

        torch.nn.utils.clip_grad_value_(self.learner_model.parameters(), 100)

        self.optimizer.step()

        loss_dict["s"] = self.learner_steps
        loss_dict["final_turn"] = (
            batch["scalars"][..., 0].max(0).values.float().mean()
        ).item()

        wandb.log(loss_dict)
        self.learner_steps += 1

        if self.learner_steps % 1000 == 0:
            self.save()

        with lock:
            self.actor_model.load_state_dict(self.learner_model.state_dict())

    @torch.no_grad()
    def _learning_forward(self, batch: TensorDict, size: int = 512) -> TensorDict:
        T, B = _get_leading_dims(batch)
        TB = T * B

        if TB <= size:
            batch = {
                k: v.to(self.replay_buffer.device, non_blocking=True)
                for k, v in batch.items()
            }
            model_output = self.learner_model.learning_forward(batch)
            return {k: v.cpu() for k, v in model_output.items()}

        num_iters = TB // size
        if num_iters != TB / size:
            num_iters += 1

        buckets = {}
        lagging_shape = {}

        flat_batch = {k: v.flatten(0, 1).unsqueeze(1) for k, v in batch.items()}

        for i in range(num_iters):
            start, end = i * size, (i + 1) * size

            minibatch = {
                k: v[start:end].to(self.replay_buffer.device, non_blocking=True)
                for k, v in flat_batch.items()
            }
            model_output = self.learner_model.learning_forward(minibatch)

            for k, v in model_output.items():
                if i == 0:
                    buckets[k] = []
                    lagging_shape[k] = v.shape[2:]

                buckets[k].append(v.cpu())

        output = {
            k: torch.cat(buckets[k]).view(T, B, *lagging_shape[k])
            for k in buckets
            if buckets[k]
        }
        return output

    @torch.no_grad()
    def _get_targets(self, batch: TensorDict) -> TensorDict:
        learner = self._learning_forward(batch)

        rewards = batch["rewards"]
        valid = batch["valid"]

        advantages = torch.zeros_like(rewards)
        curr_value = learner["value"].squeeze(-1)
        next_value = torch.zeros_like(curr_value[-1, None])
        lastgaelam = 0

        T, _ = _get_leading_dims(batch)

        for t in reversed(range(T)):
            nextvalues = next_value if t == T - 1 else curr_value[t + 1]
            nextnonterminal = valid[t]
            delta = (
                rewards[t]
                + self.config.gamma * nextvalues * nextnonterminal
                - curr_value[t]
            )
            advantages[t] = lastgaelam = (
                delta
                + self.config.gamma
                * self.config.lambda_vtrace
                * nextnonterminal
                * lastgaelam
            )
        returns = advantages + curr_value

        return returns.detach(), advantages.detach()

    def _update_params(
        self, batch: TensorDict, returns: torch.Tensor, advantages: torch.Tensor
    ) -> Dict[str, float]:
        minibatch_size = 512
        T, B = _get_leading_dims(batch)

        n_iters = ((T * B) // minibatch_size) + 1

        inds = torch.arange(T * B)
        inds = torch.randperm(inds.size(0))

        batch = {k: v.flatten(0, 1).unsqueeze(1) for k, v in batch.items()}
        batch = {k: v[inds] for k, v in batch.items() if v is not None}
        returns = returns.flatten(0, 1)[inds]
        advantages = advantages.flatten(0, 1)[inds]

        action_type_valid_sum = batch["valid"]
        action_type_valid_sum = action_type_valid_sum.sum().clamp(min=1).item()

        action_type0_valid_sum = batch["valid"] & batch["action_type_index"] == 0
        action_type0_valid_sum = action_type0_valid_sum.sum().clamp(min=1).item()

        action_type1_valid_sum = batch["valid"] & batch["action_type_index"] == 1
        action_type1_valid_sum = action_type1_valid_sum.sum().clamp(min=1).item()

        loss_dict = {"value_loss": 0}
        for field in _FIELDS:
            loss_dict[f"{field}_policy_loss"] = 0

        for i in range(n_iters):
            start_idx, end_idx = i * minibatch_size, (i + 1) * minibatch_size

            if not batch["valid"][start_idx:end_idx].sum().item():
                continue

            minibatch = {
                k: v[start_idx:end_idx].to(self.replay_buffer.device)
                for k, v in batch.items()
            }

            minireturns = returns[start_idx:end_idx].to(self.replay_buffer.device)
            miniadvantages = advantages[start_idx:end_idx].to(self.replay_buffer.device)

            loss_dict = self._step(
                minibatch,
                minireturns,
                miniadvantages,
                loss_dict,
                action_type_valid_sum,
                action_type0_valid_sum,
                action_type1_valid_sum,
            )

        return loss_dict

    def _step(
        self,
        batch: TensorDict,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        loss_dict: Dict[str, float],
        action_type_valid_sum: int,
        action_type0_valid_sum: int,
        action_type1_valid_sum: int,
    ):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model_output = self.learner_model.learning_forward(batch)

        loss = 0
        T, _ = _get_leading_dims(batch)

        for field in _FIELDS:
            pi = model_output[f"{field}_policy"].view(T, -1)
            mu = batch[f"{field}_policy"].view(T, -1)
            action = batch[f"{field}_index"].view(T)
            log_pi = model_output[f"{field}_log_policy"].view(T, -1)

            policy_valid = batch["valid"].view(T)

            if field == "move" or field == "flag":
                trajectory_mask = (batch["action_type_index"] == 0).squeeze(-1)
                policy_valid_sum = action_type0_valid_sum

            elif field == "switch":
                trajectory_mask = (batch["action_type_index"] == 1).squeeze(-1)
                policy_valid_sum = action_type1_valid_sum

            else:
                trajectory_mask = torch.ones_like(
                    batch["action_type_index"], dtype=torch.bool
                ).squeeze(-1)
                policy_valid_sum = action_type_valid_sum

            policy_valid = policy_valid & trajectory_mask

            entropy = -1e-2 * (pi * log_pi).sum(-1)
            ratio = _policy_ratio(pi, mu, F.one_hot(action, pi.shape[-1]), policy_valid)
            policy_loss_field = f"{field}_policy_loss"

            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(
                ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2)

            loss_dict[policy_loss_field] += (
                (pg_loss * policy_valid).sum() / policy_valid_sum
            ).item()

            loss += ((pg_loss - entropy) * policy_valid).sum() / policy_valid_sum

        value_loss = 0.25 * (model_output["value"].squeeze() - returns) ** 2

        valid = batch["valid"].view(T)
        valid_sum = action_type_valid_sum

        loss_dict["value_loss"] += ((value_loss * valid).sum() / valid_sum).item()

        loss += (value_loss * valid).sum() / valid_sum

        loss.backward()

        return loss_dict

    def run(self):
        while True:
            self.step()
