import os
import wandb
import traceback
import threading

import torch
import torch.optim as optim
import torch.nn.functional as F

from typing import List, Dict, Tuple

from meloetta.types import TensorDict
from meloetta.frameworks.nash_ketchum.buffer import ReplayBuffer
from meloetta.frameworks.nash_ketchum.config import NAshKetchumConfig
from meloetta.frameworks.nash_ketchum.entropy import EntropySchedule
from meloetta.frameworks.nash_ketchum.modelv2 import NAshKetchumModel
from meloetta.frameworks.nash_ketchum.utils import (
    FineTuning,
    _player_others,
    _get_leading_dims,
    v_trace,
    get_loss_nerd,
    get_loss_v,
)

_FIELDS = ["action_type", "flag", "move", "switch"]


class TargetNetSGD:
    def __init__(
        self,
        lr: float,
        target_model: NAshKetchumModel,
        learner_model: NAshKetchumModel,
    ):
        self.lr = lr
        self.target_model = target_model
        self.learner_model = learner_model

    @torch.no_grad()
    def step(self):
        for param1, param2 in zip(
            self.target_model.parameters(), self.learner_model.parameters()
        ):
            if param2.requires_grad:
                new = param1 + self.lr * (param2 - param1)
                param1[...] = new.clone().detach()


class NAshKetchumLearner:
    def __init__(
        self,
        learner_model: NAshKetchumModel = None,
        actor_model: NAshKetchumModel = None,
        target_model: NAshKetchumModel = None,
        model_prev: NAshKetchumModel = None,
        model_prev_: NAshKetchumModel = None,
        replay_buffer: ReplayBuffer = None,
        load_devices: bool = True,
        init_optimizers: bool = True,
        config: NAshKetchumConfig = NAshKetchumConfig(),
    ):
        self.config = config

        self.learner_model = (
            learner_model if learner_model is not None else self._init_model(train=True)
        )
        self.actor_model = (
            actor_model if actor_model is not None else self._init_model()
        )
        self.target_model = (
            target_model if target_model is not None else self._init_model()
        )
        self.model_prev = model_prev if model_prev is not None else self._init_model()
        self.model_prev_ = (
            model_prev_ if model_prev_ is not None else self._init_model()
        )

        for model in [
            self.actor_model,
            self.target_model,
            self.model_prev,
            self.model_prev_,
        ]:
            model.load_state_dict(self.learner_model.state_dict())
            model.requires_grad_(False)

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
                "target_model": self.target_model.state_dict(),
                "model_prev": self.model_prev.state_dict(),
                "model_prev_": self.model_prev_.state_dict(),
                "config": self.config,
                "optimizer": self.optimizer.state_dict(),
                "learner_steps": self.learner_steps,
            },
            f"cpkts/cpkt-{self.learner_steps:08}.tar",
        )

    def _init_model(self, train: bool = False):
        model = NAshKetchumModel(
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
        def _get_trainable_params(model: NAshKetchumModel, weight_decay: float):
            grad_groups = {
                pn: p for pn, p in model.named_parameters() if p.requires_grad
            }
            decay_params = [p for n, p in grad_groups.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in grad_groups.items() if p.dim() < 2]
            optim_groups = [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
            return optim_groups

        self.optimizer = optim.Adam(
            # _get_trainable_params(self.learner_model, self.config.adam.weight_decay),
            self.learner_model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam.b1, self.config.adam.b2),
            eps=self.config.adam.eps,
        )
        self.optimizer_target = TargetNetSGD(
            self.config.target_network_avg,
            self.target_model,
            self.learner_model,
        )

    def _to_devices(self):
        self.learner_model = self.learner_model.to(self.config.learner_device)
        self.actor_model = self.actor_model.to(self.config.actor_device)
        self.target_model = self.target_model.to(self.config.learner_device)
        self.model_prev = self.model_prev.to(self.config.learner_device)
        self.model_prev_ = self.model_prev_.to(self.config.learner_device)

    @classmethod
    def from_config(self, config: NAshKetchumConfig = None):
        return NAshKetchumLearner.from_fpath(config_override=config)

    @classmethod
    def from_latest(self):
        cpkts = list(
            sorted(
                os.listdir("cpkts"),
                key=lambda f: int(f.split("-")[-1].split(".")[0]),
            )
        )
        fpath = os.path.join("cpkts", cpkts[-1])
        return self.from_fpath(fpath, NAshKetchumConfig())

    @classmethod
    def from_fpath(self, fpath: str, config: NAshKetchumConfig = None):
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
                config: NAshKetchumConfig = datum["config"]

        learner = NAshKetchumLearner(
            config=config, load_devices=False, init_optimizers=False
        )

        for obj, model in [
            (learner.learner_model, "learner_model"),
            (learner.actor_model, "actor_model"),
            (learner.target_model, "target_model"),
            (learner.model_prev, "model_prev"),
            (learner.model_prev_, "model_prev_"),
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

    def collect_batch_trajectory(self) -> Tuple[List[int], TensorDict]:
        return self.replay_buffer.get_batch(self.config.batch_size)

    def step(self, lock: threading.Lock = threading.Lock()):
        _, batch = self.collect_batch_trajectory()
        alpha, update_target_net = self._entropy_schedule(self.learner_steps)

        self.optimizer.zero_grad()

        targets = self._get_targets(batch, alpha)
        loss_dict = self._update_params(batch, targets)

        loss_dict["s"] = self.learner_steps
        loss_dict["final_turn"] = (
            batch["scalars"][..., 0].max(0).values.float().mean()
        ).item()

        choice_available = batch["action_type_mask"].sum(-1, keepdim=True) > 1
        choice_available = choice_available * batch["valid"].unsqueeze(-1)

        action_type_policy = batch["action_type_policy"] * choice_available
        action_type_policy = action_type_policy.sum(0).sum(0) / choice_available.sum()

        loss_dict["move_prob"] = action_type_policy[..., 0].item()
        loss_dict["switch_prob"] = action_type_policy[..., 1].item()

        for k, v in loss_dict.items():
            if isinstance(v, list):
                loss_dict[k] = sum(v)

        norm = torch.nn.utils.clip_grad_value_(
            self.learner_model.parameters(), self.config.clip_gradient
        )
        # loss_dict["gradient_norm"] = norm.item()

        wandb.log(loss_dict)

        self.optimizer.step()
        self.optimizer_target.step()

        if update_target_net:
            print(f"Updating Regularization Policy @ Step: {self.learner_steps:,}")
            self.model_prev_.load_state_dict(self.model_prev.state_dict())
            self.model_prev.load_state_dict(self.target_model.state_dict())

        if self.learner_steps % 1000 == 0:
            self.save()

        with lock:
            self.actor_model.load_state_dict(self.learner_model.state_dict())

        self.learner_steps += 1

    @torch.no_grad()
    def _learning_forward(
        self, batch: TensorDict, size: int = 1024
    ) -> List[TensorDict]:
        T, B = _get_leading_dims(batch)
        TB = T * B

        if TB <= size:
            batch = {
                k: v.to(self.replay_buffer.device, non_blocking=True)
                for k, v in batch.items()
            }
            learner_model_output = self.learner_model.forward(
                batch, compute_value=False
            )
            target_model_output = self.target_model.forward(
                batch, compute_log_policy=False
            )
            model_prev_output = self.model_prev.forward(batch, compute_value=False)
            model_prev_output_ = self.model_prev_.forward(batch, compute_value=False)
            return [
                {k: v.cpu() for k, v in model_output.items() if v is not None}
                for model_output in [
                    learner_model_output,
                    target_model_output,
                    model_prev_output,
                    model_prev_output_,
                ]
            ]

        num_iters = TB // size
        if num_iters != TB / size:
            num_iters += 1

        buckets = [{} for _ in range(4)]
        lagging_shape = {}

        flat_batch = {k: v.flatten(0, 1).unsqueeze(1) for k, v in batch.items()}

        for i in range(num_iters):
            start, end = i * size, (i + 1) * size

            minibatch = {
                k: v[start:end].to(self.replay_buffer.device, non_blocking=True)
                for k, v in flat_batch.items()
            }
            learner_model_output = self.learner_model(minibatch, compute_value=False)
            target_model_output = self.target_model(minibatch, compute_log_policy=False)
            model_prev_output = self.model_prev(minibatch, compute_value=False)
            model_prev_output_ = self.model_prev_(minibatch, compute_value=False)

            for midx, model_output in enumerate(
                [
                    learner_model_output,
                    target_model_output,
                    model_prev_output,
                    model_prev_output_,
                ]
            ):
                for k, v in model_output.items():
                    if i == 0 and v is not None:
                        buckets[midx][k] = []
                        lagging_shape[k] = v.shape[2:]

                    if k in buckets[midx]:
                        buckets[midx][k].append(v.cpu())

        return [
            {
                k: torch.cat(bucket[k]).view(T, B, *lagging_shape[k])
                for k in bucket
                if bucket[k]
            }
            for bucket in buckets
        ]

    @torch.no_grad()
    def _get_targets(
        self, batch: TensorDict, alpha: float
    ) -> Dict[str, List[torch.Tensor]]:
        learner, target, prev, prev_ = self._learning_forward(batch)

        targets_dict = {"importance_sampling_corrections": []}

        acting_policies = []
        policies_valid = []
        policies_pprocessed = []
        log_policies_reg = []
        action_ohs = []

        for policy_select, field in enumerate(_FIELDS):
            pi = learner[f"{field}_policy"]
            log_pi = learner[f"{field}_log_policy"]
            prev_log_pi = prev[f"{field}_log_policy"]
            prev_log_pi_ = prev_[f"{field}_log_policy"]

            policy_valid = batch["valid"]
            trajectory_mask = (batch["policy_select"] == policy_select).squeeze(-1)

            policy_pprocessed = pi
            acting_policy = batch[f"{field}_policy"]

            log_policy_reg = log_pi - (alpha * prev_log_pi + (1 - alpha) * prev_log_pi_)

            action_index = batch[f"{field}_index"]
            action_oh = F.one_hot(action_index, pi.shape[-1])

            policies_valid.append(policy_valid & trajectory_mask)
            acting_policies.append(acting_policy)
            policies_pprocessed.append(policy_pprocessed)
            log_policies_reg.append(log_policy_reg)
            action_ohs.append(action_oh)

        (
            v_target_list,
            has_played_list,
            v_trace_policy_target_list,
            importance_sampling_corrections,
        ) = ([], [], [], [])

        values = [target[f"{field}_value"] for field in _FIELDS]
        for player in range(2):
            reward = batch["rewards"][:, :, player]  # [T, B, Player]

            v_target_, has_played, policy_targets_, policy_ratios = v_trace(
                values,
                batch["policy_select"],
                batch["valid"],
                policies_valid,
                batch["player_id"],
                acting_policies,
                policies_pprocessed,
                log_policies_reg,
                _player_others(batch["player_id"], batch["valid"], player),
                action_ohs,
                reward,
                player,
                lambda_=self.config.lambda_vtrace,
                c=self.config.c_vtrace,
                rho=self.config.rho_vtrace,
                eta=self.config.eta_reward_transform,
                gamma=self.config.gamma,
            )
            v_target_list.append(v_target_)
            has_played_list.append(has_played)
            v_trace_policy_target_list.append(policy_targets_)
            importance_sampling_corrections.append(
                # [torch.ones_like(policy_ratio) for policy_ratio in policy_ratios]
                policy_ratios
            )

        targets_dict["value_targets"] = v_target_list
        targets_dict["has_played"] = has_played_list

        for f, field in enumerate(_FIELDS):
            for k in range(2):
                if k == 0:
                    targets_dict[f"{field}_policy_target"] = []
                    targets_dict[f"{field}_is"] = []

                targets_dict[f"{field}_policy_target"].append(
                    v_trace_policy_target_list[k][f]
                )
                targets_dict[f"{field}_is"].append(
                    importance_sampling_corrections[k][f].unsqueeze(-1)
                )

        return targets_dict

    def _update_params(
        self,
        batch: TensorDict,
        targets: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, float]:
        minibatch_size = 1024
        T, B = _get_leading_dims(batch)

        n_iters = ((T * B) // minibatch_size) + 1

        batch = {k: v.flatten(0, 1).unsqueeze(1) for k, v in batch.items()}
        targets = {
            k: [v.flatten(0, 1).unsqueeze(1) for v in vt] for k, vt in targets.items()
        }

        valid_sum = {}
        loss_dict = {}

        for policy_select, field in enumerate(_FIELDS):
            for k in range(2):
                policy_valid_sum_field = f"{field}_policy_valid_sum"
                value_valid_sum_field = f"{field}_value_valid_sum"

                for valid_sum_field in [policy_valid_sum_field, value_valid_sum_field]:
                    if valid_sum_field not in valid_sum:
                        valid_sum[valid_sum_field] = []

                mask = batch["policy_select"] == policy_select

                policy_mask = batch["valid"] & mask & (batch["player_id"] == k)
                valid_sum[policy_valid_sum_field].append(policy_mask.sum().item())

                value_mask = targets["has_played"][k] & mask
                valid_sum[value_valid_sum_field].append(value_mask.sum().item())

            loss_dict[f"{field}_value_loss"] = [0, 0]
            loss_dict[f"{field}_policy_loss"] = [0, 0]

        for i in range(n_iters):
            start_idx, end_idx = i * minibatch_size, (i + 1) * minibatch_size

            if not batch["valid"][start_idx:end_idx].sum().item():
                continue

            minibatch = {
                k: v[start_idx:end_idx].to(self.replay_buffer.device)
                for k, v in batch.items()
            }
            minitargets = {
                k: [v[start_idx:end_idx].to(self.replay_buffer.device) for v in vt]
                for k, vt in targets.items()
            }

            loss_dict = self._step(minibatch, minitargets, loss_dict, valid_sum)

        return loss_dict

    def _step(
        self,
        batch: TensorDict,
        targets: Dict[str, List[torch.Tensor]],
        loss_dict: Dict[str, float],
        valid_sum: Dict[str, int],
    ):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model_output = self.learner_model(batch, compute_log_policy=False)

        loss = 0
        T, _ = _get_leading_dims(batch)

        for policy_select, field in enumerate(_FIELDS):
            policy_valid = batch["valid"].view(T)

            trajectory_mask = batch["policy_select"] == policy_select
            policy_valid = (policy_valid & trajectory_mask.squeeze(-1)).unsqueeze(-1)

            policy_loss_field = f"{field}_policy_loss"
            value_loss_field = f"{field}_value_loss"

            policy_field_valid_sum = valid_sum[f"{field}_policy_valid_sum"]
            value_field_valid_sum = valid_sum[f"{field}_value_valid_sum"]

            pg_losses = get_loss_nerd(
                [model_output[f"{field}_logits"]] * 2,
                [model_output[f"{field}_policy"]] * 2,
                targets[f"{field}_policy_target"],
                policy_valid,
                batch["player_id"],
                policy_field_valid_sum,
                batch[f"{field}_mask"],
                targets[f"{field}_is"],
            )

            value_losses = get_loss_v(
                [model_output[f"{field}_value"]] * 2,
                targets["value_targets"],
                [has_played & trajectory_mask for has_played in targets["has_played"]],
                value_field_valid_sum,
            )

            for k, (pg_loss, value_loss) in enumerate(zip(pg_losses, value_losses)):
                loss += pg_loss
                loss += value_loss
                loss_dict[policy_loss_field][k] += pg_loss.item()
                loss_dict[value_loss_field][k] += value_loss.item()

        loss.backward()

        return loss_dict

    def run(self):
        while True:
            self.step()
