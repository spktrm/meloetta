import os
import time
import wandb
import cProfile, pstats
import traceback
import threading

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Tuple, Dict

from meloetta.frameworks.nash_ketchum.buffer import ReplayBuffer
from meloetta.frameworks.nash_ketchum.config import NAshKetchumConfig
from meloetta.frameworks.nash_ketchum.entropy import EntropySchedule
from meloetta.frameworks.nash_ketchum.model import (
    NAshKetchumModel,
    ModelOutput,
    Batch,
    Targets,
    Loss,
    Indices,
)
from meloetta.frameworks.nash_ketchum.utils import (
    FineTuning,
    _player_others,
    v_trace,
    get_loss_nerd,
    get_loss_v,
)


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

        self.actor_model.load_state_dict(self.learner_model.state_dict())
        self.target_model.load_state_dict(self.learner_model.state_dict())
        self.model_prev.load_state_dict(self.learner_model.state_dict())
        self.model_prev_.load_state_dict(self.learner_model.state_dict())

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
        self.scaler = torch.cuda.amp.GradScaler()

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
        self.optimizer = optim.Adam(
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

    def collect_batch_trajectory(self):
        return self.replay_buffer.get_batch(self.config.batch_size)

    def step(self):
        batch = self.collect_batch_trajectory()
        alpha, update_target_net = self._entropy_schedule(self.learner_steps)

        if self.learner_steps < int(1e9):
            # if self.learner_steps < 10:
            self.update_paramters(batch, alpha, update_target_net)
        else:
            profiler = cProfile.Profile()
            profiler.enable()
            self.update_paramters(batch, alpha, update_target_net)
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats("ncalls")
            fpath = f"cprofile_{self.learner_steps}"
            stats.dump_stats(fpath)

    def update_paramters(
        self,
        batch: Batch,
        alpha: float,
        indices: Indices,
    ):
        batch_cpu = batch
        batch_gpu = batch.to(self.replay_buffer.device)

        indices = Indices(
            action_type_index=getattr(batch_cpu, "action_type_index", None),
            move_index=getattr(batch_cpu, "move_index", None),
            max_move_index=getattr(batch_cpu, "max_move_index", None),
            switch_index=getattr(batch_cpu, "switch_index", None),
            flag_index=getattr(batch_cpu, "flag_index", None),
            target_index=getattr(batch_cpu, "target_index", None),
        )

        targets = self._get_targets(batch_cpu, batch_gpu, alpha, indices)
        self.backprop(batch_cpu, targets, indices)

    def backprop(
        self,
        batch_cpu: Batch,
        targets: Targets,
        indices: Indices,
        lock=threading.Lock(),
    ):
        _, update_target_net = self._entropy_schedule(self.learner_steps)

        losses = self._backpropagate(batch_cpu, targets, indices)

        static_loss_dict = losses._asdict()
        static_loss_dict["s"] = self.learner_steps
        static_loss_dict["final_turn"] = (
            batch_cpu.scalars[..., 0].max(0).values.float().mean()
        ).item()

        wandb.log(static_loss_dict)

        self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_value_(
            self.learner_model.parameters(), self.config.clip_gradient
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        self.optimizer_target.step()

        if update_target_net:
            print(f"Updating Regularization Policy @ Step: {self.learner_steps:,}")
            self.model_prev_.load_state_dict(self.model_prev.state_dict())
            self.model_prev.load_state_dict(self.target_model.state_dict())

        self.learner_steps += 1
        if self.learner_steps % 1000 == 0:
            self.save()

        with lock:
            self.actor_model.load_state_dict(self.learner_model.state_dict())

    @torch.no_grad()
    def _learning_forward(
        self, batch: Batch, indices: Indices, size: int = 4096
    ) -> Tuple[ModelOutput, ModelOutput, ModelOutput, ModelOutput,]:

        B, T = batch.batch_size, batch.trajectory_length
        if B * T >= size:

            learner = []
            target = []
            prev = []
            prev_ = []

            def _iterate(
                batch: Dict[str, torch.Tensor],
                indices: Dict[str, torch.Tensor],
                size: int,
            ):
                n = batch["valid"].shape[1] // size
                if batch["valid"].shape[1] / size != n:
                    n += 1

                for i in range(n):
                    start, end = i * size, (i + 1) * size
                    yield (
                        {k: v[:, start:end] for k, v in batch.items()},
                        {k: v[:, start:end] for k, v in indices.items()},
                    )

            for minibatch, minindex in _iterate(
                batch.flatten(0, 1), indices.flatten(0, 1), size
            ):
                minindex = Indices(**minindex)
                learner.append(self.learner_model.learning_forward(minibatch, minindex))
                target.append(self.target_model.learning_forward(minibatch, minindex))
                prev.append(self.model_prev.learning_forward(minibatch, minindex))
                prev_.append(self.model_prev_.learning_forward(minibatch, minindex))

            learner = ModelOutput.from_list(learner)
            target = ModelOutput.from_list(target)
            prev = ModelOutput.from_list(prev)
            prev_ = ModelOutput.from_list(prev_)

            learner = learner.view(T, B)
            target = target.view(T, B)
            prev = prev.view(T, B)
            prev_ = prev_.view(T, B)

        else:
            batch = batch._asdict()
            learner = self.learner_model.learning_forward(batch, indices)
            target = self.target_model.learning_forward(batch, indices)
            prev = self.model_prev.learning_forward(batch, indices)
            prev_ = self.model_prev_.learning_forward(batch, indices)

        return learner, target, prev, prev_

    def _get_targets(
        self, batch_cpu: Batch, batch_gpu: Batch, alpha: float, indices: Indices
    ) -> Targets:
        learner, target, prev, prev_ = self._learning_forward(
            batch_gpu, indices.to(self.replay_buffer.device)
        )

        learner = learner.to("cpu")
        target = target.to("cpu")
        prev = prev.to("cpu")
        prev_ = prev_.to("cpu")

        targets_dict = {"importance_sampling_corrections": []}

        acting_policies = []
        policies_valid = []
        policies_pprocessed = []
        log_policies_reg = []
        action_ohs = []
        fields = []

        for field in sorted(learner.pi._fields):

            pi = getattr(learner.pi, field)
            if pi is None:
                continue

            fields.append(field)

            log_pi = getattr(target.log_pi, field)
            prev_log_pi = getattr(prev.log_pi, field)
            prev_log_pi_ = getattr(prev_.log_pi, field)

            # mask_field = policy_field.replace("_policy", "_mask")
            # policy_mask = getattr(batch_cpu, mask_field)
            # policy_pprocessed = self.finetune(pi, policy_mask, self.learner_steps)

            if field == "move" or field == "flag":
                policy_valid = batch_cpu.valid & (batch_cpu.action_type_index == 0)

            elif field == "switch":
                policy_valid = batch_cpu.valid & (batch_cpu.action_type_index == 1)

            else:
                policy_valid = batch_cpu.valid
            policy_valid = policy_valid.clone()

            policy_pprocessed = pi
            acting_policy = getattr(batch_cpu, field + "_policy")

            log_policy_reg = log_pi - (alpha * prev_log_pi + (1 - alpha) * prev_log_pi_)
            # log_policy_reg = log_policy_reg * policy_valid.unsqueeze(-1)

            action_index = getattr(batch_cpu, field + "_index")
            action_oh = F.one_hot(action_index, pi.shape[-1])

            policies_valid.append(policy_valid)
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

        for player in range(2):
            reward = batch_cpu.rewards[:, :, player]  # [T, B, Player]

            v_target_, has_played, policy_targets_, policy_ratios = v_trace(
                target.v,
                batch_cpu.valid,
                policies_valid,
                batch_cpu.player_id,
                acting_policies,
                policies_pprocessed,
                log_policies_reg,
                _player_others(batch_cpu.player_id, batch_cpu.valid, player),
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
            v_trace_policy_target_list.append(
                {
                    key: policy_target_
                    for key, policy_target_ in zip(fields, policy_targets_)
                }
            )
            importance_sampling_corrections.append(
                {
                    key: policy_ratio.unsqueeze(-1)
                    for key, policy_ratio in zip(fields, policy_ratios)
                }
            )

        targets_dict["value_targets"] = v_target_list
        targets_dict["has_played"] = has_played_list

        for field in fields:
            targets_dict[f"{field}_policy_target"] = [
                targ[field] for targ in v_trace_policy_target_list if field in targ
            ]
            targets_dict[f"{field}_is"] = [
                targ[field] for targ in importance_sampling_corrections if field in targ
            ]

        return Targets(**{key: value for key, value in targets_dict.items() if value})

    def _backpropagate(
        self,
        batch: Batch,
        targets: Targets,
        indices: Indices,
        remove_padding: bool = True,
    ) -> Loss:

        loss_dict = {
            key + "_loss": 0
            for key in batch._fields
            if "_policy" in key and getattr(batch, key) is not None
        }
        loss_dict["value_loss"] = 0

        batch_size = batch.batch_size * batch.trajectory_length

        if remove_padding:
            t = batch.valid.sum(0)
            # old_size = batch.trajectory_length * batch.batch_size
            # reduced_size = batch.valid.sum().item()
            # reduction = 100 * (1 - (reduced_size / old_size))
            # print(
            #     f"\n{reduction:.2f} % reduction - bs_eff = {old_size}, true_bs = {reduced_size}"
            # )

            batch = batch.flatten_without_padding(t)
            targets = targets.flatten_without_padding(t)
            indices = indices.flatten_without_padding(t)

        else:
            batch = batch.flatten(0, 1)
            targets = targets.flatten(0, 1)
            indices = indices.flatten(0, 1)

        batch = {k: v.to(self.replay_buffer.device) for k, v in batch.items()}
        indices = {k: v.to(self.replay_buffer.device) for k, v in indices.items()}
        targets = {
            k: [v.to(self.replay_buffer.device) for v in vt]
            for k, vt in targets.items()
        }

        minibatch_size = 4096

        value_loss_field = "value_loss"

        n = batch_size // minibatch_size
        if batch["valid"].shape[1] / minibatch_size != n:
            n += 1

        for i in range(n):

            start, end = i * minibatch_size, (i + 1) * minibatch_size

            minibatch = {k: v[:, start:end] for k, v in batch.items()}

            if torch.all(~minibatch["valid"]):
                continue

            loss = 0
            minitarget = {k: [v[:, start:end] for v in vt] for k, vt in targets.items()}
            minindices = Indices(**{k: vt[:, start:end] for k, vt in indices.items()})

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model_output = self.learner_model.forward(minibatch, indices=minindices)

            for pi_field in model_output.pi._fields:

                field = pi_field.replace("_policy", "")
                pi = getattr(model_output.pi, pi_field)

                if pi is None:
                    continue

                mask_field = field + "_mask"
                logit = getattr(model_output.logit, field)

                if field == "move" or field == "flag":
                    valid = minibatch["valid"] * (minibatch["action_type_index"] == 0)
                    normalization = batch["valid"] * (batch["action_type_index"] == 0)

                elif field == "switch":
                    valid = minibatch["valid"] * (minibatch["action_type_index"] == 1)
                    normalization = batch["valid"] * (batch["action_type_index"] == 1)

                else:
                    valid = minibatch["valid"]
                    normalization = batch["valid"]

                policy_mask = minibatch[mask_field]

                policy_loss_field = field + "_policy_loss"

                # Uses v-trace to define q-values for Nerd
                policy_loss = get_loss_nerd(
                    [logit] * 2,
                    [pi] * 2,
                    minitarget[f"{field}_policy_target"],
                    valid,
                    minibatch["player_id"],
                    policy_mask,
                    minitarget[f"{field}_is"],
                    normalization=normalization,
                    clip=self.config.nerd.clip,
                    threshold=self.config.nerd.beta,
                )
                loss_dict[policy_loss_field] += policy_loss.item()

                loss += policy_loss

            # repr_loss = self._get_repr_loss(
            #     model_output.state_action_emb,
            #     model_output.state_emb,
            #     (batch["valid"][:-1] * batch["valid"][1:]).flatten(0, 1),
            # )
            # loss_dict["repr_loss"] = repr_loss.item()
            # loss = loss + repr_loss

            has_played = minitarget["has_played"]
            value_loss = get_loss_v(
                [model_output.v] * 2,
                minitarget["value_targets"],
                has_played,
                normalization=targets["has_played"],
            )
            loss_dict[value_loss_field] += value_loss.item()

            loss += value_loss

            self.scaler.scale(loss).backward()

        return Loss(**loss_dict)

    def _get_repr_loss(
        self,
        state_action_emb: torch.Tensor,
        state_emb: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss(reduction="none")

        n = state_action_emb[:-1].flatten(0, 1).shape[0]
        labels = torch.arange(n, device=self.replay_buffer.device)

        logits = torch.einsum(
            "ik,jk->ij",
            state_action_emb[:-1].flatten(0, 1),
            state_emb[1:].flatten(0, 1),
        )
        loss = criterion(logits, labels)
        loss = torch.maximum(loss, torch.tensor(1))

        loss = loss * valid
        loss = loss.sum() / valid.sum().clamp(min=1)

        return loss

    def run(self):
        while True:
            self.step()
