import math
import time
import wandb
import traceback
import threading

import torch
import torch.optim as optim
import torch.nn.functional as F

from typing import Tuple

from meloetta.frameworks.mewzero import utils
from meloetta.frameworks.mewzero.buffer import ReplayBuffer
from meloetta.frameworks.mewzero.config import MewZeroConfig
from meloetta.frameworks.mewzero.entropy import EntropySchedule
from meloetta.frameworks.mewzero.model import (
    MewZeroModel,
    TrainingOutput,
    Batch,
    Targets,
    Loss,
    Policy,
)
from meloetta.frameworks.mewzero.utils import (
    _player_others,
    v_trace,
    get_loss_nerd,
    get_loss_v,
)


class TargetNetSGD:
    def __init__(
        self,
        lr: float,
        target_model: MewZeroModel,
        learner_model: MewZeroModel,
    ):
        self.lr = lr
        self.target_model = target_model
        self.learner_model = learner_model
        self.param_groups = []

        for (name, param1), param2 in zip(
            self.target_model.named_parameters(), self.learner_model.parameters()
        ):
            if param2.requires_grad:
                self.param_groups.append({"params": param1, "name": name})

    def step(self):
        learner_state_dict = self.learner_model.state_dict()
        for group in self.param_groups:
            name = group["name"]
            param = group["params"]
            diff = learner_state_dict[name] - param.data
            param.data += self.lr * diff


class MewZeroLearner:
    def __init__(
        self,
        learner_model: MewZeroModel,
        actor_model: MewZeroModel,
        target_model: MewZeroModel,
        model_prev: MewZeroModel,
        model_prev_: MewZeroModel,
        replay_buffer: ReplayBuffer,
        config: MewZeroConfig = MewZeroConfig(),
    ):
        self.learner_model = learner_model
        self.actor_model = actor_model
        self.target_model = target_model
        self.model_prev = model_prev
        self.model_prev_ = model_prev_

        self.replay_buffer = replay_buffer

        self.config = config
        self._entropy_schedule = EntropySchedule(
            sizes=self.config.entropy_schedule_size,
            repeats=self.config.entropy_schedule_repeats,
        )
        self.optimizer = optim.Adam(
            self.learner_model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam.b1, config.adam.b2),
            eps=config.adam.eps,
        )
        self.optimizer_target = TargetNetSGD(
            config.target_network_avg,
            self.target_model,
            self.learner_model,
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
                "target_model": self.target_model.state_dict(),
                "model_prev": self.model_prev.state_dict(),
                "model_prev_": self.model_prev_.state_dict(),
                "config": self.config,
                "optimizer": self.optimizer.state_dict(),
                "learner_steps": self.learner_steps,
            },
            f"cpkts/cpkt-{self.learner_steps:05}.tar",
        )

    @classmethod
    def from_config(self, config: MewZeroConfig = None):
        return MewZeroLearner.from_pretrained(config_override=config)

    @classmethod
    def from_pretrained(self, fpath: str = None, config_override: MewZeroConfig = None):
        assert not (
            fpath is None and config_override is None
        ), "please set one of `fpath` or `config`"

        if fpath is not None:
            datum = torch.load(fpath)
        else:
            datum = {}

        if datum.get("config"):
            print("Found config!")
            config: MewZeroConfig = datum["config"]

        if config_override is not None:
            config = config_override

        gen, gametype = utils.get_gen_and_gametype(config.battle_format)
        learner_model = MewZeroModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        learner_model.train()

        print(f"learnabled params: {learner_model.get_learnable_params():,}")

        actor_model = MewZeroModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        actor_model.load_state_dict(learner_model.state_dict())
        actor_model.share_memory()
        actor_model.eval()

        target_model = MewZeroModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        target_model.load_state_dict(learner_model.state_dict())
        target_model.eval()

        model_prev = MewZeroModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        model_prev.load_state_dict(learner_model.state_dict())
        model_prev.eval()

        model_prev_ = MewZeroModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        model_prev_.load_state_dict(learner_model.state_dict())
        model_prev_.eval()

        if datum.get("learner_model"):
            try:
                learner_model.load_state_dict(datum["learner_model"], strict=False)
                actor_model.load_state_dict(datum["actor_model"], strict=False)
                target_model.load_state_dict(datum["target_model"], strict=False)
                model_prev.load_state_dict(datum["model_prev"], strict=False)
                model_prev_.load_state_dict(datum["model_prev_"], strict=False)
            except Exception as e:
                traceback.print_exc()

        learner_model = learner_model.to(config.learner_device, non_blocking=True)
        actor_model = actor_model.to(config.actor_device, non_blocking=True)
        target_model = target_model.to(config.learner_device, non_blocking=True)
        model_prev = model_prev.to(config.learner_device, non_blocking=True)
        model_prev_ = model_prev_.to(config.learner_device, non_blocking=True)

        learner = MewZeroLearner(
            learner_model,
            actor_model,
            target_model,
            model_prev,
            model_prev_,
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
        alpha, update_target_net = self._entropy_schedule(self.learner_steps)
        self.update_paramters(batch, hidden_state, alpha, update_target_net)
        self.learner_steps += 1
        if self.learner_steps % 500 == 0:
            self.save()

    def update_paramters(
        self,
        batch: Batch,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        alpha: float,
        update_target_net: bool,
        lock=threading.Lock(),
    ):
        with lock:
            self.optimizer.zero_grad(set_to_none=True)

            targets = self._get_targets(batch, hidden_state, alpha)
            losses = self._backpropagate(batch, hidden_state, targets)

            static_loss_dict = losses.to_log(batch)
            static_loss_dict["s"] = self.learner_steps
            wandb.log(static_loss_dict)

            torch.nn.utils.clip_grad_value_(
                self.learner_model.parameters(), self.config.clip_gradient
            )

            self.optimizer.step()
            with torch.no_grad():
                self.optimizer_target.step()

            if update_target_net:
                self.model_prev_.load_state_dict(self.model_prev.state_dict())
                self.model_prev.load_state_dict(self.learner_model.state_dict())

            self.actor_model.load_state_dict(self.learner_model.state_dict())

    def _get_targets(
        self,
        batch: Batch,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        alpha: float,
    ) -> Targets:
        bidx = torch.arange(
            batch.batch_size, device=next(self.learner_model.parameters()).device
        )
        lengths = batch.valid.sum(0)
        final_reward = batch.rewards[lengths.squeeze() - 1, bidx]
        assert torch.all(final_reward.sum(-1) == 0).item(), "Env should be zero-sum!"

        learner: TrainingOutput
        target: TrainingOutput
        prev: TrainingOutput
        prev_: TrainingOutput

        with torch.no_grad():
            learner = self.learner_model.learning_forward(batch, hidden_state)
            target = self.target_model.learning_forward(batch, hidden_state)
            prev = self.model_prev.learning_forward(batch, hidden_state)
            prev_ = self.model_prev_.learning_forward(batch, hidden_state)

        is_vector = torch.unsqueeze(torch.ones_like(batch.valid), dim=-1)
        importance_sampling_corrections = [is_vector] * 2

        targets_dict = {
            "importance_sampling_corrections": importance_sampling_corrections
        }

        for policy_field, pi, log_pi, prev_log_pi, prev_log_pi_ in zip(
            *zip(*learner.pi._asdict().items()),
            learner.log_pi,
            prev.log_pi,
            prev_.log_pi,
        ):
            pi: torch.Tensor
            policy_field: str
            if pi is None:
                continue

            log_policy_reg = log_pi - (alpha * prev_log_pi + (1 - alpha) * prev_log_pi_)

            v_target_list, has_played_list, v_trace_policy_target_list = [], [], []

            for player in range(2):
                reward = batch.rewards[:, :, player]  # [T, B, Player]

                action_index = batch.get_index_from_policy(policy_field)
                action_oh = F.one_hot(action_index, pi.shape[-1])

                v_target_, has_played, policy_target_ = v_trace(
                    target.v,
                    batch.valid,
                    batch.player_id,
                    pi,
                    pi,
                    log_policy_reg,
                    _player_others(batch.player_id, batch.valid, player),
                    action_oh,
                    reward,
                    player,
                    lambda_=1.0,
                    c=self.config.c_vtrace,
                    rho=torch.inf,
                    eta=self.config.eta_reward_transform,
                    gamma=self.config.gamma,
                )
                v_target_list.append(v_target_)
                has_played_list.append(has_played)
                v_trace_policy_target_list.append(policy_target_)

            targets_dict[
                policy_field.replace("_policy", "_value_target")
            ] = v_target_list
            targets_dict[
                policy_field.replace("_policy", "_has_played")
            ] = has_played_list
            targets_dict[
                policy_field.replace("_policy", "_policy_target")
            ] = v_trace_policy_target_list

        return Targets(**targets_dict)

    def _backpropagate(
        self,
        batch: Batch,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        targets: Targets,
    ) -> Loss:

        loss_dict = {key + "_loss": 0 for key in Policy._fields}
        loss_dict.update({key.replace("policy", "value"): 0 for key in loss_dict})

        fields = (
            "action_type",
            "move",
            "switch",
        )
        if self.actor_model.gen >= 6:
            fields += ("flag",)

        if self.actor_model.gen == 8:
            fields += ("max_move",)

        if self.actor_model.gametype != "singles":
            fields += ("target",)

        global_action_type = batch.action_type_index

        total_valid = batch.valid.sum().item()

        total_move_valid = batch.valid * (global_action_type == 0)
        total_move_valid *= batch.move_mask.sum(-1) > 1
        total_move_valid = total_move_valid.sum().item()

        total_switch_valid = batch.valid * (global_action_type == 1)
        total_switch_valid *= batch.switch_mask.sum(-1) > 1
        total_switch_valid = total_switch_valid.sum().item()

        if batch.flag_mask is not None:
            total_flag_valid = batch.valid * (global_action_type == 0)
            total_flag_valid *= batch.flag_mask.sum(-1) > 1
            total_flag_valid = total_flag_valid.sum().item()

        minibatch_size = min(128, (4096 // batch.trajectory_length) + 1)

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
                need_log_policy=False,
            )

            for field in fields:
                pi_field = field + "_policy"
                logit_field = field + "_logits"

                pi = getattr(output.pi, pi_field)
                logit = getattr(output.logit, logit_field)

                if pi is None:
                    continue

                action_type_index = minibatch.action_type_index
                policy_mask = minibatch.get_mask_from_policy(pi_field)

                if field == "action_type":
                    valid = torch.ones_like(action_type_index)
                elif field == "switch":
                    valid = action_type_index == 1
                elif field == "target":
                    valid = action_type_index == 2
                else:
                    valid = action_type_index == 0

                valid *= policy_mask.sum(-1) > 1

                value_loss = get_loss_v(
                    [output.v] * 2,
                    targ.get_value_target(pi_field),
                    targ.get_has_played(pi_field, valid),
                )

                # Uses v-trace to define q-values for Nerd
                policy_loss = get_loss_nerd(
                    [logit] * 2,
                    [pi] * 2,
                    targ.get_policy_target(pi_field),
                    minibatch.valid * valid,
                    minibatch.player_id,
                    policy_mask,
                    targ.importance_sampling_corrections,
                    clip=self.config.nerd.clip,
                    threshold=self.config.nerd.beta,
                )

                policy_loss_field = field + "_policy_loss"
                value_loss_field = field + "_value_loss"

                loss_dict[policy_loss_field] += policy_loss.item()
                loss_dict[value_loss_field] += value_loss.item()

                if "action_type" in field:
                    policy_valid = total_valid
                elif "move" in field:
                    policy_valid = total_move_valid
                elif "flag" in field:
                    policy_valid = total_flag_valid
                elif "switch" in field:
                    policy_valid = total_switch_valid

                loss += (value_loss + policy_loss) / policy_valid

            loss.backward()

        for key in loss_dict:
            if "action_type" in key:
                loss_dict[key] /= total_valid
            elif "move" in key:
                loss_dict[key] /= total_move_valid
            elif "flag" in key:
                loss_dict[key] /= total_flag_valid
            elif "switch" in key:
                loss_dict[key] /= total_switch_valid

        return Loss(**loss_dict)

    def run(self):
        while True:
            self.step()
