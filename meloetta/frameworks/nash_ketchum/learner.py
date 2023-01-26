import wandb
import traceback
import threading
import functools

import torch
import torch.optim as optim
import torch.nn.functional as F

from meloetta.frameworks.nash_ketchum import utils
from meloetta.frameworks.nash_ketchum.buffer import ReplayBuffer
from meloetta.frameworks.nash_ketchum.config import NAshKetchumConfig
from meloetta.frameworks.nash_ketchum.entropy import EntropySchedule
from meloetta.frameworks.nash_ketchum.model import (
    NAshKetchumModel,
    TrainingOutput,
    Batch,
    Targets,
    Loss,
    Policy,
)
from meloetta.frameworks.nash_ketchum.utils import (
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
            grad = param1 - param2
            param1.add_(grad, alpha=-self.lr)


class NAshKetchumLearner:
    def __init__(
        self,
        learner_model: NAshKetchumModel,
        actor_model: NAshKetchumModel,
        target_model: NAshKetchumModel,
        model_prev: NAshKetchumModel,
        model_prev_: NAshKetchumModel,
        replay_buffer: ReplayBuffer,
        config: NAshKetchumConfig = NAshKetchumConfig(),
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
    def from_config(self, config: NAshKetchumConfig = None):
        return NAshKetchumLearner.from_pretrained(config_override=config)

    @classmethod
    def from_pretrained(
        self, fpath: str = None, config_override: NAshKetchumConfig = None
    ):
        assert not (
            fpath is None and config_override is None
        ), "please set one of `fpath` or `config`"

        if fpath is not None:
            datum = torch.load(fpath)
        else:
            datum = {}

        if datum.get("config"):
            print("Found config!")
            config: NAshKetchumConfig = datum["config"]

        if config_override is not None:
            config = config_override

        gen, gametype = utils.get_gen_and_gametype(config.battle_format)
        learner_model = NAshKetchumModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        learner_model.train()

        print(f"learnabled params: {learner_model.get_learnable_params():,}")

        actor_model = NAshKetchumModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        actor_model.load_state_dict(learner_model.state_dict())
        actor_model.share_memory()
        actor_model.eval()

        target_model = NAshKetchumModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        target_model.load_state_dict(learner_model.state_dict())
        target_model.eval()

        model_prev = NAshKetchumModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        model_prev.load_state_dict(learner_model.state_dict())
        model_prev.eval()

        model_prev_ = NAshKetchumModel(
            gen=gen, gametype=gametype, config=config.model_config
        )
        model_prev_.load_state_dict(learner_model.state_dict())
        model_prev_.eval()

        if datum.get("learner_model"):
            try:
                learner_model.load_state_dict(datum["learner_model"])
                actor_model.load_state_dict(datum["actor_model"])
                target_model.load_state_dict(datum["target_model"])
                model_prev.load_state_dict(datum["model_prev"])
                model_prev_.load_state_dict(datum["model_prev_"])
            except Exception as e:
                traceback.print_exc()

        learner_model.to(config.learner_device)
        actor_model.to(config.actor_device)
        target_model.to(config.learner_device)
        model_prev.to(config.learner_device)
        model_prev_.to(config.learner_device)

        learner = NAshKetchumLearner(
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
        return self.replay_buffer.get_batch(self.config.batch_size)

    def step(self):
        batch = self.collect_batch_trajectory()
        alpha, update_target_net = self._entropy_schedule(self.learner_steps)
        self.update_paramters(batch, alpha, update_target_net)
        self.learner_steps += 1
        if self.learner_steps % 500 == 0:
            self.save()

    def update_paramters(
        self,
        batch: Batch,
        alpha: float,
        update_target_net: bool,
        lock=threading.Lock(),
    ):
        targets = self._get_targets(batch, alpha)
        losses = self._backpropagate(batch, targets)

        static_loss_dict = losses.to_log(batch)
        static_loss_dict["s"] = self.learner_steps
        wandb.log(static_loss_dict)

        torch.nn.utils.clip_grad_value_(
            self.learner_model.parameters(), self.config.clip_gradient
        )

        self.optimizer.step()
        self.optimizer_target.step()
        self.optimizer.zero_grad(set_to_none=True)

        if update_target_net:
            self.model_prev_.load_state_dict(self.model_prev.state_dict())
            self.model_prev.load_state_dict(self.learner_model.state_dict())

        with lock:
            self.actor_model.load_state_dict(self.learner_model.state_dict())

    def _get_targets(self, batch: Batch, alpha: float) -> Targets:
        with torch.no_grad():
            learner: TrainingOutput = self.learner_model.learning_forward(batch)
            target: TrainingOutput = self.target_model.learning_forward(batch)
            prev: TrainingOutput = self.model_prev.learning_forward(batch)
            prev_: TrainingOutput = self.model_prev_.learning_forward(batch)

        is_vector = torch.unsqueeze(torch.ones_like(batch.valid), dim=-1)
        importance_sampling_corrections = [is_vector] * 2

        targets_dict = {
            "importance_sampling_corrections": importance_sampling_corrections
        }

        action_type_index = batch.action_type_index

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

                if policy_field == "switch_policy":
                    valid = action_type_index == 1
                elif policy_field == "target_policy":
                    valid = action_type_index == 2
                else:
                    valid = action_type_index == 0

                v_target_, has_played, policy_target_ = v_trace(
                    target.v,
                    batch.valid * valid,
                    batch.player_id,
                    pi,
                    pi,
                    log_policy_reg,
                    _player_others(batch.player_id, batch.valid * valid, player),
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
        targets: Targets,
        minibatch_size: int = 48,
        minitraj_size: int = 64,
    ) -> Loss:

        loss_dict = {key + "_loss": 0 for key in Policy._fields}
        loss_dict["value_loss"] = 0
        total_valid = batch.valid.sum().item()
        num_steps = 0

        # minibatch_size = int(round(batch.valid.sum(1).float().mean().item(), 0))
        # minitraj_size = int(round(batch.valid.sum(0).float().mean().item(), 0))
        # minibatch_size = 32
        # minitraj_size = 64

        for pred, targ in zip(
            batch.iterate(minibatch_size, minitraj_size),
            targets.iterate(minibatch_size, minitraj_size),
        ):
            if not pred.valid.sum():
                continue

            output = self.learner_model.forward(
                pred._asdict(), training=True, need_log_policy=False
            )

            loss = 0
            pred_valid = pred.valid.sum().item()
            accumulation_step_weight = pred_valid / total_valid

            for policy_field, pi, logit in zip(
                *zip(*output.pi._asdict().items()), output.logit
            ):
                if pi is None:
                    continue

                action_type_index = pred.action_type_index
                if policy_field == "switch_policy":
                    valid = action_type_index == 1
                elif policy_field == "target_policy":
                    valid = action_type_index == 2
                else:
                    valid = action_type_index == 0

                value_loss = get_loss_v(
                    [output.v] * 2,
                    targ.get_value_target(policy_field),
                    targ.get_has_played(policy_field),
                )

                # Uses v-trace to define q-values for Nerd
                policy_loss = get_loss_nerd(
                    [logit] * 2,
                    [pi] * 2,
                    targ.get_policy_target(policy_field),
                    pred.valid * valid,
                    pred.player_id,
                    pred.get_mask_from_policy(policy_field),
                    targ.importance_sampling_corrections,
                    clip=self.config.nerd.clip,
                    threshold=self.config.nerd.beta,
                )

                loss_field = policy_field + "_loss"
                loss_dict[loss_field] += policy_loss.item() * accumulation_step_weight
                loss_dict["value_loss"] += value_loss.item() * accumulation_step_weight

                loss += value_loss
                loss += policy_loss
                num_steps += 1

            loss *= accumulation_step_weight
            loss.backward()

        # loss_dict = {key: value  for key, value in loss_dict.items()}
        return Loss(**loss_dict)

    def run(self):
        while True:
            self.step()
