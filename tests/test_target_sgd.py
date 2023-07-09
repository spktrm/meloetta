import pytest

import torch
import torch.nn as nn

from meloetta.frameworks.nash_ketchum.learner import TargetNetSGD


def test_target_sgd():
    target_model = nn.Linear(4, 4, bias=False).requires_grad_(False)
    learner_model = nn.Linear(4, 4, bias=False)

    prev_target_weight = target_model.weight.clone()
    prev_learner_weight = learner_model.weight.clone()

    prev_dist = (prev_target_weight - prev_learner_weight) ** 2
    prev_dist = prev_dist.sum().item()

    optim = TargetNetSGD(1e-2, target_model, learner_model)
    optim.step()

    after_dist = (target_model.weight - learner_model.weight) ** 2
    after_dist = after_dist.sum().item()

    assert after_dist < prev_dist

    assert torch.all(prev_learner_weight == learner_model.weight).item()
    assert torch.all(prev_target_weight != target_model.weight).item()

    assert not target_model.weight.requires_grad
    assert learner_model.weight.requires_grad
