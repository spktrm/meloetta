import torch

from typing import NamedTuple

from meloetta.actors.types import Choices


class PostProcess(NamedTuple):
    data: Choices
    index: torch.Tensor
