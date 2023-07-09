import torch

from typing import NamedTuple

from meloetta.types import Choices


class PostProcess(NamedTuple):
    data: Choices
    index: torch.Tensor
