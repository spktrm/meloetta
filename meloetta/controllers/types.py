import torch

from typing import Callable, Tuple, Dict, List, Any

State = Dict[str, torch.Tensor]
Choices = Dict[str, Tuple[Callable, List[Any], Dict[str, Any]]]
