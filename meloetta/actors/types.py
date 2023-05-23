import torch

from typing import Callable, Tuple, Union, Dict, List, Any

State = Dict[str, Union[None, torch.Tensor]]
Choices = Dict[str, Dict[str, Tuple[Callable, List[Any], Dict[str, Any]]]]
Battle = Dict[str, Dict[str, Dict[str, Any]]]
