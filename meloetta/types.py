import torch

from typing import Callable, Tuple, Union, Dict, List, Any

TensorDict = Dict[str, Union[None, torch.Tensor]]
State = TensorDict
Choices = Dict[str, Dict[str, Tuple[Callable, List[Any], Dict[str, Any]]]]
Battle = Dict[str, Dict[str, Dict[str, Any]]]
