import torch


def to_id(string: str):
    return "".join(c for c in string if c.isalnum()).lower()


def expand_bt(tensor: torch.Tensor, time: int = 1, batch: int = 1) -> torch.Tensor:
    shape = tensor.shape
    if not shape:
        shape = (1,)
    return tensor.view(time, batch, *(tensor.shape or shape))
