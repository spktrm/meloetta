import torch
import torch.nn as nn
import numpy as np

from typing import Dict, Tuple, Any, List


Action = Any


class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


def ucb_scores(
    parent: Node, children: List[Node], pb_c_base: int, pb_c_init: int
) -> Tuple[Action, Node]:
    child_priors = torch.tensor([child.prior for child in children])
    child_visit_counts = torch.tensor([child.visit_count for child in children])

    pb_c = np.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= np.sqrt(parent.visit_count) / (child_visit_counts + 1)

    prior_scores = pb_c * child_priors

    child_value_sums = torch.tensor([child.value_sum for child in children])
    value_scores = child_value_sums / child_visit_counts.clamp(min=1)
    value_scores *= child_visit_counts > 0

    return prior_scores + value_scores


class Tree:
    def __init__(self, model: nn.Module, num_simulations: int = 50):
        self.model = model
        self.root = Node(0)
        self.num_simulations = num_simulations

    def select_child(self, node: Node) -> Tuple:
        actions = list(node.children.keys())
        children = list(node.children.values())

        scores = ucb_scores(node, children)
        _, action, child = torch.argmax(scores)

        return action, child

    def run(self, initial_observation: Dict[str, torch.Tensor]):

        for _ in range(self.num_simulations):
            history = []
            node = self.root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node)
