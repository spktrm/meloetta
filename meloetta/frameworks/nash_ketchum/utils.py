import re
import torch
import torch.nn.functional as F

from typing import Dict, List, Any, Sequence, NamedTuple, Tuple

from jax import tree_util as tree

from meloetta.actors.types import TensorDict
from meloetta.data import CHOICE_FLAGS


def _get_leading_dims(tensor_dict: TensorDict) -> int:
    return next(iter(tensor_dict.values())).shape[:2]


class LoopVTraceCarry(NamedTuple):
    """The carry of the v-trace scan loop."""

    reward: torch.Tensor
    # The cumulated reward until the end of the episode. Uncorrected (v-trace).
    # Gamma discounted and includes eta_reg_entropy.
    reward_uncorrected: torch.Tensor
    next_value: torch.Tensor
    next_v_target: torch.Tensor
    importance_sampling: torch.Tensor


def _player_others(
    player_ids: torch.Tensor, valid: torch.Tensor, player: int
) -> torch.Tensor:
    """A vector of 1 for the current player and -1 for others.

    Args:
      player_ids: Tensor [...] containing player ids (0 <= player_id < N).
      valid: Tensor [...] containing whether these states are valid.
      player: The player id as int.

    Returns:
      player_other: is 1 for the current player and -1 for others [..., 1].
    """
    current_player_tensor = (player_ids == player).to(torch.int32)

    res = 2 * current_player_tensor - 1
    res = res * valid
    return torch.unsqueeze(res, dim=-1)


def _where(pred: torch.Tensor, true_data: Any, false_data: Any) -> Any:
    """Similar to jax.where but treats `pred` as a broadcastable prefix."""

    def _where_one(t, f):
        # Expand the dimensions of pred if true_data and false_data are higher rank.
        p = torch.reshape(pred, pred.shape + (1,) * (len(t.shape) - len(pred.shape)))
        return torch.where(p, t, f)

    return tree.tree_map(_where_one, true_data, false_data)


def pytorch_scan(f, init, xs, length=None, reverse=False):
    if length is None:
        length = xs[0].shape[0]

    def body(state, i):
        xs_i = tree.tree_map(lambda x: x[i], xs)
        state, ys = f(state, xs_i)
        return state, ys

    state = init
    ys_list = []

    indices = range(length)
    if reverse:
        indices = reversed(indices)

    for i in indices:
        state, ys = body(state, i)
        ys_list.append(ys)

    if reverse:
        ys_list = list(reversed(ys_list))

    # Stack the outputs along dimension 0
    ys = tree.tree_map(lambda *args: torch.stack(args, dim=0), *ys_list)
    return state, ys


def _has_played(
    valid: torch.Tensor, player_id: torch.Tensor, player: int
) -> torch.Tensor:
    """Compute a mask of states which have a next state in the sequence."""
    assert valid.shape == player_id.shape

    def _loop_has_played(carry, x):
        valid, player_id = x
        assert valid.shape == player_id.shape

        our_res = torch.ones_like(player_id)
        opp_res = carry
        reset_res = torch.zeros_like(carry)

        our_carry = carry
        opp_carry = carry
        reset_carry = torch.zeros_like(player_id)

        # pyformat: disable
        return _where(
            valid,
            _where(
                (player_id == player),
                (our_carry, our_res),
                (opp_carry, opp_res),
            ),
            (reset_carry, reset_res),
        )
        # pyformat: enable

    _, result = pytorch_scan(
        f=_loop_has_played,
        init=torch.zeros_like(player_id[-1]),
        xs=(valid, player_id),
        reverse=True,
    )
    return result


def _policy_ratio(
    pi: torch.Tensor, mu: torch.Tensor, actions_oh: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Returns a ratio of policy pi/mu when selecting action a.

    By convention, this ratio is 1 on non valid states
    Args:
      pi: the policy of shape [..., A].
      mu: the sampling policy of shape [..., A].
      actions_oh: a one-hot encoding of the current actions of shape [..., A].
      valid: 0 if the state is not valid and else 1 of shape [...].

    Returns:
      pi/mu on valid states and 1 otherwise. The shape is the same
      as pi, mu or actions_oh but without the last dimension A.
    """
    assert pi.shape == mu.shape == actions_oh.shape
    # assert ((valid,), actions_oh.shape[:-1])

    def _select_action_prob(pi: torch.Tensor) -> torch.Tensor:
        return torch.sum(actions_oh * pi, dim=-1) * valid + ~valid

    pi_actions_prob = _select_action_prob(pi)
    mu_actions_prob = _select_action_prob(mu)
    return pi_actions_prob / mu_actions_prob


@torch.no_grad()
def v_trace(
    v: Sequence[torch.Tensor],
    policy_select: torch.Tensor,
    valid: torch.Tensor,
    policies_valid: Sequence[torch.Tensor],
    player_id: torch.Tensor,
    acting_policies: Sequence[torch.Tensor],
    merged_policies: Sequence[torch.Tensor],
    merged_log_policies: Sequence[torch.Tensor],
    player_other: torch.Tensor,
    actions_ohs: Sequence[torch.Tensor],
    reward: torch.Tensor,
    player: int,
    # Scalars below.
    eta: float,
    lambda_: float,
    c: float,
    rho: float,
    gamma: float = 1.0,
) -> Tuple[Any, Any, Any]:
    """Custom VTrace for trajectories with a mix of different player steps."""

    has_played = _has_played(valid, player_id, player)

    policy_ratios = []
    inv_mus = []
    eta_reg_entropies = []
    eta_log_policies = []

    for (
        merged_policy,
        acting_policy,
        actions_oh,
        policy_valid,
        merged_log_policy,
    ) in zip(
        merged_policies,
        acting_policies,
        actions_ohs,
        policies_valid,
        merged_log_policies,
    ):
        policy_ratios.append(
            _policy_ratio(merged_policy, acting_policy, actions_oh, policy_valid)
        )
        inv_mus.append(
            _policy_ratio(
                torch.ones_like(merged_policy), acting_policy, actions_oh, policy_valid
            )
        )
        eta_reg_entropies.append(
            torch.sum(merged_policy * merged_log_policy, dim=-1)
            * torch.squeeze(player_other, dim=-1)
        )
        eta_log_policies.append(-eta * merged_log_policy * player_other)

    eta_reg_entropy = torch.stack(eta_reg_entropies, dim=-1)
    eta_reg_entropy = eta_reg_entropy * torch.stack(policies_valid, dim=-1)
    eta_reg_entropy = -eta * torch.sum(eta_reg_entropy, dim=-1)
    policy_ratio = torch.prod(torch.stack(policy_ratios, dim=-1), dim=-1)

    init_state_v_trace = LoopVTraceCarry(
        reward=torch.zeros_like(reward[-1]),
        reward_uncorrected=torch.zeros_like(reward[-1]),
        next_value=torch.zeros_like(v[0][-1]),
        next_v_target=torch.zeros_like(v[0][-1]),
        importance_sampling=torch.ones_like(policy_ratio[-1]),
    )

    def _loop_v_trace(carry: LoopVTraceCarry, x) -> Tuple[LoopVTraceCarry, Any]:
        (
            cs,
            player_id,
            v,
            reward,
            eta_reg_entropy,
            valid,
            inv_mus,
            actions_ohs,
            eta_log_policies,
        ) = x

        reward_uncorrected = reward + gamma * carry.reward_uncorrected + eta_reg_entropy
        discounted_reward = reward + gamma * carry.reward

        # V-target:
        our_v_target = (
            v
            + torch.unsqueeze(
                torch.minimum(torch.tensor(rho), cs * carry.importance_sampling), dim=-1
            )
            * (
                torch.unsqueeze(reward_uncorrected, dim=-1)
                + gamma * carry.next_value
                - v
            )
            + lambda_
            * torch.unsqueeze(
                torch.minimum(torch.tensor(c), cs * carry.importance_sampling), dim=-1
            )
            * gamma
            * (carry.next_v_target - carry.next_value)
        )

        opp_v_target = torch.zeros_like(our_v_target)
        reset_v_target = torch.zeros_like(our_v_target)

        # Learning output:
        our_learning_outputs = [
            (
                v
                + eta_log_policy  # value
                + actions_oh  # regularisation
                * torch.unsqueeze(inv_mu, dim=-1)
                * (
                    torch.unsqueeze(discounted_reward, dim=-1)
                    + gamma
                    * torch.unsqueeze(carry.importance_sampling, dim=-1)
                    * carry.next_v_target
                    - v
                )
            )
            for eta_log_policy, inv_mu, actions_oh in zip(
                eta_log_policies, inv_mus, actions_ohs
            )
        ]

        opp_learning_outputs = [
            torch.zeros_like(our_learning_output)
            for our_learning_output in our_learning_outputs
        ]
        reset_learning_output = [
            torch.zeros_like(our_learning_output)
            for our_learning_output in our_learning_outputs
        ]

        # State carry:
        our_carry = LoopVTraceCarry(
            reward=torch.zeros_like(carry.reward),
            next_value=v,
            next_v_target=our_v_target,
            reward_uncorrected=torch.zeros_like(carry.reward_uncorrected),
            importance_sampling=torch.ones_like(carry.importance_sampling),
        )
        opp_carry = LoopVTraceCarry(
            reward=eta_reg_entropy + cs * discounted_reward,
            reward_uncorrected=reward_uncorrected,
            next_value=gamma * carry.next_value,
            next_v_target=gamma * carry.next_v_target,
            importance_sampling=cs * carry.importance_sampling,
        )
        reset_carry = init_state_v_trace

        # Invalid turn: init_state_v_trace and (zero target, learning_output)
        # pyformat: disable
        return _where(
            valid,
            _where(
                (player_id == player),
                (our_carry, (our_v_target, our_learning_outputs)),
                (opp_carry, (opp_v_target, opp_learning_outputs)),
            ),
            (reset_carry, (reset_v_target, reset_learning_output)),
        )

    v = torch.stack(v, dim=-1).squeeze()
    v = (v * F.one_hot(policy_select, 4)).sum(-1, keepdim=True)

    _, (v_target, learning_output) = pytorch_scan(
        f=_loop_v_trace,
        init=init_state_v_trace,
        xs=(
            policy_ratio,
            player_id,
            v,
            reward,
            eta_reg_entropy,
            valid,
            inv_mus,
            actions_ohs,
            eta_log_policies,
        ),
        reverse=True,
    )

    return v_target, has_played, learning_output, policy_ratios


def apply_force_with_threshold(
    decision_outputs: torch.Tensor,
    force: torch.Tensor,
    threshold: float,
    threshold_center: torch.Tensor,
) -> torch.Tensor:
    """Apply the force with below a given threshold."""
    can_decrease = decision_outputs - threshold_center > -threshold
    can_increase = decision_outputs - threshold_center < threshold
    force_negative = torch.minimum(force, torch.zeros_like(force))
    force_positive = torch.maximum(force, torch.zeros_like(force))
    clipped_force = can_decrease * force_negative + can_increase * force_positive
    return decision_outputs * clipped_force.detach()


def renormalize(loss: torch.Tensor, mask: torch.Tensor, denom: int) -> torch.Tensor:
    """The `normalization` is the number of steps over which loss is computed."""
    loss = torch.sum(loss * mask)
    return loss / denom


def get_loss_v(
    v_list: Sequence[torch.Tensor],
    v_target_list: Sequence[torch.Tensor],
    mask_list: Sequence[torch.Tensor],
    scale_list: Sequence[int],
) -> Sequence[torch.Tensor]:
    """Define the loss function for the critic."""
    loss_v_list = []
    for v_n, v_target, mask, scale in zip(v_list, v_target_list, mask_list, scale_list):
        assert v_n.shape[0] == v_target.shape[0]

        loss_v = torch.unsqueeze(mask, dim=-1) * (v_n - v_target.detach()) ** 2
        loss_v = torch.sum(loss_v) / scale
        loss_v_list.append(loss_v)

    return loss_v_list


def get_loss_nerd(
    logit_list: Sequence[torch.Tensor],
    policy_list: Sequence[torch.Tensor],
    q_vr_list: Sequence[torch.Tensor],
    valid: torch.Tensor,
    player_ids: Sequence[torch.Tensor],
    scale: Sequence[int],
    legal_actions: torch.Tensor,
    importance_sampling_correction: Sequence[torch.Tensor],
    clip: float = 100,
    threshold: float = 2,
) -> Sequence[torch.Tensor]:
    """Define the nerd loss."""
    assert isinstance(importance_sampling_correction, list)
    loss_pi_list = []
    for k, (logit_pi, pi, q_vr, is_c) in enumerate(
        zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)
    ):
        assert logit_pi.shape[0] == q_vr.shape[0]
        # loss policy
        adv_pi = q_vr - torch.sum(pi * q_vr, dim=-1, keepdim=True)
        adv_pi = is_c * adv_pi  # importance sampling correction
        adv_pi = torch.clip(adv_pi, min=-clip, max=clip)
        adv_pi = adv_pi.detach()

        logits = logit_pi - torch.mean(logit_pi * legal_actions, dim=-1, keepdim=True)

        threshold_center = torch.zeros_like(logits)

        force = apply_force_with_threshold(
            logits,
            adv_pi,
            threshold,
            threshold_center,
        )
        nerd_loss = torch.sum(legal_actions * force, dim=-1)
        nerd_loss = -renormalize(nerd_loss, valid * (player_ids == k), scale[k])
        loss_pi_list.append(nerd_loss)
    return loss_pi_list


def get_gen_and_gametype(battle_format: str) -> Tuple[int, str]:
    gen = int(re.search(r"gen([0-9])", battle_format).groups()[0])
    if "triples" in battle_format:
        gametype = "triples"
    if "doubles" in battle_format:
        gametype = "doubles"
    else:
        gametype = "singles"
    return gen, gametype


def to_id(string: str):
    return "".join(c for c in string if c.isalnum()).lower()


def _get_private_reserve_size(gen: int):
    if gen == 9:
        return 28
    elif gen == 8:
        return 25
    else:
        return 24


def get_buffer_specs(
    trajectory_length: int,
    gen: int,
    gametype: str,
    private_reserve_size: int,
):
    if gametype != "singles":
        if gametype == "doubles":
            n_active = 2
        else:
            n_active = 3
    else:
        n_active = 1

    buffer_specs = {
        "sides": {
            "size": (trajectory_length, 3, 12, 38),
            "dtype": torch.long,
        },
        "boosts": {
            "size": (trajectory_length, 2, 8),
            "dtype": torch.long,
        },
        "volatiles": {
            "size": (trajectory_length, 2, 113),
            "dtype": torch.long,
        },
        "side_conditions": {
            "size": (trajectory_length, 2, 58),
            "dtype": torch.long,
        },
        "pseudoweathers": {
            "size": (trajectory_length, 12, 2),
            "dtype": torch.long,
        },
        "weather": {
            "size": (trajectory_length, 3),
            "dtype": torch.long,
        },
        "wisher": {
            "size": (trajectory_length, 2),
            "dtype": torch.long,
        },
        "scalars": {
            "size": (trajectory_length, 7),
            "dtype": torch.long,
        },
        "action_type_mask": {
            "size": (trajectory_length, 3),
            "dtype": torch.bool,
        },
        "move_mask": {
            "size": (trajectory_length, 4),
            "dtype": torch.bool,
        },
        "switch_mask": {
            "size": (trajectory_length, 6),
            "dtype": torch.bool,
        },
        "flag_mask": {
            "size": (trajectory_length, len(CHOICE_FLAGS)),
            "dtype": torch.bool,
        },
        "rewards": {
            "size": (trajectory_length,),
            "dtype": torch.float32,
        },
        "value": {
            "size": (trajectory_length,),
            "dtype": torch.float32,
        },
    }
    if gametype != "singles":
        buffer_specs.update(
            {
                "target_mask": {
                    "size": (trajectory_length, 4),
                    "dtype": torch.bool,
                },
                "prev_choices": {
                    "size": (trajectory_length, 2, 4),
                    "dtype": torch.long,
                },
                "choices_done": {
                    "size": (trajectory_length, 1),
                    "dtype": torch.long,
                },
                "targeting": {
                    "size": (trajectory_length, 1),
                    "dtype": torch.long,
                },
            }
        )

    # add policy...
    buffer_specs.update(
        {
            "action_type_policy": {
                "size": (trajectory_length, 3),
                "dtype": torch.float,
            },
            "flag_policy": {
                "size": (trajectory_length, len(CHOICE_FLAGS)),
                "dtype": torch.float,
            },
            "move_policy": {
                "size": (trajectory_length, 4),
                "dtype": torch.float,
            },
            "switch_policy": {
                "size": (trajectory_length, 6),
                "dtype": torch.float,
            },
        }
    )

    # ...and indices
    buffer_specs.update(
        {
            "action_type_index": {
                "size": (trajectory_length,),
                "dtype": torch.long,
            },
            "flag_index": {
                "size": (trajectory_length,),
                "dtype": torch.long,
            },
            "move_index": {
                "size": (trajectory_length,),
                "dtype": torch.long,
            },
            "switch_index": {
                "size": (trajectory_length,),
                "dtype": torch.long,
            },
        }
    )

    buffer_specs.update(
        {
            "valid": {
                "size": (trajectory_length,),
                "dtype": torch.bool,
            },
            "utc": {
                "size": (trajectory_length,),
                "dtype": torch.float64,
            },
            "hist": {
                "size": (trajectory_length, 10, 4, 4),
                "dtype": torch.long,
            },
            "policy_select": {
                "size": (trajectory_length,),
                "dtype": torch.long,
            },
        },
    )
    return buffer_specs


def create_buffers(
    num_buffers: int,
    trajectory_length: int,
    gen: int,
    gametype: str,
    num_players: int = 2,
):
    """
    num_buffers: int
        the size of the replay buffer
    trajectory_length: int
        the max length of a replay
    battle_format: str
        the type of format
    """

    private_reserve_size = _get_private_reserve_size(gen)

    buffer_specs = get_buffer_specs(
        trajectory_length, gen, gametype, private_reserve_size
    )

    buffers: Dict[str, List[List[torch.Tensor]]] = {
        key: [[], []] for key in buffer_specs
    }
    for _ in range(num_buffers):
        for k in range(num_players):
            for key in buffer_specs:
                if key.endswith("_mask"):
                    buffers[key][k].append(
                        torch.ones(**buffer_specs[key]).share_memory_()
                    )
                else:
                    buffers[key][k].append(
                        torch.zeros(**buffer_specs[key]).share_memory_()
                    )
    return buffers


def expand_bt(tensor: torch.Tensor, time: int = 1, batch: int = 1) -> torch.Tensor:
    shape = tensor.shape
    return tensor.view(time, batch, *(tensor.shape or shape))


class FineTuning:
    """Fine tuning options, aka policy post-processing.

    Even when fully trained, the resulting softmax-based policy may put
    a small probability mass on bad actions. This results in an agent
    waiting for the opponent (itself in self-play) to commit an error.

    To address that the policy is post-processed using:
    - thresholding: any action with probability smaller than self.threshold
      is simply removed from the policy.
    - discretization: the probability values are rounded to the closest
      multiple of 1/self.discretization.

    The post-processing is used on the learner, and thus must be jit-friendly.
    """

    # The learner step after which the policy post processing (aka finetuning)
    # will be enabled when learning. A strictly negative value is equivalent
    # to infinity, ie disables finetuning completely.
    from_learner_steps: int = 0
    # All policy probabilities below `threshold` are zeroed out. Thresholding
    # is disabled if this value is non-positive.
    policy_threshold: float = 0.05
    # Rounds the policy probabilities to the "closest"
    # multiple of 1/`self.discretization`.
    # Discretization is disabled for non-positive values.
    policy_discretization: int = 32

    def __call__(
        self, policy: torch.Tensor, mask: torch.Tensor, learner_steps: int
    ) -> torch.Tensor:
        """A configurable fine tuning of a policy."""
        assert policy.shape == mask.shape
        do_finetune = (
            self.from_learner_steps >= 0 and learner_steps > self.from_learner_steps
        )

        return torch.where(
            torch.tensor(do_finetune),
            self.post_process_policy(policy, mask),
            policy,
        )

    def post_process_policy(
        self,
        policy: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Unconditionally post process a given masked policy."""
        assert policy.shape == mask.shape
        policy = self._threshold(policy, mask)
        # policy = self._discretize(policy)
        return policy

    def _threshold(self, policy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Remove from the support the actions 'a' where policy(a) < threshold."""
        assert policy.shape == mask.shape
        if self.policy_threshold <= 0:
            return policy

        mask = mask * (
            # Values over the threshold.
            (policy >= self.policy_threshold)
            +
            # Degenerate case is when policy is less than threshold *everywhere*.
            # In that case we just keep the policy as-is.
            (torch.max(policy, dim=-1, keepdim=True).values < self.policy_threshold)
        )
        return mask * policy / torch.sum(mask * policy, dim=-1, keepdim=True)

    def _discretize(self, policy: torch.Tensor) -> torch.Tensor:
        """Round all action probabilities to a multiple of 1/self.discretize."""
        if self.policy_discretization <= 0:
            return policy

        # The unbatched/single policy case:
        if len(policy.shape) == 1:
            return self._discretize_single(policy)

        og_shape = policy.shape
        policy = policy.flatten(0, 1)

        for i in range(policy.shape[0]):
            policy[i] = self._discretize_single(policy[i])

        policy = policy.view(*og_shape)

        return policy

    def _discretize_single(self, mu: torch.Tensor) -> torch.Tensor:
        """A version of self._discretize but for the unbatched data."""
        # TODO(author18): try to merge _discretize and _discretize_single
        # into one function that handles both batched and unbatched cases.
        if len(mu.shape) == 2:
            mu_ = torch.squeeze(mu, dim=0)
        else:
            mu_ = mu
        n_actions = mu_.shape[-1]
        roundup = torch.ceil(mu_ * self.policy_discretization).to(torch.int32)
        result = torch.zeros_like(mu_)
        order = torch.argsort(-mu_)  # Indices of descending order.
        weight_left = torch.tensor(self.policy_discretization, dtype=torch.float)

        def f_disc(i, order, roundup, weight_left, result):
            x = torch.minimum(roundup[order[i]], weight_left).to(torch.float)
            result = torch.where(
                weight_left >= 0, result.scatter_add_(0, order[i], x), result
            )
            weight_left -= x
            return i + 1, order, roundup, weight_left, result

        def f_scan_scan(carry, x):
            i, order, roundup, weight_left, result = carry
            i_next, order_next, roundup_next, weight_left_next, result_next = f_disc(
                i, order, roundup, weight_left, result
            )
            carry_next = (
                i_next,
                order_next,
                roundup_next,
                weight_left_next,
                result_next,
            )
            return carry_next, x

        (_, _, _, weight_left_next, result_next), _ = pytorch_scan(
            f_scan_scan,
            init=(torch.asarray(0), order, roundup, weight_left, result),
            xs=None,
            length=n_actions,
        )

        result_next = torch.where(
            weight_left_next > 0,
            result_next.scatter_add_(0, order[0], weight_left_next),
            result_next,
        )
        if len(mu.shape) == 2:
            result_next = torch.unsqueeze(result_next, dim=0)
        return result_next / self.policy_discretization
