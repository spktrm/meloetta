import numpy as np

from typing import Sequence, Tuple


class EntropySchedule:
    """An increasing list of steps where the regularisation network is updated.
    Example
      EntropySchedule([3, 5, 10], [2, 4, 1])
      =>   [0, 3, 6, 11, 16, 21, 26, 10]
            | 3 x2 |      5 x4     | 10 x1
    """

    def __init__(
        self,
        *,
        sizes: Sequence[int] = (100,),
        repeats: Sequence[int] = (1,),
    ):
        """Constructs a schedule of entropy iterations.
        Args:
          sizes: the list of iteration sizes.
          repeats: the list, parallel to sizes, with the number of times for each
            size from `sizes` to repeat.
        """
        try:
            if len(repeats) != len(sizes):
                raise ValueError("`repeats` must be parallel to `sizes`.")
            if not sizes:
                raise ValueError("`sizes` and `repeats` must not be empty.")
            if any([(repeat <= 0) for repeat in repeats]):
                raise ValueError("All repeat values must be strictly positive")
            if repeats[-1] != 1:
                raise ValueError(
                    "The last value in `repeats` must be equal to 1, "
                    "ince the last iteration size is repeated forever."
                )
        except ValueError as e:
            raise ValueError(
                f"Entropy iteration schedule: repeats ({repeats}) and sizes ({sizes})."
            ) from e

        schedule = [0]
        for size, repeat in zip(sizes, repeats):
            schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])

        self.schedule = np.array(schedule, dtype=np.int32)

    def __call__(self, learner_step: int) -> Tuple[float, bool]:
        """Entropy scheduling parameters for a given `learner_step`.
        Args:
          learner_step: The current learning step.
        Returns:
          alpha: The mixing weight (from [0, 1]) of the previous policy with
            the one before for computing the intrinsic reward.
          update_target_net: A boolean indicator for updating the target network
            with the current network.
        """

        # The complexity below is because at some point we might go past
        # the explicit schedule, and then we'd need to just use the last step
        # in the schedule and apply the logic of
        # ((learner_step - last_step) % last_iteration) == 0)

        # The schedule might look like this:
        # X----X-------X--X--X--X--------X
        # learner_step | might be here ^    |
        # or there     ^                    |
        # or even past the schedule         ^

        # We need to deal with two cases below.
        # Instead of going for the complicated conditional, let's just
        # compute both and then do the A * s + B * (1 - s) with s being a bool
        # selector between A and B.

        # 1. assume learner_step is past the schedule,
        #    ie schedule[-1] <= learner_step.
        last_size = self.schedule[-1] - self.schedule[-2]
        last_start = (
            self.schedule[-1]
            + (learner_step - self.schedule[-1]) // last_size * last_size
        )
        # 2. assume learner_step is within the schedule.
        start = np.amax(self.schedule * (self.schedule <= learner_step))
        finish = np.amin(
            self.schedule * (learner_step < self.schedule),
            initial=self.schedule[-1],
            where=(learner_step < self.schedule),
        )
        size = finish - start

        # Now select between the two.
        beyond = self.schedule[-1] <= learner_step  # Are we past the schedule?
        iteration_start = last_start * beyond + start * (1 - beyond)
        iteration_size = last_size * beyond + size * (1 - beyond)

        update_target_net = np.logical_and(
            learner_step > 0, np.sum(learner_step == iteration_start)
        )
        alpha = np.minimum(
            (2.0 * (learner_step - iteration_start)) / iteration_size, 1.0
        )

        return alpha, update_target_net
