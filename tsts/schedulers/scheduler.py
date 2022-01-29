from tsts.optimizers import Optimizer

__all__ = ["Scheduler"]


class Scheduler(object):
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        warmup_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self._reset_internal_state()

    @property
    def current_lr(self) -> float:
        num_groups = len(self.optimizer.param_groups)
        mean_lr = 0.0
        for i in range(num_groups):
            mean_lr += self.optimizer.param_groups[i]["lr"]
        mean_lr /= num_groups
        return mean_lr

    def _reset_internal_state(self) -> None:
        pass

    def step(self) -> None:
        raise NotImplementedError
