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

    def _reset_internal_state(self) -> None:
        self.T = 1.0

    def warmup(self) -> bool:
        if self.warmup_steps > 0 and self.warmup_steps + 1.0 >= self.T:
            lr = self.base_lr * (self.T / (self.warmup_steps + 1.0))
            for i in range(len(self.optimizer.param_groups)):
                self.optimizer.param_groups[i]["lr"] = lr
            return True
        return False

    def step(self) -> None:
        raise NotImplementedError
