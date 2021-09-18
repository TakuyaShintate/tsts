from torch.optim import Optimizer as _Optimizer

__all__ = ["Optimizer"]


class Optimizer(_Optimizer):
    @property
    def require_second_step(self) -> bool:
        return False
