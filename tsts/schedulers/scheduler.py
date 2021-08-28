__all__ = ["Scheduler"]


class Scheduler(object):
    def step(self) -> None:
        raise NotImplementedError
