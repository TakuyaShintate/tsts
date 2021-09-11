from typing import Any, Dict

__all__ = ["ContextManager"]


class ContextManager(object):
    def __init__(self) -> None:
        self._init_context()

    def _init_context(self) -> None:
        self._context: Dict[str, Any] = {}

    def __setitem__(self, k: str, v: Any) -> None:
        self._context[k] = v

    def __getitem__(self, k: str) -> Any:
        return self._context[k]

    def __contains__(self, k: str) -> bool:
        return self._context.get(k, None) is not None
