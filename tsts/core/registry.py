from typing import Any, Callable, Dict

__all__ = ["MODELS"]


class Registry(object):
    """Table to access to registered classes."""

    def __init__(self) -> None:
        self._cls_table: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self._cls_table)

    def __contains__(self, key: str) -> bool:
        return self._cls_table.get(key, None) is not None

    def __getitem__(self, key: str) -> Any:
        return self._cls_table[key]

    def register(self) -> Callable:
        """Add new class to the table.

        Returns
        -------
        Callable
            Original class
        """

        def wrapper(cls: Any) -> Any:
            name = cls.__name__
            self._cls_table[name] = cls
            return cls

        return wrapper


MODELS = Registry()