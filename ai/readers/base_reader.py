from abc import abstractmethod
from typing import Any, Protocol


class BaseReader(Protocol):
    @abstractmethod
    def read_char(self, img: Any) -> str: ...
