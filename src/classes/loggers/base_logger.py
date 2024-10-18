from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseLogger(ABC):
    @abstractmethod
    def log(self) -> None:
        pass
