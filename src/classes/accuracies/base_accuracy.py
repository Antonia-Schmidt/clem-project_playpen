from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseAccuracy(ABC):
    @abstractmethod
    def calculate(self) -> None:
        pass
