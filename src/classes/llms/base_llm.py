from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseLlm(ABC):
    @abstractmethod
    def generate(self):
        pass
