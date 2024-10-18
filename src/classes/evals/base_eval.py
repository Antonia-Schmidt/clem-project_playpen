from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..accuracies.fine_grained_accuracy import FineGrainedAccuracy
from ..accuracies.overall_accuracy import OverallAccuracy
from ..llms.base_llm import BaseLlm


@dataclass
class BaseEval(ABC):

    @abstractmethod
    def evaluate(self, llm: BaseLlm) -> tuple[OverallAccuracy, FineGrainedAccuracy]:
        pass
