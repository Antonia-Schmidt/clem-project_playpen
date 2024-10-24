from abc import ABC, abstractmethod
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class Benchmark(ABC):
    def __init__(self, benchmark_name):
        self.benchmark_name = benchmark_name
    """@abstractmethod
    def evaluate(self, llm: BaseLlm) -> tuple[OverallAccuracy, FineGrainedAccuracy]:
        pass"""

class HuggingfaceBenchmark(Benchmark):
    def __init__(self, benchmark_name):
        super().__init__(benchmark_name)
        self.benchmark = load_dataset(self.benchmark_name)



