from abc import ABC, abstractmethod
from dataclasses import dataclass

from datasets import load_dataset

from src.eval.backends.base_model import Model

class Benchmark(ABC):
    def __init__(self, benchmark_name: str, benchmark_id:str):
        self.benchmark_name = benchmark_name
        self.benchmark_id = benchmark_id

    @abstractmethod
    def evaluate(self, model:Model):
        pass

    def get_name(self):
        return self.benchmark_name


class HuggingfaceBenchmark(Benchmark, ABC):
    def __init__(self, benchmark_name:str, benchmark_id:str):
        super().__init__(benchmark_name, benchmark_id)
        self.benchmark = load_dataset(self.benchmark_id)



