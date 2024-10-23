from dataclasses import dataclass

from ..benchmark import Benchmark






@dataclass
class EwokBenchmark(Benchmark):

    def evaluate(self, llm) -> tuple:
        pass