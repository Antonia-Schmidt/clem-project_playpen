from dataclasses import dataclass

from ..benchmark import Benchmark

@dataclass
class BbhBenchmark(Benchmark):
    
    def evaluate(self, llm) -> tuple:
        pass