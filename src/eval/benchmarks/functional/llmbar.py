from dataclasses import dataclass

from ..benchmark import HuggingfaceBenchmark


@dataclass
class LLMBarBenchmark(HuggingfaceBenchmark):

    def evaluate(self, model):
        pass