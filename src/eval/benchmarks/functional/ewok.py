from dataclasses import dataclass

from ..benchmark import HuggingfaceBenchmark


@dataclass
class EwokBenchmark(HuggingfaceBenchmark):
    benchmark_name: str = 'ewok'
    benchmark_id: str = "ewok-core/ewok-core-1.0"


    def evaluate(self, model):
        pass