from src.eval.backends.base_model import Model
from src.eval.benchmarks.implementations.fantom.eval_fantom import FantomEvalAgent

from ..benchmark import Benchmark


class FantomBenchmark(Benchmark):
    benchmark_name: str = "fantom"

    def evaluate(self, model: Model):
        args = {"model": model}
        benchmark = FantomEvalAgent(args)
        benchmark.run()
