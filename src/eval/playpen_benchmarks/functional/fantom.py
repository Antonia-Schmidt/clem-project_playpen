from ..benchmark import Benchmark
from src.eval.backends.base_model import Model
from src.eval.playpen_benchmarks.implementations.fantom.eval_fantom import FantomEvalAgent

class FantomBenchmark(Benchmark):
    benchmark_name: str

    def evaluate(self, model: Model):
        args = {
            'model': model
        }
        benchmark = FantomEvalAgent(args)
        benchmark.run()