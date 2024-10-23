from .benchmark import Benchmark

__all__ = ["Benchmark"]

benchmark_list = ['ewok-core/ewok-core-1.0']

def find_benchmark(benchmark_name: str):
    for b_cls in Benchmark.__subclasses__():
        b = b_cls()  # subclasses should only get the dialog_pair
        if b.applies_to(benchmark_name):
            return b
    raise NotImplementedError("No game benchmark for:", benchmark_name)

def run_benchmarks(model_name, benchmarks):
    pass