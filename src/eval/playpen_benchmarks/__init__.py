from typing import List
import os
import json
from .benchmark import Benchmark


__all__ = ["Benchmark"]

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_benchmarks(benchmark_names: List[str]) -> List[Benchmark]:
    pass
    # TODO

def run_benchmarks(model, benchmarks):
    pass