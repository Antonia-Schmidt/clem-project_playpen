import json
import os
from typing import List

from src.eval.playpen_benchmarks import functional
from src.eval.playpen_benchmarks.benchmark import Benchmark

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_benchmarks(benchmark_names: List[str]) -> List[Benchmark]:
    benchmarks = []
    benchmark_registry_path = os.path.join(
        project_root, "playpen_benchmarks", "benchmark_registry.json"
    )
    benchmark_registry = json.load(open(benchmark_registry_path))

    if len(benchmark_names) == 1 and benchmark_names[0] == "all":
        benchmark_names = [b["benchmark_name"] for b in benchmark_registry]
    for name in benchmark_names:
        benchmark_entry = next(
            (
                entry
                for entry in benchmark_registry
                if entry.get("benchmark_name") == name
            ),
            None,
        )
        b_cls = getattr(
            functional, benchmark_entry["benchmark_class"]
        )  # Creates an instance of my_module.MyClass
        benchmarks.append(b_cls)
    return benchmarks


def run_benchmarks(model, benchmarks):
    pass
