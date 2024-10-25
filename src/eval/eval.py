import argparse
import json
from playpen_benchmarks import find_benchmark, run_benchmarks
from backends import get_model


def main(args: argparse.Namespace):
    model_name = args.model
    benchmark_names = args.benchmarks

    model = get_model(model_name)
    benchmarks = get_benchmarks(benchmark_names)
    run_benchmarks(model, benchmarks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-b", "--benchmarks", type=str, nargs="*", help="Value can be 'all'")
    main(parser.parse_args())