import argparse
import json
from playpen_benchmarks import benchmark_list, find_benchmark, run_benchmarks


def main(args: argparse.Namespace):
    model_name = args.model
    benchmark_names = args.benchmarks

    model = get_model(model_name)

    if model_name not in model_list:
        return Exception(f"{model_name} is not in the list of available models.")
    if benchmark_names == 'all':
        benchmarks = [find_benchmark(name) for name in benchmark_list]
    else:
        benchmarks = [find_benchmark(name) for name in benchmark_names]
    run_benchmarks(model_name, benchmarks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-b", "--benchmarks", type=str, nargs="*", help="Value can be 'all'")
    main(parser.parse_args())