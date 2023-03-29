from benchmarks.base import IBenchmark


class BenchmarkRunner:
    def __init__(self, benchmarks: list[IBenchmark]):
        self.benchmarks = benchmarks

    def run(self):
        for benchmark in self.benchmarks:
            benchmark.run()
